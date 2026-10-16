#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include "coreml-block.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>

namespace tts_cpp::minimax::detail {

struct CoreMLBlock {
    MLModel * model = nil;
    CoreMLBlockConfig config;
    mutable std::mutex mutex;
    std::map<std::string, MLMultiArray *> arrays;
    std::string error;
    std::atomic<uint64_t> predictions{0};
    std::map<std::tuple<int64_t, int64_t, int64_t>, bool> validated_shapes;
};

namespace {

constexpr size_t HASH_CHUNK_BYTES = 1024 * 1024;
constexpr const char * SCHEMA_VERSION = "1";

NSString * ns_string(const std::string & value) {
    return [NSString stringWithUTF8String:value.c_str()];
}

std::string error_message(NSError * error) {
    return error ? std::string(error.localizedDescription.UTF8String) : "Core ML returned no result";
}

NSArray<NSNumber *> * array_shape(const std::vector<int64_t> & shape) {
    NSMutableArray<NSNumber *> * result = [NSMutableArray array];
    for (const auto value : shape) [result addObject:@(value)];
    return result;
}

bool shape_matches(MLMultiArray * array, const std::vector<int64_t> & shape) {
    if (array.shape.count != shape.size()) return false;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (array.shape[i].longLongValue != shape[i]) return false;
    }
    return true;
}

bool supports_shape(MLFeatureDescription * feature, const std::vector<int64_t> & shape) {
    if (!feature || feature.type != MLFeatureTypeMultiArray) return false;
    auto * constraint = feature.multiArrayConstraint;
    if (constraint.dataType != MLMultiArrayDataTypeFloat32 || constraint.shape.count != shape.size()) return false;
    auto * shape_constraint = constraint.shapeConstraint;
    if (shape_constraint.type == MLMultiArrayShapeConstraintTypeRange) {
        for (size_t i = 0; i < shape.size(); ++i) {
            NSRange range = shape_constraint.sizeRangeForDimension[i].rangeValue;
            if (shape[i] <= 0 || static_cast<uint64_t>(shape[i]) < range.location ||
                static_cast<uint64_t>(shape[i]) - range.location >= range.length) return false;
        }
        return true;
    }
    if (shape_constraint.type == MLMultiArrayShapeConstraintTypeEnumerated) {
        for (NSArray<NSNumber *> * candidate in shape_constraint.enumeratedShapes) {
            if ([candidate isEqualToArray:array_shape(shape)]) return true;
        }
        return false;
    }
    return [constraint.shape isEqualToArray:array_shape(shape)];
}

size_t array_offset(MLMultiArray * array, size_t flat) {
    size_t offset = 0;
    for (size_t i = array.shape.count; i-- > 0;) {
        const size_t extent = array.shape[i].unsignedLongLongValue;
        offset += (flat % extent) * array.strides[i].unsignedLongLongValue;
        flat /= extent;
    }
    return offset;
}

bool contiguous_array(MLMultiArray * array) {
    size_t stride = 1;
    for (size_t i = array.shape.count; i-- > 0;) {
        if (array.strides[i].unsignedLongLongValue != stride) return false;
        stride *= array.shape[i].unsignedLongLongValue;
    }
    return true;
}

bool copy_output(MLMultiArray * array, CoreMLArray & output) {
    if (!array || array.dataType != MLMultiArrayDataTypeFloat32 || !shape_matches(array, output.shape)) return false;
    output.values.resize(static_cast<size_t>(array.count));
    const auto * data = static_cast<const float *>(array.dataPointer);
    const bool contiguous = contiguous_array(array);
    for (size_t i = 0; i < output.values.size(); ++i) {
        const float value = data[contiguous ? i : array_offset(array, i)];
        if (!std::isfinite(value)) return false;
        output.values[i] = value;
    }
    return true;
}

bool copy_outputs(id<MLFeatureProvider> result, std::vector<CoreMLArray> & outputs) {
    for (auto & output : outputs) {
        if (!copy_output([result featureValueForName:ns_string(output.name)].multiArrayValue, output)) return false;
    }
    return true;
}

MLMultiArray * input_array(CoreMLBlock & block, const CoreMLArray & input) {
    MLMultiArray * array = block.arrays[input.name];
    if (array && shape_matches(array, input.shape)) return array;
    NSError * error = nil;
    array = [[MLMultiArray alloc] initWithShape:array_shape(input.shape) dataType:MLMultiArrayDataTypeFloat32 error:&error];
    if (!array || error) {
        block.error = error_message(error);
        return nil;
    }
    block.arrays[input.name] = array;
    return array;
}

MLDictionaryFeatureProvider * make_provider(CoreMLBlock & block, const std::vector<CoreMLArray> & inputs) {
    NSMutableDictionary<NSString *, id> * dictionary = [NSMutableDictionary dictionary];
    for (const auto & input : inputs) {
        auto * feature = block.model.modelDescription.inputDescriptionsByName[ns_string(input.name)];
        if (!supports_shape(feature, input.shape)) {
            block.error = "incompatible Core ML input: " + input.name;
            return nil;
        }
        MLMultiArray * array = input_array(block, input);
        if (!array || static_cast<size_t>(array.count) != input.values.size()) return nil;
        auto * data = static_cast<float *>(array.dataPointer);
        if (contiguous_array(array)) {
            std::memcpy(data, input.values.data(), input.values.size() * sizeof(float));
        } else {
            for (size_t i = 0; i < input.values.size(); ++i) data[array_offset(array, i)] = input.values[i];
        }
        dictionary[ns_string(input.name)] = array;
    }
    NSError * error = nil;
    auto * provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:dictionary error:&error];
    if (!provider || error) block.error = error_message(error);
    return provider;
}

bool metadata_matches(MLModel * model, const std::string & fingerprint, const CoreMLBlockConfig & config) {
    NSDictionary<NSString *, NSString *> * metadata = model.modelDescription.metadata[MLModelCreatorDefinedKey];
    const std::map<std::string, std::string> expected = {
        {"mm3_schema", SCHEMA_VERSION}, {"mm3_source_sha256", fingerprint}, {"mm3_stage", config.stage},
        {"mm3_layer", std::to_string(config.layer)}, {"mm3_hidden", std::to_string(config.hidden)},
        {"mm3_heads", std::to_string(config.heads)}, {"mm3_kv_heads", std::to_string(config.kv_heads)},
        {"mm3_head_dim", std::to_string(config.head_dim)},
    };
    for (const auto & item : expected) {
        if (![metadata[ns_string(item.first)] isEqualToString:ns_string(item.second)]) return false;
    }
    return !fingerprint.empty();
}

}

std::string coreml_source_fingerprint(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    CC_SHA256_CTX hash;
    CC_SHA256_Init(&hash);
    std::vector<char> chunk(HASH_CHUNK_BYTES);
    while (input) {
        input.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        CC_SHA256_Update(&hash, chunk.data(), static_cast<CC_LONG>(input.gcount()));
    }
    if (!input.eof()) return {};
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &hash);
    char text[CC_SHA256_DIGEST_LENGTH * 2 + 1];
    for (size_t i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) std::snprintf(text + i * 2, 3, "%02x", digest[i]);
    return text;
}

std::shared_ptr<CoreMLBlock> load_coreml_block(const std::string & path, const std::string & fingerprint,
                                            const CoreMLBlockConfig & config) {
    @autoreleasepool {
        if (![[NSFileManager defaultManager] fileExistsAtPath:ns_string(path)]) return {};
        MLModelConfiguration * policy = [[MLModelConfiguration alloc] init];
        policy.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        NSError * error = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:ns_string(path)] configuration:policy error:&error];
        if (!model || error || !metadata_matches(model, fingerprint, config)) {
            std::fprintf(stderr, "[MM3-CoreML] Using ggml for %s: %s\n", path.c_str(), error ? error.localizedDescription.UTF8String : "incompatible model metadata");
            return {};
        }
        auto block = std::make_shared<CoreMLBlock>();
        block->model = model;
        block->config = config;
        return block;
    }
}

namespace {

bool output_schema_matches(CoreMLBlock & block) {
    auto * outputs = block.model.modelDescription.outputDescriptionsByName;
    const std::vector<std::string> names = block.config.stage == "dit"
        ? std::vector<std::string>{"hidden_out"} : std::vector<std::string>{"hidden_out", "new_keys", "new_values"};
    if (outputs.count != names.size()) return false;
    for (const auto & name : names) {
        MLFeatureDescription * feature = outputs[ns_string(name)];
        if (!feature || feature.type != MLFeatureTypeMultiArray ||
            feature.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat32) return false;
    }
    return true;
}

CoreMLArray zero_array(const char * name, const std::vector<int64_t> & shape) {
    size_t count = 1;
    for (const auto value : shape) count *= static_cast<size_t>(value);
    return {name, shape, std::vector<float>(count, 0.0f)};
}

bool probe_output_shapes(CoreMLBlock & block, int64_t tokens, int64_t cache, int64_t batch) {
    const auto & config = block.config;
    std::vector<CoreMLArray> inputs = {zero_array("hidden", {batch, tokens, config.hidden}), zero_array("positions", {tokens})};
    for (int64_t i = 0; i < tokens; ++i) inputs[1].values[static_cast<size_t>(i)] = static_cast<float>(i);
    std::vector<CoreMLArray> outputs = {{"hidden_out", inputs[0].shape, {}}};
    if (config.stage != "dit") {
        if (tokens > cache) return false;
        inputs.push_back(zero_array("keys", {batch, config.kv_heads, cache, config.head_dim}));
        inputs.push_back(zero_array("values", inputs.back().shape));
        inputs.push_back(zero_array("mask", {1, 1, tokens, cache}));
        const std::vector<int64_t> shape = {batch, tokens, config.kv_heads, config.head_dim};
        outputs.push_back({"new_keys", shape, {}});
        outputs.push_back({"new_values", shape, {}});
    }
    const bool valid = coreml_block_predict(block, inputs, outputs);
    if (valid) --block.predictions;
    coreml_block_set_error(block, {});
    return valid;
}

bool input_shapes_match(CoreMLBlock & block, int64_t tokens, int64_t cache, int64_t batch) {
    auto * features = block.model.modelDescription.inputDescriptionsByName;
    if (!supports_shape(features[@"hidden"], {batch, tokens, block.config.hidden}) ||
        !supports_shape(features[@"positions"], {tokens})) return false;
    if (block.config.stage == "dit") return features.count == 2;
    const std::vector<int64_t> shape = {batch, block.config.kv_heads, cache, block.config.head_dim};
    return features.count == 5 && supports_shape(features[@"keys"], shape) && supports_shape(features[@"values"], shape) &&
           supports_shape(features[@"mask"], {1, 1, tokens, cache});
}

}

bool coreml_block_supports(const std::shared_ptr<CoreMLBlock> & block, int64_t tokens, int64_t cache, int64_t batch) {
    if (!block || !input_shapes_match(*block, tokens, cache, batch) || !output_schema_matches(*block)) return false;
    const auto key = std::make_tuple(tokens, cache, batch);
    {
        std::lock_guard<std::mutex> lock(block->mutex);
        const auto found = block->validated_shapes.find(key);
        if (found != block->validated_shapes.end()) return found->second;
    }
    const bool valid = probe_output_shapes(*block, tokens, cache, batch);
    std::lock_guard<std::mutex> lock(block->mutex);
    block->validated_shapes[key] = valid;
    return valid;
}

bool coreml_block_predict(CoreMLBlock & block, const std::vector<CoreMLArray> & inputs, std::vector<CoreMLArray> & outputs) {
    std::lock_guard<std::mutex> lock(block.mutex);
    @autoreleasepool {
        block.error.clear();
        auto * provider = make_provider(block, inputs);
        if (!provider) {
            if (block.error.empty()) block.error = "invalid Core ML input buffer";
            return false;
        }
        NSError * error = nil;
        id<MLFeatureProvider> result = [block.model predictionFromFeatures:provider error:&error];
        if (!result || error || !copy_outputs(result, outputs)) {
            block.error = error ? error_message(error) : "invalid or non-finite Core ML output";
            return false;
        }
        ++block.predictions;
        return true;
    }
}

void coreml_block_set_error(CoreMLBlock & block, const std::string & error) {
    std::lock_guard<std::mutex> lock(block.mutex);
    block.error = error;
}

std::string coreml_block_error(const std::shared_ptr<CoreMLBlock> & block) {
    if (!block) return {};
    std::lock_guard<std::mutex> lock(block->mutex);
    return block->error;
}

uint64_t coreml_block_predictions(const std::shared_ptr<CoreMLBlock> & block) {
    return block ? block->predictions.load() : 0;
}

}

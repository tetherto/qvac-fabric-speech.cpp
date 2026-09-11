// Core ML Oobleck VAE decoder sidecar. See vae-decoder.h for the contract.

#if !defined(__APPLE__)
#error "vae-decoder.mm is Apple-only; compile it solely under AUDIOGEN_USE_COREML"
#endif

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "vae-decoder.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int64_t kLatentChannels = 64;
constexpr int64_t kPcmChannels    = 2;
constexpr int64_t kUpsample       = 1920;

float decode_float16(uint16_t h) {
    __fp16 v;
    std::memcpy(&v, &h, sizeof(v));
    return (float) v;
}

uint16_t encode_float16(float f) {
    __fp16 v = (__fp16) f;
    uint16_t out;
    std::memcpy(&out, &v, sizeof(out));
    return out;
}

MLComputeUnits requested_compute_units(std::string * label) {
    const char * env = std::getenv("ACESTEP_COREML_COMPUTE_UNITS");
    const std::string requested = env != nullptr ? env : "";
    if (requested == "cpu_only") {
        *label = "coreml-cpu";
        return MLComputeUnitsCPUOnly;
    }
    if (requested == "cpu_and_gpu") {
        *label = "coreml-gpu";
        return MLComputeUnitsCPUAndGPU;
    }
    if (requested == "cpu_and_ane") {
        *label = "coreml-ane";
        return MLComputeUnitsCPUAndNeuralEngine;
    }
    *label = "coreml-all";
    return MLComputeUnitsAll;
}

NSString * sole_multiarray_feature(NSDictionary<NSString *, MLFeatureDescription *> * features) {
    NSString * found = nil;
    for (NSString * name in features) {
        if (features[name].type != MLFeatureTypeMultiArray) continue;
        if (found != nil) return nil;
        found = name;
    }
    return found;
}

std::vector<int64_t> dims_of(NSArray<NSNumber *> * shape) {
    std::vector<int64_t> dims;
    for (NSNumber * n in shape) dims.push_back(n.longLongValue);
    return dims;
}

bool supported_data_type(MLMultiArrayDataType dt) {
    return dt == MLMultiArrayDataTypeFloat32 || dt == MLMultiArrayDataTypeFloat16;
}

// {1, ..., channels, frames} with every leading dim 1.
bool shape_is(const std::vector<int64_t> & dims, int64_t channels, int64_t frames) {
    if (dims.size() < 2) return false;
    for (std::size_t i = 0; i + 2 < dims.size(); ++i) {
        if (dims[i] != 1) return false;
    }
    return dims[dims.size() - 2] == channels && dims[dims.size() - 1] == frames;
}

bool fill_latent_array(MLMultiArray * arr, const float * latent, int64_t frames) {
    if (!supported_data_type(arr.dataType)) return false;
    const int64_t stride_c = arr.strides[arr.strides.count - 2].longLongValue;
    const int64_t stride_t = arr.strides[arr.strides.count - 1].longLongValue;
    void * base = arr.dataPointer;
    for (int64_t c = 0; c < kLatentChannels; ++c) {
        for (int64_t t = 0; t < frames; ++t) {
            const float   v   = latent[t * kLatentChannels + c];
            const int64_t off = c * stride_c + t * stride_t;
            if (arr.dataType == MLMultiArrayDataTypeFloat32) {
                ((float *) base)[off] = v;
            } else {
                ((uint16_t *) base)[off] = encode_float16(v);
            }
        }
    }
    return true;
}

bool copy_pcm_array(MLMultiArray * arr, float * pcm, int64_t t_audio) {
    if (!supported_data_type(arr.dataType)) return false;
    const int64_t stride_c = arr.strides[arr.strides.count - 2].longLongValue;
    const int64_t stride_t = arr.strides[arr.strides.count - 1].longLongValue;
    const void * base = arr.dataPointer;
    for (int64_t c = 0; c < kPcmChannels; ++c) {
        for (int64_t t = 0; t < t_audio; ++t) {
            const int64_t off = c * stride_c + t * stride_t;
            const float   v   = arr.dataType == MLMultiArrayDataTypeFloat32
                                    ? ((const float *) base)[off]
                                    : decode_float16(((const uint16_t *) base)[off]);
            pcm[t * kPcmChannels + c] = v;
        }
    }
    return true;
}

}  // namespace

struct acestep_coreml_vae_context {
    const void * model              = nullptr;  // CFBridgingRetain'd MLModel *
    const void * input_array        = nullptr;  // retained MLMultiArray *
    const void * input_provider     = nullptr;  // retained MLDictionaryFeatureProvider *
    const void * output_array       = nullptr;  // retained MLMultiArray *
    const void * prediction_options = nullptr;  // retained MLPredictionOptions *
    std::string  output_name;
    std::string  label;
    int64_t      window_frames = 0;
    std::mutex   mutex;
};

struct acestep_coreml_vae_context * acestep_coreml_vae_init(const char * path_mlmodelc) {
    if (path_mlmodelc == nullptr) return nullptr;

    @autoreleasepool {
        NSString * path = [[NSString alloc] initWithUTF8String:path_mlmodelc];
        if (path == nil) return nullptr;

        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        std::string label;
        config.computeUnits = requested_compute_units(&label);

        NSError * err   = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                            configuration:config
                                                    error:&err];
        if (model == nil || err != nil) return nullptr;

        NSString * in_name  = sole_multiarray_feature(model.modelDescription.inputDescriptionsByName);
        NSString * out_name = sole_multiarray_feature(model.modelDescription.outputDescriptionsByName);
        if (in_name == nil || out_name == nil) return nullptr;

        MLMultiArrayConstraint * in_constraint =
            model.modelDescription.inputDescriptionsByName[in_name].multiArrayConstraint;
        MLMultiArrayConstraint * out_constraint =
            model.modelDescription.outputDescriptionsByName[out_name].multiArrayConstraint;
        if (in_constraint == nil || out_constraint == nil) return nullptr;
        if (!supported_data_type(in_constraint.dataType) ||
            !supported_data_type(out_constraint.dataType)) return nullptr;

        const std::vector<int64_t> in_dims  = dims_of(in_constraint.shape);
        const std::vector<int64_t> out_dims = dims_of(out_constraint.shape);
        if (in_dims.size() < 2 || in_dims.back() <= 0) return nullptr;
        const int64_t frames = in_dims.back();
        if (!shape_is(in_dims, kLatentChannels, frames)) return nullptr;
        if (!shape_is(out_dims, kPcmChannels, frames * kUpsample)) return nullptr;

        MLMultiArray * in_arr =
            [[MLMultiArray alloc] initWithShape:in_constraint.shape
                                       dataType:in_constraint.dataType
                                          error:&err];
        if (in_arr == nil || err != nil) return nullptr;
        MLDictionaryFeatureProvider * provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ in_name : in_arr } error:&err];
        if (provider == nil || err != nil) return nullptr;

        MLMultiArray * out_arr =
            [[MLMultiArray alloc] initWithShape:out_constraint.shape
                                       dataType:out_constraint.dataType
                                          error:&err];
        if (out_arr == nil || err != nil) return nullptr;
        MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
        options.outputBackings = @{ out_name : out_arr };

        auto * ctx = new acestep_coreml_vae_context();
        ctx->model              = CFBridgingRetain(model);
        ctx->input_array        = CFBridgingRetain(in_arr);
        ctx->input_provider     = CFBridgingRetain(provider);
        ctx->output_array       = CFBridgingRetain(out_arr);
        ctx->prediction_options = CFBridgingRetain(options);
        ctx->output_name        = out_name.UTF8String;
        ctx->label              = std::move(label);
        ctx->window_frames      = frames;
        return ctx;
    }
}

void acestep_coreml_vae_free(struct acestep_coreml_vae_context * ctx) {
    if (ctx == nullptr) return;
    if (ctx->prediction_options != nullptr) CFRelease(ctx->prediction_options);
    if (ctx->output_array != nullptr) CFRelease(ctx->output_array);
    if (ctx->input_provider != nullptr) CFRelease(ctx->input_provider);
    if (ctx->input_array != nullptr) CFRelease(ctx->input_array);
    if (ctx->model != nullptr) CFRelease(ctx->model);
    delete ctx;
}

int64_t acestep_coreml_vae_window_frames(const struct acestep_coreml_vae_context * ctx) {
    return ctx != nullptr ? ctx->window_frames : 0;
}

int acestep_coreml_vae_decode(struct acestep_coreml_vae_context * ctx,
                              const float * latent,
                              float       * pcm) {
    if (ctx == nullptr || ctx->model == nullptr || latent == nullptr || pcm == nullptr) return 1;

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        MLMultiArray * in_arr = (__bridge MLMultiArray *) ctx->input_array;
        if (!fill_latent_array(in_arr, latent, ctx->window_frames)) return 2;

        NSError * err = nil;
        MLModel * model = (__bridge MLModel *) ctx->model;
        MLDictionaryFeatureProvider * provider =
            (__bridge MLDictionaryFeatureProvider *) ctx->input_provider;
        MLPredictionOptions * options = (__bridge MLPredictionOptions *) ctx->prediction_options;
        id<MLFeatureProvider> result = [model predictionFromFeatures:provider options:options error:&err];
        if (result == nil || err != nil) return 3;

        NSString * out_name = [NSString stringWithUTF8String:ctx->output_name.c_str()];
        MLMultiArray * out_arr = [result featureValueForName:out_name].multiArrayValue;
        if (out_arr == nil) return 4;
        if (!copy_pcm_array(out_arr, pcm, ctx->window_frames * kUpsample)) return 5;
        return 0;
    }
}

const char * acestep_coreml_vae_backend_label(const struct acestep_coreml_vae_context * ctx) {
    return ctx != nullptr ? ctx->label.c_str() : "coreml";
}

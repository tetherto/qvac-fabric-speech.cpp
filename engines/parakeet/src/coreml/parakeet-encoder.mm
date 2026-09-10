// Core ML FastConformer encoder sidecar for parakeet-cpp.
//
// Derived from the whisper.cpp Core ML wrapper pattern (src/coreml/whisper-encoder.mm),
// generalised to load an arbitrary compiled encoder via the generic MLModel API so it
// works with whatever `.mlmodelc` the mobius export produces (no code-generated model
// interface required). Tensor orientation is discovered from the model description, so
// the export only has to agree on dimension sizes, not their order.

#if !__has_feature(objc_arc)
#error "parakeet-encoder.mm must be compiled with -fobjc-arc"
#endif

#import "coreml/parakeet-encoder.h"
#include "coreml/parakeet_coreml_shape.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

float decode_float16(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int      shift = 0;
            uint32_t m     = mant;
            do { m <<= 1; ++shift; } while ((m & 0x400u) == 0);
            bits = sign | ((uint32_t) (127 - 15 - shift + 1) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t encode_float16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exp  = (bits >> 23) & 0xffu;
    uint32_t       mant = bits & 0x7fffffu;
    if (exp == 0xffu) {
        return (uint16_t) (sign | (mant == 0 ? 0x7c00u : 0x7e00u));
    }

    const int32_t half_exp = (int32_t) exp - 127 + 15;
    if (half_exp >= 31) return (uint16_t) (sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return (uint16_t) sign;
        mant |= 0x800000u;
        const uint32_t shift   = (uint32_t) (14 - half_exp);
        uint32_t       rounded = mant >> shift;
        const uint32_t halfway = 1u << (shift - 1);
        if ((mant & (halfway - 1)) != 0 || ((mant & halfway) != 0 && (rounded & 1u) != 0)) {
            ++rounded;
        }
        return (uint16_t) (sign | rounded);
    }

    uint32_t rounded = mant + 0xfffu + ((mant >> 13) & 1u);
    if ((rounded & 0x800000u) != 0) {
        rounded = 0;
        if (half_exp + 1 >= 31) return (uint16_t) (sign | 0x7c00u);
        return (uint16_t) (sign | ((uint32_t) (half_exp + 1) << 10));
    }
    return (uint16_t) (sign | ((uint32_t) half_exp << 10) | (rounded >> 13));
}

NSString * sole_multiarray_feature(NSDictionary<NSString *, MLFeatureDescription *> * descs) {
    NSString * found = nil;
    for (NSString * key in descs) {
        if (descs[key].type == MLFeatureTypeMultiArray) {
            if (found != nil) return nil;
            found = key;
        }
    }
    return found;
}

std::vector<int64_t> dims_of(NSArray<NSNumber *> * shape) {
    std::vector<int64_t> dims;
    if (shape != nil) {
        dims.reserve(shape.count);
        for (NSNumber * dim in shape) {
            dims.push_back(dim.longLongValue);
        }
    }
    return dims;
}

bool constraint_has_one_fixed_shape(MLMultiArrayConstraint * constraint) {
    if (constraint == nil) return false;

    MLMultiArrayShapeConstraint * shape_constraint = constraint.shapeConstraint;
    if (shape_constraint == nil) return false;

    if (shape_constraint.type == MLMultiArrayShapeConstraintTypeEnumerated) {
        return shape_constraint.enumeratedShapes.count == 1;
    }
    if (shape_constraint.type != MLMultiArrayShapeConstraintTypeRange) {
        return false;
    }

    NSArray<NSValue *> * ranges = shape_constraint.sizeRangeForDimension;
    if (ranges.count != constraint.shape.count) return false;
    for (NSValue * value in ranges) {
        if (value.rangeValue.length != 1) return false;
    }
    return true;
}

// Resolves the concrete input shape to allocate and whether it is feature-major.
// Thin Objective-C adapter over parakeet::coreml_resolve_input_dims (unit-tested in
// test-coreml-shapes): a fixed-shape model already sized to this input is honoured
// verbatim, a flexible export is rebuilt at the requested mel length in the model's
// declared orientation, and an unknown/non-concrete shape falls back to
// features-major [1, n_mels, n_mel_frames].
NSArray<NSNumber *> * resolve_input_shape(NSArray<NSNumber *> * declared,
                                          int64_t n_mel_frames, int64_t n_mels,
                                          bool * transpose) {
    const parakeet::CoremlInputDims resolved =
        parakeet::coreml_resolve_input_dims(dims_of(declared), n_mel_frames, n_mels);
    *transpose = resolved.transpose;
    NSMutableArray<NSNumber *> * shape = [NSMutableArray arrayWithCapacity:resolved.dims.size()];
    for (int64_t dim : resolved.dims) {
        [shape addObject:@(dim)];
    }
    return shape;
}

MLMultiArray * build_array(NSArray<NSNumber *> * shape, MLMultiArrayDataType data_type,
                           NSError ** err) {
    if (data_type != MLMultiArrayDataTypeFloat32 && data_type != MLMultiArrayDataTypeFloat16) {
        return nil;
    }
    return [[MLMultiArray alloc] initWithShape:shape dataType:data_type error:err];
}

bool fill_input_array(
    MLMultiArray * arr,
    bool transpose,
    const float * mel,
    int64_t actual_mel_frames,
    int64_t allocated_mel_frames,
    int64_t n_mels) {
    if (arr == nil || mel == nullptr ||
        actual_mel_frames <= 0 ||
        allocated_mel_frames < actual_mel_frames ||
        n_mels <= 0) {
        return false;
    }

    const int64_t allocated_count = allocated_mel_frames * n_mels;
    if ((int64_t) arr.count != allocated_count) return false;

    if (arr.dataType == MLMultiArrayDataTypeFloat32) {
        float * dst = (float *) arr.dataPointer;
        std::fill(dst, dst + allocated_count, 0.0f);

        if (!transpose) {
            std::memcpy(
                dst,
                mel,
                (size_t) actual_mel_frames * n_mels * sizeof(float));
            return true;
        }

        for (int64_t m = 0; m < n_mels; ++m) {
            for (int64_t t = 0; t < actual_mel_frames; ++t) {
                dst[m * allocated_mel_frames + t] =
                    mel[t * n_mels + m];
            }
        }
        return true;
    }

    if (arr.dataType != MLMultiArrayDataTypeFloat16) return false;

    uint16_t * dst = (uint16_t *) arr.dataPointer;
    std::fill(dst, dst + allocated_count, uint16_t{0});

    if (!transpose) {
        const int64_t actual_count = actual_mel_frames * n_mels;
        for (int64_t i = 0; i < actual_count; ++i) {
            dst[i] = encode_float16(mel[i]);
        }
        return true;
    }

    for (int64_t m = 0; m < n_mels; ++m) {
        for (int64_t t = 0; t < actual_mel_frames; ++t) {
            dst[m * allocated_mel_frames + t] =
                encode_float16(mel[t * n_mels + m]);
        }
    }
    return true;
}

bool read_scalar(const void * base, MLMultiArrayDataType dt, int64_t off, float * out) {
    switch (dt) {
        case MLMultiArrayDataTypeFloat32: *out = ((const float *)    base)[off];                 return true;
        case MLMultiArrayDataTypeDouble:  *out = (float) ((const double *) base)[off];            return true;
        case MLMultiArrayDataTypeFloat16: *out = decode_float16(((const uint16_t *) base)[off]);  return true;
        default:                          return false;
    }
}

// Copies the model output into `dst` as row-major (n_enc_frames, d_model), adapting to the
// output tensor orientation and dtype and honouring its element strides.
bool copy_output_array(MLMultiArray * arr, float * dst, int64_t n_enc_frames, int64_t d_model) {
    if (arr == nil) return false;

    const void               * base  = arr.dataPointer;
    const MLMultiArrayDataType dt    = arr.dataType;
    const int64_t              total = n_enc_frames * d_model;

    const parakeet::CoremlTrailingMatch match =
        parakeet::coreml_match_trailing_capacity(dims_of(arr.shape), n_enc_frames, d_model);
    if (match.matched) {
        const bool            transpose = match.transpose;
        const NSUInteger      n       = arr.shape.count;
        NSArray<NSNumber *> * strides = arr.strides;
        const int64_t         s_outer = strides[n - 2].longLongValue;
        const int64_t         s_inner = strides[n - 1].longLongValue;
        for (int64_t t = 0; t < n_enc_frames; ++t) {
            for (int64_t f = 0; f < d_model; ++f) {
                const int64_t outer = transpose ? f : t;
                const int64_t inner = transpose ? t : f;
                float value;
                if (!read_scalar(base, dt, outer * s_outer + inner * s_inner, &value)) return false;
                dst[t * d_model + f] = value;
            }
        }
        return true;
    }

    // Unmatched shape but matching element count: assume a contiguous row-major
    // (n_enc_frames, d_model) buffer -- the natural layout for a flattened export.
    if ((int64_t) arr.count != total) return false;
    for (int64_t i = 0; i < total; ++i) {
        float value;
        if (!read_scalar(base, dt, i, &value)) return false;
        dst[i] = value;
    }
    return true;
}

}  // namespace

struct parakeet_coreml_context {
    const void * model              = nullptr;  // CFBridgingRetain'd MLModel *
    const void * input_array        = nullptr;  // retained MLMultiArray *
    const void * input_provider     = nullptr;  // retained MLDictionaryFeatureProvider *
    const void * output_array       = nullptr;  // retained MLMultiArray * (when shape is fixed)
    const void * prediction_options = nullptr;  // retained MLPredictionOptions *
    std::string  input_name;
    std::string  output_name;
    std::string  label;
    std::vector<int64_t> declared_input_dims;
    bool         input_shape_fixed = false;
    int64_t      cached_mel_frames = 0;
    int64_t      cached_mels       = 0;
    int64_t      cached_d_model    = 0;
    bool         input_transpose   = false;
    std::mutex   mutex;
};

int64_t fixed_mel_frames_for(
    const parakeet_coreml_context * ctx,
    int64_t n_mels) {
    if (ctx == nullptr || !ctx->input_shape_fixed || n_mels <= 0) return 0;

    const std::vector<int64_t> & dims = ctx->declared_input_dims;
    if (dims.size() < 2) return 0;

    const std::size_t n = dims.size();
    if (dims[n - 1] == n_mels && dims[n - 2] > 0) {
        return dims[n - 2];
    }
    if (dims[n - 2] == n_mels && dims[n - 1] > 0) {
        return dims[n - 1];
    }
    return 0;
}

namespace {

void release_cached_io(parakeet_coreml_context * ctx) {
    if (ctx->prediction_options != nullptr) CFRelease(ctx->prediction_options);
    if (ctx->output_array       != nullptr) CFRelease(ctx->output_array);
    if (ctx->input_provider     != nullptr) CFRelease(ctx->input_provider);
    if (ctx->input_array        != nullptr) CFRelease(ctx->input_array);
    ctx->prediction_options = nullptr;
    ctx->output_array       = nullptr;
    ctx->input_provider     = nullptr;
    ctx->input_array        = nullptr;
    ctx->cached_mel_frames  = 0;
    ctx->cached_mels        = 0;
    ctx->cached_d_model     = 0;
}

bool prepare_cached_io(parakeet_coreml_context * ctx, MLModel * model,
                       NSString * in_name, NSString * out_name,
                       int64_t n_mel_frames, int64_t n_mels,
                       int64_t n_enc_frames, int64_t d_model) {
    const int64_t fixed_frames = fixed_mel_frames_for(ctx, n_mels);
    if (fixed_frames > 0 && n_mel_frames > fixed_frames) {
        return false;
    }

    const int64_t input_frames = fixed_frames > 0 ? fixed_frames : n_mel_frames;

    if (ctx->input_array != nullptr &&
        ctx->cached_mel_frames == input_frames &&
        ctx->cached_mels == n_mels &&
        ctx->cached_d_model == d_model) {
        return true;
    }

    release_cached_io(ctx);
    MLMultiArrayConstraint * in_constraint =
        model.modelDescription.inputDescriptionsByName[in_name].multiArrayConstraint;
    bool                  in_transpose = true;
    NSArray<NSNumber *> * in_shape =
        resolve_input_shape(in_constraint.shape, input_frames, n_mels, &in_transpose);
    if (in_shape == nil) return false;

    NSError      * err    = nil;
    MLMultiArray * in_arr = build_array(in_shape, in_constraint.dataType, &err);
    if (in_arr == nil || err != nil) return false;
    MLDictionaryFeatureProvider * provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ in_name : in_arr } error:&err];
    if (provider == nil || err != nil) return false;

    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
    MLMultiArrayConstraint * out_constraint =
        model.modelDescription.outputDescriptionsByName[out_name].multiArrayConstraint;
    MLMultiArray * out_arr = nil;
    const parakeet::CoremlTrailingMatch out_match =
        parakeet::coreml_match_trailing_capacity(dims_of(out_constraint.shape), n_enc_frames, d_model);
    if (out_match.matched) {
        out_arr = build_array(out_constraint.shape, out_constraint.dataType, &err);
        if (out_arr == nil || err != nil) return false;
        options.outputBackings = @{ out_name : out_arr };
    }

    ctx->input_array        = CFBridgingRetain(in_arr);
    ctx->input_provider     = CFBridgingRetain(provider);
    ctx->prediction_options = CFBridgingRetain(options);
    if (out_arr != nil) ctx->output_array = CFBridgingRetain(out_arr);
    ctx->cached_mel_frames = input_frames;
    ctx->cached_mels       = n_mels;
    ctx->cached_d_model    = d_model;
    ctx->input_transpose   = in_transpose;
    return true;
}

MLComputeUnits requested_compute_units(std::string * label) {
    const char * env = std::getenv("PARAKEET_COREML_COMPUTE_UNITS");
    if (env != nullptr && std::strcmp(env, "all") == 0) {
        *label = "coreml-all";
        return MLComputeUnitsAll;
    }
    if (env != nullptr && std::strcmp(env, "cpu_and_gpu") == 0) {
        *label = "coreml-cpu-gpu";
        return MLComputeUnitsCPUAndGPU;
    }
    if (env != nullptr && std::strcmp(env, "cpu_only") == 0) {
        *label = "coreml-cpu";
        return MLComputeUnitsCPUOnly;
    }
    if (env != nullptr && std::strcmp(env, "cpu_and_ane") == 0) {
        if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)) {
            *label = "coreml-ane";
            return MLComputeUnitsCPUAndNeuralEngine;
        }
    }
    // The fixed Parakeet graph is predominantly ANE-backed but retains a small
    // set of GPU-preferred operations. Let Core ML use all three processors by
    // default; forcing CPU+ANE makes those operations fall back to CPU and can
    // erase the encoder speedup on current Apple silicon.
    *label = "coreml-all";
    return MLComputeUnitsAll;
}

}  // namespace

struct parakeet_coreml_context * parakeet_coreml_init(const char * path_mlmodelc) {
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

        MLMultiArrayConstraint * input_constraint = model.modelDescription.inputDescriptionsByName[in_name].multiArrayConstraint;
        if (input_constraint == nil) return nullptr;

        auto * ctx = new parakeet_coreml_context();
        ctx->model       = CFBridgingRetain(model);
        ctx->input_name  = in_name.UTF8String;
        ctx->output_name = out_name.UTF8String;
        ctx->label       = std::move(label);
        ctx->declared_input_dims = dims_of(input_constraint.shape);
        ctx->input_shape_fixed = constraint_has_one_fixed_shape(input_constraint);
        return ctx;
    }
}

void parakeet_coreml_free(struct parakeet_coreml_context * ctx) {
    if (ctx == nullptr) return;
    release_cached_io(ctx);
    if (ctx->model != nullptr) CFRelease(ctx->model);
    delete ctx;
}

int64_t parakeet_coreml_fixed_mel_frames(
    const struct parakeet_coreml_context * ctx,
    int64_t n_mels) {
    return fixed_mel_frames_for(ctx, n_mels);
}

int parakeet_coreml_encode(struct parakeet_coreml_context * ctx,
                           int64_t       n_mel_frames,
                           int64_t       n_mels,
                           const float * mel,
                           int64_t       n_enc_frames,
                           int64_t       d_model,
                           float       * encoder_out) {
    if (ctx == nullptr || ctx->model == nullptr || mel == nullptr || encoder_out == nullptr) return 1;
    if (n_mel_frames <= 0 || n_mels <= 0 || n_enc_frames <= 0 || d_model <= 0) return 1;

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        MLModel  * model    = (__bridge MLModel *) ctx->model;
        NSString * in_name  = [NSString stringWithUTF8String:ctx->input_name.c_str()];
        NSString * out_name = [NSString stringWithUTF8String:ctx->output_name.c_str()];

        if (!prepare_cached_io(ctx, model, in_name, out_name, n_mel_frames, n_mels,
                               n_enc_frames, d_model)) return 2;
        MLMultiArray * in_arr = (__bridge MLMultiArray *) ctx->input_array;
        if (!fill_input_array(in_arr, ctx->input_transpose, mel, n_mel_frames, ctx->cached_mel_frames, n_mels)) return 3;

        NSError * err = nil;
        MLDictionaryFeatureProvider * provider =
            (__bridge MLDictionaryFeatureProvider *) ctx->input_provider;
        MLPredictionOptions * options = (__bridge MLPredictionOptions *) ctx->prediction_options;
        id<MLFeatureProvider> result = [model predictionFromFeatures:provider options:options error:&err];
        if (result == nil || err != nil) return 5;

        MLMultiArray * out_arr = [result featureValueForName:out_name].multiArrayValue;
        if (out_arr == nil) return 6;
        if (!copy_output_array(out_arr, encoder_out, n_enc_frames, d_model)) return 7;
        return 0;
    }
}

const char * parakeet_coreml_backend_label(const struct parakeet_coreml_context * ctx) {
    return ctx != nullptr ? ctx->label.c_str() : "coreml";
}

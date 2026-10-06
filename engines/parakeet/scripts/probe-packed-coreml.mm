// Execute a compiled preservation probe and write its FP16 output verbatim.
// clang++ -O2 -fobjc-arc -framework Foundation -framework CoreML \
//   probe-packed-coreml.mm -o probe-packed-coreml
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

static uint64_t resident_bytes() {
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) return 0;
    return info.resident_size;
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s model.mlmodelc output.bin [input.bin]\n", argv[0]);
        return 2;
    }
    @autoreleasepool {
        NSError * error = nil;
        MLModelConfiguration * config = [MLModelConfiguration new];
        config.computeUnits = MLComputeUnitsCPUOnly;
        uint64_t rss0 = resident_bytes();
        auto start = std::chrono::steady_clock::now();
        MLModel * model = [MLModel modelWithContentsOfURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
            configuration:config error:&error];
        auto loaded = std::chrono::steady_clock::now();
        if (!model) {
            fprintf(stderr, "Core ML load: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        uint64_t rss1 = resident_bytes();
        NSString * input_name = model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
        MLFeatureDescription * input_desc = model.modelDescription.inputDescriptionsByName[input_name];
        NSArray<NSNumber *> * shape = input_desc.multiArrayConstraint.shape;
        MLMultiArrayDataType input_type = input_desc.multiArrayConstraint.dataType;
        MLMultiArray * input = [[MLMultiArray alloc] initWithShape:shape
            dataType:input_type error:&error];
        if (!input) {
            fprintf(stderr, "Core ML input: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        size_t input_element_bytes = input_type == MLMultiArrayDataTypeFloat32 ? 4 : 2;
        memset(input.dataPointer, 0, input.count * input_element_bytes);
        if (argc >= 4) {
            FILE * data = fopen(argv[3], "rb");
            if (!data || fread(input.dataPointer, input_element_bytes,
                               input.count, data) != input.count) {
                fprintf(stderr, "failed to read input tensor\n");
                if (data) fclose(data);
                return 1;
            }
            fclose(data);
        }
        MLDictionaryFeatureProvider * provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{input_name: [MLFeatureValue featureValueWithMultiArray:input]}
            error:&error];
        auto inference_start = std::chrono::steady_clock::now();
        id<MLFeatureProvider> output = [model predictionFromFeatures:provider error:&error];
        auto inference_end = std::chrono::steady_clock::now();
        if (!output) {
            fprintf(stderr, "Core ML predict: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        NSString * output_name = model.modelDescription.outputDescriptionsByName.allKeys.firstObject;
        MLMultiArray * values = [output featureValueForName:output_name].multiArrayValue;
        if (!values || (values.dataType != MLMultiArrayDataTypeFloat16 &&
                        values.dataType != MLMultiArrayDataTypeFloat32)) {
            fprintf(stderr, "unexpected Core ML output dtype\n");
            return 1;
        }
        FILE * file = fopen(argv[2], "wb");
        if (!file) return 1;
        size_t output_element_bytes = values.dataType == MLMultiArrayDataTypeFloat32 ? 4 : 2;
        fwrite(values.dataPointer, output_element_bytes, values.count, file);
        fclose(file);
        uint64_t rss2 = resident_bytes();
        double load_ms = std::chrono::duration<double, std::milli>(loaded-start).count();
        double predict_ms = std::chrono::duration<double, std::milli>(inference_end-inference_start).count();
        printf("{\"load_ms\":%.3f,\"predict_ms\":%.3f,\"rss_before\":%llu,\"rss_loaded\":%llu,\"rss_after_predict\":%llu,\"elements\":%ld,\"input_bytes\":%zu,\"output_bytes\":%zu}\n",
               load_ms, predict_ms, (unsigned long long)rss0,
               (unsigned long long)rss1, (unsigned long long)rss2, (long)values.count,
               input_element_bytes, output_element_bytes);
    }
    return 0;
}

#import "ParakeetBridge.h"

#include "LongAudioRunner.h"

#include <parakeet/engine.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

namespace {

NSError * bridge_error(NSInteger code, NSString * message) {
    return [NSError errorWithDomain:@"to.tether.qvac.ParakeetLab"
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey: message}];
}

NSString * model_path() {
    return [[NSBundle mainBundle] pathForResource:@"parakeet-tdt-0.6b-v3.q8_0"
                                           ofType:@"gguf"];
}

}  // namespace

@interface PKTranscriptionResult ()
@property(nonatomic, readwrite) NSString *text;
@property(nonatomic, readwrite) NSString *backendDescription;
@property(nonatomic, readwrite) NSTimeInterval inferenceSeconds;
@property(nonatomic, readwrite) NSTimeInterval modelLoadSeconds;
@property(nonatomic, readwrite) double realtimeMultiplier;
@property(nonatomic, readwrite) BOOL encoderUsedCoreML;
@end
@implementation PKTranscriptionResult
@end

@implementation ParakeetBridge {
    dispatch_queue_t _queue;
    std::unique_ptr<parakeet::Engine> _engine;
    PKBackend _loadedBackend;
    NSTimeInterval _modelLoadSeconds;
    std::mutex _engineMutex;
    std::atomic<bool> _cancelRequested;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _queue = dispatch_queue_create("to.tether.qvac.parakeet.inference", DISPATCH_QUEUE_SERIAL);
        _loadedBackend = (PKBackend) -1;
        _cancelRequested.store(false);
    }
    return self;
}

- (parakeet::Engine *)engineForBackend:(PKBackend)backend error:(NSError **)error {
    NSString * path = model_path();
    if (path == nil) {
        if (error) *error = bridge_error(1, @"Bundled Parakeet GGUF was not found.");
        return nullptr;
    }

    if (_engine != nullptr && _loadedBackend == backend) {
        return _engine.get();
    }

    {
        std::lock_guard<std::mutex> guard(_engineMutex);
        _engine.reset();
    }

    if (backend == PKBackendMetal) {
        setenv("PARAKEET_COREML_DISABLE", "1", 1);
    } else {
        unsetenv("PARAKEET_COREML_DISABLE");
    }

    // ggml's desktop-oriented Metal residency sets keep every recently used
    // graph buffer wired for 180 seconds. A long iOS transcription refreshes
    // that timer on every batch, causing otherwise released buffers to
    // accumulate until jetsam terminates the app. Let Metal manage residency
    // normally on iOS so completed batch buffers can be reclaimed promptly.
    setenv("GGML_METAL_NO_RESIDENCY", "1", 1);

    const auto started = std::chrono::steady_clock::now();
    try {
        parakeet::EngineOptions options;
        options.model_gguf_path = path.UTF8String;
        options.n_gpu_layers = 999;
        options.n_threads = 0;
        options.verbose = true;
        // The desktop default allows encoder windows that are too large for
        // iOS memory limits (self-attention grows quadratically with length).
        // Keep both Metal and Core ML encoder windows bounded to roughly
        // 14 seconds inside each app-level audio batch.
        options.long_form_window_frames = 180;
        options.long_form_context_frames = 32;
        auto engine = std::make_unique<parakeet::Engine>(options);
        _modelLoadSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();

        if (backend == PKBackendCoreML && !engine->encoder_on_coreml()) {
            if (error) {
                *error = bridge_error(
                    2,
                    @"The compiled Core ML encoder did not load. Verify that the .mlmodelc was passed to the app build."
                );
            }
            return nullptr;
        }

        std::lock_guard<std::mutex> guard(_engineMutex);
        _engine = std::move(engine);
        _loadedBackend = backend;
        return _engine.get();
    } catch (const std::exception & exception) {
        if (error) *error = bridge_error(3, [NSString stringWithUTF8String:exception.what()]);
        return nullptr;
    }
}

- (void)transcribeWAVAtURL:(NSURL *)url
                   backend:(PKBackend)backend
                 onSegment:(PKSegmentHandler)onSegment
                completion:(PKCompletionHandler)completion {
    _cancelRequested.store(false);
    dispatch_async(_queue, ^{
        @autoreleasepool {
            NSError * loadError = nil;
            parakeet::Engine * engine = [self engineForBackend:backend error:&loadError];
            if (engine == nullptr) {
                dispatch_async(dispatch_get_main_queue(), ^{ completion(nil, loadError); });
                return;
            }

            try {
                const auto result = run_long_audio_wav(
                    *engine,
                    url.fileSystemRepresentation,
                    self->_cancelRequested,
                    backend == PKBackendCoreML,
                    [onSegment](const parakeet::StreamingSegment & segment,
                                double start,
                                double end) {
                        NSString * text = [NSString stringWithUTF8String:segment.text.c_str()];
                        dispatch_async(dispatch_get_main_queue(), ^{
                            onSegment(text, start, end);
                        });
                    });
                if (self->_cancelRequested.load()) return;

                PKTranscriptionResult * output = [PKTranscriptionResult new];
                output.text = [NSString stringWithUTF8String:result.text.c_str()];
                output.inferenceSeconds = result.inference_ms / 1000.0;
                output.modelLoadSeconds = self->_modelLoadSeconds;
                output.realtimeMultiplier = result.inference_ms > 0.0
                    ? (static_cast<double>(result.audio_samples) / 16000.0) /
                        (result.inference_ms / 1000.0)
                    : 0.0;
                output.encoderUsedCoreML = result.encoder_used_coreml;
                const std::string backendName = engine->backend_name();
                if (result.encoder_used_coreml) {
                    output.backendDescription = [NSString stringWithFormat:
                        @"Core ML encoder + %s decoder", backendName.c_str()];
                } else {
                    output.backendDescription = [NSString stringWithFormat:
                        @"QVAC Metal (%s)", backendName.c_str()];
                }

                dispatch_async(dispatch_get_main_queue(), ^{ completion(output, nil); });
            } catch (const std::exception & exception) {
                NSError * error = bridge_error(4, [NSString stringWithUTF8String:exception.what()]);
                dispatch_async(dispatch_get_main_queue(), ^{ completion(nil, error); });
            }
        }
    });
}

- (void)cancel {
    _cancelRequested.store(true);
    std::lock_guard<std::mutex> guard(_engineMutex);
    if (_engine != nullptr) _engine->cancel();
}

@end

#import "ParakeetBridge.h"
#import <AVFoundation/AVFoundation.h>

#include <parakeet/engine.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

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
                NSError * audioError = nil;
                AVAudioFile * audioFile = [[AVAudioFile alloc]
                    initForReading:url
                    commonFormat:AVAudioPCMFormatFloat32
                    interleaved:NO
                    error:&audioError];
                if (audioFile == nil) {
                    const char * message = audioError.localizedDescription.UTF8String;
                    throw std::runtime_error(message != nullptr ? message : "Could not open audio file");
                }

                AVAudioFormat * format = audioFile.processingFormat;
                if (format.channelCount != 1 || std::llround(format.sampleRate) != 16000) {
                    throw std::runtime_error("Prepared audio must be mono 16 kHz PCM");
                }

                parakeet::StreamingOptions options;
                options.chunk_ms = 8000;
                options.emit_partials = false;

                // Keep both encoder and decoder intermediates bounded. The
                // engine's file API windows the encoder, but intentionally
                // retains the complete encoder output until decoding. That is
                // appropriate on desktop and can exceed the iOS process limit
                // for long recordings. Thirty-second batches release all
                // intermediate tensors before the next section starts.
                constexpr AVAudioFrameCount batchFrames = 30 * 16000;
                std::string transcript;
                int64_t totalSamples = 0;
                double totalInferenceMs = 0.0;
                bool usedCoreML = backend == PKBackendCoreML;
                bool processedAudio = false;

                while (!self->_cancelRequested.load()) {
                    @autoreleasepool {
                        const AVAudioFramePosition remainingFrames =
                            audioFile.length - audioFile.framePosition;
                        if (remainingFrames <= 0) break;
                        const AVAudioFrameCount framesToRead = static_cast<AVAudioFrameCount>(
                            std::min<AVAudioFramePosition>(remainingFrames, batchFrames)
                        );
                        AVAudioPCMBuffer * buffer = [[AVAudioPCMBuffer alloc]
                            initWithPCMFormat:format
                            frameCapacity:framesToRead];
                        NSError * readError = nil;
                        if (![audioFile readIntoBuffer:buffer
                                            frameCount:framesToRead
                                                 error:&readError]) {
                            if (audioFile.framePosition >= audioFile.length && readError == nil) break;
                            const char * message = readError.localizedDescription.UTF8String;
                            throw std::runtime_error(message != nullptr ? message : "Could not read audio file");
                        }
                        if (buffer.frameLength == 0) break;

                        const int sampleCount = static_cast<int>(buffer.frameLength);
                        const double timeOffset = static_cast<double>(totalSamples) / 16000.0;
                        const double validBatchSeconds = static_cast<double>(sampleCount) / 16000.0;
                        std::vector<float> paddedSamples;
                        const float * inferenceSamples = buffer.floatChannelData[0];
                        int inferenceSampleCount = sampleCount;
                        if (sampleCount < static_cast<int>(batchFrames)) {
                            // Keep the final native graph identical to every
                            // full batch. A short tail otherwise makes ggml
                            // reallocate its graph while the previous buffer
                            // is resident, which can exceed the iOS limit.
                            paddedSamples.assign(batchFrames, 0.0f);
                            std::copy_n(inferenceSamples, sampleCount, paddedSamples.data());
                            inferenceSamples = paddedSamples.data();
                            inferenceSampleCount = static_cast<int>(batchFrames);
                        }
                        const auto batch = engine->transcribe_samples_stream(
                            inferenceSamples,
                            inferenceSampleCount,
                            16000,
                            options,
                            [onSegment, timeOffset, validBatchSeconds](const parakeet::StreamingSegment & segment) {
                                if (segment.text.empty()) return;
                                if (segment.start_s >= validBatchSeconds) return;
                                NSString * text = [NSString stringWithUTF8String:segment.text.c_str()];
                                const double end = std::min(segment.end_s, validBatchSeconds);
                                dispatch_async(dispatch_get_main_queue(), ^{
                                    onSegment(
                                        text,
                                        timeOffset + segment.start_s,
                                        timeOffset + end
                                    );
                                });
                            }
                        );

                        if (!batch.text.empty()) {
                            if (!transcript.empty() && transcript.back() != ' ') transcript.push_back(' ');
                            transcript.append(batch.text);
                        }
                        totalSamples += sampleCount;
                        totalInferenceMs += batch.total_ms;
                        usedCoreML = usedCoreML && batch.encoder_used_coreml;
                        processedAudio = true;
                    }
                }

                if (self->_cancelRequested.load()) return;
                if (!processedAudio) throw std::runtime_error("Audio file is empty");

                PKTranscriptionResult * output = [PKTranscriptionResult new];
                output.text = [NSString stringWithUTF8String:transcript.c_str()];
                output.inferenceSeconds = totalInferenceMs / 1000.0;
                output.modelLoadSeconds = self->_modelLoadSeconds;
                output.realtimeMultiplier = totalInferenceMs > 0.0
                    ? (static_cast<double>(totalSamples) / 16000.0) / (totalInferenceMs / 1000.0)
                    : 0.0;
                output.encoderUsedCoreML = usedCoreML;
                const std::string backendName = engine->backend_name();
                if (usedCoreML) {
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

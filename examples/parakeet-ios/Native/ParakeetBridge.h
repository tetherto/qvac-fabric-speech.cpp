#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, PKBackend) {
    PKBackendCoreML = 0,
    PKBackendMetal = 1,
};

@interface PKTranscriptionResult : NSObject
@property(nonatomic, readonly) NSString *text;
@property(nonatomic, readonly) NSString *backendDescription;
@property(nonatomic, readonly) NSTimeInterval inferenceSeconds;
@property(nonatomic, readonly) NSTimeInterval modelLoadSeconds;
@property(nonatomic, readonly) double realtimeMultiplier;
@property(nonatomic, readonly) BOOL encoderUsedCoreML;
@end

typedef void (^PKSegmentHandler)(NSString *text, NSTimeInterval start, NSTimeInterval end);
typedef void (^PKCompletionHandler)(PKTranscriptionResult * _Nullable result, NSError * _Nullable error);

@interface ParakeetBridge : NSObject
- (void)transcribeWAVAtURL:(NSURL *)url
                   backend:(PKBackend)backend
                 onSegment:(PKSegmentHandler)onSegment
                 completion:(PKCompletionHandler)completion
    NS_SWIFT_NAME(transcribe(wavURL:backend:onSegment:completion:));
- (void)cancel;
@end

NS_ASSUME_NONNULL_END

import AVFoundation
import Foundation

enum AudioPreparationError: LocalizedError {
    case tooLarge
    case tooLong
    case unsupportedFormat
    case conversionFailed(String)

    var errorDescription: String? {
        switch self {
        case .tooLarge: return "Audio must be 512 MB or smaller."
        case .tooLong: return "Audio must be 60 minutes or shorter."
        case .unsupportedFormat: return "The selected audio format is not supported by iOS."
        case .conversionFailed(let message): return "Audio conversion failed: \(message)"
        }
    }
}
struct PreparedAudio {
    let url: URL
    let displayName: String
    let duration: TimeInterval
}

enum AudioPreparation {
    static let maximumBytes: Int64 = 512 * 1024 * 1024
    static let maximumDuration: TimeInterval = 60 * 60

    static func prepare(
        url sourceURL: URL,
        displayName: String? = nil,
        useSourceIfCompatible: Bool = false
    ) throws -> PreparedAudio {
        let values = try sourceURL.resourceValues(forKeys: [.fileSizeKey])
        if Int64(values.fileSize ?? 0) > maximumBytes { throw AudioPreparationError.tooLarge }

        let input: AVAudioFile
        do {
            input = try AVAudioFile(forReading: sourceURL)
        } catch {
            throw AudioPreparationError.unsupportedFormat
        }

        let duration = Double(input.length) / input.processingFormat.sampleRate
        if duration > maximumDuration { throw AudioPreparationError.tooLong }

        let isNativeWAV = sourceURL.pathExtension.lowercased() == "wav"
            && input.fileFormat.commonFormat == .pcmFormatInt16
            && input.fileFormat.sampleRate == 16_000
            && input.fileFormat.channelCount == 1
        if isNativeWAV {
            let preparedURL: URL
            if useSourceIfCompatible {
                preparedURL = sourceURL
            } else {
                preparedURL = temporaryWAVURL()
                try FileManager.default.copyItem(at: sourceURL, to: preparedURL)
            }
            return PreparedAudio(
                url: preparedURL,
                displayName: displayName ?? sourceURL.deletingPathExtension().lastPathComponent,
                duration: duration
            )
        }

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: 16_000,
            channels: 1,
            interleaved: true
        ), let converter = AVAudioConverter(from: input.processingFormat, to: outputFormat) else {
            throw AudioPreparationError.unsupportedFormat
        }

        let outputURL = temporaryWAVURL()
        var conversionCompleted = false
        defer {
            if !conversionCompleted {
                try? FileManager.default.removeItem(at: outputURL)
            }
        }
        let output = try AVAudioFile(
            forWriting: outputURL,
            settings: outputFormat.settings,
            commonFormat: .pcmFormatInt16,
            interleaved: true
        )

        let outputCapacity: AVAudioFrameCount = 32_768
        while true {
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity: outputCapacity
            ) else {
                throw AudioPreparationError.unsupportedFormat
            }

            var conversionError: NSError?
            var readError: Error?
            let status = converter.convert(to: outputBuffer, error: &conversionError) { requestedFrames, status in
                if input.framePosition >= input.length {
                    status.pointee = .endOfStream
                    return nil
                }

                let remaining = input.length - input.framePosition
                let frameCount = min(requestedFrames, AVAudioFrameCount(remaining))
                guard let inputBuffer = AVAudioPCMBuffer(
                    pcmFormat: input.processingFormat,
                    frameCapacity: frameCount
                ) else {
                    readError = AudioPreparationError.unsupportedFormat
                    status.pointee = .noDataNow
                    return nil
                }
                do {
                    try input.read(into: inputBuffer, frameCount: frameCount)
                } catch {
                    readError = error
                    status.pointee = .noDataNow
                    return nil
                }
                if inputBuffer.frameLength == 0 {
                    status.pointee = .endOfStream
                    return nil
                }
                status.pointee = .haveData
                return inputBuffer
            }

            if let readError { throw readError }
            if status == .error {
                throw AudioPreparationError.conversionFailed(
                    conversionError?.localizedDescription ?? "unknown converter error"
                )
            }
            if outputBuffer.frameLength > 0 { try output.write(from: outputBuffer) }
            if status == .endOfStream { break }
            if outputBuffer.frameLength == 0 && status == .haveData {
                throw AudioPreparationError.conversionFailed("converter made no progress")
            }
        }

        conversionCompleted = true
        return PreparedAudio(
            url: outputURL,
            displayName: displayName ?? sourceURL.deletingPathExtension().lastPathComponent,
            duration: duration
        )
    }

    private static func temporaryWAVURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("parakeet-\(UUID().uuidString)")
            .appendingPathExtension("wav")
    }
}

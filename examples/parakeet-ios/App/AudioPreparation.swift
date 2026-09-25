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

    static func prepare(url sourceURL: URL, displayName: String? = nil) throws -> PreparedAudio {
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

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: 16_000,
            channels: 1,
            interleaved: true
        ), let converter = AVAudioConverter(from: input.processingFormat, to: outputFormat) else {
            throw AudioPreparationError.unsupportedFormat
        }

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("parakeet-\(UUID().uuidString)")
            .appendingPathExtension("wav")
        let output = try AVAudioFile(
            forWriting: outputURL,
            settings: outputFormat.settings,
            commonFormat: .pcmFormatInt16,
            interleaved: true
        )

        let inputCapacity: AVAudioFrameCount = 32_768
        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: input.processingFormat,
            frameCapacity: inputCapacity
        ) else {
            throw AudioPreparationError.unsupportedFormat
        }

        while input.framePosition < input.length {
            inputBuffer.frameLength = 0
            try input.read(into: inputBuffer, frameCount: inputCapacity)
            if inputBuffer.frameLength == 0 { break }

            let ratio = outputFormat.sampleRate / input.processingFormat.sampleRate
            let outputCapacity = AVAudioFrameCount(ceil(Double(inputBuffer.frameLength) * ratio)) + 64
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity: outputCapacity
            ) else {
                throw AudioPreparationError.unsupportedFormat
            }

            var supplied = false
            var conversionError: NSError?
            let status = converter.convert(to: outputBuffer, error: &conversionError) { _, status in
                if supplied {
                    status.pointee = .noDataNow
                    return nil
                }
                supplied = true
                status.pointee = .haveData
                return inputBuffer
            }

            if status == .error {
                throw AudioPreparationError.conversionFailed(
                    conversionError?.localizedDescription ?? "unknown converter error"
                )
            }
            if outputBuffer.frameLength > 0 { try output.write(from: outputBuffer) }
        }

        return PreparedAudio(
            url: outputURL,
            displayName: displayName ?? sourceURL.deletingPathExtension().lastPathComponent,
            duration: duration
        )
    }
}

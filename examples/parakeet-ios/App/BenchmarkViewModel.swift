import AVFoundation
import Foundation
import SwiftUI
import UniformTypeIdentifiers

enum DemoBackend: String, CaseIterable, Identifiable {
    case coreML = "Core ML"
    case metal = "Metal"

    var id: Self { self }
    var native: PKBackend { self == .coreML ? .coreML : .metal }
    var explanation: String {
        switch self {
        case .coreML: return "Core ML encoder + Metal decoder. The compiled encoder ships inside this app."
        case .metal: return "The Q8_0 encoder and TDT decoder run through QVAC Metal."
        }
    }
}

struct BenchmarkResult {
    let backendDescription: String
    let inferenceSeconds: Double
    let realtimeMultiplier: Double
    let modelLoadSeconds: Double
    let transcript: String
}

@MainActor
final class BenchmarkViewModel: NSObject, ObservableObject {
    @Published var backend: DemoBackend = .coreML
    @Published var audio: PreparedAudio?
    @Published var liveTranscript = ""
    @Published var result: BenchmarkResult?
    @Published var status = "Choose audio, then run the benchmark."
    @Published var isRunning = false
    @Published var isPreparingAudio = false
    @Published var errorMessage: String?
    @Published var isImporting = false
    @Published var isEnteringURL = false
    @Published var remoteURL = ""

    private let bridge = ParakeetBridge()
    private var player: AVAudioPlayer?

    override init() {
        super.init()
        loadBundledSample()
    }

    var audioTitle: String { audio?.displayName ?? "No audio selected" }
    var audioDuration: String { Self.formatDuration(audio?.duration ?? 0) }

    func loadBundledSample() {
        guard let url = Bundle.main.url(forResource: "parakeet-demo-sample", withExtension: "wav") else {
            errorMessage = "Bundled sample audio is missing."
            return
        }
        prepare(url: url, name: "Sample audio")
    }

    func importAudio(_ url: URL) {
        guard !isRunning else { return }
        isPreparingAudio = true
        result = nil
        status = "Preparing 16 kHz mono audio…"
        let name = url.deletingPathExtension().lastPathComponent
        Task {
            do {
                let prepared = try await Task.detached(priority: .userInitiated) {
                    let hasAccess = url.startAccessingSecurityScopedResource()
                    defer { if hasAccess { url.stopAccessingSecurityScopedResource() } }
                    return try AudioPreparation.prepare(url: url, displayName: name)
                }.value
                audio = prepared
                isPreparingAudio = false
                status = "Ready to transcribe on this device."
            } catch {
                isPreparingAudio = false
                errorMessage = error.localizedDescription
                status = "Audio preparation failed."
            }
        }
    }

    func downloadAudio() {
        guard let url = URL(string: remoteURL), url.scheme?.lowercased() == "https" else {
            errorMessage = "Enter a direct HTTPS audio URL."
            return
        }
        isEnteringURL = false
        isPreparingAudio = true
        status = "Downloading audio…"
        Task {
            do {
                let (temporaryURL, response) = try await URLSession.shared.download(from: url)
                let expected = response.expectedContentLength
                if expected > AudioPreparation.maximumBytes {
                    throw AudioPreparationError.tooLarge
                }
                prepare(url: temporaryURL, name: url.deletingPathExtension().lastPathComponent)
            } catch {
                isPreparingAudio = false
                errorMessage = error.localizedDescription
                status = "Audio download failed."
            }
        }
    }

    private func prepare(url: URL, name: String) {
        guard !isRunning else { return }
        isPreparingAudio = true
        result = nil
        status = "Preparing 16 kHz mono audio…"
        Task.detached(priority: .userInitiated) {
            do {
                let prepared = try AudioPreparation.prepare(url: url, displayName: name)
                await MainActor.run {
                    self.audio = prepared
                    self.isPreparingAudio = false
                    self.status = "Ready to transcribe on this device."
                }
            } catch {
                await MainActor.run {
                    self.isPreparingAudio = false
                    self.errorMessage = error.localizedDescription
                    self.status = "Audio preparation failed."
                }
            }
        }
    }

    func togglePlayback() {
        guard let url = audio?.url else { return }
        if player?.isPlaying == true {
            player?.stop()
            return
        }
        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.play()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func runBenchmark() {
        guard let audio else { return }
        isRunning = true
        result = nil
        liveTranscript = ""
        status = "Loading the \(backend.rawValue) model…"
        run(audio: audio)
    }

    private func run(audio: PreparedAudio) {
        status = "Transcribing…"

        bridge.transcribe(
            wavURL: audio.url,
            backend: backend.native,
            onSegment: { [weak self] text, start, end in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    let stamp = String(format: "%5.1f–%5.1f", start, end)
                    self.liveTranscript += "[\(stamp)] \(text)\n"
                }
            },
            completion: { [weak self] nativeResult, error in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    if let error {
                        self.isRunning = false
                        self.errorMessage = error.localizedDescription
                        self.status = "Benchmark failed."
                        return
                    }
                    guard let nativeResult else { return }
                    self.result = BenchmarkResult(
                        backendDescription: nativeResult.backendDescription,
                        inferenceSeconds: nativeResult.inferenceSeconds,
                        realtimeMultiplier: audio.duration / nativeResult.inferenceSeconds,
                        modelLoadSeconds: nativeResult.modelLoadSeconds,
                        transcript: nativeResult.text
                    )
                    self.isRunning = false
                    self.status = "Transcription complete."
                }
            }
        )
    }

    func cancel() {
        bridge.cancel()
        isRunning = false
        status = "Benchmark cancelled."
    }

    static func formatDuration(_ duration: TimeInterval) -> String {
        let seconds = max(0, Int(duration.rounded()))
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }
}

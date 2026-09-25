import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var model: BenchmarkViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 28) {
                    inferenceSection
                    audioSection
                    actionSection
                    if model.isRunning || !model.liveTranscript.isEmpty {
                        liveSection
                    }
                    if let result = model.result {
                        resultSection(result)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 48)
            }
            .background(Color(uiColor: .systemGroupedBackground))
            .navigationTitle("Parakeet Lab")
            .navigationBarTitleDisplayMode(.inline)
        }
        .fileImporter(
            isPresented: $model.isImporting,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first { model.importAudio(url) }
            case .failure(let error):
                model.errorMessage = error.localizedDescription
            }
        }
        .alert("Audio from a link", isPresented: $model.isEnteringURL) {
            TextField("https://example.com/speech.wav", text: $model.remoteURL)
                .textInputAutocapitalization(.never)
                .keyboardType(.URL)
            Button("Cancel", role: .cancel) {}
            Button("Download") { model.downloadAudio() }
        } message: {
            Text("Use a direct HTTPS link to an audio file.")
        }
        .alert("Parakeet Lab", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "Unknown error")
        }
    }

    private var inferenceSection: some View {
        section("Inference") {
            VStack(spacing: 16) {
                Picker("Backend", selection: $model.backend) {
                    ForEach(DemoBackend.allCases) { backend in
                        Text(backend.rawValue).tag(backend)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(model.isRunning)

                Divider()
                Text(model.backend.explanation)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Divider()
                Label("Transcription runs entirely on this device.", systemImage: "iphone")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var audioSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Audio")
                .font(.title3.weight(.semibold))
                .foregroundStyle(.secondary)
                .padding(.leading, 20)

            card {
                VStack(spacing: 14) {
                    HStack(spacing: 14) {
                        Image(systemName: "waveform")
                            .font(.title2)
                            .foregroundStyle(.blue)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.audioTitle).font(.headline).lineLimit(1)
                            Text(model.audioDuration).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button(action: model.togglePlayback) {
                            Image(systemName: "play.fill").font(.title3)
                        }
                        .disabled(model.audio == nil)
                    }
                    Divider()
                    HStack {
                        Button("Sample", action: model.loadBundledSample)
                        Spacer()
                        Button("Import file") { model.isImporting = true }
                    }
                    .font(.body)
                    Divider()
                    Button {
                        model.isEnteringURL = true
                    } label: {
                        HStack {
                            Text("Audio from a link").foregroundStyle(.primary)
                            Spacer()
                            Image(systemName: "chevron.right").foregroundStyle(.secondary)
                        }
                    }
                }
            }

            Text("WAV, M4A, MP3 and other iOS-supported audio. Up to 60 minutes / 512 MB. Links must point directly to audio files.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 20)
        }
    }

    private var actionSection: some View {
        card {
            VStack(spacing: 15) {
                if model.isRunning {
                    Button("Cancel", role: .destructive, action: model.cancel)
                        .font(.headline)
                } else {
                    Button("Transcribe", action: model.runBenchmark)
                        .font(.headline)
                        .disabled(model.audio == nil || model.isPreparingAudio)
                }
                if model.isRunning || model.isPreparingAudio { ProgressView() }
                Divider()
                Text(model.status)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxWidth: .infinity)
        }
    }

    private var liveSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Live transcript")
                .font(.title3.weight(.semibold))
                .foregroundStyle(.secondary)
                .padding(.leading, 20)
            card {
                ScrollViewReader { proxy in
                    ScrollView {
                        Text(model.liveTranscript.isEmpty ? "Waiting for the first segment…" : model.liveTranscript)
                            .font(.system(.body, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id("transcript-end")
                    }
                    .frame(maxHeight: 280)
                    .onChange(of: model.liveTranscript) { _ in
                        withAnimation { proxy.scrollTo("transcript-end", anchor: .bottom) }
                    }
                }
            }
        }
    }

    private func resultSection(_ result: BenchmarkResult) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("\(model.backend.rawValue) · \(model.audioTitle) · \(model.audioDuration)")
                .font(.title3.weight(.semibold))
                .foregroundStyle(.secondary)
                .padding(.leading, 20)
            card {
                VStack(alignment: .leading, spacing: 16) {
                    Text(result.backendDescription).font(.headline)
                    Divider()
                    HStack(alignment: .top) {
                        metric(String(format: "%.2f s", result.medianSeconds), "Median · \(result.runs) runs")
                        Spacer()
                        metric(String(format: "%.1f×", result.realtimeMultiplier), "Speed")
                        Spacer()
                        metric(String(format: "%.2f s", result.modelLoadSeconds), "Model load")
                    }
                    Divider()
                    Text(result.transcript)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    private func metric(_ value: String, _ label: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value).font(.title2.weight(.semibold))
            Text(label).font(.footnote).foregroundStyle(.secondary)
        }
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.title3.weight(.semibold))
                .foregroundStyle(.secondary)
                .padding(.leading, 20)
            card(content: content)
        }
    }

    private func card<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content()
            .padding(20)
            .background(.background)
            .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
    }
}

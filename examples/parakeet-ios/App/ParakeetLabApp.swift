import SwiftUI

@main
struct ParakeetLabApp: App {
    @StateObject private var model = BenchmarkViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
        }
    }
}

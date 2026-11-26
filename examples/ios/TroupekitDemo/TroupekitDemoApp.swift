import SwiftUI

/// Entry point of the iOS demo: one chat screen and the key sheet.
/// The agent wiring (start, stream, resume, background window) lands in the next M4 step.
/// Input: none. Output: the app scene.
@main
struct TroupekitDemoApp: App {
    @StateObject private var model = ChatModel()
    @State private var showingKey = false

    private let keychain = KeychainStore(service: "dev.troupekit.demo")

    var body: some Scene {
        WindowGroup {
            ChatView(
                model: model,
                onSend: { question in
                    model.fail(message: "The agent is not wired yet.", resumable: false)
                    _ = question
                },
                onKey: { showingKey = true }
            )
            .sheet(isPresented: $showingKey) {
                KeySheet(store: keychain, onDone: { showingKey = false })
            }
        }
    }
}

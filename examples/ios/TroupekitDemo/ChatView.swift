import SwiftUI

/// The demo's single screen: the conversation, the answer streaming in, and the input row.
/// The send action is injected so the shell runs before the agent is wired.
/// Input: the chat model and a send action. Output: the SwiftUI view.
struct ChatView: View {
    @ObservedObject var model: ChatModel
    let onSend: (String) -> Void
    let onKey: () -> Void

    @State private var draft = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollViewReader { proxy in
                    List {
                        ForEach(model.lines) { line in
                            bubble(line.text, author: line.author, identifier: line.author == .agent ? "answer" : "line")
                                .listRowSeparator(.hidden)
                        }
                        if model.isRunning {
                            bubble(model.streaming.isEmpty ? "…" : model.streaming, author: .agent, identifier: "streaming")
                                .listRowSeparator(.hidden)
                                .id("streaming")
                        }
                    }
                    .listStyle(.plain)
                    .onChange(of: model.streaming) { _ in
                        proxy.scrollTo("streaming", anchor: .bottom)
                    }
                }
                if let runId = model.runId {
                    HStack(spacing: 12) {
                        Text("run \(runId)").accessibilityIdentifier("runId")
                        Text("steps \(model.steps)").accessibilityIdentifier("steps")
                        if let progress = model.progress {
                            Text(progress).accessibilityIdentifier("progress")
                        }
                    }
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                }
                Divider()
                HStack(spacing: 8) {
                    TextField("Ask support", text: $draft, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...4)
                        .accessibilityIdentifier("question")
                    Button("Send") {
                        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !text.isEmpty else { return }
                        draft = ""
                        onSend(text)
                    }
                    .disabled(model.isRunning || draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityIdentifier("send")
                }
                .padding(12)
            }
            .navigationTitle("Support")
            .toolbar {
                Button("Key", action: onKey)
                    .accessibilityLabel("Cloud API key")
            }
        }
    }

    private func bubble(_ text: String, author: ChatLine.Author, identifier: String) -> some View {
        HStack {
            if author == .user { Spacer(minLength: 40) }
            Text(text)
                .accessibilityIdentifier(identifier)
                .padding(10)
                .foregroundStyle(author == .user ? Color.white : Color.primary)
                .background(background(for: author), in: RoundedRectangle(cornerRadius: 12))
                .textSelection(.enabled)
            if author != .user { Spacer(minLength: 40) }
        }
    }

    private func background(for author: ChatLine.Author) -> Color {
        switch author {
        case .user: return .accentColor
        case .agent: return Color(.secondarySystemBackground)
        case .notice: return Color(.tertiarySystemBackground)
        }
    }
}

/// Sheet where the user types a cloud API key, saved to the Keychain.
/// Input: the store and a dismiss action. Output: the SwiftUI view.
struct KeySheet: View {
    let store: KeychainStore
    let onDone: () -> Void

    @State private var key = ""
    @State private var failed = false

    var body: some View {
        NavigationStack {
            Form {
                SecureField("API key", text: $key)
                    .textContentType(.password)
                    .autocorrectionDisabled()
                if failed {
                    Text("The key could not be saved.").foregroundStyle(.red)
                }
                Button("Remove saved key", role: .destructive) {
                    store.delete()
                    onDone()
                }
            }
            .navigationTitle("Cloud API key")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel", action: onDone) }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        if store.save(key) { onDone() } else { failed = true }
                    }
                    .disabled(key.isEmpty)
                }
            }
        }
    }
}

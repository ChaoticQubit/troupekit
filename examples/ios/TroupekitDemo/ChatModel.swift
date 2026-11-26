import Foundation

/// One line of the conversation shown in the chat view.
/// Input: who wrote it and the text. Output: a value the list renders by `id`.
struct ChatLine: Identifiable, Equatable {
    enum Author: Equatable {
        case user
        case agent
        case notice
    }

    let id = UUID()
    let author: Author
    let text: String
}

/// UI state of the demo chat: finished lines, the streaming answer, and the stored run id.
/// The run id is kept in `UserDefaults` so a relaunch after a kill can resume the same run.
/// Input: user text and agent events. Output: published state for SwiftUI.
@MainActor
final class ChatModel: ObservableObject {
    static let runIdKey = "troupekit.demo.runId"
    static let maxStreamingBytes = 1 << 20

    @Published private(set) var lines: [ChatLine] = []
    @Published private(set) var streaming: String = ""
    @Published private(set) var isRunning = false
    @Published private(set) var runId: String?
    @Published private(set) var steps = 0
    @Published private(set) var progress: String?

    private let defaults: UserDefaults

    /// Creates the model over a defaults store.
    /// Input: the store holding the pending run id. Output: an idle model.
    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    /// The run id of a run started and not yet finished, if any.
    /// Input: none. Output: the stored id, or nil when no run is pending.
    var pendingRunId: String? {
        defaults.string(forKey: Self.runIdKey)
    }

    /// Records the user's question and the id of the run that answers it.
    /// Input: the question and the run id returned by `start`. Output: none.
    func begin(question: String, runId: String) {
        precondition(!runId.isEmpty, "run id must not be empty")
        defaults.set(runId, forKey: Self.runIdKey)
        lines.append(ChatLine(author: .user, text: question))
        self.runId = runId
        steps = 0
        progress = nil
        streaming = ""
        isRunning = true
    }

    /// Marks a stored run as resuming after a relaunch.
    /// Input: none. Output: none; no-op when nothing is pending.
    func beginResume() {
        guard let pending = pendingRunId else { return }
        lines.append(ChatLine(author: .notice, text: "Resuming the last run"))
        runId = pending
        progress = nil
        streaming = ""
        isRunning = true
    }

    /// Appends one streamed text delta to the answer in progress.
    /// Input: the delta text. Output: none; text past `maxStreamingBytes` is dropped.
    func apply(delta: String) {
        guard streaming.utf8.count + delta.utf8.count <= Self.maxStreamingBytes else { return }
        streaming += delta
    }

    /// Counts one finished step of the current run.
    /// Input: none. Output: none.
    func stepFinished() {
        steps += 1
    }

    /// Ends a background window that left the run unfinished; the run id stays stored.
    /// Input: none. Output: none.
    func pending() {
        progress = "pending"
        streaming = ""
        isRunning = false
    }

    /// Forgets any stored run, for a clean start.
    /// Input: none. Output: none.
    func reset() {
        defaults.removeObject(forKey: Self.runIdKey)
        lines = []
        runId = nil
        steps = 0
        progress = nil
        streaming = ""
        isRunning = false
    }

    /// Shows the final answer once and clears the stored run id.
    /// Input: the answer text. Output: none.
    func finish(answer: String) {
        defaults.removeObject(forKey: Self.runIdKey)
        lines.append(ChatLine(author: .agent, text: answer))
        progress = "done"
        streaming = ""
        isRunning = false
    }

    /// Shows a failure; keeps the run id when the run can still be resumed.
    /// Input: the message and whether the run stays resumable. Output: none.
    func fail(message: String, resumable: Bool) {
        if !resumable {
            defaults.removeObject(forKey: Self.runIdKey)
        }
        lines.append(ChatLine(author: .notice, text: message))
        streaming = ""
        isRunning = false
    }
}

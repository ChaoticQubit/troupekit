import Foundation
import Troupekit

/// Failure inside an Apple adapter before it crosses the boundary as a `HostError`.
/// Its text is the message the agent's run reports, so it names the cause plainly.
/// Input: the message. Output: an `Error` whose description is that message.
struct AppleAdapterError: Error, CustomStringConvertible {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var description: String { message }
}

/// Builds the boundary error type from a message, in one place for every adapter.
/// Input: the message. Output: the generated `HostError` carrying it.
func makeHostError(_ message: String) -> HostError {
    HostError.Failed(message: message)
}

/// Maps any error raised inside an adapter to the boundary error type.
/// Input: the error and a prefix for errors the adapter did not name. Output: a `HostError`.
func hostError(from error: Error, prefix: String) -> HostError {
    if let error = error as? HostError {
        return error
    }
    if let error = error as? AppleAdapterError {
        return makeHostError(error.message)
    }
    return makeHostError("\(prefix): \(error)")
}

/// Holds one result handed from a detached task to the waiting thread.
/// Input: one `set` call. Output: the stored result on `get`.
final class ResultBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<T, Error>?

    func set(_ value: Result<T, Error>) {
        lock.lock()
        defer { lock.unlock() }
        precondition(result == nil, "result set twice")
        result = value
    }

    func get() -> Result<T, Error> {
        lock.lock()
        defer { lock.unlock() }
        guard let result else { preconditionFailure("result read before it was set") }
        return result
    }
}

/// Runs async platform work from a sync host callback and blocks until it ends.
/// Host callbacks run on a tokio blocking thread, never the main thread (plan P6).
/// Input: the async operation. Output: its value, or its error rethrown.
enum Blocking {
    static func run<T>(_ operation: @escaping () async throws -> T) throws -> T {
        precondition(!Thread.isMainThread, "host callbacks must not run on the main thread")
        let box = ResultBox<T>()
        let done = DispatchSemaphore(value: 0)
        Task.detached {
            do {
                box.set(.success(try await operation()))
            } catch {
                box.set(.failure(error))
            }
            done.signal()
        }
        done.wait()
        return try box.get().get()
    }
}

/// Turns the cumulative snapshots of a streamed reply into text deltas.
/// A snapshot that does not extend the previous one is pushed whole.
/// Input: each snapshot in order. Output: the text the sink has not seen yet.
struct SnapshotDiffer {
    private var previous = ""

    mutating func delta(for snapshot: String) -> String {
        defer { previous = snapshot }
        guard snapshot.hasPrefix(previous) else { return snapshot }
        return String(snapshot.dropFirst(previous.count))
    }
}

import Foundation

/// The events of a `RunStream` as an `AsyncSequence`, so a run reads as
/// `for try await event in stream.events { ... }`.
/// Input: the stream. Output: each event until the run ends; type sugar only.
public struct RunEvents: AsyncSequence, Sendable {
    public typealias Element = Event

    let stream: RunStream

    /// Pulls the next event from the stream on each `next()` call.
    /// Input: the stream. Output: the next event, or nil once the run ended.
    /// A `TroupekitError` from the stream is rethrown unchanged.
    public struct AsyncIterator: AsyncIteratorProtocol {
        let stream: RunStream

        public mutating func next() async throws -> Event? {
            try await stream.next()
        }
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(stream: stream)
    }
}

extension RunStream {
    /// The run's events as an `AsyncSequence` over `next()`.
    /// Input: none. Output: a `RunEvents` sequence for `for try await`.
    /// Pull and listener are exclusive, as in the neutral layer.
    public var events: RunEvents {
        RunEvents(stream: self)
    }
}

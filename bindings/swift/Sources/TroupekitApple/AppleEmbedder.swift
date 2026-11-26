import Foundation
import NaturalLanguage
import Troupekit

/// Apple's built-in sentence embedding as a troupekit `HostEmbedder` (iOS 14, macOS 11).
/// One vector per text, `dims()` long; the Rust side checks the shape and batches calls.
/// Input: the embedding language. Output: the adapter, or nil when the OS has no model for it.
public final class AppleEmbedder: HostEmbedder, @unchecked Sendable {
    private let embedding: NLEmbedding
    private let lock = NSLock()

    /// Loads the sentence embedding for a language.
    /// Input: the language, English by default. Output: nil when this OS ships none for it.
    public init?(language: NLLanguage = .english) {
        guard let embedding = NLEmbedding.sentenceEmbedding(for: language) else {
            return nil
        }
        self.embedding = embedding
    }

    public func dims() -> UInt32 {
        UInt32(embedding.dimension)
    }

    public func embed(texts: [String]) throws -> [[Float]] {
        lock.lock()
        defer { lock.unlock() }
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        for text in texts {
            guard let vector = embedding.vector(for: text) else {
                throw makeHostError("apple embedder: no vector for a text")
            }
            assert(vector.count == embedding.dimension, "sentence vector length differs from dimension")
            vectors.append(vector.map { Float($0) })
        }
        return vectors
    }
}

import Foundation
import Troupekit
#if canImport(FoundationModels)
import FoundationModels
#endif

/// The Apple on-device system model as a troupekit `HostModel`.
/// Tools are emulated by the Rust side through `schema_json` (guided generation), so the
/// tool loop and every checkpoint stay in troupekit. Safe to create on any OS; it reports unavailable.
public final class AppleFoundationModel: HostModel, @unchecked Sendable {
    static let fallbackContextTokens = 4096

    /// Creates the adapter over `SystemLanguageModel.default`.
    /// Input: none. Output: the adapter; availability is checked on every call.
    public init() {}

    public func id() -> String {
        "apple/system"
    }

    public func capabilities() -> HostModelCaps {
        HostModelCaps(tools: false, jsonSchema: true, streaming: true, contextTokens: UInt32(Self.contextTokens()))
    }

    public func generate(request: HostRequest, sink: TokenSink) throws -> HostReply {
        do {
            let text = try generateText(request: request) { delta in sink.delta(text: delta) }
            return HostReply(text: text, inputTokens: 0, outputTokens: 0)
        } catch {
            throw hostError(from: error, prefix: "apple model error")
        }
    }

    /// Why the model cannot answer on this device right now, or nil when it can.
    /// Input: none. Output: a message starting `apple model unavailable: `, or nil.
    func unavailableMessage() -> String? {
        #if canImport(FoundationModels)
        if #available(iOS 26, macOS 26, *) {
            guard case let .unavailable(reason) = SystemLanguageModel.default.availability else {
                return nil
            }
            switch reason {
            case .deviceNotEligible:
                return "apple model unavailable: device not eligible"
            case .appleIntelligenceNotEnabled:
                return "apple model unavailable: apple intelligence off"
            case .modelNotReady:
                return "apple model unavailable: model not ready"
            @unknown default:
                return "apple model unavailable: \(reason)"
            }
        }
        return "apple model unavailable: needs iOS 26 or macOS 26"
        #else
        return "apple model unavailable: sdk without FoundationModels"
        #endif
    }

    /// Runs one model turn for a request, pushing streamed text through `onDelta`.
    /// With `schemaJson` the call is guided and returns the JSON text without deltas.
    /// Input: the request and a delta callback. Output: the reply text; throws a named error.
    func generateText(request: HostRequest, onDelta: @escaping (String) -> Void) throws -> String {
        if let message = unavailableMessage() {
            throw AppleAdapterError(message)
        }
        #if canImport(FoundationModels)
        if #available(iOS 26, macOS 26, *) {
            let mapped = Self.render(request)
            let schema = try request.schemaJson.map { try AppleSchema.build($0) }
            let options = GenerationOptions(sampling: nil, temperature: nil, maximumResponseTokens: Int(request.maxTokens))
            return try Blocking.run {
                do {
                    let session = LanguageModelSession(model: .default) { mapped.instructions }
                    if let schema {
                        let response = try await session.respond(to: mapped.prompt, schema: schema, includeSchemaInPrompt: true, options: options)
                        return response.content.jsonString
                    }
                    var differ = SnapshotDiffer()
                    var last = ""
                    for try await snapshot in session.streamResponse(to: mapped.prompt, options: options) {
                        last = snapshot.content
                        let delta = differ.delta(for: last)
                        if !delta.isEmpty {
                            onDelta(delta)
                        }
                    }
                    return last
                } catch let error as LanguageModelSession.GenerationError {
                    throw AppleAdapterError("apple model error: \(Self.name(of: error))")
                }
            }
        }
        #endif
        throw AppleAdapterError("apple model unavailable: needs iOS 26 or macOS 26")
    }

    /// Renders a request as instructions plus one prompt holding every turn as text.
    /// Tool calls and tool results become labelled lines; the prompt ends with `assistant:`.
    /// Input: the request. Output: the instructions (empty without a system) and the prompt.
    static func render(_ request: HostRequest) -> (instructions: String, prompt: String) {
        var lines: [String] = []
        lines.reserveCapacity(request.messages.count + 1)
        for message in request.messages {
            if let calls = message.toolCallsJson {
                if !message.text.isEmpty {
                    lines.append("assistant: \(message.text)")
                }
                lines.append("assistant tool call: \(calls)")
            } else if let callId = message.toolCallId {
                lines.append("tool result \(callId): \(message.text)")
            } else {
                lines.append("\(message.role): \(message.text)")
            }
        }
        lines.append("assistant:")
        return (request.system ?? "", lines.joined(separator: "\n"))
    }

    private static func contextTokens() -> Int {
        #if canImport(FoundationModels)
        if #available(iOS 26, macOS 26, *) {
            return SystemLanguageModel.default.contextSize
        }
        #endif
        return fallbackContextTokens
    }

    #if canImport(FoundationModels)
    @available(iOS 26, macOS 26, *)
    private static func name(of error: LanguageModelSession.GenerationError) -> String {
        switch error {
        case .exceededContextWindowSize: return "exceeded context window"
        case .assetsUnavailable: return "assets unavailable"
        case .guardrailViolation: return "guardrail violation"
        case .unsupportedGuide: return "unsupported guide"
        case .unsupportedLanguageOrLocale: return "unsupported language or locale"
        case .decodingFailure: return "decoding failure"
        case .rateLimited: return "rate limited"
        case .concurrentRequests: return "concurrent requests"
        case .refusal: return "refusal"
        @unknown default: return "\(error)"
        }
    }
    #endif
}

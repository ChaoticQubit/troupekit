import Foundation
import XCTest
@testable import Troupekit

let selector = "stub/stub-model"
let stubBase = "http://127.0.0.1:9/v1"
let answerParts = ["Hello", " from", " the stub"]
let eventsMax = 10_000
let embedDims: UInt32 = 64
let concurrentRuns = 8
let secret = "sk-contract-secret-value-0001"
let missingEnv = "TROUPEKIT_CONTRACT_KEY_THAT_IS_NOT_SET"
let codes: [String: UInt32] = [
    "Config": 1, "MissingKey": 2, "Catalog": 3, "Persona": 4, "Agent": 5, "Tool": 6,
    "Workflow": 7, "Graph": 8, "Store": 9, "Context": 10, "Failed": 11, "Paused": 12,
    "Retries": 13, "Cancelled": 14, "HostCallback": 15, "InvalidArgument": 16,
    "Unsupported": 17, "Panic": 18, "Internal": 99,
]

/// A failure a Swift host callback throws that is not a `HostError`.
/// Input: none. Output: an error UniFFI reports to Rust as unexpected,
/// which reaches the model as a `HostError`.
struct Boom: Error {}

/// A host model that answers from a script and records every request.
/// Input: the replies, one per `generate` call; the last one repeats.
/// Output: `requests` holds the rendered text of each request.
final class ScriptedModel: HostModel, @unchecked Sendable {
    private let lock = NSLock()
    private let replies: [String]
    private var seen: [String] = []

    init(_ replies: [String]) {
        precondition(!replies.isEmpty, "the script needs at least one reply")
        self.replies = replies
    }

    var requests: [String] {
        lock.withLock { seen }
    }

    func id() -> String {
        "scripted"
    }

    func capabilities() -> HostModelCaps {
        HostModelCaps(tools: false, jsonSchema: false, streaming: true, contextTokens: 4096)
    }

    func generate(request: HostRequest, sink: TokenSink) throws -> HostReply {
        let text = lock.withLock { () -> String in
            let turn = seen.count
            seen.append(render(request))
            return replies[min(turn, replies.count - 1)]
        }
        sink.delta(text: text)
        return HostReply(text: text, inputTokens: 3, outputTokens: 1)
    }
}

/// What a `RecordingTool` does when it is called.
/// Input: none. Output: echo the `text` argument, throw a `HostError`,
/// throw a non-`HostError`, or return fixed output.
enum ToolBehaviour {
    case echo
    case hostError(String)
    case crash
    case output(String)
}

/// A host tool that records its calls and the threads they ran on.
/// Input: its name, argument name and behaviour. Output: `calls` holds
/// the decoded arguments and whether the call ran on the main thread.
final class RecordingTool: HostTool, @unchecked Sendable {
    private let lock = NSLock()
    private let toolName: String
    private let behaviour: ToolBehaviour
    private let argument: String
    private var seen: [([String: String], Bool)] = []

    init(name: String = "echo", behaviour: ToolBehaviour = .echo, argument: String = "text") {
        self.toolName = name
        self.behaviour = behaviour
        self.argument = argument
    }

    var calls: [([String: String], Bool)] {
        lock.withLock { seen }
    }

    func name() -> String {
        toolName
    }

    func description() -> String {
        "Repeats the text it is given."
    }

    func schemaJson() -> String {
        #"{"type":"object","properties":{"\#(argument)":{"type":"string"}}}"#
    }

    func call(argsJson: String) throws -> String {
        let decoded = try? JSONSerialization.jsonObject(with: Data(argsJson.utf8))
        let args = decoded as? [String: String] ?? [:]
        lock.withLock { seen.append((args, Thread.isMainThread)) }
        switch behaviour {
        case .echo: return args[argument] ?? ""
        case .hostError(let message): throw HostError.Failed(detail: message)
        case .crash: throw Boom()
        case .output(let text): return text
        }
    }
}

/// A deterministic host embedder: a hashed bag of lowercase words.
/// Input: none. Output: `calls` counts `embed` calls; equal word sets give
/// equal vectors, so a query finds the document holding its words.
final class HashEmbedder: HostEmbedder, @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    var calls: Int {
        lock.withLock { count }
    }

    func dims() -> UInt32 {
        embedDims
    }

    func embed(texts: [String]) throws -> [[Float]] {
        lock.withLock { count += 1 }
        return texts.map(bagOfWords)
    }
}

/// A host transport that answers every request from a fixed script.
/// Input: body chunks, an optional head URL override, an optional failure.
/// Output: `requests` holds each request URL it got.
final class StubTransport: HostTransport, @unchecked Sendable {
    private let lock = NSLock()
    private let chunks: [Data]
    private let headUrl: String?
    private let failure: String?
    private var seen: [String] = []

    init(chunks: [Data], headUrl: String? = nil, failure: String? = nil) {
        self.chunks = chunks
        self.headUrl = headUrl
        self.failure = failure
    }

    var requests: [String] {
        lock.withLock { seen }
    }

    func send(request: HostHttpRequest, sink: HttpSink) throws {
        lock.withLock { seen.append(request.url) }
        if let failure {
            throw HostError.Failed(detail: failure)
        }
        let header = HostHeader(name: "content-type", value: "text/event-stream")
        sink.head(status: 200, url: headUrl ?? request.url, headers: [header])
        for chunk in chunks {
            sink.chunk(bytes: chunk)
        }
    }
}

/// An event listener that stores the kind of every event it gets.
/// Input: none. Output: `kinds` in arrival order; `finished` is fulfilled
/// once the run ends.
final class Collector: EventListener, @unchecked Sendable {
    private let lock = NSLock()
    private var seen: [String] = []
    let finished = XCTestExpectation(description: "run ended")

    var kinds: [String] {
        lock.withLock { seen }
    }

    func onEvent(event: Event) {
        let kind = kindName(event.kind)
        lock.withLock { seen.append(kind) }
        if ["runFinished", "runFailed", "runPaused"].contains(kind) {
            finished.fulfill()
        }
    }
}

/// Joins every text field of a host request for substring checks.
/// Input: a `HostRequest`. Output: one string with the system text, the
/// tool list and every message text, tool call and tool call id.
func render(_ request: HostRequest) -> String {
    var parts = [request.system ?? "", request.toolsJson ?? ""]
    for message in request.messages {
        parts += [message.role, message.text, message.toolCallsJson ?? "", message.toolCallId ?? ""]
    }
    return parts.joined(separator: "\n")
}

/// Hashes the lowercase words of `text` into a unit vector.
/// Input: a string. Output: `embedDims` floats, never all zero.
/// FNV-1a keeps the hash stable across runs and platforms.
func bagOfWords(_ text: String) -> [Float] {
    var vector = [Float](repeating: 0, count: Int(embedDims))
    for word in text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }) {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in word.utf8 {
            hash = (hash ^ UInt64(byte)) &* 0x100_0000_01b3
        }
        vector[Int(hash % UInt64(embedDims))] += 1
    }
    vector[0] += 0.001
    let norm = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
    return vector.map { $0 / norm }
}

/// Names an event kind the way the Python runner does, in lowerCamelCase.
/// Input: the kind. Output: its case name, for example `textDelta`.
/// Comparing names keeps pull and listener checks independent of payloads.
func kindName(_ kind: EventKind) -> String {
    switch kind {
    case .runStarted: return "runStarted"
    case .stepStarted: return "stepStarted"
    case .textDelta: return "textDelta"
    case .toolCallStarted: return "toolCallStarted"
    case .toolCallFinished: return "toolCallFinished"
    case .stepFinished: return "stepFinished"
    case .runPaused: return "runPaused"
    case .runFinished: return "runFinished"
    case .runFailed: return "runFailed"
    }
}

/// Renders an emulated tool call as the host model writes it (K8).
/// Input: the tool name and arguments. Output: the JSON text.
/// Arguments are strings only, which every case here needs.
func toolReply(_ name: String, _ arguments: [String: String]) -> String {
    let body: [String: Any] = ["tool": name, "arguments": arguments]
    let data = try! JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
    return String(decoding: data, as: UTF8.self)
}

/// Renders an emulated final answer as the host model writes it (K8).
/// Input: the answer text. Output: the JSON text.
/// Same shape as the Python runner's `answer_reply`.
func answerReply(_ text: String) -> String {
    let data = try! JSONSerialization.data(withJSONObject: ["answer": text])
    return String(decoding: data, as: UTF8.self)
}

/// Encodes one SSE `data:` event of the OpenAI-compatible chat stream.
/// Input: the event object. Output: its bytes, ending in a blank line.
/// Used by the in-process transport that stands in for the cloud stub.
func sse(_ payload: [String: Any]) -> Data {
    let json = try! JSONSerialization.data(withJSONObject: payload)
    return Data("data: ".utf8) + json + Data("\n\n".utf8)
}

/// The streamed reply of the stub vendor: one delta per answer part.
/// Input: none. Output: the SSE chunks, ending with stop and `[DONE]`.
/// Each chunk is one `HttpSink.chunk` call.
func answerChunks() -> [Data] {
    var chunks = answerParts.map { part -> Data in
        let delta: [String: Any] = ["index": 0, "delta": ["content": part]]
        return sse(["choices": [delta]])
    }
    let stop: [String: Any] = ["index": 0, "delta": [String: String](), "finish_reason": "stop"]
    chunks.append(sse(["choices": [stop]]))
    chunks.append(Data("data: [DONE]\n\n".utf8))
    return chunks
}

/// Builds a catalog TOML naming the stub vendor at `stubBase`.
/// Input: the auth style, `none` or `bearer`, and the base URL.
/// Output: catalog text with vendor `stub` and model `stub-model`.
func catalog(auth: String = "none", base: String = stubBase) -> String {
    """
    [[vendor]]
    id = "stub"
    name = "Stub"
    protocol = "openai_compatible"
    base_url = "\(base)"
    auth = "\(auth)"

    [[vendor.model]]
    id = "stub-model"
    context_window = 8192
    input_usd_micros_per_mtok = 0
    output_usd_micros_per_mtok = 0
    price_verified = true
    """
}

/// Builds an agent on a host model with the given host tools.
/// Input: the model, tools and an optional SQLite path.
/// Output: the unbuilt `AgentBuilder`.
func hostAgent(
    _ model: HostModel, _ tools: [HostTool] = [], sqlite: String? = nil
) -> AgentBuilder {
    var builder = AgentBuilder(name: "helper").instructions(text: "Help the user.")
        .hostModel(model: model)
    for tool in tools {
        builder = builder.tool(tool: tool)
    }
    if let sqlite {
        builder = builder.sqlite(path: sqlite)
    }
    return builder
}

/// Builds an agent on the stub vendor through the cloud path and a transport.
/// Input: the transport and an optional SQLite path.
/// Output: the unbuilt `AgentBuilder`.
func cloudAgent(_ transport: HostTransport, sqlite: String? = nil) -> AgentBuilder {
    var builder = AgentBuilder(name: "helper").instructions(text: "Help the user.")
        .model(selector: selector).catalogToml(text: catalog()).hostTransport(transport: transport)
    if let sqlite {
        builder = builder.sqlite(path: sqlite)
    }
    return builder
}

/// Pulls every event of a stream until it ends.
/// Input: a `RunStream`. Output: the events; fails past `eventsMax`.
/// Uses the `events` sugar from `Troupekit+Async.swift`.
func pullAll(_ stream: RunStream) async throws -> [Event] {
    var events: [Event] = []
    for try await event in stream.events {
        events.append(event)
        if events.count > eventsMax {
            XCTFail("stream did not end within eventsMax events")
            break
        }
    }
    return events
}

/// Returns the numeric code and message of a thrown `TroupekitError`.
/// Input: any error. Output: (code, message); fails the test otherwise.
/// The code is the stable number of plan section 6.1.
func failure(_ error: Error) -> (UInt32, String) {
    guard case let TroupekitError.Failure(code, message, _) = error else {
        XCTFail("expected TroupekitError, got \(error)")
        return (0, "")
    }
    return (code, message)
}

/// Runs `body` and returns its `TroupekitError` as (code, message).
/// Input: a throwing async body. Output: the failure; fails the test when
/// the body returns normally.
func expectFailure(_ body: () async throws -> Void) async -> (UInt32, String) {
    do {
        try await body()
    } catch {
        return failure(error)
    }
    XCTFail("expected a TroupekitError")
    return (0, "")
}

/// A fresh temporary directory for one case.
/// Input: none. Output: its path; the system removes it later.
/// Each case gets its own SQLite files.
func scratch() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

/// The 25 contract cases of `troupekit-ffi-core/contract/cases.txt`, through
/// the UniFFI Swift bindings. The runtime is set once with a 16-event
/// buffer, small enough for the full-buffer case and enough for the others.
final class ContractTests: XCTestCase {
    /// Whether `initialize` took the 16-event buffer. It answers
    /// `InvalidArgument` when another test in this process used the runtime
    /// first; only the full-buffer case depends on the buffer, so it skips.
    static var bufferSet = false

    override class func setUp() {
        super.setUp()
        let settings = RuntimeSettings(workerThreads: 2, maxInputBytes: 1_048_576, eventBuffer: 16)
        do {
            try initialize(settings: settings)
            bufferSet = true
        } catch {
            guard case let TroupekitError.Failure(code, _, _) = error else {
                fatalError("initialize threw a non-TroupekitError: \(error)")
            }
            precondition(code == codes["InvalidArgument"], "initialize failed: \(error)")
        }
    }

    func test_version_matches_core() {
        let parts = version().split(separator: ".")
        XCTAssertEqual(parts.count, 3)
        XCTAssertTrue(parts.allSatisfy { UInt($0) != nil })
    }

    func test_missing_model_is_config_error() async throws {
        let (code, _) = await expectFailure {
            _ = try await AgentBuilder(name: "helper").instructions(text: "Help.").build()
        }
        XCTAssertEqual(code, codes["Config"])
    }

    func test_missing_key_names_env_not_value() async throws {
        unsetenv(missingEnv)
        let keyed = { AgentBuilder(name: "helper").instructions(text: "Help.")
            .model(selector: selector)
            .catalogToml(text: catalog(auth: "bearer", base: "https://127.0.0.1:9/v1")) }
        let (code, message) = await expectFailure {
            _ = try await keyed().apiKeyEnv(name: missingEnv).build().run(input: "hi")
        }
        XCTAssertEqual(code, codes["MissingKey"])
        XCTAssertTrue(message.contains(missingEnv))
        let transport = StubTransport(chunks: [], failure: "connection refused")
        let (_, leaked) = await expectFailure {
            let agent = keyed().apiKey(key: secret).hostTransport(transport: transport)
            _ = try await agent.build().run(input: "hi")
        }
        XCTAssertFalse(leaked.contains(secret))
    }

    func test_host_model_run_returns_text() async throws {
        let model = ScriptedModel(["Hello from the host model"])
        let answer = try await hostAgent(model).build().run(input: "hi")
        XCTAssertEqual(answer, "Hello from the host model")
        XCTAssertEqual(model.requests.count, 1)
    }

    func test_host_tool_called_with_json_args() async throws {
        let model = ScriptedModel([toolReply("echo", ["text": "hi"]), answerReply("done")])
        let tool = RecordingTool()
        let answer = try await hostAgent(model, [tool]).build().run(input: "echo hi")
        XCTAssertEqual(answer, "done")
        XCTAssertEqual(tool.calls.map { $0.0 }, [["text": "hi"]])
        XCTAssertEqual(model.requests.count, 2)
    }

    func test_host_tool_error_reaches_model() async throws {
        let model = ScriptedModel([toolReply("echo", ["text": "hi"]), answerReply("recovered")])
        let tool = RecordingTool(behaviour: .hostError("echo is offline"))
        let answer = try await hostAgent(model, [tool]).build().run(input: "echo hi")
        XCTAssertEqual(answer, "recovered")
        XCTAssertTrue(model.requests[1].contains("echo is offline"))
    }

    func test_host_tool_throw_does_not_crash() async throws {
        let call = toolReply("echo", ["text": "hi"])
        let model = ScriptedModel([call, answerReply("still here")])
        let tool = RecordingTool(behaviour: .crash)
        let answer = try await hostAgent(model, [tool]).build().run(input: "echo hi")
        XCTAssertEqual(answer, "still here")
        XCTAssertEqual(tool.calls.count, 1)
        XCTAssertEqual(model.requests.count, 2)
    }

    func test_stream_yields_deltas_then_finished() async throws {
        let transport = StubTransport(chunks: answerChunks())
        let stream = try await cloudAgent(transport).build().stream(input: "hi")
        let events = try await pullAll(stream)
        let deltas = events.compactMap { event -> String? in
            if case let .textDelta(text) = event.kind { return text }
            return nil
        }
        XCTAssertEqual(deltas.joined(), answerParts.joined())
        XCTAssertEqual(events.first.map { kindName($0.kind) }, "runStarted")
        XCTAssertEqual(events.last.map { kindName($0.kind) }, "runFinished")
        XCTAssertEqual(stream.droppedDeltas(), 0)
    }

    func test_listener_sees_same_events_as_pull() async throws {
        let first = try await hostAgent(ScriptedModel(["same"])).build()
        let pulled = try await pullAll(first.stream(input: "hi"))
        let collector = Collector()
        let stream = try await hostAgent(ScriptedModel(["same"])).build().stream(input: "hi")
        try stream.setListener(listener: collector)
        _ = try await stream.result()
        await fulfillment(of: [collector.finished], timeout: 30)
        XCTAssertEqual(collector.kinds, pulled.map { kindName($0.kind) })
    }

    func test_start_returns_run_id_before_model_call() async throws {
        let transport = StubTransport(chunks: answerChunks())
        let database = try scratch().appendingPathComponent("runs.db").path
        let runId = try await cloudAgent(transport, sqlite: database).build().start(input: "hi")
        XCTAssertFalse(runId.isEmpty)
        XCTAssertEqual(transport.requests.count, 0)
    }

    func test_resume_after_handles_dropped() async throws {
        let transport = StubTransport(chunks: answerChunks())
        let database = try scratch().appendingPathComponent("runs.db").path
        var first: AgentHandle? = try await cloudAgent(transport, sqlite: database).build()
        let runId = try await first!.start(input: "hi")
        first = nil
        let second = try await cloudAgent(transport, sqlite: database).build()
        let answer = try await second.resume(runId: runId)
        XCTAssertNil(first)
        XCTAssertEqual(answer, answerParts.joined())
        XCTAssertEqual(transport.requests.count, 1)
    }

    func test_resume_for_zero_budget_is_pending() async throws {
        let transport = StubTransport(chunks: answerChunks())
        let database = try scratch().appendingPathComponent("runs.db").path
        let agent = try await cloudAgent(transport, sqlite: database).build()
        let runId = try await agent.start(input: "hi")
        let progress = try await agent.resumeFor(runId: runId, budgetMs: 0)
        guard case let .pending(pendingId) = progress else {
            return XCTFail("expected Progress.pending, got \(progress)")
        }
        XCTAssertEqual(pendingId, runId)
        XCTAssertEqual(transport.requests.count, 0)
    }

    func test_cancel_keeps_checkpoint_and_resumes() async throws {
        let transport = StubTransport(chunks: answerChunks())
        let database = try scratch().appendingPathComponent("runs.db").path
        let agent = try await cloudAgent(transport, sqlite: database).build()
        let stream = try await agent.stream(input: "hi")
        stream.cancel()
        let (code, _) = await expectFailure { _ = try await stream.result() }
        XCTAssertEqual(code, codes["Cancelled"])
        let answer = try await agent.resume(runId: stream.runId())
        XCTAssertEqual(answer, answerParts.joined())
    }

    func test_toml_agent_runs() async throws {
        let text = try String(contentsOf: fixture("agent.toml"), encoding: .utf8)
        let builder = AgentBuilder.fromToml(text: text).catalogToml(text: catalog())
            .hostTransport(transport: StubTransport(chunks: answerChunks()))
        let answer = try await builder.build().run(input: "hi")
        XCTAssertEqual(answer, answerParts.joined())
    }

    func test_toml_over_cap_refused() async throws {
        let text = try String(contentsOf: fixture("agent.toml"), encoding: .utf8)
            + "\n" + String(repeating: "a", count: 1_048_576)
        let (code, message) = await expectFailure {
            _ = try await AgentBuilder.fromToml(text: text).build()
        }
        XCTAssertTrue([codes["Config"], codes["InvalidArgument"]].contains(code))
        XCTAssertTrue(message.contains("max_input_bytes") || message.contains("MiB"))
    }

    func test_graph_agent_then_step_runs() async throws {
        let writer = hostAgent(ScriptedModel(["draft"]))
        let stamp = RecordingTool(name: "stamp", behaviour: .output("stamped"), argument: "writer")
        let graph = GraphBuilder(name: "flow").agent(id: "writer", agent: writer)
            .step(id: "stamp", tool: stamp).edge(from: "writer", to: "stamp")
        let result = try await graph.build().runResult(input: "go")
        XCTAssertEqual(stamp.calls.count, 1)
        XCTAssertEqual(stamp.calls.first?.0, ["writer": "draft"])
        XCTAssertTrue(result.outputJson.contains("stamped"))
    }

    func test_knowledge_with_host_embedder_finds_doc() async throws {
        let root = try scratch()
        let docs = root.appendingPathComponent("docs")
        try FileManager.default.createDirectory(at: docs, withIntermediateDirectories: true)
        try "The secret fruit is pineapple.\n".write(
            to: docs.appendingPathComponent("fruit.md"), atomically: true, encoding: .utf8)
        let embedder = HashEmbedder()
        let search = toolReply("search_knowledge", ["query": "secret fruit pineapple"])
        let model = ScriptedModel([search, answerReply("pineapple")])
        let config = KnowledgeConfig(dir: docs.path, index: nil, store: nil, allowFailed: false)
        let agent = hostAgent(model, sqlite: root.appendingPathComponent("kb.db").path)
            .knowledge(config: config, embedder: embedder)
        let answer = try await agent.build().run(input: "What is the secret fruit?")
        XCTAssertEqual(answer, "pineapple")
        XCTAssertGreaterThan(embedder.calls, 0)
        XCTAssertTrue(model.requests[1].contains("The secret fruit is pineapple"))
    }

    func test_memory_facts_persist_between_runs() async throws {
        let database = try scratch().appendingPathComponent("memory.db").path
        let config = MemoryConfig(
            user: "user-1", session: "session-1", sqlite: database, preamble: true)
        let remember = toolReply("remember_fact", ["key": "name", "value": "Ada Lovelace"])
        let first = ScriptedModel([remember, answerReply("noted")])
        let rememberer = try await hostAgent(first).memory(config: config).build()
        let noted = try await rememberer.run(input: "My name is Ada Lovelace")
        XCTAssertEqual(noted, "noted")
        let second = ScriptedModel([answerReply("Hello Ada")])
        let recaller = try await hostAgent(second).memory(config: config).build()
        let hello = try await recaller.run(input: "Who am I?")
        XCTAssertEqual(hello, "Hello Ada")
        XCTAssertTrue(second.requests[0].contains("Ada Lovelace"))
    }

    func test_error_codes_are_stable() {
        let mirrored: [(ErrorCode, String)] = [
            (.config, "Config"), (.missingKey, "MissingKey"), (.catalog, "Catalog"),
            (.persona, "Persona"), (.agent, "Agent"), (.tool, "Tool"), (.workflow, "Workflow"),
            (.graph, "Graph"), (.store, "Store"), (.context, "Context"), (.failed, "Failed"),
            (.paused, "Paused"), (.retries, "Retries"), (.cancelled, "Cancelled"),
            (.hostCallback, "HostCallback"), (.invalidArgument, "InvalidArgument"),
            (.unsupported, "Unsupported"), (.panic, "Panic"), (.internal, "Internal"),
        ]
        XCTAssertEqual(mirrored.count, codes.count)
        for (code, name) in mirrored {
            XCTAssertEqual(code.rawValue, codes[name], name)
        }
    }

    func test_laws_path_with_dotdot_refused() async throws {
        let agent = hostAgent(ScriptedModel(["x"])).laws(path: "../laws.bend")
        let (code, message) = await expectFailure { _ = try await agent.build() }
        XCTAssertTrue([codes["Config"], codes["InvalidArgument"]].contains(code))
        XCTAssertTrue(message.contains(".."))
    }

    func test_host_transport_refuses_other_origin() async throws {
        let other = "https://elsewhere.invalid/v1/chat/completions"
        let transport = StubTransport(chunks: [Data("data: [DONE]\n\n".utf8)], headUrl: other)
        let (code, _) = await expectFailure {
            _ = try await cloudAgent(transport).build().run(input: "hi")
        }
        XCTAssertNotEqual(code, 0)
        XCTAssertEqual(transport.requests.count, 1)
    }

    func test_host_transport_streams_chunks() async throws {
        let transport = StubTransport(chunks: answerChunks())
        let events = try await pullAll(cloudAgent(transport).build().stream(input: "hi"))
        let deltas = events.compactMap { event -> String? in
            if case let .textDelta(text) = event.kind { return text }
            return nil
        }
        XCTAssertEqual(deltas.joined(), answerParts.joined())
        XCTAssertTrue(transport.requests[0].hasPrefix(stubBase))
    }

    func test_full_buffer_drops_deltas_not_lifecycle() async throws {
        try XCTSkipUnless(ContractTests.bufferSet, "runtime set up before this runner")
        let chatty = ScriptedModel([String(repeating: "x", count: 500)])
        let stream = try await hostAgent(chatty).build().stream(input: "hi")
        _ = try await stream.result()
        let kinds = try await pullAll(stream).map { kindName($0.kind) }
        XCTAssertEqual(kinds.first, "runStarted")
        XCTAssertEqual(kinds.last, "runFinished")
        XCTAssertLessThanOrEqual(kinds.filter { $0 == "textDelta" }.count, 16)
    }

    func test_callback_from_worker_thread_no_deadlock() async throws {
        let model = ScriptedModel([toolReply("echo", ["text": "hi"]), answerReply("done")])
        let tool = RecordingTool()
        let agent = try await hostAgent(model, [tool]).build()
        let answer = try await agent.run(input: "echo hi")
        XCTAssertEqual(answer, "done")
        XCTAssertEqual(tool.calls.count, 1)
        XCTAssertFalse(tool.calls[0].1)
    }

    func test_concurrent_runs_on_one_handle() async throws {
        let model = ScriptedModel(["parallel"])
        let agent = try await hostAgent(model).build()
        let answers = try await withThrowingTaskGroup(of: String.self) { group in
            for _ in 0..<concurrentRuns {
                group.addTask { try await agent.run(input: "hi") }
            }
            var out: [String] = []
            for try await answer in group {
                out.append(answer)
            }
            return out
        }
        XCTAssertEqual(answers, Array(repeating: "parallel", count: concurrentRuns))
        XCTAssertEqual(model.requests.count, concurrentRuns)
    }

    func test_host_request_debug_hides_key() {
        let headers = [
            HostHeader(name: "Authorization", value: "Bearer \(secret)", sensitive: false),
            HostHeader(name: "x-trace", value: "sk-flagged-secret", sensitive: true),
            HostHeader(name: "content-type", value: "application/json", sensitive: false),
        ]
        let request = HostHttpRequest(
            method: "POST", url: "https://api.stub.test/v1/chat/completions",
            headers: headers, body: Data())
        let texts = [
            String(describing: request), String(reflecting: request), "\(request)",
            String(describing: headers[0]), String(reflecting: headers[1]),
        ]
        for text in texts {
            XCTAssertFalse(text.contains(secret), text)
            XCTAssertFalse(text.contains("sk-flagged-secret"), text)
        }
        XCTAssertTrue(texts[0].contains("application/json"), texts[0])
        XCTAssertTrue(texts[1].contains("application/json"), texts[1])
    }

    func testEveryCaseIsImplemented() throws {
        let text = try String(contentsOf: casesFile(), encoding: .utf8)
        let names = text.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        XCTAssertFalse(names.isEmpty)
        let methods = Set(ContractTests.defaultTestSuite.tests.compactMap { test -> String? in
            test.name.split(separator: " ").last.map { String($0.dropLast()) }
        })
        let missing = names.filter { !methods.contains("test_\($0)") }
        XCTAssertEqual(missing, [])
    }
}

/// The repository root, found from this file's path.
/// Input: none. Output: the root URL; the tests run from the source tree.
/// This file sits at `bindings/swift/Tests/TroupekitTests/`.
func repoRoot() -> URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
}

/// The shared contract case list of `troupekit-ffi-core`.
/// Input: none. Output: the URL of `contract/cases.txt`.
/// Every runner reads the same file.
func casesFile() -> URL {
    repoRoot().appendingPathComponent("crates/troupekit-ffi-core/contract/cases.txt")
}

/// A persona fixture shared with the Python runner.
/// Input: the file name. Output: its URL under the troupekit-py tests.
/// One fixture keeps the TOML case identical in every language.
func fixture(_ name: String) -> URL {
    repoRoot().appendingPathComponent("crates/troupekit-py/tests/fixtures")
        .appendingPathComponent(name)
}

import Foundation
import NaturalLanguage
import XCTest
import Troupekit
@testable import TroupekitApple

/// Serves canned replies to URLSession in tests, keyed by absolute URL.
/// A reply is a status, headers, body, and an optional redirect target.
/// Input: `StubProtocol.set` called before the request. Output: the reply.
final class StubProtocol: URLProtocol {
    struct Reply {
        var status: Int
        var headers: [String: String]
        var body: Data
        var redirect: URL?
    }

    private static let lock = NSLock()
    private static var table: [String: Reply] = [:]

    static func set(_ url: String, _ reply: Reply) {
        lock.lock()
        table[url] = reply
        lock.unlock()
    }

    static func reply(for url: String) -> Reply? {
        lock.lock()
        defer { lock.unlock() }
        return table[url]
    }

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let reply = Self.reply(for: url.absoluteString) else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        let response = HTTPURLResponse(url: url, statusCode: reply.status, httpVersion: "HTTP/1.1", headerFields: reply.headers)!
        if let target = reply.redirect {
            client?.urlProtocol(self, wasRedirectedTo: URLRequest(url: target), redirectResponse: response)
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: reply.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// Tests of the Apple platform adapters: model mapping, availability, transport, embedder.
/// CI runners have no Apple Intelligence, so the model tests cover mapping and errors,
/// and the end-to-end test skips with a printed reason when the model is unavailable.
final class AppleAdapterTests: XCTestCase {
    private func stubbedTransport() -> URLSessionTransport {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubProtocol.self]
        return URLSessionTransport(configuration: configuration)
    }

    func testUnavailableModelThrowsNamedHostError() throws {
        let model = AppleFoundationModel()
        guard let message = model.unavailableMessage() else {
            throw XCTSkip("apple model available on this host; unavailable path not reachable")
        }
        XCTAssertTrue(message.hasPrefix("apple model unavailable: "), message)
        let known = [
            "apple model unavailable: device not eligible",
            "apple model unavailable: apple intelligence off",
            "apple model unavailable: model not ready",
            "apple model unavailable: needs iOS 26 or macOS 26",
            "apple model unavailable: sdk without FoundationModels",
        ]
        XCTAssertTrue(known.contains(message), message)
        let request = HostRequest(system: nil, messages: [HostMessage(role: "user", text: "hi", toolCallsJson: nil, toolCallId: nil)], toolsJson: nil, schemaJson: nil, maxTokens: 64)
        XCTAssertThrowsError(try model.generateText(request: request, onDelta: { _ in })) { error in
            XCTAssertTrue("\(error)".contains(message), "\(error)")
        }
    }

    func testRequestMappingKeepsSystemAndTurns() {
        let request = HostRequest(
            system: "You are support.",
            messages: [
                HostMessage(role: "user", text: "Where is my order 42?", toolCallsJson: nil, toolCallId: nil),
                HostMessage(role: "assistant", text: "", toolCallsJson: "[{\"id\":\"c1\",\"name\":\"lookup\",\"arguments\":{\"id\":42}}]", toolCallId: nil),
                HostMessage(role: "tool", text: "shipped", toolCallsJson: nil, toolCallId: "c1"),
            ],
            toolsJson: nil,
            schemaJson: nil,
            maxTokens: 128
        )
        let mapped = AppleFoundationModel.render(request)
        XCTAssertEqual(mapped.instructions, "You are support.")
        let user = mapped.prompt.range(of: "user: Where is my order 42?")
        let call = mapped.prompt.range(of: "assistant tool call: [{\"id\":\"c1\",\"name\":\"lookup\",\"arguments\":{\"id\":42}}]")
        let result = mapped.prompt.range(of: "tool result c1: shipped")
        XCTAssertNotNil(user)
        XCTAssertNotNil(call)
        XCTAssertNotNil(result)
        XCTAssertLessThan(user!.lowerBound, call!.lowerBound)
        XCTAssertLessThan(call!.lowerBound, result!.lowerBound)
        XCTAssertTrue(mapped.prompt.hasSuffix("assistant:"), mapped.prompt)
    }

    func testRequestMappingWithoutSystemUsesEmptyInstructions() {
        let request = HostRequest(system: nil, messages: [HostMessage(role: "user", text: "hi", toolCallsJson: nil, toolCallId: nil)], toolsJson: nil, schemaJson: nil, maxTokens: 16)
        let mapped = AppleFoundationModel.render(request)
        XCTAssertEqual(mapped.instructions, "")
        XCTAssertEqual(mapped.prompt, "user: hi\nassistant:")
    }

    func testDeltaDifferPushesOnlyNewSuffix() {
        var differ = SnapshotDiffer()
        XCTAssertEqual(differ.delta(for: "Hel"), "Hel")
        XCTAssertEqual(differ.delta(for: "Hello"), "lo")
        XCTAssertEqual(differ.delta(for: "Hello"), "")
        XCTAssertEqual(differ.delta(for: "Help"), "Help")
    }

    func testSchemaWithUnsupportedKeywordIsRefused() {
        let schema = "{\"type\":\"object\",\"patternProperties\":{\"^a\":{\"type\":\"string\"}}}"
        XCTAssertThrowsError(try AppleSchema.check(schema)) { error in
            XCTAssertTrue("\(error)".contains("apple model: schema keyword patternProperties unsupported"), "\(error)")
        }
    }

    func testSchemaWithFreeFormObjectIsRefused() {
        let schema = "{\"type\":\"object\",\"properties\":{\"arguments\":{\"type\":\"object\"}},\"required\":[\"arguments\"]}"
        XCTAssertThrowsError(try AppleSchema.check(schema)) { error in
            XCTAssertTrue("\(error)".contains("free-form object"), "\(error)")
        }
    }

    func testSchemaAnswerOrToolShapeIsAccepted() throws {
        let schema = """
        {"anyOf":[
          {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]},
          {"type":"object","properties":{"tool":{"type":"string","enum":["lookup"]},"arguments":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}},"required":["tool","arguments"]}
        ]}
        """
        XCTAssertNoThrow(try AppleSchema.check(schema))
    }

    func testSchemaConstIsOneValueEnumAndRequiredOrderIsKept() throws {
        let schema = "{\"type\":\"object\",\"properties\":{\"arguments\":{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"integer\"}}},\"tool\":{\"type\":\"string\",\"const\":\"lookup\"}},\"required\":[\"tool\",\"arguments\"]}"
        guard case let .object(_, _, properties) = try AppleSchema.check(schema) else {
            return XCTFail("root is not an object")
        }
        XCTAssertEqual(properties.map(\.name), ["tool", "arguments"])
        XCTAssertEqual(properties.first?.node, .string(values: ["lookup"]))
    }

    func testTransportReportsFinalURL() async throws {
        let url = "https://api.example.test/v1/chat"
        StubProtocol.set(url, .init(status: 200, headers: ["content-type": "text/event-stream"], body: Data("data: hi\n\n".utf8), redirect: nil))
        var heads: [(Int, String)] = []
        var body = Data()
        let request = HostHttpRequest(method: "POST", url: url, headers: [HostHeader(name: "authorization", value: "Bearer k")], body: Data("{}".utf8))
        try await stubbedTransport().perform(request, head: { status, final, _ in heads.append((status, final)) }, chunk: { body.append($0) })
        XCTAssertEqual(heads.count, 1)
        XCTAssertEqual(heads.first?.0, 200)
        XCTAssertEqual(heads.first?.1, url)
        XCTAssertEqual(String(decoding: body, as: UTF8.self), "data: hi\n\n")
    }

    func testTransportRefusesRedirects() async throws {
        let url = "https://api.example.test/v1/redirect"
        StubProtocol.set(url, .init(status: 302, headers: ["location": "https://evil.example.test/"], body: Data(), redirect: URL(string: "https://evil.example.test/")!))
        StubProtocol.set("https://evil.example.test/", .init(status: 200, headers: [:], body: Data("leaked".utf8), redirect: nil))
        var finals: [String] = []
        var body = Data()
        let request = HostHttpRequest(method: "GET", url: url, headers: [], body: Data())
        try await stubbedTransport().perform(request, head: { _, final, _ in finals.append(final) }, chunk: { body.append($0) })
        XCTAssertEqual(finals, [url])
        XCTAssertFalse(String(decoding: body, as: UTF8.self).contains("leaked"))
    }

    func testEmbedderReturnsDeclaredDims() throws {
        guard let embedder = AppleEmbedder(language: .english) else {
            throw XCTSkip("no english sentence embedding on this host")
        }
        let dims = Int(embedder.dims())
        XCTAssertGreaterThan(dims, 0)
        let vectors = try embedder.embed(texts: ["refund policy", "shipping times"])
        XCTAssertEqual(vectors.count, 2)
        XCTAssertTrue(vectors.allSatisfy { $0.count == dims })
    }

    func testAppleModelAnswersTomlAgent() async throws {
        let model = AppleFoundationModel()
        if let message = model.unavailableMessage() {
            print("SKIP testAppleModelAnswersTomlAgent: \(message)")
            throw XCTSkip(message)
        }
        let toml = """
        name = "support"
        display_name = "Support"
        role = "Answers customers"
        goal = "Answer in one sentence"
        instructions = "Answer in one short sentence."
        """
        let agent = try await AgentBuilder.fromToml(text: toml).hostModel(model: model).build()
        let answer = try await agent.run(input: "Say hello.")
        XCTAssertFalse(answer.isEmpty)
    }
}

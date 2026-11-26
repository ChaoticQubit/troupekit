import Foundation
import XCTest
@testable import Troupekit

let benchToolOutput = "noop-ok"
let benchIterationsMax = 1_000_000

/// A host model that calls the no-op tool once per run, then answers.
/// Input: none. Output: emulated tool calls (K8); stateless across runs,
/// the same workload as `BenchModel` in `troupekit-ffi/examples/overhead.rs`.
final class BenchModel: HostModel, @unchecked Sendable {
    func id() -> String {
        "bench"
    }

    func capabilities() -> HostModelCaps {
        HostModelCaps(tools: false, jsonSchema: false, streaming: true, contextTokens: 4096)
    }

    func generate(request: HostRequest, sink: TokenSink) throws -> HostReply {
        let called = request.messages.contains { $0.text.contains(benchToolOutput) }
        let text = called ? #"{"answer": "done"}"# : #"{"tool": "noop", "arguments": {}}"#
        sink.delta(text: text)
        return HostReply(text: text, inputTokens: 1, outputTokens: 1)
    }
}

/// A host model that answers at once with one short text.
/// Input: none. Output: `x` for every request.
/// Used by the event benches, where only the run's events matter.
final class PlainModel: HostModel, @unchecked Sendable {
    func id() -> String {
        "plain"
    }

    func capabilities() -> HostModelCaps {
        HostModelCaps(tools: false, jsonSchema: false, streaming: true, contextTokens: 4096)
    }

    func generate(request: HostRequest, sink: TokenSink) throws -> HostReply {
        sink.delta(text: "x")
        return HostReply(text: "x", inputTokens: 1, outputTokens: 1)
    }
}

/// A host tool that does nothing and counts its calls.
/// Input: none. Output: `benchToolOutput`; `calls` is the number of calls,
/// which proves every iteration crossed the tool boundary.
final class NoopTool: HostTool, @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    var calls: Int {
        lock.withLock { count }
    }

    func name() -> String {
        "noop"
    }

    func description() -> String {
        "Does nothing."
    }

    func schemaJson() -> String {
        #"{"type":"object","properties":{}}"#
    }

    func call(argsJson: String) throws -> String {
        lock.withLock { count += 1 }
        return benchToolOutput
    }
}

/// Signals the end of each listened run.
/// Input: none. Output: `ended` is signalled once per terminal event.
/// One instance serves every iteration.
final class Ender: EventListener, @unchecked Sendable {
    let ended = DispatchSemaphore(value: 0)

    func onEvent(event: Event) {
        switch event.kind {
        case .runFinished, .runFailed, .runPaused: ended.signal()
        default: break
        }
    }
}

/// Times `body` for `count` runs and records one result line.
/// Input: the bench name, count, label and body. Output: the line, also
/// appended to `TROUPEKIT_BENCH_OUT` when set; fails unless `count` ran.
func benchmark(_ name: String, _ count: Int, _ label: String, _ body: () async throws -> Void)
    async throws -> String
{
    var samples: [UInt64] = []
    samples.reserveCapacity(count)
    for _ in 0..<count {
        let start = DispatchTime.now().uptimeNanoseconds
        try await body()
        samples.append(DispatchTime.now().uptimeNanoseconds - start)
    }
    XCTAssertEqual(samples.count, count)
    samples.sort()
    let p50 = samples[count / 2]
    let p99 = samples[min(count * 99 / 100, count - 1)]
    let line = "overhead lang=swift bench=\(name) n=\(count)"
        + " p50_ns=\(p50) p99_ns=\(p99) label=\(label)"
    print(line)
    if let path = ProcessInfo.processInfo.environment["TROUPEKIT_BENCH_OUT"] {
        let handle = FileHandle(forWritingAtPath: path) ?? {
            FileManager.default.createFile(atPath: path, contents: nil)
            return FileHandle(forWritingAtPath: path)!
        }()
        handle.seekToEndOfFile()
        handle.write(Data((line + "\n").utf8))
        try handle.close()
    }
    return line
}

/// B8: the cost of crossing the UniFFI boundary from Swift, best and worst
/// case per the label. Runs only with `TROUPEKIT_BENCH=1`, in CI.
final class OverheadTests: XCTestCase {
    func testOverhead() async throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["TROUPEKIT_BENCH"] == "1", "benchmarks run in CI only")
        let count = Int(environment["TROUPEKIT_BENCH_N"] ?? "10000") ?? 0
        XCTAssertTrue(count > 0 && count <= benchIterationsMax)
        let label = environment["TROUPEKIT_BENCH_LABEL"] ?? "quiet"

        var versions = 0
        _ = try await benchmark("version", count, label) {
            XCTAssertFalse(version().isEmpty)
            versions += 1
        }
        XCTAssertEqual(versions, count)

        let tool = NoopTool()
        let agent = try await AgentBuilder(name: "bench").instructions(text: "Benchmark.")
            .hostModel(model: BenchModel()).tool(tool: tool).build()
        _ = try await benchmark("tool_round_trip", count, label) {
            let answer = try await agent.run(input: "go")
            XCTAssertEqual(answer, "done")
        }
        XCTAssertEqual(tool.calls, count)

        let plain = try await AgentBuilder(name: "bench").instructions(text: "Benchmark.")
            .hostModel(model: PlainModel()).build()
        var pulled = 0
        _ = try await benchmark("event_pull", count, label) {
            let stream = try await plain.stream(input: "go")
            while try await stream.next() != nil {}
            pulled += 1
        }
        XCTAssertEqual(pulled, count)

        let ender = Ender()
        var listened = 0
        _ = try await benchmark("event_listener", count, label) {
            let stream = try await plain.stream(input: "go")
            try stream.setListener(listener: ender)
            _ = try await stream.result()
            XCTAssertEqual(ender.ended.wait(timeout: .now() + 30), .success)
            listened += 1
        }
        XCTAssertEqual(listened, count)
    }
}

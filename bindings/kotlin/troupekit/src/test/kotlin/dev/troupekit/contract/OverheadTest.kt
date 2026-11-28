package dev.troupekit.contract

import dev.troupekit.AgentBuilder
import dev.troupekit.Event
import dev.troupekit.EventKind
import dev.troupekit.EventListener
import dev.troupekit.HostModel
import dev.troupekit.HostModelCaps
import dev.troupekit.HostReply
import dev.troupekit.HostRequest
import dev.troupekit.HostTool
import dev.troupekit.TokenSink
import dev.troupekit.version
import java.io.File
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

const val BENCH_TOOL_OUTPUT = "noop-ok"
const val BENCH_ITERATIONS_MAX = 1_000_000
const val ANSWER_REPLY = """{"answer": "done"}"""
const val TOOL_CALL_REPLY = """{"tool": "noop", "arguments": {}}"""

/**
 * Untimed runs before each bench's samples: a tenth of the count, at
 * least one. Input: the sample count. Output: the warm-up count.
 * Same rule as `warmup_of` in `troupekit-ffi/examples/overhead.rs`.
 */
fun warmupOf(count: Int): Int {
    check(count > 0)
    val warmup = maxOf(count / 10, 1)
    check(warmup <= count)
    return warmup
}

/**
 * A host model that calls the no-op tool once per run, then answers.
 * Input: none. Output: emulated tool calls (K8); stateless across runs,
 * the same workload as `BenchModel` in `troupekit-ffi/examples/overhead.rs`.
 */
class BenchModel : HostModel {
    override fun id(): String = "bench"

    override fun capabilities(): HostModelCaps =
        HostModelCaps(tools = false, jsonSchema = false, streaming = true, contextTokens = 4096u)

    override fun generate(request: HostRequest, sink: TokenSink): HostReply {
        val called = request.messages.any { it.text.contains(BENCH_TOOL_OUTPUT) }
        val text = if (called) ANSWER_REPLY else TOOL_CALL_REPLY
        sink.delta(text)
        return HostReply(text = text, inputTokens = 1uL, outputTokens = 1uL)
    }
}

/**
 * A host model that answers at once with one short text.
 * Input: none. Output: `x` for every request.
 * Used by the event benches, where only the run's events matter.
 */
class PlainModel : HostModel {
    override fun id(): String = "plain"

    override fun capabilities(): HostModelCaps = BenchModel().capabilities()

    override fun generate(request: HostRequest, sink: TokenSink): HostReply {
        sink.delta("x")
        return HostReply(text = "x", inputTokens = 1uL, outputTokens = 1uL)
    }
}

/**
 * A host tool that does nothing and counts its calls.
 * Input: none. Output: `BENCH_TOOL_OUTPUT`; `calls` proves every iteration
 * crossed the tool boundary.
 */
class NoopTool : HostTool {
    val calls = AtomicInteger(0)

    override fun name(): String = "noop"

    override fun description(): String = "Does nothing."

    override fun schemaJson(): String = """{"type":"object","properties":{}}"""

    override fun call(argsJson: String): String {
        calls.incrementAndGet()
        return BENCH_TOOL_OUTPUT
    }
}

/**
 * Signals the end of each listened run.
 * Input: none. Output: `ended` gets one permit per terminal event.
 * One instance serves every iteration.
 */
class Ender : EventListener {
    val ended = Semaphore(0)

    override fun onEvent(event: Event) {
        val kind = event.kind
        val terminal = kind is EventKind.RunFinished || kind is EventKind.RunFailed ||
            kind is EventKind.RunPaused
        if (terminal) {
            ended.release()
        }
    }
}

/**
 * Runs `body` untimed `warmupOf(count)` times, then times it for `count`
 * runs and records one result line. Input: the bench name, count, label,
 * first pass and body. Output: the line, also appended to
 * `TROUPEKIT_BENCH_OUT` when set.
 */
suspend fun benchmark(
    name: String,
    count: Int,
    label: String,
    first: String,
    body: suspend () -> Unit,
): String {
    repeat(warmupOf(count)) { body() }
    val samples = LongArray(count)
    for (index in 0 until count) {
        val start = System.nanoTime()
        body()
        samples[index] = System.nanoTime() - start
    }
    samples.sort()
    val p50 = samples[count / 2]
    val p99 = samples[minOf(count * 99 / 100, count - 1)]
    val line = "overhead lang=kotlin bench=$name n=$count p50_ns=$p50 p99_ns=$p99 " +
        "label=$label first=$first"
    println(line)
    System.getenv("TROUPEKIT_BENCH_OUT")?.let { File(it).appendText(line + "\n") }
    return line
}

/**
 * B8: the cost of crossing the UniFFI boundary from Kotlin on the JVM, best
 * and worst case per the label. Runs only with `TROUPEKIT_BENCH=1`, in CI.
 */
class OverheadTest {
    @Test
    fun overhead() = runBlocking {
        assumeTrue("benchmarks run in CI only", System.getenv("TROUPEKIT_BENCH") == "1")
        ContractRuntime.ready()
        val count = (System.getenv("TROUPEKIT_BENCH_N") ?: "10000").toInt()
        assertTrue(count in 1..BENCH_ITERATIONS_MAX)
        val label = System.getenv("TROUPEKIT_BENCH_LABEL") ?: "quiet"
        val first = System.getenv("TROUPEKIT_BENCH_FIRST") ?: "quiet"
        assertTrue(first in setOf("quiet", "loaded"))
        val runs = count + warmupOf(count)

        var versions = 0
        benchmark("version", count, label, first) {
            assertTrue(version().isNotEmpty())
            versions += 1
        }
        assertEquals(runs, versions)

        val tool = NoopTool()
        val agent = AgentBuilder("bench").instructions("Benchmark.").hostModel(BenchModel())
            .tool(tool).build()
        benchmark("tool_round_trip", count, label, first) { assertEquals("done", agent.run("go")) }
        assertEquals(runs, tool.calls.get())

        val plain = AgentBuilder("bench").instructions("Benchmark.").hostModel(PlainModel())
            .build()
        var pulled = 0
        benchmark("event_pull", count, label, first) {
            val stream = plain.stream("go")
            while (stream.next() != null) {
                continue
            }
            pulled += 1
        }
        assertEquals(runs, pulled)

        val ender = Ender()
        var listened = 0
        benchmark("event_listener", count, label, first) {
            val stream = plain.stream("go")
            stream.setListener(ender)
            stream.result()
            assertTrue(ender.ended.tryAcquire(30, TimeUnit.SECONDS))
            listened += 1
        }
        assertEquals(runs, listened)
    }
}

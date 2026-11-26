package dev.troupekit.contract

import dev.troupekit.AgentBuilder
import dev.troupekit.AgentHandle
import dev.troupekit.ErrorCode
import dev.troupekit.Event
import dev.troupekit.EventKind
import dev.troupekit.EventListener
import dev.troupekit.GraphBuilder
import dev.troupekit.HostEmbedder
import dev.troupekit.HostException
import dev.troupekit.HostHeader
import dev.troupekit.HostHttpRequest
import dev.troupekit.HostModel
import dev.troupekit.HostModelCaps
import dev.troupekit.HostReply
import dev.troupekit.HostRequest
import dev.troupekit.HostTool
import dev.troupekit.HostTransport
import dev.troupekit.HttpSink
import dev.troupekit.KnowledgeConfig
import dev.troupekit.MemoryConfig
import dev.troupekit.Progress
import dev.troupekit.RunStream
import dev.troupekit.RuntimeSettings
import dev.troupekit.TokenSink
import dev.troupekit.TroupekitException
import dev.troupekit.initialize
import dev.troupekit.version
import java.io.File
import java.util.Collections
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test

const val SELECTOR = "stub/stub-model"
const val STUB_BASE = "http://127.0.0.1:9/v1"
const val EVENTS_MAX = 10_000
const val EMBED_DIMS = 64
const val CONCURRENT_RUNS = 8
const val SECRET = "sk-contract-secret-value-0001"
const val MISSING_ENV = "TROUPEKIT_CONTRACT_KEY_THAT_IS_NOT_SET"
const val EVENT_BUFFER = 16
val ANSWER_PARTS = listOf("Hello", " from", " the stub")
val CODES = mapOf(
    "Config" to 1u, "MissingKey" to 2u, "Catalog" to 3u, "Persona" to 4u, "Agent" to 5u,
    "Tool" to 6u, "Workflow" to 7u, "Graph" to 8u, "Store" to 9u, "Context" to 10u,
    "Failed" to 11u, "Paused" to 12u, "Retries" to 13u, "Cancelled" to 14u,
    "HostCallback" to 15u, "InvalidArgument" to 16u, "Unsupported" to 17u, "Panic" to 18u,
    "Internal" to 99u,
)

/**
 * A host model that answers from a script and records every request.
 * Input: the replies, one per `generate` call; the last one repeats.
 * Output: `requests` holds the rendered text of each request.
 */
class ScriptedModel(private val replies: List<String>) : HostModel {
    private val seen = Collections.synchronizedList(mutableListOf<String>())

    init {
        require(replies.isNotEmpty()) { "the script needs at least one reply" }
    }

    val requests: List<String>
        get() = synchronized(seen) { seen.toList() }

    override fun id(): String = "scripted"

    override fun capabilities(): HostModelCaps =
        HostModelCaps(tools = false, jsonSchema = false, streaming = true, contextTokens = 4096u)

    override fun generate(request: HostRequest, sink: TokenSink): HostReply {
        val text = synchronized(seen) {
            val turn = seen.size
            seen.add(render(request))
            replies[minOf(turn, replies.size - 1)]
        }
        sink.delta(text)
        return HostReply(text = text, inputTokens = 3uL, outputTokens = 1uL)
    }
}

/**
 * What a `RecordingTool` does when it is called.
 * Input: none. Output: echo the `text` argument, throw a `HostException`,
 * throw a non-`HostException`, or return fixed output.
 */
sealed class ToolBehaviour {
    object Echo : ToolBehaviour()
    data class Fails(val detail: String) : ToolBehaviour()
    object Crash : ToolBehaviour()
    data class Output(val text: String) : ToolBehaviour()
}

/**
 * A host tool that records its calls and the thread each one ran on.
 * Input: its name and behaviour. Output: `calls` holds the raw argument
 * JSON and the calling thread of every call.
 */
class RecordingTool(
    private val toolName: String = "echo",
    private val behaviour: ToolBehaviour = ToolBehaviour.Echo,
) : HostTool {
    private val seen = Collections.synchronizedList(mutableListOf<Pair<String, Thread>>())

    val calls: List<Pair<String, Thread>>
        get() = synchronized(seen) { seen.toList() }

    override fun name(): String = toolName

    override fun description(): String = "Repeats the text it is given."

    override fun schemaJson(): String =
        """{"type":"object","properties":{"text":{"type":"string"}}}"""

    override fun call(argsJson: String): String {
        seen.add(argsJson to Thread.currentThread())
        return when (behaviour) {
            is ToolBehaviour.Echo -> stringField(argsJson, "text")
            is ToolBehaviour.Fails -> throw HostException.Failed(behaviour.detail)
            is ToolBehaviour.Crash -> throw IllegalStateException("tool crashed")
            is ToolBehaviour.Output -> behaviour.text
        }
    }
}

/**
 * A deterministic host embedder: a hashed bag of lowercase words.
 * Input: none. Output: `calls` counts `embed` calls; equal word sets give
 * equal vectors, so a query finds the document holding its words.
 */
class HashEmbedder : HostEmbedder {
    @Volatile
    var calls = 0
        private set

    override fun dims(): UInt = EMBED_DIMS.toUInt()

    @Synchronized
    override fun embed(texts: List<String>): List<List<Float>> {
        calls += 1
        return texts.map(::bagOfWords)
    }
}

/**
 * A host transport that answers every request from a fixed script.
 * Input: body chunks, an optional head URL override, an optional failure.
 * Output: `requests` holds each request URL it got.
 */
class StubTransport(
    private val chunks: List<ByteArray>,
    private val headUrl: String? = null,
    private val failure: String? = null,
) : HostTransport {
    private val seen = Collections.synchronizedList(mutableListOf<String>())

    val requests: List<String>
        get() = synchronized(seen) { seen.toList() }

    override fun send(request: HostHttpRequest, sink: HttpSink) {
        seen.add(request.url)
        if (failure != null) {
            throw HostException.Failed(failure)
        }
        val header = HostHeader(name = "content-type", value = "text/event-stream")
        sink.head(200u, headUrl ?: request.url, listOf(header))
        for (chunk in chunks) {
            sink.chunk(chunk)
        }
    }
}

/**
 * An event listener that stores the kind of every event it gets.
 * Input: none. Output: `kinds` in arrival order; `finished` counts down
 * once the run ends.
 */
class Collector : EventListener {
    private val seen = Collections.synchronizedList(mutableListOf<String>())
    val finished = CountDownLatch(1)

    val kinds: List<String>
        get() = synchronized(seen) { seen.toList() }

    override fun onEvent(event: Event) {
        val kind = kindName(event.kind)
        seen.add(kind)
        if (kind in setOf("RunFinished", "RunFailed", "RunPaused")) {
            finished.countDown()
        }
    }
}

/**
 * Joins every text field of a host request for substring checks.
 * Input: a `HostRequest`. Output: one string with the system text, the
 * tool list and every message text, tool call and tool call id.
 */
fun render(request: HostRequest): String {
    val parts = mutableListOf(request.system ?: "", request.toolsJson ?: "")
    for (message in request.messages) {
        parts += listOf(
            message.role, message.text, message.toolCallsJson ?: "", message.toolCallId ?: "",
        )
    }
    return parts.joinToString("\n")
}

/**
 * Hashes the lowercase words of `text` into a unit vector.
 * Input: a string. Output: `EMBED_DIMS` floats, never all zero.
 * FNV-1a keeps the hash stable across runs and platforms.
 */
fun bagOfWords(text: String): List<Float> {
    val vector = FloatArray(EMBED_DIMS)
    for (word in text.lowercase().split(Regex("[^\\p{L}\\p{N}]+")).filter { it.isNotEmpty() }) {
        var hash = 0xcbf29ce484222325uL
        for (byte in word.toByteArray(Charsets.UTF_8)) {
            hash = (hash xor byte.toUByte().toULong()) * 0x100000001b3uL
        }
        vector[(hash % EMBED_DIMS.toULong()).toInt()] += 1f
    }
    vector[0] += 0.001f
    val norm = kotlin.math.sqrt(vector.fold(0f) { sum, value -> sum + value * value })
    return vector.map { it / norm }
}

/**
 * Names an event kind by its generated class, for example `TextDelta`.
 * Input: the kind. Output: the simple class name.
 * Comparing names keeps pull and listener checks independent of payloads.
 */
fun kindName(kind: EventKind): String =
    checkNotNull(kind::class.simpleName) { "EventKind variants are named classes" }

/**
 * Quotes a string as a JSON string literal.
 * Input: any text. Output: the literal with quotes, backslashes and
 * control characters escaped, so the script replies stay valid JSON.
 */
fun jsonString(text: String): String {
    val out = StringBuilder("\"")
    for (char in text) {
        when {
            char == '"' -> out.append("\\\"")
            char == '\\' -> out.append("\\\\")
            char < ' ' -> out.append("\\u%04x".format(char.code))
            else -> out.append(char)
        }
    }
    return out.append('"').toString()
}

/**
 * Reads one string field of a flat JSON object the neutral layer sent.
 * Input: the JSON text and the key. Output: the value, or "" when absent.
 * Enough for the tool arguments of these cases; no JSON library is approved.
 */
fun stringField(json: String, key: String): String {
    val match = Regex("\"${Regex.escape(key)}\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"").find(json)
    return match?.groupValues?.get(1) ?: ""
}

/**
 * Renders an emulated tool call as the host model writes it (K8).
 * Input: the tool name and string arguments. Output: the JSON text.
 * Same shape as the Python runner's `tool_reply`.
 */
fun toolReply(name: String, arguments: Map<String, String>): String {
    val args = arguments.entries.sortedBy { it.key }
        .joinToString(",") { "${jsonString(it.key)}:${jsonString(it.value)}" }
    return """{"arguments":{$args},"tool":${jsonString(name)}}"""
}

/**
 * Renders an emulated final answer as the host model writes it (K8).
 * Input: the answer text. Output: the JSON text.
 * Same shape as the Python runner's `answer_reply`.
 */
fun answerReply(text: String): String = """{"answer":${jsonString(text)}}"""

/**
 * The streamed reply of the stub vendor: one delta per answer part.
 * Input: none. Output: SSE chunks, ending with stop and `[DONE]`; each
 * chunk is one `HttpSink.chunk` call.
 */
fun answerChunks(): List<ByteArray> {
    val deltas = ANSWER_PARTS.map {
        """{"choices":[{"index":0,"delta":{"content":${jsonString(it)}}}]}"""
    }
    val stop = """{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"""
    return (deltas + stop + "[DONE]").map { "data: $it\n\n".toByteArray(Charsets.UTF_8) }
}

/**
 * Builds a catalog TOML naming the stub vendor.
 * Input: the auth style, `none` or `bearer`, and the base URL.
 * Output: catalog text with vendor `stub` and model `stub-model`.
 */
fun catalog(auth: String = "none", base: String = STUB_BASE): String = """
    [[vendor]]
    id = "stub"
    name = "Stub"
    protocol = "openai_compatible"
    base_url = "$base"
    auth = "$auth"

    [[vendor.model]]
    id = "stub-model"
    context_window = 8192
    input_usd_micros_per_mtok = 0
    output_usd_micros_per_mtok = 0
    price_verified = true
""".trimIndent()

/**
 * Builds an agent on a host model with the given host tools.
 * Input: the model, tools and an optional SQLite path.
 * Output: the unbuilt `AgentBuilder`.
 */
fun hostAgent(model: HostModel, tools: List<HostTool> = emptyList(), sqlite: String? = null) =
    tools.fold(AgentBuilder("helper").instructions("Help the user.").hostModel(model)) { b, t ->
        b.tool(t)
    }.let { builder -> if (sqlite == null) builder else builder.sqlite(sqlite) }

/**
 * Builds an agent on the stub vendor through the cloud path and a transport.
 * Input: the transport and an optional SQLite path.
 * Output: the unbuilt `AgentBuilder`.
 */
fun cloudAgent(transport: HostTransport, sqlite: String? = null): AgentBuilder {
    val builder = AgentBuilder("helper").instructions("Help the user.").model(SELECTOR)
        .catalogToml(catalog()).hostTransport(transport)
    return if (sqlite == null) builder else builder.sqlite(sqlite)
}

/**
 * Pulls every event of a stream until it ends.
 * Input: a `RunStream`. Output: the events; fails past `EVENTS_MAX`.
 * The same bound as the Swift and Python runners.
 */
suspend fun pullAll(stream: RunStream): List<Event> {
    val events = mutableListOf<Event>()
    while (true) {
        val event = stream.next() ?: return events
        events += event
        if (events.size > EVENTS_MAX) {
            fail("stream did not end within EVENTS_MAX events")
        }
    }
}

/**
 * The text deltas of a list of events, joined.
 * Input: the events. Output: the concatenated `TextDelta` texts.
 * Used by both streaming cases.
 */
fun deltaText(events: List<Event>): String =
    events.mapNotNull { (it.kind as? EventKind.TextDelta)?.text }.joinToString("")

/**
 * Runs `body` and returns its `TroupekitException.Failure`.
 * Input: a suspending body. Output: the failure; fails the test when the
 * body returns normally. Every body is built (K10), so no code skips.
 */
suspend fun expectFailure(body: suspend () -> Unit): TroupekitException.Failure {
    try {
        body()
    } catch (error: TroupekitException.Failure) {
        return error
    }
    throw AssertionError("expected a TroupekitException")
}

/**
 * The 25 contract cases of `troupekit-ffi-core/contract/cases.txt`, through
 * the UniFFI Kotlin bindings. The JVM and Android runners subclass it and
 * supply the shared files and a scratch directory.
 */
abstract class ContractCases {
    /**
     * Reads one shared file by name: `cases.txt` or `agent.toml`.
     * Input: the file name. Output: its UTF-8 text.
     * The JVM reads the repository; Android reads the test APK's assets.
     */
    abstract fun shared(name: String): String

    /**
     * A directory the case may write into.
     * Input: none. Output: an existing directory.
     * Each case makes its own subdirectory under it.
     */
    abstract fun scratchRoot(): File

    @Before
    fun runtime() {
        ContractRuntime.ready()
    }

    /**
     * A fresh scratch directory for one case.
     * Input: none. Output: its absolute path.
     * Each case gets its own SQLite files.
     */
    fun scratch(): File =
        File(scratchRoot(), UUID.randomUUID().toString()).also { check(it.mkdirs()) }

    @Test
    fun test_version_matches_core() {
        val parts = version().split(".")
        assertEquals(3, parts.size)
        assertTrue(parts.all { it.toUIntOrNull() != null })
    }

    @Test
    fun test_missing_model_is_config_error() = runBlocking<Unit> {
        val failure = expectFailure { AgentBuilder("helper").instructions("Help.").build() }
        assertEquals(CODES["Config"], failure.code)
    }

    @Test
    fun test_missing_key_names_env_not_value() = runBlocking<Unit> {
        assertEquals(null, System.getenv(MISSING_ENV))
        val keyed = {
            AgentBuilder("helper").instructions("Help.").model(SELECTOR)
                .catalogToml(catalog(auth = "bearer", base = "https://127.0.0.1:9/v1"))
        }
        val missing = expectFailure { keyed().apiKeyEnv(MISSING_ENV).build().run("hi") }
        assertEquals(CODES["MissingKey"], missing.code)
        assertTrue(missing.detail.contains(MISSING_ENV))
        val transport = StubTransport(emptyList(), failure = "connection refused")
        val leaked = expectFailure {
            keyed().apiKey(SECRET).hostTransport(transport).build().run("hi")
        }
        assertFalse(leaked.detail.contains(SECRET))
    }

    @Test
    fun test_host_model_run_returns_text() = runBlocking<Unit> {
        val model = ScriptedModel(listOf("Hello from the host model"))
        assertEquals("Hello from the host model", hostAgent(model).build().run("hi"))
        assertEquals(1, model.requests.size)
    }

    @Test
    fun test_host_tool_called_with_json_args() = runBlocking<Unit> {
        val call = toolReply("echo", mapOf("text" to "hi"))
        val model = ScriptedModel(listOf(call, answerReply("done")))
        val tool = RecordingTool()
        assertEquals("done", hostAgent(model, listOf(tool)).build().run("echo hi"))
        assertEquals(listOf("hi"), tool.calls.map { stringField(it.first, "text") })
        assertEquals(2, model.requests.size)
    }

    @Test
    fun test_host_tool_error_reaches_model() = runBlocking<Unit> {
        val call = toolReply("echo", mapOf("text" to "hi"))
        val model = ScriptedModel(listOf(call, answerReply("recovered")))
        val tool = RecordingTool(behaviour = ToolBehaviour.Fails("echo is offline"))
        assertEquals("recovered", hostAgent(model, listOf(tool)).build().run("echo hi"))
        assertTrue(model.requests[1].contains("echo is offline"))
    }

    @Test
    fun test_host_tool_throw_does_not_crash() = runBlocking<Unit> {
        val call = toolReply("echo", mapOf("text" to "hi"))
        val model = ScriptedModel(listOf(call, answerReply("still here")))
        val tool = RecordingTool(behaviour = ToolBehaviour.Crash)
        assertEquals("still here", hostAgent(model, listOf(tool)).build().run("echo hi"))
        assertEquals(1, tool.calls.size)
        assertEquals(2, model.requests.size)
    }

    @Test
    fun test_stream_yields_deltas_then_finished() = runBlocking<Unit> {
        val stream = cloudAgent(StubTransport(answerChunks())).build().stream("hi")
        val events = pullAll(stream)
        assertEquals(ANSWER_PARTS.joinToString(""), deltaText(events))
        assertEquals("RunStarted", kindName(events.first().kind))
        assertEquals("RunFinished", kindName(events.last().kind))
        assertEquals(0uL, stream.droppedDeltas())
    }

    @Test
    fun test_listener_sees_same_events_as_pull() = runBlocking<Unit> {
        val pulled = pullAll(hostAgent(ScriptedModel(listOf("same"))).build().stream("hi"))
        val collector = Collector()
        val stream = hostAgent(ScriptedModel(listOf("same"))).build().stream("hi")
        stream.setListener(collector)
        stream.result()
        assertTrue(collector.finished.await(30, TimeUnit.SECONDS))
        assertEquals(pulled.map { kindName(it.kind) }, collector.kinds)
    }

    @Test
    fun test_start_returns_run_id_before_model_call() = runBlocking<Unit> {
        val transport = StubTransport(answerChunks())
        val database = File(scratch(), "runs.db").path
        val runId = cloudAgent(transport, database).build().start("hi")
        assertTrue(runId.isNotEmpty())
        assertEquals(0, transport.requests.size)
    }

    @Test
    fun test_resume_after_handles_dropped() = runBlocking<Unit> {
        val transport = StubTransport(answerChunks())
        val database = File(scratch(), "runs.db").path
        val runId = cloudAgent(transport, database).build().use { first -> first.start("hi") }
        val answer = cloudAgent(transport, database).build().resume(runId)
        assertEquals(ANSWER_PARTS.joinToString(""), answer)
        assertEquals(1, transport.requests.size)
    }

    @Test
    fun test_resume_for_zero_budget_is_pending() = runBlocking<Unit> {
        val transport = StubTransport(answerChunks())
        val agent = cloudAgent(transport, File(scratch(), "runs.db").path).build()
        val runId = agent.start("hi")
        val progress = agent.resumeFor(runId, 0uL)
        assertEquals(Progress.Pending(runId), progress)
        assertEquals(0, transport.requests.size)
    }

    @Test
    fun test_cancel_keeps_checkpoint_and_resumes() = runBlocking<Unit> {
        val transport = StubTransport(answerChunks())
        val agent = cloudAgent(transport, File(scratch(), "runs.db").path).build()
        val stream = agent.stream("hi")
        stream.cancel()
        val cancelled = expectFailure { stream.result() }
        assertEquals(CODES["Cancelled"], cancelled.code)
        assertEquals(ANSWER_PARTS.joinToString(""), agent.resume(stream.runId()))
    }

    @Test
    fun test_toml_agent_runs() = runBlocking<Unit> {
        val builder = AgentBuilder.fromToml(shared("agent.toml")).catalogToml(catalog())
            .hostTransport(StubTransport(answerChunks()))
        assertEquals(ANSWER_PARTS.joinToString(""), builder.build().run("hi"))
    }

    @Test
    fun test_toml_over_cap_refused() = runBlocking<Unit> {
        val text = shared("agent.toml") + "\n" + "a".repeat(1_048_576)
        val failure = expectFailure { AgentBuilder.fromToml(text).build() }
        assertTrue(failure.code in setOf(CODES["Config"], CODES["InvalidArgument"]))
        assertTrue(failure.detail.contains("MiB"))
    }

    @Test
    fun test_graph_agent_then_step_runs() = runBlocking<Unit> {
        val writer = hostAgent(ScriptedModel(listOf("draft")))
        val stamp = RecordingTool("stamp", ToolBehaviour.Output("stamped"))
        val graph = GraphBuilder("flow").agent("writer", writer).step("stamp", stamp)
            .edge("writer", "stamp")
        val result = graph.build().runResult("go")
        assertEquals(1, stamp.calls.size)
        assertTrue(result.outputJson.contains("stamped"))
    }

    @Test
    fun test_knowledge_with_host_embedder_finds_doc() = runBlocking<Unit> {
        val root = scratch()
        val docs = File(root, "docs").also { check(it.mkdirs()) }
        File(docs, "fruit.md").writeText("The secret fruit is pineapple.\n")
        val embedder = HashEmbedder()
        val search = toolReply("search_knowledge", mapOf("query" to "secret fruit pineapple"))
        val model = ScriptedModel(listOf(search, answerReply("pineapple")))
        val config = KnowledgeConfig(
            dir = docs.path, index = null, store = null, allowFailed = false,
        )
        val agent = hostAgent(model, sqlite = File(root, "kb.db").path).knowledge(config, embedder)
        assertEquals("pineapple", agent.build().run("What is the secret fruit?"))
        assertTrue(embedder.calls > 0)
        assertTrue(model.requests[1].contains("The secret fruit is pineapple"))
    }

    @Test
    fun test_memory_facts_persist_between_runs() = runBlocking<Unit> {
        val database = File(scratch(), "memory.db").path
        val config = MemoryConfig(
            user = "user-1", session = "session-1", sqlite = database, preamble = true,
        )
        val remember = toolReply("remember_fact", mapOf("key" to "name", "value" to "Ada Lovelace"))
        val first = ScriptedModel(listOf(remember, answerReply("noted")))
        val noted = hostAgent(first).memory(config).build().run("My name is Ada Lovelace")
        assertEquals("noted", noted)
        val second = ScriptedModel(listOf("Hello Ada"))
        assertEquals("Hello Ada", hostAgent(second).memory(config).build().run("Who am I?"))
        assertTrue(second.requests[0].contains("Ada Lovelace"))
    }

    @Test
    fun test_error_codes_are_stable() {
        val mirrored = ErrorCode.values().associate { code ->
            code.name.split("_").joinToString("") { part ->
                part.lowercase().replaceFirstChar { it.uppercase() }
            } to code.value
        }
        assertEquals(CODES, mirrored)
    }

    @Test
    fun test_laws_path_with_dotdot_refused() = runBlocking<Unit> {
        val agent = hostAgent(ScriptedModel(listOf("x"))).laws("../laws.bend")
        val failure = expectFailure { agent.build() }
        assertTrue(failure.code in setOf(CODES["Config"], CODES["InvalidArgument"]))
        assertTrue(failure.detail.contains(".."))
    }

    @Test
    fun test_host_transport_refuses_other_origin() = runBlocking<Unit> {
        val other = "https://elsewhere.invalid/v1/chat/completions"
        val done = listOf("data: [DONE]\n\n".toByteArray(Charsets.UTF_8))
        val transport = StubTransport(done, headUrl = other)
        val failure = expectFailure { cloudAgent(transport).build().run("hi") }
        assertNotEquals(0u, failure.code)
        assertEquals(1, transport.requests.size)
    }

    @Test
    fun test_host_transport_streams_chunks() = runBlocking<Unit> {
        val transport = StubTransport(answerChunks())
        val events = pullAll(cloudAgent(transport).build().stream("hi"))
        assertEquals(ANSWER_PARTS.joinToString(""), deltaText(events))
        assertTrue(transport.requests[0].startsWith(STUB_BASE))
    }

    @Test
    fun test_full_buffer_drops_deltas_not_lifecycle() = runBlocking<Unit> {
        assumeTrue("runtime set up before this runner", ContractRuntime.bufferSet)
        val chatty = ScriptedModel(listOf("x".repeat(500)))
        val stream = hostAgent(chatty).build().stream("hi")
        stream.result()
        val kinds = pullAll(stream).map { kindName(it.kind) }
        assertEquals("RunStarted", kinds.first())
        assertEquals("RunFinished", kinds.last())
        assertTrue(kinds.count { it == "TextDelta" } <= EVENT_BUFFER)
    }

    @Test
    fun test_callback_from_worker_thread_no_deadlock() = runBlocking<Unit> {
        val caller = Thread.currentThread()
        val call = toolReply("echo", mapOf("text" to "hi"))
        val model = ScriptedModel(listOf(call, answerReply("done")))
        val tool = RecordingTool()
        assertEquals("done", hostAgent(model, listOf(tool)).build().run("echo hi"))
        assertEquals(1, tool.calls.size)
        assertNotEquals(caller, tool.calls[0].second)
    }

    @Test
    fun test_concurrent_runs_on_one_handle() = runBlocking<Unit> {
        val model = ScriptedModel(listOf("parallel"))
        val agent: AgentHandle = hostAgent(model).build()
        val answers = coroutineScope {
            (1..CONCURRENT_RUNS).map { async { agent.run("hi") } }.awaitAll()
        }
        assertEquals(List(CONCURRENT_RUNS) { "parallel" }, answers)
        assertEquals(CONCURRENT_RUNS, model.requests.size)
    }

    @Test
    fun test_host_request_debug_hides_key() {
        val headers = listOf(
            HostHeader(name = "Authorization", value = "Bearer $SECRET", sensitive = false),
            HostHeader(name = "x-trace", value = "sk-flagged-secret", sensitive = true),
            HostHeader(name = "content-type", value = "application/json", sensitive = false),
        )
        val request = HostHttpRequest(
            method = "POST",
            url = "https://api.stub.test/v1/chat/completions",
            headers = headers,
            body = ByteArray(0),
        )
        val texts = listOf(request.toString(), "$request", headers[0].toString(), "${headers[1]}")
        for (text in texts) {
            assertFalse(text, text.contains(SECRET))
            assertFalse(text, text.contains("sk-flagged-secret"))
        }
        assertTrue(texts[0], texts[0].contains("application/json"))
    }

    @Test
    fun everyCaseIsImplemented() {
        val names = shared("cases.txt").lines().map { it.trim() }.filter { it.isNotEmpty() }
        assertTrue(names.isNotEmpty())
        val methods = javaClass.methods.filter { it.isAnnotationPresent(Test::class.java) }
            .map { it.name }.toSet()
        assertEquals(emptyList<String>(), names.filter { "test_$it" !in methods })
    }
}

/**
 * Sets the runtime once per process with a 16-event buffer, small enough for
 * the full-buffer case and enough for the others.
 * Input: none. Output: `ready()` returns once `initialize` has run.
 */
object ContractRuntime {
    /**
     * Whether `initialize` took the 16-event buffer. It answers
     * `InvalidArgument` when another test in this process used the runtime
     * first; only the full-buffer case depends on the buffer, so it skips.
     */
    var bufferSet = false
        private set

    init {
        try {
            val settings = RuntimeSettings(
                workerThreads = 2u,
                maxInputBytes = 1_048_576uL,
                eventBuffer = EVENT_BUFFER.toUInt(),
            )
            initialize(settings)
            bufferSet = true
        } catch (error: TroupekitException.Failure) {
            check(error.code == CODES["InvalidArgument"]) { "initialize failed: ${error.detail}" }
        }
    }

    /**
     * Forces the one-time `initialize` call.
     * Input: none. Output: none.
     * Called before every case.
     */
    fun ready() = Unit
}

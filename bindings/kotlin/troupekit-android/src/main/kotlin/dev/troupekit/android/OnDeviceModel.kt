package dev.troupekit.android

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import dev.troupekit.HostException
import dev.troupekit.HostModel
import dev.troupekit.HostModelCaps
import dev.troupekit.HostReply
import dev.troupekit.HostRequest
import dev.troupekit.TokenSink
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * A LiteRT-LM model file on the device as a troupekit `HostModel`.
 * Tools are emulated by the Rust side, so the tool loop and every checkpoint stay in troupekit.
 * Input: the `.litertlm` path, the backend (null means CPU), the context size and the reply timeout,
 * which bounds one generate call after the engine is loaded. Creating it loads no LiteRT-LM class.
 */
class OnDeviceModel(
    private val modelPath: String,
    private val backend: Backend? = null,
    private val maxNumTokens: Int = 4096,
    private val cacheDir: String? = null,
    private val replyTimeoutMs: Long = DEFAULT_REPLY_TIMEOUT_MS,
) : HostModel, AutoCloseable {
    private val lock = Any()
    private var engine: Engine? = null

    init {
        require(maxNumTokens > 0) { "maxNumTokens must be above 0, got $maxNumTokens" }
        require(replyTimeoutMs > 0) { "replyTimeoutMs must be above 0, got $replyTimeoutMs" }
    }

    override fun id(): String = "android/litertlm"

    override fun capabilities(): HostModelCaps =
        HostModelCaps(tools = false, jsonSchema = false, streaming = true, contextTokens = maxNumTokens.toUInt())

    override fun generate(request: HostRequest, sink: TokenSink): HostReply {
        try {
            val text = generateText(request) { delta -> sink.delta(delta) }
            return HostReply(text = text, inputTokens = 0uL, outputTokens = 0uL)
        } catch (error: HostException) {
            throw error
        } catch (error: Exception) {
            throw HostException.Failed("on-device model error: ${error.message ?: error.javaClass.simpleName}")
        }
    }

    /**
     * Why the model cannot answer right now, or null when it can be loaded.
     * Input: none. Output: a message starting `on-device model unavailable: `, or null.
     */
    fun unavailableMessage(): String? {
        val file = File(modelPath)
        if (!file.isFile) {
            return "on-device model unavailable: model file missing"
        }
        if (!file.canRead()) {
            return "on-device model unavailable: model file unreadable"
        }
        return null
    }

    /**
     * Runs one model turn on a fresh conversation, pushing each streamed chunk through `onDelta`.
     * Calls are serialized on one engine, loaded on the first call, never on the main thread.
     * Input: the request and a chunk callback. Output: the reply text; throws a named `HostException`.
     */
    fun generateText(request: HostRequest, onDelta: (String) -> Unit): String {
        unavailableMessage()?.let { throw HostException.Failed(it) }
        val (instructions, prompt) = render(request)
        synchronized(lock) {
            val config = ConversationConfig(
                systemInstruction = if (instructions.isEmpty()) null else Contents.of(instructions),
                maxOutputToken = request.maxTokens.toInt(),
            )
            return loaded().createConversation(config).use { conversation -> stream(conversation, prompt, onDelta) }
        }
    }

    /**
     * Releases the loaded engine, if any; the next call loads it again.
     * Input: none. Output: none.
     */
    override fun close() {
        synchronized(lock) {
            engine?.close()
            engine = null
        }
    }

    private fun loaded(): Engine {
        engine?.let { return it }
        val created = Engine(EngineConfig(modelPath = modelPath, backend = backend ?: Backend.CPU(), maxNumTokens = maxNumTokens, cacheDir = cacheDir))
        try {
            created.initialize()
        } catch (error: Exception) {
            created.close()
            throw HostException.Failed("on-device model unavailable: ${error.message ?: error.javaClass.simpleName}")
        }
        check(created.isInitialized()) { "engine reports ready after initialize" }
        engine = created
        return created
    }

    private fun stream(conversation: Conversation, prompt: String, onDelta: (String) -> Unit): String =
        collect(replyTimeoutMs, { chunk, done, fail ->
            conversation.sendMessageAsync(prompt, object : MessageCallback {
                override fun onMessage(message: Message) {
                    chunk(message.toString())
                }

                override fun onDone() {
                    done()
                }

                override fun onError(throwable: Throwable) {
                    fail(throwable)
                }
            })
        }, { conversation.cancelProcess() }, onDelta)

    companion object {
        /** Default bound of one reply after the engine is loaded: two minutes. */
        const val DEFAULT_REPLY_TIMEOUT_MS: Long = 120_000

        /**
         * Starts one streamed reply and waits for it at most `timeoutMs`; on timeout it calls `cancel`.
         * Non-empty chunks reach `onDelta` in order; none reach it once the wait has ended.
         * Input: the timeout, the start and cancel hooks, the delta callback. Output: the reply text;
         * throws `HostException.Failed` naming the timeout or the stream error.
         */
        internal fun collect(
            timeoutMs: Long,
            start: (chunk: (String) -> Unit, done: () -> Unit, fail: (Throwable) -> Unit) -> Unit,
            cancel: () -> Unit,
            onDelta: (String) -> Unit,
        ): String {
            require(timeoutMs > 0) { "timeoutMs must be above 0, got $timeoutMs" }
            val finished = CountDownLatch(1)
            val failure = AtomicReference<Throwable?>(null)
            val gate = Any()
            val text = StringBuilder()
            var open = true
            start(
                { chunk ->
                    synchronized(gate) {
                        if (open && chunk.isNotEmpty()) {
                            text.append(chunk)
                            onDelta(chunk)
                        }
                    }
                },
                { finished.countDown() },
                { error ->
                    failure.compareAndSet(null, error)
                    finished.countDown()
                },
            )
            val inTime = finished.await(timeoutMs, TimeUnit.MILLISECONDS)
            synchronized(gate) { open = false }
            if (!inTime) {
                cancel()
                throw HostException.Failed("on-device model error: no reply within $timeoutMs ms")
            }
            failure.get()?.let { throw HostException.Failed("on-device model error: ${it.message ?: it.javaClass.simpleName}") }
            return synchronized(gate) { text.toString() }
        }

        /**
         * Renders a request as instructions plus one prompt holding every turn as text.
         * Tool calls and tool results become labelled lines; the prompt ends with `assistant:`.
         * Input: the request. Output: the instructions (empty without a system) and the prompt.
         */
        fun render(request: HostRequest): Pair<String, String> {
            val lines = ArrayList<String>(request.messages.size + 1)
            for (message in request.messages) {
                val calls = message.toolCallsJson
                val callId = message.toolCallId
                if (calls != null) {
                    if (message.text.isNotEmpty()) {
                        lines.add("assistant: ${message.text}")
                    }
                    lines.add("assistant tool call: $calls")
                } else if (callId != null) {
                    lines.add("tool result $callId: ${message.text}")
                } else {
                    lines.add("${message.role}: ${message.text}")
                }
            }
            lines.add("assistant:")
            return (request.system ?: "") to lines.joinToString("\n")
        }
    }
}

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
import java.util.concurrent.atomic.AtomicReference

/**
 * A LiteRT-LM model file on the device as a troupekit `HostModel`.
 * Tools are emulated by the Rust side, so the tool loop and every checkpoint stay in troupekit.
 * Input: the `.litertlm` path, the backend and the context size. Safe to create anywhere; it reports unavailable.
 */
class OnDeviceModel(
    private val modelPath: String,
    private val backend: Backend = Backend.CPU(),
    private val maxNumTokens: Int = 4096,
    private val cacheDir: String? = null,
) : HostModel, AutoCloseable {
    private val lock = Any()
    private var engine: Engine? = null

    init {
        require(maxNumTokens > 0) { "maxNumTokens must be above 0, got $maxNumTokens" }
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
        val created = Engine(EngineConfig(modelPath = modelPath, backend = backend, maxNumTokens = maxNumTokens, cacheDir = cacheDir))
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

    private fun stream(conversation: Conversation, prompt: String, onDelta: (String) -> Unit): String {
        val done = CountDownLatch(1)
        val failure = AtomicReference<Throwable?>(null)
        val text = StringBuilder()
        conversation.sendMessageAsync(prompt, object : MessageCallback {
            override fun onMessage(message: Message) {
                val chunk = message.toString()
                if (chunk.isNotEmpty()) {
                    text.append(chunk)
                    onDelta(chunk)
                }
            }

            override fun onDone() {
                done.countDown()
            }

            override fun onError(throwable: Throwable) {
                failure.set(throwable)
                done.countDown()
            }
        })
        done.await()
        failure.get()?.let { throw HostException.Failed("on-device model error: ${it.message ?: it.javaClass.simpleName}") }
        return text.toString()
    }

    companion object {
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

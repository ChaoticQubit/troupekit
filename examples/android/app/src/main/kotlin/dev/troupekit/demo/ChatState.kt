package dev.troupekit.demo

import android.content.Context

/** Who wrote a line of the conversation. */
enum class Author { USER, AGENT, NOTICE }

/** One line of the conversation shown on screen. Input: author and text. Output: an immutable value. */
data class ChatLine(val author: Author, val text: String)

/**
 * Where the id of a started, unfinished run is kept between process launches.
 * Input: the id to save. Output: the saved id on load, or null when none is pending.
 */
interface RunIdStore {
    fun load(): String?
    fun save(runId: String)
    fun clear()
}

/**
 * Run id store over SharedPreferences, so a relaunch after `am force-stop` finds the run.
 * Input: the app context. Output: a store that commits each write before returning.
 */
class PrefsRunIdStore(context: Context) : RunIdStore {
    private val prefs = context.getSharedPreferences("troupekit.demo", Context.MODE_PRIVATE)

    override fun load(): String? = prefs.getString(KEY, null)

    override fun save(runId: String) {
        check(prefs.edit().putString(KEY, runId).commit()) { "run id not saved" }
    }

    override fun clear() {
        prefs.edit().remove(KEY).commit()
    }

    private companion object {
        const val KEY = "runId"
    }
}

/**
 * UI state of the demo chat: finished lines, the streaming answer, run id, steps and window progress.
 * Plain Kotlin so the JVM tests cover it; the activity redraws after every change through `onChange`.
 * Input: the run id store. Output: the state the activity renders.
 */
class ChatState(private val store: RunIdStore) {
    var onChange: () -> Unit = {}

    private val mutableLines = mutableListOf<ChatLine>()
    val lines: List<ChatLine> get() = mutableLines

    var streaming: String = ""
        private set
    var isRunning: Boolean = false
        private set
    var runId: String? = null
        private set
    var steps: Int = 0
        private set
    var progress: String? = null
        private set

    /** The id of a run started and not yet finished. Output: the stored id, or null. */
    val pendingRunId: String? get() = store.load()

    /** Records the question and the run id returned by `start`. Input: both. Output: none. */
    fun begin(question: String, runId: String) {
        require(runId.isNotEmpty()) { "run id must not be empty" }
        store.save(runId)
        mutableLines.add(ChatLine(Author.USER, question))
        this.runId = runId
        steps = 0
        progress = null
        streaming = ""
        isRunning = true
        onChange()
    }

    /** Marks the stored run as resuming after a relaunch. Output: none; no-op when nothing is pending. */
    fun beginResume() {
        val pending = store.load() ?: return
        mutableLines.add(ChatLine(Author.NOTICE, "Resuming the last run"))
        runId = pending
        progress = null
        streaming = ""
        isRunning = true
        onChange()
    }

    /** Appends one streamed delta. Input: the text. Output: none; text past the bound is dropped. */
    fun apply(delta: String) {
        val room = MAX_STREAMING_CHARS - streaming.length
        if (room <= 0) return
        streaming += if (delta.length <= room) delta else delta.substring(0, room)
        onChange()
    }

    /** Counts one finished step of the current run. Output: none. */
    fun stepFinished() {
        steps += 1
        onChange()
    }

    /** Ends a background window that left the run unfinished; the id stays stored. Output: none. */
    fun pending() {
        progress = "pending"
        streaming = ""
        isRunning = false
        onChange()
    }

    /** Shows the final answer once and clears the stored id. Input: the answer. Output: none. */
    fun finish(answer: String) {
        store.clear()
        mutableLines.add(ChatLine(Author.AGENT, answer))
        progress = "done"
        streaming = ""
        isRunning = false
        onChange()
    }

    /** Shows a failure; keeps the id only when the run can still resume. Input: message, flag. Output: none. */
    fun fail(message: String, resumable: Boolean) {
        if (!resumable) store.clear()
        mutableLines.add(ChatLine(Author.NOTICE, message))
        streaming = ""
        isRunning = false
        onChange()
    }

    /** Forgets any stored run, for a clean start. Output: none. */
    fun reset() {
        store.clear()
        mutableLines.clear()
        runId = null
        steps = 0
        progress = null
        streaming = ""
        isRunning = false
        onChange()
    }

    companion object {
        const val MAX_STREAMING_CHARS = 1 shl 20
    }
}

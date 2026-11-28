package dev.troupekit.demo

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * JVM tests of the demo's chat state: the stored run id survives a new state over the same store,
 * the answer appears once, and a failure keeps the id only when the run can still resume.
 * The store is a plain map, standing in for SharedPreferences, which needs a device.
 */
class ChatStateTest {
    private class MapStore : RunIdStore {
        var value: String? = null
        override fun load(): String? = value
        override fun save(runId: String) { value = runId }
        override fun clear() { value = null }
    }

    @Test
    fun runIdSurvivesNewStateOverSameStore() {
        val store = MapStore()
        ChatState(store).begin("q", "r1")
        val relaunched = ChatState(store)
        assertEquals("r1", relaunched.pendingRunId)
        relaunched.beginResume()
        assertEquals("r1", relaunched.runId)
        assertTrue(relaunched.isRunning)
    }

    @Test
    fun finishShowsAnswerOnceAndClearsId() {
        val store = MapStore()
        val state = ChatState(store)
        state.begin("q", "r1")
        state.apply("Hel")
        state.apply("lo")
        assertEquals("Hello", state.streaming)
        state.finish("Hello")
        assertEquals(1, state.lines.count { it.author == Author.AGENT })
        assertNull(store.value)
        assertEquals("done", state.progress)
        assertFalse(state.isRunning)
    }

    @Test
    fun failKeepsIdOnlyWhenResumable() {
        val store = MapStore()
        val state = ChatState(store)
        state.begin("q", "r1")
        state.fail("offline", resumable = true)
        assertEquals("r1", store.value)
        state.fail("bad key", resumable = false)
        assertNull(store.value)
    }

    @Test
    fun pendingKeepsIdAndStopsRunning() {
        val store = MapStore()
        val state = ChatState(store)
        state.begin("q", "r1")
        state.pending()
        assertEquals("pending", state.progress)
        assertEquals("r1", store.value)
        assertFalse(state.isRunning)
    }

    @Test
    fun streamingTextIsBounded() {
        val state = ChatState(MapStore())
        state.begin("q", "r1")
        state.apply("a".repeat(ChatState.MAX_STREAMING_CHARS))
        state.apply("b")
        assertEquals(ChatState.MAX_STREAMING_CHARS, state.streaming.length)
    }

    @Test
    fun emptyRunIdIsRefused() {
        assertFailsWith<IllegalArgumentException> { ChatState(MapStore()).begin("q", "") }
    }
}

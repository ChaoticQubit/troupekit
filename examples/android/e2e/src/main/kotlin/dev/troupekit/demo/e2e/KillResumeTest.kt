package dev.troupekit.demo.e2e

import android.accessibilityservice.AccessibilityServiceInfo
import android.app.UiAutomation
import android.os.Bundle
import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo
import androidx.test.platform.app.InstrumentationRegistry
import java.io.FileInputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * End-to-end tests of the Android demo on an emulator, run from a separate test APK so that
 * `am force-stop` of the app does not stop the test. The app runs its debug-only scripted model;
 * the UI is read and driven through the platform `UiAutomation` accessibility tree.
 */
class KillResumeTest {
    private val pkg = "dev.troupekit.demo"
    private val timeoutMs = 60_000L
    private val automation: UiAutomation = InstrumentationRegistry.getInstrumentation().uiAutomation

    @Before
    fun reportViewIds() {
        automation.serviceInfo = automation.serviceInfo.apply {
            flags = flags or AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS
        }
    }

    private fun shell(command: String): String {
        val descriptor = automation.executeShellCommand(command)
        return FileInputStream(descriptor.fileDescriptor).use { it.readBytes().decodeToString() }.also { descriptor.close() }
    }

    private fun launch(reset: Boolean, budgetMs: Long? = null) {
        val extras = buildString {
            append(" --ez scripted true")
            if (reset) append(" --ez resetState true")
            if (budgetMs != null) append(" --el resumeForBudgetMs $budgetMs")
        }
        shell("am start -W -n $pkg/.MainActivity$extras")
    }

    private fun nodes(id: String): List<AccessibilityNodeInfo> {
        val root = automation.rootInActiveWindow ?: return emptyList()
        return root.findAccessibilityNodeInfosByViewId("$pkg:id/$id")
    }

    private fun text(id: String): String? = nodes(id).firstOrNull()?.text?.toString()

    private fun waitUntil(what: String, check: () -> Boolean) {
        val deadline = SystemClock.uptimeMillis() + timeoutMs
        while (SystemClock.uptimeMillis() < deadline) {
            if (check()) return
            SystemClock.sleep(200)
        }
        throw AssertionError("timed out waiting for $what")
    }

    private fun ask(question: String) {
        waitUntil("question field") { nodes("question").isNotEmpty() }
        val field = nodes("question").first()
        val arguments = Bundle().apply {
            putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, question)
        }
        assertTrue(field.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments))
        assertTrue(nodes("send").first().performAction(AccessibilityNodeInfo.ACTION_CLICK))
    }

    @Test
    fun android_force_stop_mid_run_resumes() {
        launch(reset = true)
        ask("How do I reset my password?")
        waitUntil("steps 2") { text("steps") == "steps 2" }
        val runId = text("runId")
        assertTrue("$runId", runId != null && runId.startsWith("run "))
        shell("am force-stop $pkg")

        launch(reset = false)
        waitUntil("run id after relaunch") { text("runId") != null }
        assertEquals(runId, text("runId"))
        waitUntil("answer") { nodes("answer").any { it.text?.contains("account-password-reset.md") == true } }
        assertEquals(1, nodes("answer").size)
    }

    @Test
    fun android_stream_shows_deltas() {
        launch(reset = true)
        ask("How do I reset my password?")
        waitUntil("partial answer") { text("streaming")?.contains("Open Settings") == true }
        val partial = text("streaming")!!
        waitUntil("answer") { nodes("answer").any { it.text?.contains("account-password-reset.md") == true } }
        val answer = nodes("answer").first().text.toString()
        assertTrue("$partial is not a prefix of $answer", answer.startsWith(partial))
        assertTrue(partial.length < answer.length)
    }

    @Test
    fun android_worker_resume_for_pending() {
        launch(reset = true, budgetMs = 0)
        ask("How do I reset my password?")
        waitUntil("run id") { text("runId") != null }
        shell("cmd jobscheduler run -f $pkg 1")
        waitUntil("pending") { text("progress") == "pending" }
        assertEquals(0, nodes("answer").size)
    }
}

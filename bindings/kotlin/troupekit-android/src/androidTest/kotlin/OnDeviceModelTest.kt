package dev.troupekit.android

import androidx.test.platform.app.InstrumentationRegistry
import dev.troupekit.AgentBuilder
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.Test
import kotlin.test.assertTrue

/**
 * Instrumented test: a TOML agent answers through the on-device model.
 * Needs the model file pushed to the app's files dir; skipped with a printed reason otherwise,
 * which is the case on CI runners (plan R3, M1 risk on the 584 MB model download).
 */
class OnDeviceModelTest {
    @Test
    fun onDeviceModelAnswers() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val path = File(context.filesDir, "gemma3-1b-it-int4.litertlm").path
        val model = OnDeviceModel(path)
        val reason = model.unavailableMessage()
        if (reason != null) {
            println("SKIP onDeviceModelAnswers: $reason")
        }
        assumeTrue(reason == null)
        val toml = """
            name = "support"
            display_name = "Support"
            role = "Answers customers"
            goal = "Answer in one sentence"
            instructions = "Answer in one short sentence."
        """.trimIndent()
        val agent = AgentBuilder.fromToml(toml).hostModel(model).build()
        val answer = agent.run("Say hello.")
        assertTrue(answer.isNotEmpty())
    }
}

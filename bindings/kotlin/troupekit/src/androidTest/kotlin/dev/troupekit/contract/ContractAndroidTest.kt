package dev.troupekit.contract

import androidx.test.platform.app.InstrumentationRegistry
import java.io.File

/**
 * The contract cases on an Android device or emulator, against the AAR's
 * `libtroupekit_ffi.so`. The shared case list and fixture are packed into
 * the test APK as assets by `troupekit/build.gradle.kts`.
 */
class ContractAndroidTest : ContractCases() {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    override fun shared(name: String): String =
        instrumentation.context.assets.open(name).use { it.readBytes().toString(Charsets.UTF_8) }

    override fun scratchRoot(): File = instrumentation.targetContext.cacheDir
}

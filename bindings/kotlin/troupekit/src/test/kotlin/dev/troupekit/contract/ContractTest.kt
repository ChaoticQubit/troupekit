package dev.troupekit.contract

import java.io.File
import java.nio.file.Files

/**
 * The contract cases on the JVM, against a host-built `libtroupekit_ffi`
 * found through `jna.library.path`. Gradle sets `troupekit.repo` to the
 * repository root, where the shared case list and fixture live.
 */
class ContractTest : ContractCases() {
    private val repo = File(checkNotNull(System.getProperty("troupekit.repo")) {
        "Gradle sets troupekit.repo for unit tests"
    })
    private val files = mapOf(
        "cases.txt" to "crates/troupekit-ffi-core/contract/cases.txt",
        "agent.toml" to "crates/troupekit-py/tests/fixtures/agent.toml",
    )
    private val root: File by lazy { Files.createTempDirectory("troupekit-contract").toFile() }

    override fun shared(name: String): String {
        val path = checkNotNull(files[name]) { "no shared file named $name" }
        return File(repo, path).readText(Charsets.UTF_8)
    }

    override fun scratchRoot(): File = root
}

package dev.troupekit.android

import com.sun.net.httpserver.HttpServer
import dev.troupekit.HostHeader
import dev.troupekit.HostHttpRequest
import dev.troupekit.HostMessage
import dev.troupekit.HostRequest
import java.io.File
import java.net.InetSocketAddress
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * JVM tests of the Android platform adapters: request mapping, unavailable runtime, transport.
 * The transport runs against the JDK's built-in HTTP server on 127.0.0.1, no mock.
 * The on-device model itself needs a device or emulator and is tested in androidTest.
 */
class AndroidAdapterTest {
    private lateinit var server: HttpServer
    private var hitsOnTarget = 0

    private fun base(): String = "http://127.0.0.1:${server.address.port}"

    @BeforeTest
    fun startServer() {
        server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        server.createContext("/v1/chat") { exchange ->
            val body = "data: hi\n\n".toByteArray()
            exchange.responseHeaders.add("content-type", "text/event-stream")
            exchange.sendResponseHeaders(200, body.size.toLong())
            exchange.responseBody.use { it.write(body) }
        }
        server.createContext("/v1/redirect") { exchange ->
            exchange.responseHeaders.add("location", "${base()}/target")
            exchange.sendResponseHeaders(302, -1)
            exchange.close()
        }
        server.createContext("/target") { exchange ->
            hitsOnTarget += 1
            val body = "leaked".toByteArray()
            exchange.sendResponseHeaders(200, body.size.toLong())
            exchange.responseBody.use { it.write(body) }
        }
        server.start()
    }

    @AfterTest
    fun stopServer() {
        server.stop(0)
    }

    @Test
    fun unavailableRuntimeThrowsNamedHostError() {
        val missing = File(System.getProperty("java.io.tmpdir"), "troupekit-missing-model.litertlm")
        missing.delete()
        val model = OnDeviceModel(missing.path)
        val message = model.unavailableMessage()
        assertEquals("on-device model unavailable: model file missing", message)
        val request = HostRequest(null, listOf(HostMessage("user", "hi", null, null)), null, null, 16u)
        val error = assertFailsWith<Exception> { model.generateText(request) { } }
        assertTrue(error.toString().contains("on-device model unavailable: model file missing"), error.toString())
    }

    @Test
    fun requestMappingKeepsSystemAndTurns() {
        val request = HostRequest(
            "You are support.",
            listOf(
                HostMessage("user", "Where is my order 42?", null, null),
                HostMessage("assistant", "", "[{\"id\":\"c1\",\"name\":\"lookup\",\"arguments\":{\"id\":42}}]", null),
                HostMessage("tool", "shipped", null, "c1"),
            ),
            null,
            null,
            128u,
        )
        val (instructions, prompt) = OnDeviceModel.render(request)
        assertEquals("You are support.", instructions)
        val user = prompt.indexOf("user: Where is my order 42?")
        val call = prompt.indexOf("assistant tool call: [{\"id\":\"c1\",\"name\":\"lookup\",\"arguments\":{\"id\":42}}]")
        val result = prompt.indexOf("tool result c1: shipped")
        assertTrue(user >= 0 && call > user && result > call, prompt)
        assertTrue(prompt.endsWith("assistant:"), prompt)
    }

    @Test
    fun requestMappingWithoutSystemUsesEmptyInstructions() {
        val request = HostRequest(null, listOf(HostMessage("user", "hi", null, null)), null, null, 16u)
        val (instructions, prompt) = OnDeviceModel.render(request)
        assertEquals("", instructions)
        assertEquals("user: hi\nassistant:", prompt)
    }

    @Test
    fun transportReportsFinalUrlAndStreamsBody() {
        val url = "${base()}/v1/chat"
        val heads = mutableListOf<Pair<Int, String>>()
        val body = java.io.ByteArrayOutputStream()
        val request = HostHttpRequest("POST", url, listOf(HostHeader("authorization", "Bearer k")), "{}".toByteArray())
        HttpUrlConnectionTransport().perform(request, { status, final, _ -> heads.add(status to final) }, { body.write(it) })
        assertEquals(listOf(200 to url), heads)
        assertEquals("data: hi\n\n", body.toString(Charsets.UTF_8))
    }

    @Test
    fun transportDoesNotFollowRedirects() {
        val url = "${base()}/v1/redirect"
        val heads = mutableListOf<Pair<Int, String>>()
        val request = HostHttpRequest("GET", url, listOf(HostHeader("authorization", "Bearer k")), ByteArray(0))
        HttpUrlConnectionTransport().perform(request, { status, final, _ -> heads.add(status to final) }, { })
        assertEquals(listOf(302 to url), heads)
        assertEquals(0, hitsOnTarget)
    }

    @Test
    fun transportChunksAreAtMost16KiB() {
        val big = ByteArray(40 * 1024) { 'a'.code.toByte() }
        server.createContext("/v1/big") { exchange ->
            exchange.sendResponseHeaders(200, big.size.toLong())
            exchange.responseBody.use { it.write(big) }
        }
        val sizes = mutableListOf<Int>()
        val request = HostHttpRequest("GET", "${base()}/v1/big", emptyList(), ByteArray(0))
        HttpUrlConnectionTransport().perform(request, { _, _, _ -> }, { sizes.add(it.size) })
        assertEquals(big.size, sizes.sum())
        assertTrue(sizes.all { it in 1..16 * 1024 }, sizes.toString())
    }
}

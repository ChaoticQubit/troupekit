package dev.troupekit.android

import dev.troupekit.HostException
import dev.troupekit.HostHeader
import dev.troupekit.HostHttpRequest
import dev.troupekit.HostMessage
import dev.troupekit.HostRequest
import java.io.BufferedInputStream
import java.io.File
import java.io.InputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketException
import java.util.concurrent.ConcurrentHashMap
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * JVM tests of the Android platform adapters: request mapping, unavailable runtime, transport.
 * The transport runs against a real HTTP/1.1 server on a 127.0.0.1 socket, no mock; the JDK's
 * `com.sun.net.httpserver` is not on the Android unit-test compile classpath.
 * The on-device model itself needs a device or emulator and is tested in androidTest.
 */
class AndroidAdapterTest {
    private lateinit var server: TestHttpServer
    @Volatile private var hitsOnTarget = 0

    private fun base(): String = "http://127.0.0.1:${server.port}"

    @BeforeTest
    fun startServer() {
        server = TestHttpServer()
        server.route("/v1/chat") { Reply(200, mapOf("content-type" to "text/event-stream"), "data: hi\n\n".toByteArray()) }
        server.route("/v1/redirect") { Reply(302, mapOf("location" to "${base()}/target"), ByteArray(0)) }
        server.route("/target") {
            hitsOnTarget += 1
            Reply(200, emptyMap(), "leaked".toByteArray())
        }
    }

    @AfterTest
    fun stopServer() {
        server.close()
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
    fun replyTimeoutCancelsAndThrowsNamedHostError() {
        var cancelled = false
        val deltas = mutableListOf<String>()
        var late: ((String) -> Unit)? = null
        val error = assertFailsWith<HostException.Failed> {
            OnDeviceModel.collect(50, { chunk, _, _ -> late = chunk }, { cancelled = true }) { deltas.add(it) }
        }
        assertEquals("on-device model error: no reply within 50 ms", error.detail)
        assertTrue(cancelled)
        late?.invoke("too late")
        assertEquals(emptyList(), deltas)
    }

    @Test
    fun replyBeforeTimeoutStreamsEveryChunk() {
        val deltas = mutableListOf<String>()
        val text = OnDeviceModel.collect(5_000, { chunk, done, _ ->
            chunk("hel")
            chunk("")
            chunk("lo")
            done()
        }, { error("no cancel on a finished reply") }) { deltas.add(it) }
        assertEquals("hello", text)
        assertEquals(listOf("hel", "lo"), deltas)
    }

    @Test
    fun streamErrorThrowsNamedHostError() {
        val error = assertFailsWith<HostException.Failed> {
            OnDeviceModel.collect(5_000, { _, _, fail -> fail(IllegalStateException("busy")) }, { }) { }
        }
        assertEquals("on-device model error: busy", error.detail)
    }

    @Test
    fun nonPositiveReplyTimeoutIsRefused() {
        assertFailsWith<IllegalArgumentException> { OnDeviceModel("unused.litertlm", replyTimeoutMs = 0) }
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
        server.route("/v1/big") { Reply(200, emptyMap(), big) }
        val sizes = mutableListOf<Int>()
        val request = HostHttpRequest("GET", "${base()}/v1/big", emptyList(), ByteArray(0))
        HttpUrlConnectionTransport().perform(request, { _, _, _ -> }, { sizes.add(it.size) })
        assertEquals(big.size, sizes.sum())
        assertTrue(sizes.all { it in 1..16 * 1024 }, sizes.toString())
    }
}

/** One canned HTTP response: status, headers and body. */
private class Reply(val status: Int, val headers: Map<String, String>, val body: ByteArray)

/**
 * A minimal HTTP/1.1 server on an ephemeral 127.0.0.1 port, one request per connection.
 * Input: routes from path to reply, registered before the request. Output: real socket replies;
 * an unknown path gets 404. Closing stops the accept loop.
 */
private class TestHttpServer : AutoCloseable {
    private val socket = ServerSocket(0, 16, InetAddress.getByName("127.0.0.1"))
    private val routes = ConcurrentHashMap<String, () -> Reply>()
    val port: Int = socket.localPort

    init {
        Thread { serve() }.apply { isDaemon = true }.start()
    }

    fun route(path: String, reply: () -> Reply) {
        routes[path] = reply
    }

    override fun close() {
        socket.close()
    }

    private fun serve() {
        while (true) {
            val client = try {
                socket.accept()
            } catch (closed: SocketException) {
                return
            }
            client.use { answer(it) }
        }
    }

    private fun answer(client: Socket) {
        val input = BufferedInputStream(client.getInputStream())
        val path = line(input).split(" ")[1]
        var length = 0
        while (true) {
            val header = line(input)
            if (header.isEmpty()) {
                break
            }
            val name = header.substringBefore(":").trim()
            if (name.equals("content-length", ignoreCase = true)) {
                length = header.substringAfter(":").trim().toInt()
            }
        }
        repeat(length) { check(input.read() >= 0) { "request body ended early" } }
        val reply = routes[path]?.invoke() ?: Reply(404, emptyMap(), ByteArray(0))
        val head = StringBuilder("HTTP/1.1 ${reply.status} Status\r\n")
        for ((name, value) in reply.headers) {
            head.append("$name: $value\r\n")
        }
        head.append("content-length: ${reply.body.size}\r\nconnection: close\r\n\r\n")
        val output = client.getOutputStream()
        output.write(head.toString().toByteArray(Charsets.ISO_8859_1))
        output.write(reply.body)
        output.flush()
    }

    private fun line(input: InputStream): String {
        val bytes = StringBuilder()
        while (true) {
            val next = input.read()
            if (next < 0 || next == '\n'.code) {
                return bytes.toString().trimEnd('\r')
            }
            bytes.append(next.toChar())
        }
    }
}

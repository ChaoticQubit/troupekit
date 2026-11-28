package dev.troupekit.android

import dev.troupekit.HostException
import dev.troupekit.HostHeader
import dev.troupekit.HostHttpRequest
import dev.troupekit.HostTransport
import dev.troupekit.HttpSink
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * The platform HTTP stack, `java.net.HttpURLConnection`, as a troupekit `HostTransport`.
 * For model and embedding calls only: no cache, redirects refused, body bytes passed on as they
 * arrive in chunks of at most 16 KiB. URL, origin and size checks stay on the Rust side.
 */
class HttpUrlConnectionTransport(
    private val connectTimeoutMs: Int = 30_000,
    private val readTimeoutMs: Int = 300_000,
) : HostTransport {
    init {
        require(connectTimeoutMs > 0) { "connectTimeoutMs must be above 0, got $connectTimeoutMs" }
        require(readTimeoutMs > 0) { "readTimeoutMs must be above 0, got $readTimeoutMs" }
    }

    override fun send(request: HostHttpRequest, sink: HttpSink) {
        try {
            perform(
                request,
                { status, url, headers -> sink.head(status.toUShort(), url, headers) },
                { bytes -> sink.chunk(bytes) },
            )
        } catch (error: HostException) {
            throw error
        } catch (error: Exception) {
            throw HostException.Failed("transport error: ${error.javaClass.simpleName}: ${error.message ?: ""}")
        }
    }

    /**
     * Sends one request and streams the reply: `head` once, then `chunk` per body piece.
     * Input: the request and the two callbacks. Output: returns when the body ends; throws on failure.
     * A redirect is reported as its own status, never followed.
     */
    fun perform(
        request: HostHttpRequest,
        head: (Int, String, List<HostHeader>) -> Unit,
        chunk: (ByteArray) -> Unit,
    ) {
        val connection = URL(request.url).openConnection() as? HttpURLConnection
            ?: throw HostException.Failed("transport error: not an http url")
        try {
            connection.instanceFollowRedirects = false
            connection.useCaches = false
            connection.connectTimeout = connectTimeoutMs
            connection.readTimeout = readTimeoutMs
            connection.requestMethod = request.method
            for (header in request.headers) {
                connection.addRequestProperty(header.name, header.value)
            }
            if (request.body.isNotEmpty()) {
                connection.doOutput = true
                connection.setFixedLengthStreamingMode(request.body.size)
                connection.outputStream.use { it.write(request.body) }
            }
            val status = connection.responseCode
            if (status !in 100..999) {
                throw HostException.Failed("transport error: invalid status line")
            }
            head(status, connection.url.toString(), responseHeaders(connection))
            val body = if (status >= 400) connection.errorStream else connection.inputStream
            body?.use { copy(it, chunk) }
        } finally {
            connection.disconnect()
        }
    }

    private fun responseHeaders(connection: HttpURLConnection): List<HostHeader> {
        val headers = ArrayList<HostHeader>()
        for ((name, values) in connection.headerFields) {
            if (name == null) {
                continue
            }
            for (value in values) {
                headers.add(HostHeader(name.lowercase(), value))
            }
        }
        return headers
    }

    private fun copy(input: InputStream, chunk: (ByteArray) -> Unit) {
        val buffer = ByteArray(CHUNK_BYTES)
        while (true) {
            val read = input.read(buffer)
            if (read < 0) {
                return
            }
            if (read > 0) {
                chunk(buffer.copyOf(read))
            }
        }
    }

    private companion object {
        const val CHUNK_BYTES = 16 * 1024
    }
}

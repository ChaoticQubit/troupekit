import Foundation
import Troupekit

/// The platform HTTP stack as a troupekit `HostTransport`, for model and embedding calls only.
/// Ephemeral session: no cookies, no cache, redirects refused. Body bytes reach the sink as they
/// arrive, in chunks of at most 16 KiB. URL, origin and size checks stay on the Rust side (P7).
public final class URLSessionTransport: HostTransport, @unchecked Sendable {
    static let chunkBytes = 16 * 1024

    private let session: URLSession
    private let router: TransportRouter

    /// Creates the transport over a fresh ephemeral session.
    /// Input: none. Output: the transport.
    public convenience init() {
        self.init(configuration: .ephemeral)
    }

    init(configuration: URLSessionConfiguration) {
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        let router = TransportRouter()
        self.router = router
        session = URLSession(configuration: configuration, delegate: router, delegateQueue: nil)
    }

    deinit {
        session.invalidateAndCancel()
    }

    public func send(request: HostHttpRequest, sink: HttpSink) throws {
        do {
            try Blocking.run {
                try await self.perform(
                    request,
                    head: { status, url, headers in sink.head(status: UInt16(clamping: status), url: url, headers: headers) },
                    chunk: { bytes in sink.chunk(bytes: bytes) }
                )
            }
        } catch {
            throw hostError(from: error, prefix: "transport error")
        }
    }

    /// Sends one request and streams the reply: `head` once, then `chunk` per body piece.
    /// Input: the request and the two callbacks. Output: returns when the body ends; throws on failure.
    func perform(
        _ request: HostHttpRequest,
        head: @escaping (Int, String, [HostHeader]) -> Void,
        chunk: @escaping (Data) -> Void
    ) async throws {
        guard let url = URL(string: request.url) else {
            throw AppleAdapterError("transport error: bad url")
        }
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = request.method
        for header in request.headers {
            urlRequest.addValue(header.value, forHTTPHeaderField: header.name)
        }
        if !request.body.isEmpty {
            urlRequest.httpBody = request.body
        }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let task = session.dataTask(with: urlRequest)
            router.register(TransportTask(requestURL: request.url, head: head, chunk: chunk, continuation: continuation), for: task.taskIdentifier)
            task.resume()
        }
    }
}

/// Session delegate that routes each task's events to its `TransportTask` by task id.
/// Redirects are refused for every task. An entry is removed when its task completes.
/// Input: URLSession delegate calls. Output: calls on the registered task, or none.
final class TransportRouter: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var tasks: [Int: TransportTask] = [:]

    func register(_ task: TransportTask, for id: Int) {
        lock.lock()
        defer { lock.unlock() }
        precondition(tasks[id] == nil, "task id registered twice")
        tasks[id] = task
    }

    private func task(_ id: Int, remove: Bool = false) -> TransportTask? {
        lock.lock()
        defer { lock.unlock() }
        return remove ? tasks.removeValue(forKey: id) : tasks[id]
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let entry = task(dataTask.taskIdentifier), let http = response as? HTTPURLResponse else {
            completionHandler(.cancel)
            return
        }
        entry.receive(http)
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        task(dataTask.taskIdentifier)?.receive(data)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        self.task(task.taskIdentifier, remove: true)?.complete(error)
    }
}

/// One request in flight: reports the head, splits the body, ends the caller's wait once.
/// Input: the callbacks and the continuation to resume. Output: callbacks in arrival order.
final class TransportTask: @unchecked Sendable {
    private let requestURL: String
    private let head: (Int, String, [HostHeader]) -> Void
    private let chunk: (Data) -> Void
    private let continuation: CheckedContinuation<Void, Error>

    init(
        requestURL: String,
        head: @escaping (Int, String, [HostHeader]) -> Void,
        chunk: @escaping (Data) -> Void,
        continuation: CheckedContinuation<Void, Error>
    ) {
        self.requestURL = requestURL
        self.head = head
        self.chunk = chunk
        self.continuation = continuation
    }

    func receive(_ response: HTTPURLResponse) {
        let headers = response.allHeaderFields.compactMap { key, value -> HostHeader? in
            guard let name = key as? String else { return nil }
            return HostHeader(name: name, value: "\(value)")
        }
        head(response.statusCode, response.url?.absoluteString ?? requestURL, headers)
    }

    func receive(_ data: Data) {
        var start = data.startIndex
        while start < data.endIndex {
            let end = data.index(start, offsetBy: URLSessionTransport.chunkBytes, limitedBy: data.endIndex) ?? data.endIndex
            chunk(data.subdata(in: start..<end))
            start = end
        }
    }

    func complete(_ error: Error?) {
        if let error {
            continuation.resume(throwing: AppleAdapterError("transport error: \(error.localizedDescription)"))
        } else {
            continuation.resume()
        }
    }
}

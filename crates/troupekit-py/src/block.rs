use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

/// Wakes the thread that is blocked in `block_on`.
/// Input: the thread to unpark. Output: none.
/// One per `block_on` call.
struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark()
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark()
    }
}

/// Drives one neutral future to completion on the calling thread.
/// The neutral futures only await a tokio `JoinHandle` (decision P4), which
/// needs no executor of its own, so parking between polls is enough.
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        thread::park();
    }
}

/// Runs one neutral call with the GIL released, then raises its error.
/// Input: the GIL token and the neutral future. Output: its value, or the
/// `TroupekitError` its `FfiError` becomes; Python threads run meanwhile.
pub(crate) fn wait<T, F>(py: pyo3::Python<'_>, future: F) -> pyo3::PyResult<T>
where
    T: Send,
    F: Future<Output = Result<T, troupekit_ffi_core::FfiError>> + Send,
{
    py.detach(|| block_on(future))
        .map_err(|error| crate::error::raise(py, error))
}

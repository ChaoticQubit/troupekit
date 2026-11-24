use crate::catalog::Price;
use crate::model::Model;
use crate::request::{CompletionRequest, CompletionResponse, Delta};
use crate::status::bad_request;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use troupekit_core::error::ProviderError;

/// Waits between retries. `TokioSleep` is the real clock; tests inject a
/// recorder so no test sleeps.
/// `ms` is the wait in milliseconds.
#[async_trait]
pub trait Sleep: Send + Sync {
    async fn sleep(&self, ms: u64);
}

/// Sleeps on the tokio timer, the default for `RetryModel`.
/// Needs a running tokio runtime with the timer enabled.
/// Holds no state.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioSleep;

#[async_trait]
impl Sleep for TokioSleep {
    async fn sleep(&self, ms: u64) {
        tokio::time::sleep(Duration::from_millis(ms)).await;
    }
}

/// How often and how long `RetryModel` retries: `max_attempts` tries in
/// all, exponential waits from `base_delay_ms`, never waiting longer than
/// `max_delay_ms`. Defaults to 3 tries, 500 ms base, 30 s cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 500,
            max_delay_ms: 30_000,
        }
    }
}

impl RetryPolicy {
    /// Wait before the next try after `failures` failed tries, the last
    /// ending in `err`. Returns `None` to give up: not retryable, tries
    /// used up, or the server asked to wait past `max_delay_ms`.
    pub fn next_delay_ms(&self, failures: u32, err: &ProviderError) -> Option<u64> {
        assert!(failures >= 1, "asked for a retry delay before any failure");
        if !err.is_retryable() || failures >= self.max_attempts {
            return None;
        }
        match err.retry_after_ms {
            Some(ms) if ms <= self.max_delay_ms => Some(ms),
            Some(_) => None,
            None => Some(
                self.base_delay_ms
                    .saturating_mul(1u64 << (failures - 1).min(32))
                    .min(self.max_delay_ms),
            ),
        }
    }
}

/// Retries an inner model on transient failures, honouring server retry
/// delays. A stream is retried only while no delta has reached the
/// caller, so a UI never sees a reply twice.
pub struct RetryModel {
    inner: Box<dyn Model>,
    policy: RetryPolicy,
    sleep: Arc<dyn Sleep>,
}

impl RetryModel {
    /// Wraps `inner` with `policy`, waiting on the tokio clock.
    /// Returns `BadRequest` when `max_attempts` is not 1 to 10 or the
    /// base delay is above the max delay.
    pub fn new(inner: Box<dyn Model>, policy: RetryPolicy) -> Result<Self, ProviderError> {
        if !(1..=10).contains(&policy.max_attempts) {
            return Err(bad_request("retry max_attempts must be 1 to 10"));
        }
        if policy.base_delay_ms > policy.max_delay_ms {
            return Err(bad_request("retry base delay is above its max delay"));
        }
        Ok(Self {
            inner,
            policy,
            sleep: Arc::new(TokioSleep),
        })
    }

    /// Replaces the clock with `sleep`, so tests record waits instead of
    /// waiting. Returns the model.
    /// The policy is unchanged.
    pub fn with_sleep(mut self, sleep: Arc<dyn Sleep>) -> Self {
        self.sleep = sleep;
        self
    }
}

#[async_trait]
impl Model for RetryModel {
    fn id(&self) -> &str {
        self.inner.id()
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        let mut failures = 0;
        loop {
            let err = match self.inner.complete(req).await {
                Ok(reply) => return Ok(reply),
                Err(err) => err,
            };
            failures += 1;
            match self.policy.next_delay_ms(failures, &err) {
                Some(ms) => self.sleep.sleep(ms).await,
                None => return Err(err),
            }
        }
    }

    async fn stream(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        let mut failures = 0;
        loop {
            let mut emitted = false;
            let result = self
                .inner
                .stream(req, &mut |d: &Delta| {
                    emitted = true;
                    on_delta(d);
                })
                .await;
            let err = match result {
                Ok(reply) => return Ok(reply),
                Err(err) if emitted => return Err(err),
                Err(err) => err,
            };
            failures += 1;
            match self.policy.next_delay_ms(failures, &err) {
                Some(ms) => self.sleep.sleep(ms).await,
                None => return Err(err),
            }
        }
    }
}

/// Tries an ordered list of models and returns the first success, or
/// the last error when all fail. A stream moves on only while no delta
/// has reached the caller.
pub struct FallbackModel {
    id: String,
    models: Vec<Box<dyn Model>>,
}

impl FallbackModel {
    /// Builds a fallback chain from `first` then each of `rest`, so the
    /// chain is never empty. Its id is the model ids joined by commas.
    /// Returns the chain.
    pub fn new(first: Box<dyn Model>, rest: Vec<Box<dyn Model>>) -> Self {
        let mut models = vec![first];
        models.extend(rest);
        let id = models.iter().map(|m| m.id()).collect::<Vec<_>>().join(",");
        Self { id, models }
    }
}

#[async_trait]
impl Model for FallbackModel {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        assert!(
            !self.models.is_empty(),
            "a fallback chain always holds a model"
        );
        let mut last = None;
        for model in &self.models {
            match model.complete(req).await {
                Ok(reply) => return Ok(reply),
                Err(err) => last = Some(err),
            }
        }
        Err(last.expect("the loop ran at least once"))
    }

    async fn stream(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        assert!(
            !self.models.is_empty(),
            "a fallback chain always holds a model"
        );
        let mut last = None;
        for model in &self.models {
            let mut emitted = false;
            let result = model
                .stream(req, &mut |d: &Delta| {
                    emitted = true;
                    on_delta(d);
                })
                .await;
            match result {
                Ok(reply) => return Ok(reply),
                Err(err) if emitted => return Err(err),
                Err(err) => last = Some(err),
            }
        }
        Err(last.expect("the loop ran at least once"))
    }
}

/// Fills in `usage.cost_usd_micros` from catalog prices on every reply.
/// `Catalog::connect` wraps each model in one; `price` exposes the rates
/// and whether they were verified.
pub struct MeteredModel {
    inner: Box<dyn Model>,
    price: Price,
}

impl MeteredModel {
    /// Wraps `inner`, pricing its replies with `price`.
    /// Returns the metered model.
    /// The inner model's id is kept.
    pub fn new(inner: Box<dyn Model>, price: Price) -> Self {
        Self { inner, price }
    }

    /// Returns the price this meter applies, including `verified`, so a
    /// budget can refuse to treat an unverified zero as free.
    /// Never changes after `new`.
    pub fn price(&self) -> Price {
        self.price
    }

    fn priced(&self, mut reply: CompletionResponse) -> CompletionResponse {
        reply.usage.cost_usd_micros = self.price.reply_cost_usd_micros(&reply);
        reply
    }
}

#[async_trait]
impl Model for MeteredModel {
    fn id(&self) -> &str {
        self.inner.id()
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        self.inner
            .complete(req)
            .await
            .map(|reply| self.priced(reply))
    }

    async fn stream(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        self.inner
            .stream(req, on_delta)
            .await
            .map(|reply| self.priced(reply))
    }
}

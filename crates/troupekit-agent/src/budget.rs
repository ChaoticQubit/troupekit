use crate::persona::Budget;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use troupekit_core::error::CoreError;
use troupekit_core::event::Usage;
use troupekit_core::tool::WEB_SEARCH;
use troupekit_providers::{CompletionRequest, Price};

/// A spend limit for one period, such as a goal's or a user's month.
/// `key` names the ledger total (for example `goal:42`); the ledger decides
/// what the period is. Amounts are millionths of a US dollar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeriodLimit {
    pub key: String,
    pub max_usd_micros: u64,
}

/// The layered limits a run checks before every model call: one call,
/// the whole run, the scope's period and the user's period. `None` is no
/// limit. Amounts are millionths of a US dollar.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub call: Option<u64>,
    pub run: Option<u64>,
    pub scope: Option<PeriodLimit>,
    pub user: Option<PeriodLimit>,
}

impl Limits {
    /// Applies a persona `budget` on top: each of `call` and `run` becomes
    /// the stricter of the two, so neither side can loosen the other.
    /// Returns the combined limits.
    pub fn with_budget(mut self, budget: &Budget) -> Self {
        let stricter = |a: Option<u64>, b: Option<u64>| match (a, b) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        };
        self.call = stricter(self.call, budget.call_usd_micros);
        self.run = stricter(self.run, budget.run_usd_micros);
        self
    }

    /// Reports whether no layer has a limit.
    /// An empty set skips every budget check, including the
    /// unverified price check.
    pub fn is_empty(&self) -> bool {
        self.call.is_none() && self.run.is_none() && self.scope.is_none() && self.user.is_none()
    }
}

pub use troupekit_core::store::UsageLedger;

/// In-memory `UsageLedger` for tests and examples.
/// It has one period that never ends; `usage` reads a key's totals.
/// Safe to share across threads.
#[derive(Debug, Default)]
pub struct InMemoryLedger(Mutex<HashMap<String, Usage>>);

impl InMemoryLedger {
    /// Returns the usage recorded under `key`.
    /// A key never recorded reads as zero usage.
    /// Tests use it to check what a run charged.
    pub fn usage(&self, key: &str) -> Usage {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(key)
            .copied()
            .unwrap_or_default()
    }
}

#[async_trait]
impl UsageLedger for InMemoryLedger {
    async fn period_total(&self, key: &str) -> Result<u64, CoreError> {
        Ok(self.usage(key).cost_usd_micros)
    }

    async fn record(&self, key: &str, usage: Usage) -> Result<(), CoreError> {
        *self
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(key.to_string())
            .or_default() += usage;
        Ok(())
    }
}

/// Worst-case cost of sending `req` at `price`: input tokens counted as
/// a quarter of the JSON bytes of system text, messages and tool specs,
/// rounded up, plus `max_tokens` output tokens and every web search the
/// hosted tools allow. Never under-counts output.
pub fn estimate(req: &CompletionRequest, price: &Price) -> u64 {
    let bytes = req.system.as_ref().map_or(0, String::len)
        + serde_json::to_string(&req.messages)
            .expect("messages serialize")
            .len()
        + serde_json::to_string(&req.tools)
            .expect("tool specs serialize")
            .len();
    let input_tokens = u64::try_from(bytes.div_ceil(4)).expect("byte count fits in u64");
    let searches: u64 = req
        .tools
        .iter()
        .filter_map(|t| t.hosted)
        .filter(|h| h.name() == WEB_SEARCH)
        .map(|h| u64::from(h.max_uses()))
        .sum();
    price
        .cost_usd_micros(&Usage {
            input_tokens,
            output_tokens: u64::from(req.max_tokens),
            cost_usd_micros: 0,
        })
        .saturating_add(searches.saturating_mul(price.search_usd_micros))
}

/// Checks a call costing up to `estimate` against every layer of `limits`,
/// given `run_spent` so far and the ledger's period totals. Returns the
/// first crossed layer: `unverified_price`, `call`, `run`, `scope`, `user`.
pub async fn check(
    limits: &Limits,
    price: &Price,
    run_spent: u64,
    estimate: u64,
    ledger: &dyn UsageLedger,
) -> Result<Option<&'static str>, CoreError> {
    if limits.is_empty() {
        return Ok(None);
    }
    if !price.verified {
        return Ok(Some("unverified_price"));
    }
    if limits.call.is_some_and(|max| estimate > max) {
        return Ok(Some("call"));
    }
    if limits
        .run
        .is_some_and(|max| run_spent.saturating_add(estimate) > max)
    {
        return Ok(Some("run"));
    }
    for (name, limit) in [("scope", &limits.scope), ("user", &limits.user)] {
        if let Some(l) = limit {
            if ledger.period_total(&l.key).await?.saturating_add(estimate) > l.max_usd_micros {
                return Ok(Some(name));
            }
        }
    }
    Ok(None)
}

/// Adds one call's `usage` to the ledger under the scope and user keys of
/// `limits`, when they are set.
/// Returns the first ledger failure.
pub async fn record(
    limits: &Limits,
    ledger: &dyn UsageLedger,
    usage: Usage,
) -> Result<(), CoreError> {
    for l in [&limits.scope, &limits.user].into_iter().flatten() {
        ledger.record(&l.key, usage).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRICE: Price = Price {
        input_usd_micros_per_mtok: 1_000_000,
        output_usd_micros_per_mtok: 2_000_000,
        verified: true,
        long_context: None,
        search_usd_micros: 0,
    };

    fn req() -> CompletionRequest {
        CompletionRequest::user("x".repeat(400), 1000)
    }

    #[test]
    fn estimate_is_worst_case() {
        let r = req();
        let bytes = serde_json::to_string(&r.messages).unwrap().len() + 2;
        let input = bytes.div_ceil(4) as u64;
        assert_eq!(
            estimate(&r, &PRICE),
            PRICE.cost_usd_micros(&Usage {
                input_tokens: input,
                output_tokens: 1000,
                cost_usd_micros: 0
            })
        );
        assert!(estimate(&r, &PRICE) >= 2000 + 100);
        assert_eq!(estimate(&r, &Price::FREE), 0);
    }

    #[test]
    fn estimate_adds_every_allowed_search() {
        use troupekit_core::tool::{HostedTool, ToolSpec};
        let mut r = req();
        let priced = Price {
            search_usd_micros: 10_000,
            ..PRICE
        };
        let base = estimate(&r, &priced);
        r.tools.push(ToolSpec {
            name: "web_search".into(),
            description: String::new(),
            schema: serde_json::json!({}),
            hosted: Some(HostedTool::WebSearch { max_uses: 3 }),
        });
        let tools = serde_json::to_string(&r.tools).unwrap().len() as u64;
        assert!(estimate(&r, &priced) >= base + 30_000);
        assert!(estimate(&r, &priced) <= base + 30_000 + tools.div_ceil(4) + 1);
    }

    #[tokio::test]
    async fn each_layer_pauses_by_name() {
        let ledger = InMemoryLedger::default();
        ledger
            .record(
                "goal:1",
                Usage {
                    cost_usd_micros: 900,
                    ..Usage::default()
                },
            )
            .await
            .unwrap();
        ledger
            .record(
                "user:1",
                Usage {
                    cost_usd_micros: 5_000,
                    ..Usage::default()
                },
            )
            .await
            .unwrap();
        let scope = Some(PeriodLimit {
            key: "goal:1".into(),
            max_usd_micros: 1_000,
        });
        let user = Some(PeriodLimit {
            key: "user:1".into(),
            max_usd_micros: 5_050,
        });
        let cases = [
            (
                Limits {
                    call: Some(99),
                    ..Limits::default()
                },
                0,
                Some("call"),
            ),
            (
                Limits {
                    call: Some(100),
                    ..Limits::default()
                },
                0,
                None,
            ),
            (
                Limits {
                    run: Some(150),
                    ..Limits::default()
                },
                60,
                Some("run"),
            ),
            (
                Limits {
                    run: Some(150),
                    ..Limits::default()
                },
                50,
                None,
            ),
            (
                Limits {
                    scope: scope.clone(),
                    ..Limits::default()
                },
                0,
                None,
            ),
            (
                Limits {
                    scope: Some(PeriodLimit {
                        key: "goal:1".into(),
                        max_usd_micros: 999,
                    }),
                    ..Limits::default()
                },
                0,
                Some("scope"),
            ),
            (
                Limits {
                    scope,
                    user,
                    ..Limits::default()
                },
                0,
                Some("user"),
            ),
        ];
        for (limits, spent, want) in cases {
            assert_eq!(
                check(&limits, &PRICE, spent, 100, &ledger).await.unwrap(),
                want,
                "{limits:?}"
            );
        }
    }

    #[test]
    fn persona_budget_only_tightens() {
        let b = Budget {
            call_usd_micros: Some(10),
            run_usd_micros: None,
        };
        let host = Limits {
            call: Some(20),
            run: Some(30),
            ..Limits::default()
        };
        assert_eq!(
            host.clone().with_budget(&b),
            Limits {
                call: Some(10),
                run: Some(30),
                ..Limits::default()
            }
        );
        let tight = Limits {
            call: Some(5),
            ..Limits::default()
        };
        assert_eq!(tight.with_budget(&b).call, Some(5));
        assert_eq!(Limits::default().with_budget(&b).call, Some(10));
    }

    #[tokio::test]
    async fn unverified_price_pauses_only_with_limits() {
        let ledger = InMemoryLedger::default();
        let unverified = Price {
            verified: false,
            ..Price::FREE
        };
        assert_eq!(
            check(&Limits::default(), &unverified, 0, 0, &ledger)
                .await
                .unwrap(),
            None
        );
        let limits = Limits {
            run: Some(1),
            ..Limits::default()
        };
        assert_eq!(
            check(&limits, &unverified, 0, 0, &ledger).await.unwrap(),
            Some("unverified_price")
        );
        assert_eq!(
            check(&limits, &Price::FREE, 0, 0, &ledger).await.unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn ledger_sums_per_key() {
        let ledger = InMemoryLedger::default();
        let limits = Limits {
            scope: Some(PeriodLimit {
                key: "goal:1".into(),
                max_usd_micros: 10,
            }),
            user: Some(PeriodLimit {
                key: "user:1".into(),
                max_usd_micros: 10,
            }),
            ..Limits::default()
        };
        let u = Usage {
            input_tokens: 3,
            output_tokens: 4,
            cost_usd_micros: 7,
        };
        record(&limits, &ledger, u).await.unwrap();
        record(&limits, &ledger, u).await.unwrap();
        record(&Limits::default(), &ledger, u).await.unwrap();
        assert_eq!(ledger.usage("goal:1"), u + u);
        assert_eq!(ledger.period_total("user:1").await.unwrap(), 14);
        assert_eq!(ledger.usage("other"), Usage::default());
    }
}

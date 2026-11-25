use crate::stores::{corrupt, immediate};
use crate::{db, SqliteStore};
use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde_json::Value;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::store::{JobQueue, JobState, Lease};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Row {
    state: String,
    claim_at_ms: Option<i64>,
    token: Option<i64>,
    next_due_ms: Option<i64>,
    error: Option<String>,
}

fn to_state(row: Row) -> Result<JobState, CoreError> {
    let token = row
        .token
        .map(|t| u64::try_from(t).map_err(|_| corrupt(format!("job token {t}"))))
        .transpose()?;
    let state = match (
        row.state.as_str(),
        row.claim_at_ms,
        token,
        row.next_due_ms,
        row.error,
    ) {
        ("pending", Some(due_ms), None, None, None) => JobState::Pending { due_ms },
        ("leased", Some(until_ms), Some(token), next_due_ms, None) if token > 0 => {
            JobState::Leased {
                token,
                until_ms,
                next_due_ms,
            }
        }
        ("done", None, None, None, None) => JobState::Done,
        ("failed", None, None, None, Some(error)) => JobState::Failed { error },
        (state, ..) => {
            return Err(corrupt(format!(
                "job row in state {state:?} has a bad shape"
            )))
        }
    };
    Ok(state)
}

fn to_row(state: &JobState) -> Result<Row, CoreError> {
    let row = match state {
        JobState::Pending { due_ms } => Row {
            state: "pending".into(),
            claim_at_ms: Some(*due_ms),
            token: None,
            next_due_ms: None,
            error: None,
        },
        JobState::Leased {
            token,
            until_ms,
            next_due_ms,
        } => Row {
            state: "leased".into(),
            claim_at_ms: Some(*until_ms),
            token: Some(crate::stores::to_sql_int("token", *token)?),
            next_due_ms: *next_due_ms,
            error: None,
        },
        JobState::Done => Row {
            state: "done".into(),
            claim_at_ms: None,
            token: None,
            next_due_ms: None,
            error: None,
        },
        JobState::Failed { error } => Row {
            state: "failed".into(),
            claim_at_ms: None,
            token: None,
            next_due_ms: None,
            error: Some(error.clone()),
        },
    };
    Ok(row)
}

fn to_attempts(value: i64) -> Result<u32, CoreError> {
    u32::try_from(value).map_err(|_| corrupt(format!("job attempts {value}")))
}

fn parse_run(text: &str) -> Result<RunId, CoreError> {
    serde_json::from_value(Value::String(text.to_string()))
        .map_err(|_| corrupt(format!("job run id {text:?}")))
}

fn read_job(conn: &Connection, run: &str) -> Result<Option<(JobState, u32)>, CoreError> {
    let row = conn
        .query_row(
            "SELECT state, claim_at_ms, token, next_due_ms, error, attempts FROM job \
             WHERE run_id = ?1",
            [run],
            |r| {
                let row = Row {
                    state: r.get(0)?,
                    claim_at_ms: r.get(1)?,
                    token: r.get(2)?,
                    next_due_ms: r.get(3)?,
                    error: r.get(4)?,
                };
                Ok((row, r.get::<_, i64>(5)?))
            },
        )
        .optional()
        .map_err(db)?;
    row.map(|(row, attempts)| Ok((to_state(row)?, to_attempts(attempts)?)))
        .transpose()
}

fn write_job(
    tx: &Transaction<'_>,
    run: &str,
    state: &JobState,
    attempts: u32,
) -> Result<(), CoreError> {
    let row = to_row(state)?;
    let written = tx
        .execute(
            "INSERT INTO job (run_id, state, claim_at_ms, token, next_due_ms, error, attempts) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) ON CONFLICT (run_id) DO UPDATE SET \
             state = excluded.state, claim_at_ms = excluded.claim_at_ms, \
             token = excluded.token, next_due_ms = excluded.next_due_ms, \
             error = excluded.error, attempts = excluded.attempts",
            params![
                run,
                row.state,
                row.claim_at_ms,
                row.token,
                row.next_due_ms,
                row.error,
                attempts
            ],
        )
        .map_err(db)?;
    assert_eq!(written, 1);
    assert_eq!(read_job(tx, run)?, Some((state.clone(), attempts)));
    Ok(())
}

fn next_token(tx: &Transaction<'_>) -> Result<u64, CoreError> {
    let last: i64 = tx
        .query_row(
            "UPDATE lease_token SET last = last + 1 WHERE id = 1 RETURNING last",
            [],
            |r| r.get(0),
        )
        .map_err(db)?;
    let token = u64::try_from(last).map_err(|_| corrupt(format!("lease token {last}")))?;
    assert!(token > 0, "tokens start at 1");
    Ok(token)
}

fn earlier(a: Option<i64>, b: i64) -> i64 {
    a.map_or(b, |a| a.min(b))
}

fn enqueued(state: Option<JobState>, due_ms: i64) -> JobState {
    match state {
        Some(JobState::Pending { due_ms: d }) => JobState::Pending {
            due_ms: d.min(due_ms),
        },
        Some(JobState::Leased {
            token,
            until_ms,
            next_due_ms,
        }) => JobState::Leased {
            token,
            until_ms,
            next_due_ms: Some(earlier(next_due_ms, due_ms)),
        },
        None | Some(JobState::Done) | Some(JobState::Failed { .. }) => JobState::Pending { due_ms },
    }
}

fn due_runs(tx: &Transaction<'_>, now: i64, limit: usize) -> Result<Vec<String>, CoreError> {
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let mut stmt = tx
        .prepare_cached(
            "SELECT run_id FROM job WHERE claim_at_ms IS NOT NULL AND claim_at_ms <= ?1 \
             ORDER BY claim_at_ms, run_id LIMIT ?2",
        )
        .map_err(db)?;
    let rows = stmt
        .query_map(params![now, limit], |r| r.get::<_, String>(0))
        .map_err(db)?;
    rows.map(|r| r.map_err(db)).collect()
}

fn claim_one(tx: &Transaction<'_>, run: &str, now: i64, until_ms: i64) -> Result<Lease, CoreError> {
    let (state, attempts) =
        read_job(tx, run)?.ok_or_else(|| corrupt(format!("due job {run} vanished")))?;
    let next_due_ms = match state {
        JobState::Leased {
            until_ms: held,
            next_due_ms,
            ..
        } => {
            assert!(held <= now, "only expired leases are claimed");
            next_due_ms
        }
        JobState::Pending { due_ms } => {
            assert!(due_ms <= now, "only due jobs are claimed");
            None
        }
        JobState::Done | JobState::Failed { .. } => {
            return Err(corrupt(format!("ended job {run} has a claim time")))
        }
    };
    let attempts = attempts
        .checked_add(1)
        .ok_or_else(|| corrupt(format!("job {run} attempts at u32::MAX")))?;
    let token = next_token(tx)?;
    write_job(
        tx,
        run,
        &JobState::Leased {
            token,
            until_ms,
            next_due_ms,
        },
        attempts,
    )?;
    Ok(Lease {
        run_id: parse_run(run)?,
        token,
        attempts,
    })
}

impl SqliteStore {
    /// Returns the state of `run_id`'s job, as `InMemQueue::job` does.
    /// Input: the run. Output: `None` when the run was never enqueued, or
    /// `CoreError::Store` for a row the file should not hold.
    pub async fn job(&self, run_id: RunId) -> Result<Option<JobState>, CoreError> {
        let run = run_id.to_string();
        self.call(move |conn| Ok(read_job(conn, &run)?.map(|(state, _)| state)))
            .await
    }

    async fn end_lease(
        &self,
        lease: &Lease,
        f: impl FnOnce(Option<i64>) -> JobState + Send + 'static,
    ) -> Result<(), CoreError> {
        let run_id = lease.run_id;
        let actual = lease.token;
        self.call(move |conn| {
            let tx = immediate(conn)?;
            let run = run_id.to_string();
            let (state, _) = read_job(&tx, &run)?
                .ok_or_else(|| CoreError::NotFound(format!("job for run {run_id}")))?;
            let JobState::Leased {
                token, next_due_ms, ..
            } = state
            else {
                return Err(CoreError::Conflict {
                    expected: 0,
                    actual,
                });
            };
            if token != actual {
                return Err(CoreError::Conflict {
                    expected: token,
                    actual,
                });
            }
            write_job(&tx, &run, &f(next_due_ms), 0)?;
            tx.commit().map_err(db)
        })
        .await
    }
}

#[async_trait]
impl JobQueue for SqliteStore {
    async fn enqueue(&self, run_id: RunId, due_ms: i64) -> Result<(), CoreError> {
        let run = run_id.to_string();
        self.call(move |conn| {
            let tx = immediate(conn)?;
            let current = read_job(&tx, &run)?;
            let attempts = current.as_ref().map_or(0, |(_, a)| *a);
            let state = enqueued(current.map(|(s, _)| s), due_ms);
            write_job(&tx, &run, &state, attempts)?;
            tx.commit().map_err(db)
        })
        .await
    }

    async fn claim_due(&self, limit: usize, lease_ms: i64) -> Result<Vec<Lease>, CoreError> {
        assert!(lease_ms > 0, "a lease lasts some time");
        let clock = self.inner.clock.clone();
        self.call(move |conn| {
            let tx = immediate(conn)?;
            let now = clock.now_ms();
            let until_ms = now.checked_add(lease_ms).ok_or_else(|| {
                CoreError::Store(format!("lease of {lease_ms} ms overflows the clock"))
            })?;
            assert!(until_ms > now, "a lease ends after it starts");
            let runs = due_runs(&tx, now, limit)?;
            let mut leases = Vec::with_capacity(runs.len());
            for run in &runs {
                leases.push(claim_one(&tx, run, now, until_ms)?);
            }
            assert!(leases.len() <= limit);
            assert_eq!(leases.len(), runs.len());
            tx.commit().map_err(db)?;
            Ok(leases)
        })
        .await
    }

    async fn complete(&self, lease: &Lease) -> Result<(), CoreError> {
        self.end_lease(lease, |next| match next {
            Some(due_ms) => JobState::Pending { due_ms },
            None => JobState::Done,
        })
        .await
    }

    async fn fail(&self, lease: &Lease, error: &str) -> Result<(), CoreError> {
        let error = error.to_string();
        self.end_lease(lease, move |_| JobState::Failed { error })
            .await
    }

    async fn reschedule(&self, lease: &Lease, due_ms: i64) -> Result<(), CoreError> {
        self.end_lease(lease, move |next| JobState::Pending {
            due_ms: earlier(next, due_ms),
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_row_shape_maps_to_its_job_state() {
        let states = [
            JobState::Pending { due_ms: 5 },
            JobState::Leased {
                token: 3,
                until_ms: 9,
                next_due_ms: None,
            },
            JobState::Leased {
                token: 4,
                until_ms: 9,
                next_due_ms: Some(7),
            },
            JobState::Done,
            JobState::Failed {
                error: "boom".into(),
            },
        ];
        for state in states {
            assert_eq!(to_state(to_row(&state).unwrap()).unwrap(), state);
        }
        let bad = Row {
            state: "leased".into(),
            claim_at_ms: Some(1),
            token: None,
            next_due_ms: None,
            error: None,
        };
        assert!(
            matches!(to_state(bad), Err(CoreError::Store(m)) if m.starts_with("corrupt store"))
        );
        let unknown = Row {
            state: "lost".into(),
            claim_at_ms: None,
            token: None,
            next_due_ms: None,
            error: None,
        };
        assert!(to_state(unknown).is_err());
        assert!(parse_run("not a uuid").is_err());
        let run = RunId::new();
        assert_eq!(parse_run(&run.to_string()).unwrap(), run);
    }
}

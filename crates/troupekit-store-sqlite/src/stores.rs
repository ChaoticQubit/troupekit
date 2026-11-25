use crate::{db, SqliteStore};
use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde_json::Value;
use troupekit_core::error::CoreError;
use troupekit_core::event::Usage;
use troupekit_core::ids::RunId;
use troupekit_core::message::Message;
use troupekit_core::store::{BlobStore, Checkpoint, CheckpointStore, MemoryLog, UsageLedger};

/// Starts the write transaction every check-then-write uses. `IMMEDIATE`
/// takes the write lock at once, so two handles never both pass a check.
/// Input: the connection. Output: the transaction guard.
pub(crate) fn immediate(conn: &mut Connection) -> Result<Transaction<'_>, CoreError> {
    conn.transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(db)
}

/// Converts a caller's `u64` to a SQLite integer.
/// Input: the value and its name for the message.
/// Output: the `i64`, or `CoreError::Store` when it does not fit.
pub(crate) fn to_sql_int(name: &str, value: u64) -> Result<i64, CoreError> {
    i64::try_from(value).map_err(|_| CoreError::Store(format!("{name} {value} is above i64::MAX")))
}

/// Converts a stored SQLite integer back to `u64`.
/// Input: the value and its name. Output: the `u64`, or
/// `CoreError::Store("corrupt store: ...")` for a negative value.
pub(crate) fn from_sql_int(name: &str, value: i64) -> Result<u64, CoreError> {
    u64::try_from(value).map_err(|_| corrupt(format!("negative {name} {value}")))
}

/// Builds the error for a value read from the file that cannot be right.
/// Input: what was wrong. Output: `CoreError::Store("corrupt store: ...")`.
pub(crate) fn corrupt(what: String) -> CoreError {
    CoreError::Store(format!("corrupt store: {what}"))
}

const STATE_DEPTH_MAX: usize = 127;

/// Tells whether `value` nests arrays and objects at most `levels_left`
/// deep. Recursion stops one level past the limit, so it is bounded.
/// Input: the value and the levels it may still use. Output: true when
/// the value fits, which is what `decode_state` can read back.
fn fits_depth(value: &Value, levels_left: usize) -> bool {
    assert!(
        levels_left <= STATE_DEPTH_MAX,
        "depth budget above the limit"
    );
    match value {
        Value::Array(items) => {
            levels_left > 0 && items.iter().all(|v| fits_depth(v, levels_left - 1))
        }
        Value::Object(map) => {
            levels_left > 0 && map.values().all(|v| fits_depth(v, levels_left - 1))
        }
        _ => true,
    }
}

fn decode_state(bytes: &[u8]) -> Result<Value, CoreError> {
    serde_json::from_slice(bytes).map_err(|e| corrupt(format!("checkpoint state is not JSON: {e}")))
}

fn max_step(tx: &Transaction<'_>, run: &str) -> Result<Option<u64>, CoreError> {
    let max: Option<i64> = tx
        .query_row(
            "SELECT MAX(step) FROM checkpoint WHERE run_id = ?1",
            [run],
            |row| row.get(0),
        )
        .map_err(db)?;
    max.map(|m| from_sql_int("step", m)).transpose()
}

fn row_checkpoint(run_id: RunId, step: i64, bytes: &[u8]) -> Result<Checkpoint, CoreError> {
    Ok(Checkpoint {
        run_id,
        step: from_sql_int("step", step)?,
        state: decode_state(bytes)?,
    })
}

#[async_trait]
impl CheckpointStore for SqliteStore {
    async fn save(&self, cp: Checkpoint) -> Result<(), CoreError> {
        let step = to_sql_int("step", cp.step)?;
        if !fits_depth(&cp.state, STATE_DEPTH_MAX) {
            return Err(CoreError::Store(format!(
                "checkpoint state nests deeper than {STATE_DEPTH_MAX} levels"
            )));
        }
        let bytes = serde_json::to_vec(&cp.state)
            .map_err(|e| CoreError::Store(format!("cannot encode checkpoint state: {e}")))?;
        let max = self.limits().state_bytes_max;
        if bytes.len() as u64 > max {
            return Err(CoreError::Store(format!(
                "checkpoint state is {} bytes, above max_state_bytes {max}",
                bytes.len()
            )));
        }
        let run = cp.run_id.to_string();
        let actual = cp.step;
        self.call(move |conn| {
            let tx = immediate(conn)?;
            let expected = match max_step(&tx, &run)? {
                None => 0,
                Some(m) => m
                    .checked_add(1)
                    .ok_or_else(|| corrupt("step at u64::MAX".into()))?,
            };
            if actual != expected {
                return Err(CoreError::Conflict { expected, actual });
            }
            let inserted = tx
                .execute(
                    "INSERT INTO checkpoint (run_id, step, state_json) VALUES (?1, ?2, ?3)",
                    params![run, step, bytes],
                )
                .map_err(db)?;
            assert_eq!(inserted, 1);
            assert_eq!(max_step(&tx, &run)?, Some(actual));
            tx.commit().map_err(db)
        })
        .await
    }

    async fn latest(&self, run_id: RunId) -> Result<Option<Checkpoint>, CoreError> {
        let run = run_id.to_string();
        self.call(move |conn| {
            let row: Option<(i64, Vec<u8>)> = conn
                .query_row(
                    "SELECT step, state_json FROM checkpoint WHERE run_id = ?1 \
                     ORDER BY step DESC LIMIT 1",
                    [&run],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()
                .map_err(db)?;
            row.map(|(step, bytes)| row_checkpoint(run_id, step, &bytes))
                .transpose()
        })
        .await
    }

    async fn history(&self, run_id: RunId) -> Result<Vec<Checkpoint>, CoreError> {
        let run = run_id.to_string();
        let max = self.limits().history_steps_max;
        self.call(move |conn| {
            let tx = conn.transaction().map_err(db)?;
            let count: i64 = tx
                .query_row(
                    "SELECT COUNT(*) FROM checkpoint WHERE run_id = ?1",
                    [&run],
                    |r| r.get(0),
                )
                .map_err(db)?;
            let count = from_sql_int("count", count)?;
            if count > max {
                return Err(CoreError::Store(format!(
                    "run has {count} kept steps, above max_history_steps {max}"
                )));
            }
            let mut stmt = tx
                .prepare_cached(
                    "SELECT step, state_json FROM checkpoint WHERE run_id = ?1 ORDER BY step",
                )
                .map_err(db)?;
            let rows = stmt
                .query_map([&run], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
                })
                .map_err(db)?;
            let mut out = Vec::with_capacity(count as usize);
            for row in rows {
                let (step, bytes) = row.map_err(db)?;
                out.push(row_checkpoint(run_id, step, &bytes)?);
            }
            assert_eq!(out.len() as u64, count);
            assert!(out.windows(2).all(|w| w[0].step < w[1].step));
            Ok(out)
        })
        .await
    }

    async fn prune(&self, run_id: RunId, keep_last: u64) -> Result<u64, CoreError> {
        if keep_last == 0 {
            return Err(CoreError::Store(
                "prune keep_last must be at least 1".into(),
            ));
        }
        let run = run_id.to_string();
        self.call(move |conn| {
            let tx = immediate(conn)?;
            let Some(max) = max_step(&tx, &run)? else {
                return Ok(0);
            };
            let cutoff = (max + 1).saturating_sub(keep_last);
            assert!(cutoff <= max, "prune never reaches the latest step");
            let deleted = tx
                .execute(
                    "DELETE FROM checkpoint WHERE run_id = ?1 AND step < ?2",
                    params![run, to_sql_int("step", cutoff)?],
                )
                .map_err(db)?;
            assert_eq!(max_step(&tx, &run)?, Some(max));
            tx.commit().map_err(db)?;
            Ok(deleted as u64)
        })
        .await
    }
}

fn log_len(tx: &Transaction<'_>, log: &str) -> Result<u64, CoreError> {
    let count: i64 = tx
        .query_row(
            "SELECT COUNT(*) FROM memory_entry WHERE log = ?1",
            [log],
            |row| row.get(0),
        )
        .map_err(db)?;
    from_sql_int("log length", count)
}

#[async_trait]
impl MemoryLog for SqliteStore {
    async fn append(
        &self,
        log: &str,
        expected_len: u64,
        entries: Vec<Message>,
    ) -> Result<u64, CoreError> {
        let mut encoded = Vec::with_capacity(entries.len());
        for message in &entries {
            encoded.push(
                serde_json::to_vec(message)
                    .map_err(|e| CoreError::Store(format!("cannot encode message: {e}")))?,
            );
        }
        let log = log.to_string();
        self.call(move |conn| {
            let tx = immediate(conn)?;
            let actual = log_len(&tx, &log)?;
            if actual != expected_len {
                return Err(CoreError::Conflict {
                    expected: expected_len,
                    actual,
                });
            }
            let added = encoded.len() as u64;
            {
                let mut insert = tx
                    .prepare_cached(
                        "INSERT INTO memory_entry (log, idx, message_json) VALUES (?1, ?2, ?3)",
                    )
                    .map_err(db)?;
                for (offset, bytes) in encoded.iter().enumerate() {
                    let idx = to_sql_int("log index", actual + offset as u64)?;
                    let inserted = insert.execute(params![log, idx, bytes]).map_err(db)?;
                    assert_eq!(inserted, 1);
                }
            }
            let len = log_len(&tx, &log)?;
            assert_eq!(len, actual + added);
            tx.commit().map_err(db)?;
            Ok(len)
        })
        .await
    }

    async fn read(&self, log: &str) -> Result<Vec<Message>, CoreError> {
        let log = log.to_string();
        self.call(move |conn| {
            let mut stmt = conn
                .prepare_cached(
                    "SELECT idx, message_json FROM memory_entry WHERE log = ?1 ORDER BY idx",
                )
                .map_err(db)?;
            let rows = stmt
                .query_map([&log], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
                })
                .map_err(db)?;
            let mut out = Vec::new();
            for row in rows {
                let (idx, bytes) = row.map_err(db)?;
                if from_sql_int("log index", idx)? != out.len() as u64 {
                    return Err(corrupt(format!("log has a gap at index {}", out.len())));
                }
                let message = serde_json::from_slice(&bytes)
                    .map_err(|e| corrupt(format!("message is not valid: {e}")))?;
                out.push(message);
            }
            Ok(out)
        })
        .await
    }
}

#[async_trait]
impl BlobStore for SqliteStore {
    async fn put(&self, path: &str, bytes: Vec<u8>) -> Result<(), CoreError> {
        let max = self.limits().blob_bytes_max;
        if bytes.len() as u64 > max {
            return Err(CoreError::Store(format!(
                "blob is {} bytes, above max_blob_bytes {max}",
                bytes.len()
            )));
        }
        let path = path.to_string();
        self.call(move |conn| {
            let written = conn
                .execute(
                    "INSERT INTO blob_entry (path, bytes) VALUES (?1, ?2) \
                     ON CONFLICT (path) DO UPDATE SET bytes = excluded.bytes",
                    params![path, bytes],
                )
                .map_err(db)?;
            assert_eq!(written, 1);
            Ok(())
        })
        .await
    }

    async fn get(&self, path: &str) -> Result<Vec<u8>, CoreError> {
        let path = path.to_string();
        self.call(move |conn| {
            let bytes: Option<Vec<u8>> = conn
                .query_row(
                    "SELECT bytes FROM blob_entry WHERE path = ?1",
                    [&path],
                    |row| row.get(0),
                )
                .optional()
                .map_err(db)?;
            bytes.ok_or(CoreError::NotFound(path))
        })
        .await
    }
}

fn read_usage(conn: &Connection, key: &str) -> Result<Usage, CoreError> {
    let row: Option<(i64, i64, i64)> = conn
        .query_row(
            "SELECT input_tokens, output_tokens, cost_usd_micros FROM usage_total \
             WHERE ledger_key = ?1",
            [key],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()
        .map_err(db)?;
    let Some((input, output, cost)) = row else {
        return Ok(Usage::default());
    };
    Ok(Usage {
        input_tokens: from_sql_int("input_tokens", input)?,
        output_tokens: from_sql_int("output_tokens", output)?,
        cost_usd_micros: from_sql_int("cost_usd_micros", cost)?,
    })
}

fn add_counter(name: &str, total: u64, more: u64) -> Result<i64, CoreError> {
    let sum = total
        .checked_add(more)
        .ok_or_else(|| CoreError::Store(format!("{name} total would overflow")))?;
    to_sql_int(name, sum)
}

#[async_trait]
impl UsageLedger for SqliteStore {
    async fn period_total(&self, key: &str) -> Result<u64, CoreError> {
        let key = key.to_string();
        self.call(move |conn| Ok(read_usage(conn, &key)?.cost_usd_micros))
            .await
    }

    async fn record(&self, key: &str, usage: Usage) -> Result<(), CoreError> {
        let key = key.to_string();
        self.call(move |conn| {
            let tx = immediate(conn)?;
            let total = read_usage(&tx, &key)?;
            let input = add_counter("input_tokens", total.input_tokens, usage.input_tokens)?;
            let output = add_counter("output_tokens", total.output_tokens, usage.output_tokens)?;
            let cost = add_counter(
                "cost_usd_micros",
                total.cost_usd_micros,
                usage.cost_usd_micros,
            )?;
            let written = tx
                .execute(
                    "INSERT INTO usage_total \
                     (ledger_key, input_tokens, output_tokens, cost_usd_micros) \
                     VALUES (?1, ?2, ?3, ?4) ON CONFLICT (ledger_key) DO UPDATE SET \
                     input_tokens = excluded.input_tokens, \
                     output_tokens = excluded.output_tokens, \
                     cost_usd_micros = excluded.cost_usd_micros",
                    params![key, input, output, cost],
                )
                .map_err(db)?;
            assert_eq!(written, 1);
            assert_eq!(read_usage(&tx, &key)?, total + usage);
            tx.commit().map_err(db)
        })
        .await
    }
}

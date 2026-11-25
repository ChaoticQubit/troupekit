use crate::{db, SqliteStore};
use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde_json::Value;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::store::{Checkpoint, CheckpointStore};

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

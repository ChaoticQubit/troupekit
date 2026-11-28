use crate::{db, SqliteStore};
use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde_json::Value;
use troupekit_core::error::CoreError;
use troupekit_core::event::Usage;
use troupekit_core::ids::RunId;
use troupekit_core::message::Message;
use troupekit_core::store::{
    BlobStore, BranchWrite, Checkpoint, CheckpointStore, ForkLink, MemoryLog, UsageLedger,
};

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

/// Refuses JSON the read path could not decode because it nests too
/// deep. Input: the value and what it is, for the message. Output: `Ok`
/// when it fits, `CoreError::Store` naming the limit otherwise.
fn check_depth(value: &Value, what: &str) -> Result<(), CoreError> {
    if fits_depth(value, STATE_DEPTH_MAX) {
        return Ok(());
    }
    Err(CoreError::Store(format!(
        "{what} nests deeper than {STATE_DEPTH_MAX} levels"
    )))
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

/// Counts the branch writes of `run` below step `below`.
/// Input: the open transaction, the run and the step. Output: the count;
/// `drop_branch_writes` asserts it is zero after its delete.
fn branch_writes_below(tx: &Transaction<'_>, run: &str, below: i64) -> Result<i64, CoreError> {
    tx.query_row(
        "SELECT COUNT(*) FROM branch_write WHERE run_id = ?1 AND step < ?2",
        params![run, below],
        |row| row.get(0),
    )
    .map_err(db)
}

/// Parses a run id stored as text by this store.
/// Input: the text and the column it came from, for the message.
/// Output: the id, or `CoreError::Store("corrupt store: ...")`.
fn parse_run_id(text: String, column: &str) -> Result<RunId, CoreError> {
    serde_json::from_value(Value::String(text))
        .map_err(|_| corrupt(format!("{column} is not a run id")))
}

/// Refuses a list longer than `max` rows, read with `LIMIT max + 1`.
/// Input: the rows read, the cap and what they are. Output: `Ok` when
/// the list fits, `CoreError::Store` naming `max_history_steps` otherwise.
fn check_rows<T>(rows: &[T], max: u64, what: &str) -> Result<(), CoreError> {
    if rows.len() as u64 > max {
        return Err(CoreError::Store(format!(
            "{what} has more than max_history_steps {max} rows"
        )));
    }
    Ok(())
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
        check_depth(&cp.state, "checkpoint state")?;
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
            let cutoff = to_sql_int("step", cutoff)?;
            let deleted = tx
                .execute(
                    "DELETE FROM checkpoint WHERE run_id = ?1 AND step < ?2",
                    params![run, cutoff],
                )
                .map_err(db)?;
            assert_eq!(max_step(&tx, &run)?, Some(max));
            tx.commit().map_err(db)?;
            Ok(deleted as u64)
        })
        .await
    }

    async fn save_branch(&self, run_id: RunId, step: u64, write: BranchWrite) -> Result<bool, CoreError> {
        let step_sql = to_sql_int("step", step)?;
        check_depth(write.write(), "branch write")?;
        let bytes = serde_json::to_vec(write.write())
            .map_err(|e| CoreError::Store(format!("cannot encode branch write: {e}")))?;
        let max = self.limits().branch_write_bytes();
        let size = bytes.len() as u64 + write.node().len() as u64;
        if size > max {
            return Err(CoreError::Store(format!(
                "branch write is {size} bytes, above max_branch_write_bytes {max}"
            )));
        }
        let run = run_id.to_string();
        let branch = i64::from(write.branch());
        let node = write.node().to_string();
        self.call(move |conn| {
            let tx = immediate(conn)?;
            if let Some(latest) = max_step(&tx, &run)? {
                if step <= latest {
                    let expected = latest
                        .checked_add(1)
                        .ok_or_else(|| corrupt("step at u64::MAX".into()))?;
                    return Err(CoreError::Conflict { expected, actual: step });
                }
            }
            let kept: Option<(String, Vec<u8>)> = tx
                .query_row(
                    "SELECT node, write_json FROM branch_write \
                     WHERE run_id = ?1 AND step = ?2 AND branch = ?3",
                    params![run, step_sql, branch],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()
                .map_err(db)?;
            match kept {
                Some((kept_node, kept_bytes)) if kept_node == node && kept_bytes == bytes => Ok(true),
                Some(_) => Err(CoreError::Conflict { expected: step, actual: step }),
                None => {
                    let inserted = tx
                        .execute(
                            "INSERT INTO branch_write (run_id, step, branch, node, write_json) \
                             VALUES (?1, ?2, ?3, ?4, ?5)",
                            params![run, step_sql, branch, node, bytes],
                        )
                        .map_err(db)?;
                    assert_eq!(inserted, 1);
                    tx.commit().map_err(db)?;
                    Ok(true)
                }
            }
        })
        .await
    }

    async fn branch_writes(&self, run_id: RunId, step: u64) -> Result<Vec<BranchWrite>, CoreError> {
        let step = to_sql_int("step", step)?;
        let run = run_id.to_string();
        let max = self.limits().history_steps_max;
        self.call(move |conn| {
            let mut stmt = conn
                .prepare_cached(
                    "SELECT branch, node, write_json FROM branch_write \
                     WHERE run_id = ?1 AND step = ?2 ORDER BY branch LIMIT ?3",
                )
                .map_err(db)?;
            let limit = to_sql_int("limit", max.saturating_add(1))?;
            let rows = stmt
                .query_map(params![run, step, limit], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, Vec<u8>>(2)?))
                })
                .map_err(db)?;
            let mut out = Vec::new();
            for row in rows {
                let (branch, node, bytes) = row.map_err(db)?;
                let branch = u32::try_from(branch)
                    .map_err(|_| corrupt(format!("branch index {branch}")))?;
                let write: Value = serde_json::from_slice(&bytes)
                    .map_err(|e| corrupt(format!("branch write is not JSON: {e}")))?;
                out.push(BranchWrite::new(branch, node, write));
                check_rows(&out, max, "step")?;
            }
            assert!(out.windows(2).all(|w| w[0].branch() < w[1].branch()));
            Ok(out)
        })
        .await
    }

    async fn drop_branch_writes(&self, run_id: RunId, below: u64) -> Result<u64, CoreError> {
        let below = to_sql_int("step", below)?;
        let run = run_id.to_string();
        self.call(move |conn| {
            let tx = immediate(conn)?;
            let deleted = tx
                .execute(
                    "DELETE FROM branch_write WHERE run_id = ?1 AND step < ?2",
                    params![run, below],
                )
                .map_err(db)?;
            assert_eq!(branch_writes_below(&tx, &run, below)?, 0);
            tx.commit().map_err(db)?;
            Ok(deleted as u64)
        })
        .await
    }

    async fn link_fork(&self, link: ForkLink) -> Result<bool, CoreError> {
        if link.child() == link.parent() {
            return Err(CoreError::Store("a run cannot be forked from itself".into()));
        }
        let parent_step = to_sql_int("parent_step", link.parent_step())?;
        let child = link.child().to_string();
        let parent = link.parent().to_string();
        self.call(move |conn| {
            let tx = immediate(conn)?;
            let kept: Option<(String, i64)> = tx
                .query_row(
                    "SELECT parent, parent_step FROM run_fork WHERE child = ?1",
                    [&child],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()
                .map_err(db)?;
            match kept {
                Some((kept_parent, kept_step)) if kept_parent == parent && kept_step == parent_step => Ok(true),
                Some((_, kept_step)) => Err(CoreError::Conflict {
                    expected: from_sql_int("parent_step", kept_step)?,
                    actual: link.parent_step(),
                }),
                None => {
                    let inserted = tx
                        .execute(
                            "INSERT INTO run_fork (child, parent, parent_step) VALUES (?1, ?2, ?3)",
                            params![child, parent, parent_step],
                        )
                        .map_err(db)?;
                    assert_eq!(inserted, 1);
                    tx.commit().map_err(db)?;
                    Ok(true)
                }
            }
        })
        .await
    }

    async fn forks(&self, parent: RunId) -> Result<Vec<ForkLink>, CoreError> {
        let parent_text = parent.to_string();
        let max = self.limits().history_steps_max;
        self.call(move |conn| {
            let mut stmt = conn
                .prepare_cached(
                    "SELECT child, parent_step FROM run_fork WHERE parent = ?1 \
                     ORDER BY child LIMIT ?2",
                )
                .map_err(db)?;
            let limit = to_sql_int("limit", max.saturating_add(1))?;
            let rows = stmt
                .query_map(params![parent_text, limit], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })
                .map_err(db)?;
            let mut out = Vec::new();
            for row in rows {
                let (child, step) = row.map_err(db)?;
                let child = parse_run_id(child, "run_fork child")?;
                out.push(ForkLink::new(child, parent, from_sql_int("parent_step", step)?));
                check_rows(&out, max, "fork list")?;
            }
            assert!(out.windows(2).all(|w| w[0].child() < w[1].child()));
            Ok(out)
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
            let value = serde_json::to_value(message)
                .map_err(|e| CoreError::Store(format!("cannot encode message: {e}")))?;
            check_depth(&value, "message")?;
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

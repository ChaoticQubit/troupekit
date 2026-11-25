use crate::db;
use rusqlite::{Connection, TransactionBehavior};
use troupekit_core::error::CoreError;

/// The `application_id` of every troupekit store file.
/// It is `0x54524B54`, the ASCII bytes `TRKT`.
/// A file with another non-zero id is refused.
pub(crate) const APPLICATION_ID: i32 = 1_414_679_380;

const STEP_1: &str = "
CREATE TABLE checkpoint (
  run_id TEXT NOT NULL,
  step INTEGER NOT NULL CHECK (step >= 0),
  state_json BLOB NOT NULL,
  PRIMARY KEY (run_id, step)
) STRICT;

CREATE TABLE memory_entry (
  log TEXT NOT NULL,
  idx INTEGER NOT NULL CHECK (idx >= 0),
  message_json BLOB NOT NULL,
  PRIMARY KEY (log, idx)
) STRICT;

CREATE TABLE blob_entry (
  path TEXT NOT NULL PRIMARY KEY,
  bytes BLOB NOT NULL
) STRICT;

CREATE TABLE job (
  run_id TEXT NOT NULL PRIMARY KEY,
  state TEXT NOT NULL CHECK (state IN ('pending', 'leased', 'done', 'failed')),
  claim_at_ms INTEGER,
  token INTEGER CHECK (token IS NULL OR token > 0),
  next_due_ms INTEGER,
  error TEXT,
  attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
  CHECK ((state IN ('pending', 'leased')) = (claim_at_ms IS NOT NULL)),
  CHECK ((state = 'leased') = (token IS NOT NULL)),
  CHECK (next_due_ms IS NULL OR state = 'leased'),
  CHECK ((state = 'failed') = (error IS NOT NULL))
) STRICT;

CREATE INDEX job_claim ON job (claim_at_ms, run_id) WHERE claim_at_ms IS NOT NULL;

CREATE TABLE lease_token (
  id INTEGER NOT NULL PRIMARY KEY CHECK (id = 1),
  last INTEGER NOT NULL CHECK (last >= 0)
) STRICT;

INSERT INTO lease_token (id, last) VALUES (1, 0);

CREATE TABLE usage_total (
  ledger_key TEXT NOT NULL PRIMARY KEY,
  input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
  output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
  cost_usd_micros INTEGER NOT NULL CHECK (cost_usd_micros >= 0)
) STRICT;

PRAGMA application_id = 1414679380;
";

/// Ordered migration scripts. Step `n` takes a file from version `n - 1`
/// to `n`. A released step is never edited; a schema change is a new step
/// appended at the end.
pub(crate) const STEPS: &[&str] = &[STEP_1];

/// What the header of a store file says before any migration.
/// Read by `read_header` on any connection, keyed or not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Header {
    pub(crate) application_id: i32,
    pub(crate) user_version: u32,
    pub(crate) has_tables: bool,
}

/// Reads `application_id`, `user_version` and whether the file has tables.
/// Input: an open connection. Output: the header, or `CoreError::Store`
/// when the file cannot be read, for example a wrong key.
pub(crate) fn read_header(conn: &Connection) -> Result<Header, CoreError> {
    let application_id: i32 = conn
        .pragma_query_value(None, "application_id", |row| row.get(0))
        .map_err(db)?;
    let user_version: i64 = conn
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(db)?;
    let has_tables: bool = conn
        .query_row(
            "SELECT EXISTS (SELECT 1 FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%')",
            [],
            |row| row.get(0),
        )
        .map_err(db)?;
    let user_version = u32::try_from(user_version)
        .map_err(|_| CoreError::Store(format!("corrupt store: bad version {user_version}")))?;
    Ok(Header {
        application_id,
        user_version,
        has_tables,
    })
}

/// Refuses a header this troupekit must not touch: a foreign
/// `application_id`, a versioned file without the troupekit id, an
/// unversioned file that already has tables, or a version above `latest`.
/// Output: `Ok` when migrating is safe.
pub(crate) fn check_header(header: Header, latest: u32) -> Result<(), CoreError> {
    if header.application_id != 0 && header.application_id != APPLICATION_ID {
        return Err(CoreError::Store("not a troupekit store".into()));
    }
    if header.user_version > 0 && header.application_id != APPLICATION_ID {
        return Err(CoreError::Store("not a troupekit store".into()));
    }
    if header.user_version == 0 && header.has_tables {
        return Err(CoreError::Store("not a troupekit store".into()));
    }
    if header.user_version > latest {
        return Err(CoreError::Store(format!(
            "store is version {}, this troupekit reads up to {latest}",
            header.user_version
        )));
    }
    Ok(())
}

/// Applies the missing `steps` in one `BEGIN IMMEDIATE` transaction and
/// sets `user_version` to `steps.len()`. Any error rolls back every step of
/// this call. Input: the connection and the ordered steps. Output: the
/// version the file is at afterwards.
pub(crate) fn migrate(conn: &mut Connection, steps: &[&str]) -> Result<u32, CoreError> {
    let latest = u32::try_from(steps.len()).expect("step list fits in u32");
    assert!(latest > 0, "migration step list is empty");
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(db)?;
    let header = read_header(&tx)?;
    check_header(header, latest)?;
    let from = header.user_version;
    for (index, step) in steps.iter().enumerate().skip(from as usize) {
        tx.execute_batch(step)
            .map_err(|e| CoreError::Store(format!("migration step {} failed: {e}", index + 1)))?;
    }
    tx.pragma_update(None, "user_version", latest).map_err(db)?;
    let after = read_header(&tx)?;
    assert_eq!(after.user_version, latest, "user_version not written");
    assert!(
        from == latest || after.application_id == APPLICATION_ID,
        "step 1 sets the id"
    );
    tx.commit().map_err(db)?;
    Ok(latest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory() -> Connection {
        Connection::open_in_memory().unwrap()
    }

    fn tables(conn: &Connection) -> Vec<String> {
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name")
            .unwrap();
        let rows = stmt.query_map([], |row| row.get(0)).unwrap();
        rows.map(Result::unwrap).collect()
    }

    #[test]
    fn fresh_database_migrates_to_latest() {
        let mut conn = memory();
        assert_eq!(migrate(&mut conn, STEPS).unwrap(), 1);
        let header = read_header(&conn).unwrap();
        assert_eq!(header.user_version, 1);
        assert_eq!(header.application_id, APPLICATION_ID);
        assert_eq!(
            tables(&conn),
            [
                "blob_entry",
                "checkpoint",
                "job",
                "lease_token",
                "memory_entry",
                "usage_total"
            ]
        );
        let last: i64 = conn
            .query_row("SELECT last FROM lease_token", [], |r| r.get(0))
            .unwrap();
        assert_eq!(last, 0);
    }

    #[test]
    fn second_migration_changes_nothing() {
        let mut conn = memory();
        migrate(&mut conn, STEPS).unwrap();
        let before = tables(&conn);
        assert_eq!(migrate(&mut conn, STEPS).unwrap(), 1);
        assert_eq!(tables(&conn), before);
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM lease_token", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 1);
    }

    #[test]
    fn newer_file_is_refused_with_both_versions() {
        let mut conn = memory();
        migrate(&mut conn, STEPS).unwrap();
        conn.pragma_update(None, "user_version", 5).unwrap();
        let err = migrate(&mut conn, STEPS).unwrap_err();
        assert_eq!(
            err,
            CoreError::Store("store is version 5, this troupekit reads up to 1".into())
        );
        assert_eq!(read_header(&conn).unwrap().user_version, 5);
    }

    #[test]
    fn foreign_application_id_is_refused() {
        let mut conn = memory();
        conn.pragma_update(None, "application_id", 42).unwrap();
        let err = migrate(&mut conn, STEPS).unwrap_err();
        assert_eq!(err, CoreError::Store("not a troupekit store".into()));
        assert!(tables(&conn).is_empty());
    }

    #[test]
    fn unversioned_database_with_tables_is_refused() {
        let mut conn = memory();
        conn.execute_batch("CREATE TABLE other (x INTEGER)")
            .unwrap();
        let err = migrate(&mut conn, STEPS).unwrap_err();
        assert_eq!(err, CoreError::Store("not a troupekit store".into()));
        assert_eq!(tables(&conn), ["other"]);
        assert_eq!(read_header(&conn).unwrap().user_version, 0);
    }

    #[test]
    fn versioned_database_without_the_id_is_refused() {
        let mut conn = memory();
        conn.execute_batch("CREATE TABLE other (x INTEGER)")
            .unwrap();
        conn.pragma_update(None, "user_version", 1).unwrap();
        let err = migrate(&mut conn, STEPS).unwrap_err();
        assert_eq!(err, CoreError::Store("not a troupekit store".into()));
        assert_eq!(tables(&conn), ["other"]);
        let header = read_header(&conn).unwrap();
        assert_eq!(header.user_version, 1);
        assert_eq!(header.application_id, 0);
    }

    #[test]
    fn failing_step_rolls_back_every_step() {
        let mut conn = memory();
        let steps = [STEP_1, "CREATE TABLE broken (;"];
        let err = migrate(&mut conn, &steps).unwrap_err();
        assert!(
            matches!(&err, CoreError::Store(m) if m.contains("step 2")),
            "{err:?}"
        );
        let header = read_header(&conn).unwrap();
        assert_eq!(header.user_version, 0);
        assert_eq!(header.application_id, 0);
        assert!(tables(&conn).is_empty());
    }
}

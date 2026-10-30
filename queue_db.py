# queue_db.py
from __future__ import annotations
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional, List, Tuple


DEFAULT_DB_PATH = os.path.expanduser("~/code/SpinesGUI/queue/jobs.sqlite")


@dataclass
class Job:
    id: int
    exp_id: str
    root_folder: str
    mode: str
    force: bool
    status: str
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    log_path: Optional[str]
    error: Optional[str]
    cancel_requested: bool
    post_requested: bool
    post_status: str
    post_started_at: Optional[str]
    post_finished_at: Optional[str]
    post_error: Optional[str]


class QueueDB:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None lets us manage transactions manually
        con = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL;")  # robust for concurrent reads
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _init_db(self) -> None:
        con = self._connect()
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              exp_id TEXT NOT NULL,
              root_folder TEXT NOT NULL,
              mode TEXT NOT NULL,
              force INTEGER NOT NULL DEFAULT 0,

              status TEXT NOT NULL DEFAULT 'queued', -- queued|running|done|failed|canceled
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              started_at TEXT,
              finished_at TEXT,

              log_path TEXT,
              error TEXT,
              cancel_requested INTEGER NOT NULL DEFAULT 0,
              post_requested INTEGER NOT NULL DEFAULT 0,
              post_status TEXT NOT NULL DEFAULT 'not_requested',
              post_started_at TEXT,
              post_finished_at TEXT,
              post_error TEXT
            );
            """
        )
        columns = {row[1] for row in con.execute("PRAGMA table_info(jobs)")}
        migrations = {
            "cancel_requested": "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0",
            "post_requested": "ALTER TABLE jobs ADD COLUMN post_requested INTEGER NOT NULL DEFAULT 0",
            "post_status": "ALTER TABLE jobs ADD COLUMN post_status TEXT NOT NULL DEFAULT 'not_requested'",
            "post_started_at": "ALTER TABLE jobs ADD COLUMN post_started_at TEXT",
            "post_finished_at": "ALTER TABLE jobs ADD COLUMN post_finished_at TEXT",
            "post_error": "ALTER TABLE jobs ADD COLUMN post_error TEXT",
        }
        for name, statement in migrations.items():
            if name not in columns:
                con.execute(statement)
        con.close()

    # ---------- GUI-side ----------
    def enqueue_job(
        self, exp_id: str, root_folder: str, mode: str, force: bool,
        post_requested: bool = False,
    ) -> int:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO jobs (exp_id, root_folder, mode, force, status, post_requested, post_status)
            VALUES (?, ?, ?, ?, 'queued', ?, ?)
            """,
            (
                exp_id, root_folder, mode, 1 if force else 0,
                1 if post_requested else 0,
                "queued" if post_requested else "not_requested",
            ),
        )
        job_id = cur.lastrowid
        con.close()
        return int(job_id)

    def get_running(self) -> Optional[Job]:
        running = self.get_running_jobs(limit=1)
        return running[0] if running else None

    def get_running_jobs(self, limit: int = 200) -> List[Job]:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, exp_id, root_folder, mode, force, status, created_at, started_at, finished_at, log_path, error, cancel_requested, post_requested, post_status, post_started_at, post_finished_at, post_error
            FROM jobs
            WHERE status='running' OR post_status='running'
            ORDER BY started_at DESC
            LIMIT ?
            """, (limit,)
        )
        rows = cur.fetchall()
        con.close()
        return [Job(*self._convert_row(row)) for row in rows]

    def get_queued(self, limit: int = 200) -> List[Job]:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, exp_id, root_folder, mode, force, status, created_at, started_at, finished_at, log_path, error, cancel_requested, post_requested, post_status, post_started_at, post_finished_at, post_error
            FROM jobs
            WHERE status='queued'
            ORDER BY id ASC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        con.close()
        return [Job(*self._convert_row(r)) for r in rows]

    def get_last_finished(self, n: int = 4) -> List[Job]:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, exp_id, root_folder, mode, force, status, created_at, started_at, finished_at, log_path, error, cancel_requested, post_requested, post_status, post_started_at, post_finished_at, post_error
            FROM jobs
            WHERE status IN ('done','failed','canceled')
              AND post_status != 'running'
            ORDER BY finished_at DESC
            LIMIT ?
            """,
            (n,),
        )
        rows = cur.fetchall()
        con.close()
        return [Job(*self._convert_row(r)) for r in rows]

    def cancel_job(self, job_id: int) -> None:
        # only cancel queued jobs
        con = self._connect()
        con.execute(
            "UPDATE jobs SET status='canceled', finished_at=datetime('now') WHERE id=? AND status='queued'",
            (job_id,),
        )
        con.close()

    def request_cancel(self, job_id: int) -> bool:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            "UPDATE jobs SET cancel_requested=1 WHERE id=? AND (status='running' OR post_status='running')",
            (job_id,),
        )
        changed = cur.rowcount > 0
        con.close()
        return changed

    def is_cancel_requested(self, job_id: int) -> bool:
        con = self._connect()
        row = con.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        con.close()
        return bool(row and row[0])

    # ---------- Worker-side ----------
    def claim_next_job(self) -> Optional[Job]:
        """
        Atomically claim the next queued job (FIFO by id).
        Returns the claimed job with status still 'queued' in object (we update separately),
        or None if no queued jobs.
        """
        con = self._connect()
        cur = con.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE;")  # lock for writers
            cur.execute(
                """
                SELECT id, exp_id, root_folder, mode, force, status, created_at, started_at, finished_at, log_path, error, cancel_requested, post_requested, post_status, post_started_at, post_finished_at, post_error
                FROM jobs
                WHERE status='queued'
                ORDER BY id ASC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if not row:
                cur.execute("COMMIT;")
                return None

            job_id = row[0]
            cur.execute(
                "UPDATE jobs SET status='running', started_at=datetime('now') WHERE id=?",
                (job_id,),
            )
            cur.execute("COMMIT;")
            return Job(*self._convert_row(row))
        except Exception:
            cur.execute("ROLLBACK;")
            raise
        finally:
            con.close()

    def set_log_path(self, job_id: int, log_path: str) -> None:
        con = self._connect()
        con.execute("UPDATE jobs SET log_path=? WHERE id=?", (log_path, job_id))
        con.close()

    def mark_done(self, job_id: int) -> bool:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            "UPDATE jobs SET status='done', finished_at=datetime('now'), error=NULL WHERE id=? AND status='running' AND cancel_requested=0",
            (job_id,),
        )
        changed = cur.rowcount > 0
        con.close()
        return changed

    def mark_failed(self, job_id: int, error_text: str) -> None:
        con = self._connect()
        con.execute(
            "UPDATE jobs SET status='failed', finished_at=datetime('now'), error=? WHERE id=? AND status='running'",
            (error_text, job_id),
        )
        con.close()

    def mark_canceled(self, job_id: int) -> None:
        con = self._connect()
        con.execute(
            """UPDATE jobs SET status='canceled',
                      post_status=CASE WHEN post_status IN ('queued', 'running') THEN 'canceled' ELSE post_status END,
                      finished_at=datetime('now'), error=NULL
               WHERE id=? AND (status='running' OR post_status='running')""",
            (job_id,),
        )
        con.close()

    def recover_interrupted_jobs(self, reason: str) -> int:
        """Mark jobs left running by a previous worker process as canceled."""
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            """UPDATE jobs SET status='canceled',
                      post_status=CASE WHEN post_status IN ('queued', 'running') THEN 'canceled' ELSE post_status END,
                      finished_at=datetime('now'), error=?
               WHERE status='running' OR post_status='running'""",
            (reason,),
        )
        changed = cur.rowcount
        con.close()
        return changed

    def mark_post_running(self, job_id: int) -> None:
        con = self._connect()
        con.execute(
            "UPDATE jobs SET post_status='running', post_started_at=datetime('now'), post_error=NULL WHERE id=? AND status='done'",
            (job_id,),
        )
        con.close()

    def mark_post_done(self, job_id: int) -> None:
        con = self._connect()
        con.execute(
            "UPDATE jobs SET post_status='done', post_finished_at=datetime('now'), post_error=NULL WHERE id=?",
            (job_id,),
        )
        con.close()

    def mark_post_failed(self, job_id: int, error_text: str) -> None:
        con = self._connect()
        con.execute(
            "UPDATE jobs SET post_status='failed', post_finished_at=datetime('now'), post_error=? WHERE id=?",
            (error_text, job_id),
        )
        con.close()

    # ---------- utilities ----------
    @staticmethod
    def _convert_row(row: Tuple) -> Tuple:
        # force is int -> bool
        if row is None:
            return row
        row = list(row)
        row[4] = bool(row[4])
        row[11] = bool(row[11])
        row[12] = bool(row[12])
        return tuple(row)

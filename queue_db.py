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
              error TEXT
            );
            """
        )
        con.close()

    # ---------- GUI-side ----------
    def enqueue_job(self, exp_id: str, root_folder: str, mode: str, force: bool) -> int:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO jobs (exp_id, root_folder, mode, force, status)
            VALUES (?, ?, ?, ?, 'queued')
            """,
            (exp_id, root_folder, mode, 1 if force else 0),
        )
        job_id = cur.lastrowid
        con.close()
        return int(job_id)

    def get_running(self) -> Optional[Job]:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, exp_id, root_folder, mode, force, status, created_at, started_at, finished_at, log_path, error
            FROM jobs
            WHERE status='running'
            ORDER BY started_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        con.close()
        return Job(*self._convert_row(row)) if row else None

    def get_queued(self, limit: int = 200) -> List[Job]:
        con = self._connect()
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, exp_id, root_folder, mode, force, status, created_at, started_at, finished_at, log_path, error
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
            SELECT id, exp_id, root_folder, mode, force, status, created_at, started_at, finished_at, log_path, error
            FROM jobs
            WHERE status IN ('done','failed','canceled')
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
                SELECT id, exp_id, root_folder, mode, force, status, created_at, started_at, finished_at, log_path, error
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

    def mark_done(self, job_id: int) -> None:
        con = self._connect()
        con.execute(
            "UPDATE jobs SET status='done', finished_at=datetime('now'), error=NULL WHERE id=?",
            (job_id,),
        )
        con.close()

    def mark_failed(self, job_id: int, error_text: str) -> None:
        con = self._connect()
        con.execute(
            "UPDATE jobs SET status='failed', finished_at=datetime('now'), error=? WHERE id=?",
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
        return tuple(row)

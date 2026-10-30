import unittest

import sqlite3

from queue_db import QueueDB


class QueueDBTests(unittest.TestCase):
    def _running_job(self, q, exp_id="exp-running"):
        job_id = q.enqueue_job(exp_id, "/tmp/root", "normal", False)
        claimed = q.claim_next_job()
        self.assertEqual(claimed.id, job_id)
        return job_id

    def test_running_cancel_request_and_completion_race(self):
        with self.subTest("running cancellation"):
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as d:
                q = QueueDB(str(Path(d) / "jobs.sqlite"))
                job_id = self._running_job(q)
                self.assertTrue(q.request_cancel(job_id))
                self.assertTrue(q.is_cancel_requested(job_id))
                self.assertTrue(q.get_running().cancel_requested)
                q.mark_canceled(job_id)
                self.assertIsNone(q.get_running())
                self.assertEqual(q.get_last_finished(1)[0].status, "canceled")
                self.assertFalse(q.request_cancel(job_id))

    def test_queued_cancel_remains_immediate_and_terminal_jobs_are_noop(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            q = QueueDB(str(Path(d) / "jobs.sqlite"))
            queued_id = q.enqueue_job("exp-queued", "/tmp/root", "normal", False)
            q.cancel_job(queued_id)
            self.assertEqual(q.get_queued(), [])
            self.assertEqual(q.get_last_finished(1)[0].status, "canceled")
            self.assertFalse(q.request_cancel(queued_id))

    def test_existing_database_migrates_cancel_column(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "legacy.sqlite"
            con = sqlite3.connect(db_path)
            con.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, exp_id TEXT NOT NULL, root_folder TEXT NOT NULL, mode TEXT NOT NULL, force INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'queued', created_at TEXT NOT NULL DEFAULT (datetime('now')), started_at TEXT, finished_at TEXT, log_path TEXT, error TEXT, post_requested INTEGER NOT NULL DEFAULT 0, post_status TEXT NOT NULL DEFAULT 'not_requested', post_started_at TEXT, post_finished_at TEXT, post_error TEXT)")
            con.commit()
            con.close()
            q = QueueDB(str(db_path))
            columns = {row[1] for row in sqlite3.connect(db_path).execute("PRAGMA table_info(jobs)")}
            self.assertIn("cancel_requested", columns)
            self.assertEqual(q.enqueue_job("exp", "/tmp/root", "normal", False), 1)


if __name__ == "__main__":
    unittest.main()

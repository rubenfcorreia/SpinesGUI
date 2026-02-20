# worker.py
from __future__ import annotations
import argparse
import os
import sys
import time
import traceback
from datetime import datetime

from queue_db import QueueDB
from spines_extraction import run_extraction


class Tee:
    """Write to multiple streams (stdout + file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def main(db_path: str, poll: int):
    q = QueueDB(db_path=db_path)
    print(f"[worker] started at {datetime.now().isoformat()} db={db_path}")

    while True:
        job = q.claim_next_job()
        if job is None:
            time.sleep(poll)
            continue

        # job returned is the row as selected; status in DB already set to running
        job_id = job.id
        exp_id = job.exp_id
        root = job.root_folder
        mode = job.mode
        force = job.force

        # per-job log file
        queue_dir = os.path.dirname(db_path)
        logs_dir = os.path.join(queue_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        safe_exp = "".join(c if c.isalnum() or c in "-_." else "_" for c in exp_id)
        log_path = os.path.join(logs_dir, f"job_{job_id:06d}_{safe_exp}.log")

        q.set_log_path(job_id, log_path)

        with open(log_path, "a", encoding="utf-8", buffering=1) as lf:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = Tee(old_out, lf)
            sys.stderr = Tee(old_err, lf)
            try:
                print(f"[worker] running job={job_id} exp_id={exp_id}")
                print(f"[worker] root={root} mode={mode} force={force}")
                run_extraction(root_folder=root, mode=mode, force=force, log_path=log_path)
                q.mark_done(job_id)
                print(f"[worker] done job={job_id}")
            except Exception:
                err = traceback.format_exc()
                q.mark_failed(job_id, err)
                print(f"[worker] failed job={job_id}\n{err}")
            finally:
                sys.stdout, sys.stderr = old_out, old_err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to SQLite jobs DB")
    ap.add_argument("--poll", type=int, default=2, help="Poll interval seconds")
    args = ap.parse_args()
    main(db_path=args.db, poll=args.poll)

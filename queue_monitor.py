# queue_monitor.py
from __future__ import annotations
import os
import subprocess

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QTextEdit, QPushButton
)

from queue_db import QueueDB


def tmux_worker_running(session_name: str = "spines_queue") -> bool:
    try:
        r = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return r.returncode == 0
    except Exception:
        return False


class QueueMonitorWindow(QWidget):
    def __init__(self, db_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SpinesGUI Queue")
        self.resize(900, 700)

        self.db = QueueDB(db_path=db_path)

        self.worker_status = QLabel("Worker: (unknown)")
        self.running_label = QLabel("Running: (none)")

        self.queued_table = QTableWidget(0, 4)
        self.queued_table.setHorizontalHeaderLabels(["Job ID", "Exp ID", "Created", "Mode"])
        self.queued_table.setSelectionBehavior(self.queued_table.SelectRows)

        self.cancel_btn = QPushButton("Cancel selected queued job")
        self.cancel_btn.clicked.connect(self.cancel_selected)

        self.done_table = QTableWidget(0, 4)
        self.done_table.setHorizontalHeaderLabels(["Job ID", "Exp ID", "Status", "Finished"])

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        layout = QVBoxLayout()

        top = QHBoxLayout()
        top.addWidget(self.worker_status)
        top.addStretch(1)
        layout.addLayout(top)

        layout.addWidget(self.running_label)
        layout.addWidget(QLabel("Waiting (FIFO)"))
        layout.addWidget(self.queued_table)
        layout.addWidget(self.cancel_btn)

        layout.addWidget(QLabel("Last 4 finished"))
        layout.addWidget(self.done_table)

        layout.addWidget(QLabel("Live log (current running job)"))
        layout.addWidget(self.log_view)

        self.setLayout(layout)

        self._current_log_path = None
        self._log_offset = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(750)

        self.refresh()

    def refresh(self):
        # tmux status
        self.worker_status.setText("Worker: RUNNING" if tmux_worker_running() else "Worker: STOPPED")

        # running
        running = self.db.get_running()
        if running:
            self.running_label.setText(
                f"Running: #{running.id}  {running.exp_id}  ({running.mode})  started {running.started_at}"
            )
            if running.log_path and running.log_path != self._current_log_path:
                self._current_log_path = running.log_path
                self._log_offset = 0
                self.log_view.clear()
        else:
            self.running_label.setText("Running: (none)")
            self._current_log_path = None
            self._log_offset = 0

        # queued
        queued = self.db.get_queued()
        self._fill_table(self.queued_table, [(j.id, j.exp_id, j.created_at, j.mode) for j in queued])

        # last finished 4
        done = self.db.get_last_finished(4)
        self._fill_table(self.done_table, [(j.id, j.exp_id, j.status, j.finished_at) for j in done])

        # live log tail
        self._tail_log()

    def _fill_table(self, table: QTableWidget, rows):
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem("" if val is None else str(val))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()

    def _tail_log(self):
        p = self._current_log_path
        if not p or not os.path.exists(p):
            return
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._log_offset)
                new = f.read()
                self._log_offset = f.tell()
            if new:
                self.log_view.moveCursor(self.log_view.textCursor().End)
                self.log_view.insertPlainText(new)
                self.log_view.moveCursor(self.log_view.textCursor().End)
        except Exception:
            # don't crash monitor if log file is being rotated/locked
            pass

    def cancel_selected(self):
        row = self.queued_table.currentRow()
        if row < 0:
            return
        job_id_item = self.queued_table.item(row, 0)
        if not job_id_item:
            return
        try:
            job_id = int(job_id_item.text())
            self.db.cancel_job(job_id)
        except Exception:
            pass
        self.refresh()

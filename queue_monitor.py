# queue/queue_monitor.py
from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, List

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QMessageBox, QSplitter
)

from queue_db import QueueDB, Job


def _set_item(table: QTableWidget, row: int, col: int, text: str) -> None:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
    table.setItem(row, col, item)


def tail_text(path: str, max_bytes: int = 80_000) -> str:
    """
    Read up to the last max_bytes of a text file.
    Robust + fast for logs.
    """
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_bytes)
            f.seek(start, os.SEEK_SET)
            data = f.read()
        # try utf-8; fall back ignoring errors
        return data.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[queue-monitor] Could not read log: {path}\n{type(e).__name__}: {e}\n"


class QueueMonitorDialog(QDialog):
    """
    Monitor window for the extraction queue:
    - running job
    - queued jobs
    - last finished jobs
    - live log tail
    """

    def __init__(self, parent=None, db_path: Optional[str] = None, refresh_ms: int = 1000):
        super().__init__(parent)
        self.setWindowTitle("SpinesGUI Queue Monitor")
        self.resize(1100, 700)

        self.qdb = QueueDB(db_path) if db_path else QueueDB()

        self._last_log_path: Optional[str] = None
        self._worker_stdout_path = os.path.expanduser("~/code/SpinesGUI/queue/worker_stdout.log")

        # ---- UI ----
        root = QVBoxLayout(self)

        # top row: status + buttons
        top = QHBoxLayout()
        self.lbl_running = QLabel("RUNNING: (none)")
        self.lbl_running.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.lbl_running, stretch=1)

        self.btn_refresh = QPushButton("Refresh now")
        self.btn_refresh.clicked.connect(self.refresh)
        top.addWidget(self.btn_refresh)

        self.btn_cancel = QPushButton("Cancel selected queued")
        self.btn_cancel.clicked.connect(self.cancel_selected)
        top.addWidget(self.btn_cancel)

        root.addLayout(top)

        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter, stretch=1)

        # Middle area: tables
        tables_widget = QDialog(self)
        tables_layout = QHBoxLayout(tables_widget)

        # queued table
        self.tbl_queued = QTableWidget(0, 5)
        self.tbl_queued.setHorizontalHeaderLabels(["Job ID", "Exp ID", "Mode", "Force", "Created"])
        self.tbl_queued.setSelectionBehavior(self.tbl_queued.SelectRows)
        self.tbl_queued.setSelectionMode(self.tbl_queued.ExtendedSelection)
        self.tbl_queued.horizontalHeader().setStretchLastSection(True)
        tables_layout.addWidget(self._with_title("Queued", self.tbl_queued), stretch=2)

        # last finished table
        self.tbl_last = QTableWidget(0, 6)
        self.tbl_last.setHorizontalHeaderLabels(["Job ID", "Exp ID", "Status", "Mode", "Started", "Finished"])
        self.tbl_last.setSelectionBehavior(self.tbl_last.SelectRows)
        self.tbl_last.setSelectionMode(self.tbl_last.SingleSelection)
        self.tbl_last.horizontalHeader().setStretchLastSection(True)
        tables_layout.addWidget(self._with_title("Last finished (4)", self.tbl_last), stretch=2)

        splitter.addWidget(tables_widget)

        # Bottom area: logs
        logs_widget = QDialog(self)
        logs_layout = QVBoxLayout(logs_widget)

        self.lbl_log_source = QLabel("LOG: (none)")
        self.lbl_log_source.setTextInteractionFlags(Qt.TextSelectableByMouse)
        logs_layout.addWidget(self.lbl_log_source)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setLineWrapMode(QTextEdit.NoWrap)
        logs_layout.addWidget(self.txt_log, stretch=1)

        splitter.addWidget(logs_widget)

        splitter.setSizes([350, 350])

        # ---- timer ----
        self.timer = QTimer(self)
        self.timer.setInterval(refresh_ms)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

        self.refresh()

    def _with_title(self, title: str, widget):
        box = QDialog(self)
        lay = QVBoxLayout(box)
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(lbl)
        lay.addWidget(widget)
        return box

    def refresh(self) -> None:
        running = self.qdb.get_running()
        queued = self.qdb.get_queued(limit=500)
        last = self.qdb.get_last_finished(4)

        # ---- running label ----
        if running is None:
            self.lbl_running.setText("RUNNING: (none)")
        else:
            self.lbl_running.setText(
                f"RUNNING: job={running.id} exp_id={running.exp_id} mode={running.mode} "
                f"force={running.force} started_at={running.started_at} log={running.log_path or '(none)'}"
            )

        # ---- queued table ----
        self.tbl_queued.setRowCount(len(queued))
        for r, j in enumerate(queued):
            _set_item(self.tbl_queued, r, 0, str(j.id))
            _set_item(self.tbl_queued, r, 1, j.exp_id)
            _set_item(self.tbl_queued, r, 2, j.mode)
            _set_item(self.tbl_queued, r, 3, "Yes" if j.force else "No")
            _set_item(self.tbl_queued, r, 4, j.created_at or "")

        # ---- last table ----
        self.tbl_last.setRowCount(len(last))
        for r, j in enumerate(last):
            _set_item(self.tbl_last, r, 0, str(j.id))
            _set_item(self.tbl_last, r, 1, j.exp_id)
            _set_item(self.tbl_last, r, 2, j.status)
            _set_item(self.tbl_last, r, 3, j.mode)
            _set_item(self.tbl_last, r, 4, j.started_at or "")
            _set_item(self.tbl_last, r, 5, j.finished_at or "")

        # ---- log tail ----
        # Prefer running job log if present; else show worker stdout
        log_path = None
        if running and running.log_path:
            log_path = running.log_path
        else:
            log_path = self._worker_stdout_path

        if log_path != self._last_log_path:
            self._last_log_path = log_path
            self.lbl_log_source.setText(f"LOG: {log_path}")

        text = tail_text(log_path)
        if text:
            # only update if changed; avoids annoying cursor jumps
            if text != self.txt_log.toPlainText():
                self.txt_log.setPlainText(text)
                self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())
        else:
            self.txt_log.setPlainText("(no log output yet)")

    def cancel_selected(self) -> None:
        rows = set([idx.row() for idx in self.tbl_queued.selectionModel().selectedRows()])
        if not rows:
            QMessageBox.information(self, "Cancel", "Select one or more queued jobs to cancel.")
            return

        job_ids: List[int] = []
        for r in sorted(rows):
            item = self.tbl_queued.item(r, 0)
            if item:
                try:
                    job_ids.append(int(item.text()))
                except ValueError:
                    pass

        if not job_ids:
            QMessageBox.warning(self, "Cancel", "Could not parse selected job IDs.")
            return

        if QMessageBox.question(
            self,
            "Cancel jobs",
            f"Cancel {len(job_ids)} queued job(s)?\n\n{job_ids}",
            QMessageBox.Yes | QMessageBox.No
        ) != QMessageBox.Yes:
            return

        for jid in job_ids:
            self.qdb.cancel_job(jid)

        self.refresh()

if __name__ == "__main__":
    # Allow running directly: python queue/queue_monitor.py
    # Make sure project root is on sys.path so we can import queue_db.py
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="SpinesGUI Queue Monitor")
    parser.add_argument("--db", default=None, help="Path to jobs.sqlite (optional)")
    parser.add_argument("--refresh-ms", type=int, default=1000, help="Refresh interval in ms")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    dlg = QueueMonitorDialog(parent=None, db_path=args.db, refresh_ms=args.refresh_ms)
    dlg.show()
    sys.exit(app.exec_())
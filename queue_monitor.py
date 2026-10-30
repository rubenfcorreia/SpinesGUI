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


def _set_check_item(table: QTableWidget, row: int, col: int, checked: bool = False) -> None:
    item = QTableWidgetItem()
    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
    item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
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
        self.btn_refresh = QPushButton("Refresh now")
        self.btn_refresh.clicked.connect(self.refresh)
        top.addWidget(self.btn_refresh)

        self.btn_cancel = QPushButton("Cancel selected")
        self.btn_cancel.clicked.connect(self.cancel_selected)
        top.addWidget(self.btn_cancel)

        root.addLayout(top)

        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter, stretch=1)

        # Middle area: tables
        tables_widget = QDialog(self)
        tables_layout = QHBoxLayout(tables_widget)
        running_queued_layout = QVBoxLayout()

        # running table
        self.tbl_running = QTableWidget(0, 8)
        self.tbl_running.setHorizontalHeaderLabels(["Select", "Job ID", "Exp ID", "Mode", "Extraction", "Post", "Started", "Cancel"])
        self.tbl_running.setSelectionBehavior(self.tbl_running.SelectRows)
        self.tbl_running.setSelectionMode(self.tbl_running.SingleSelection)
        self.tbl_running.horizontalHeader().setStretchLastSection(True)
        running_queued_layout.addWidget(self._with_title("Running", self.tbl_running), stretch=1)

        # queued table
        self.tbl_queued = QTableWidget(0, 7)
        self.tbl_queued.setHorizontalHeaderLabels(["Select", "Job ID", "Exp ID", "Mode", "Force", "Post", "Created"])
        self.tbl_queued.setSelectionBehavior(self.tbl_queued.SelectRows)
        self.tbl_queued.setSelectionMode(self.tbl_queued.ExtendedSelection)
        self.tbl_queued.horizontalHeader().setStretchLastSection(True)
        running_queued_layout.addWidget(self._with_title("Queued", self.tbl_queued), stretch=3)
        tables_layout.addLayout(running_queued_layout, stretch=2)

        # last finished table
        self.tbl_last = QTableWidget(0, 7)
        self.tbl_last.setHorizontalHeaderLabels(["Job ID", "Exp ID", "Extraction", "Post", "Mode", "Started", "Finished"])
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
        running = self.qdb.get_running_jobs(limit=500)
        queued = self.qdb.get_queued(limit=500)
        last = self.qdb.get_last_finished(4)

        # Preserve checkbox selections while the auto-refresh rebuilds the tables.
        checked_running_ids = {
            int(self.tbl_running.item(row, 1).text())
            for row in range(self.tbl_running.rowCount())
            if self.tbl_running.item(row, 0)
            and self.tbl_running.item(row, 0).checkState() == Qt.Checked
            and self.tbl_running.item(row, 1)
        }
        checked_queued_ids = {
            int(self.tbl_queued.item(row, 1).text())
            for row in range(self.tbl_queued.rowCount())
            if self.tbl_queued.item(row, 0)
            and self.tbl_queued.item(row, 0).checkState() == Qt.Checked
            and self.tbl_queued.item(row, 1)
        }

        # ---- running table ----
        self.tbl_running.setRowCount(len(running))
        for r, j in enumerate(running):
            _set_check_item(self.tbl_running, r, 0, j.id in checked_running_ids)
            _set_item(self.tbl_running, r, 1, str(j.id))
            _set_item(self.tbl_running, r, 2, j.exp_id)
            _set_item(self.tbl_running, r, 3, j.mode)
            _set_item(self.tbl_running, r, 4, j.status)
            _set_item(self.tbl_running, r, 5, j.post_status)
            _set_item(self.tbl_running, r, 6, j.started_at or "")
            _set_item(self.tbl_running, r, 7, "Requested" if j.cancel_requested else "")

        # ---- queued table ----
        self.tbl_queued.setRowCount(len(queued))
        for r, j in enumerate(queued):
            _set_check_item(self.tbl_queued, r, 0, j.id in checked_queued_ids)
            _set_item(self.tbl_queued, r, 1, str(j.id))
            _set_item(self.tbl_queued, r, 2, j.exp_id)
            _set_item(self.tbl_queued, r, 3, j.mode)
            _set_item(self.tbl_queued, r, 4, "Yes" if j.force else "No")
            _set_item(self.tbl_queued, r, 5, j.post_status)
            _set_item(self.tbl_queued, r, 6, j.created_at or "")

        # ---- last table ----
        self.tbl_last.setRowCount(len(last))
        for r, j in enumerate(last):
            _set_item(self.tbl_last, r, 0, str(j.id))
            _set_item(self.tbl_last, r, 1, j.exp_id)
            _set_item(self.tbl_last, r, 2, j.status)
            _set_item(self.tbl_last, r, 3, j.post_status)
            _set_item(self.tbl_last, r, 4, j.mode)
            _set_item(self.tbl_last, r, 5, j.started_at or "")
            _set_item(self.tbl_last, r, 6, j.finished_at or "")

        # ---- log tail ----
        # Prefer running job log if present; else show worker stdout
        log_path = None
        current_running = running[0] if running else None
        if current_running and current_running.log_path:
            log_path = current_running.log_path
        else:
            log_path = self._worker_stdout_path

        if log_path != self._last_log_path:
            self._last_log_path = log_path
            self.lbl_log_source.setText(f"LOG: {log_path}")

        text = tail_text(log_path)
        if text:
            # only update if changed; avoids annoying cursor jumps
            log_changed = text != self.txt_log.toPlainText()
            if log_changed:
                self.txt_log.setPlainText(text)
            if current_running is None or log_changed:
                self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())
        else:
            self.txt_log.setPlainText("(no log output yet)")
            if current_running is None:
                self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def cancel_selected(self) -> None:
        queued_rows = [
            row for row in range(self.tbl_queued.rowCount())
            if self.tbl_queued.item(row, 0)
            and self.tbl_queued.item(row, 0).checkState() == Qt.Checked
        ]
        running_rows = [
            row for row in range(self.tbl_running.rowCount())
            if self.tbl_running.item(row, 0)
            and self.tbl_running.item(row, 0).checkState() == Qt.Checked
        ]

        if queued_rows and running_rows:
            QMessageBox.information(
                self, "Cancel", "Check either a running job or queued jobs, not both."
            )
            return
        if not queued_rows and not running_rows:
            QMessageBox.information(self, "Cancel", "Check a running or queued job.")
            return

        if running_rows:
            item = self.tbl_running.item(running_rows[0], 1)
            try:
                job_id = int(item.text())
            except (AttributeError, ValueError):
                QMessageBox.warning(self, "Cancel", "Could not parse the checked job ID.")
                return
            if QMessageBox.question(
                self,
                "Cancel running job",
                f"Request a safe cancellation for running job #{job_id}?\n\n"
                "The current Suite2p stage will finish before the worker stops.",
                QMessageBox.Yes | QMessageBox.No,
            ) != QMessageBox.Yes:
                return
            self.qdb.request_cancel(job_id)
        else:
            job_ids: List[int] = []
            for row in queued_rows:
                item = self.tbl_queued.item(row, 1)
                if item:
                    try:
                        job_ids.append(int(item.text()))
                    except ValueError:
                        pass
            if not job_ids:
                QMessageBox.warning(self, "Cancel", "Could not parse checked job IDs.")
                return
            if QMessageBox.question(
                self,
                "Cancel queued jobs",
                f"Cancel {len(job_ids)} queued job(s)?\n\n{job_ids}",
                QMessageBox.Yes | QMessageBox.No,
            ) != QMessageBox.Yes:
                return
            for job_id in job_ids:
                self.qdb.cancel_job(job_id)

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

"""Read-only Suite2p registration and registration-PC viewer for SpinesGUI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from matplotlib.figure import Figure
from suite2p_numpy_compat import load_suite2p_dict

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (
        QFormLayout, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
        QPushButton, QSlider, QSpinBox, QSplitter, QTabWidget,
        QVBoxLayout, QWidget,
    )
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    FigureCanvas = NavigationToolbar = QTimer = QImage = QPixmap = None
    Qt = QFormLayout = QGridLayout = QHBoxLayout = QLabel = object
    QMainWindow = QPushButton = QSlider = QSpinBox = QSplitter = QTabWidget = object
    QVBoxLayout = QWidget = object


def _as_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value)
    return array if array.size else None


def normalize_pc_image(image: Any) -> np.ndarray:
    """Return a finite 0..1 image, safely handling constant/all-zero arrays."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D PC image, got shape {values.shape}")
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.float32)
    clean = np.where(finite, values, 0.0)
    lo, hi = np.percentile(clean[finite], [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(clean[finite])), float(np.max(clean[finite]))
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((clean - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def compose_pc_images(top: Any, bottom: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose Suite2p-style difference, merged, and normalized top/bottom images."""
    top_norm = normalize_pc_image(top)
    bottom_norm = normalize_pc_image(bottom)
    difference = np.clip((bottom_norm - top_norm) / 2.0 + 0.5, 0.0, 1.0)
    merged = np.stack((bottom_norm, top_norm, bottom_norm), axis=-1)
    return difference, merged, np.stack((top_norm, bottom_norm), axis=0)


def registration_pc_curves(reg_dx: Any) -> Optional[np.ndarray]:
    """Return the Suite2p rigid/nonrigid/nonrigid-max curves."""
    values = _as_array(reg_dx)
    if values is None or values.ndim != 2 or values.shape[1] < 3:
        return None
    return np.asarray(values[:, :3], dtype=np.float32)


def selected_pc_values(reg_dx: Any, pc_index: int) -> Optional[np.ndarray]:
    curves = registration_pc_curves(reg_dx)
    if curves is None or pc_index < 0 or pc_index >= curves.shape[0]:
        return None
    return curves[pc_index].copy()


def resample_trace(values: Any, target_length: int) -> Optional[np.ndarray]:
    """Resample a one-dimensional trace onto a target sample count."""
    array = _as_array(values)
    target_length = int(target_length)
    if array is None or array.ndim != 1 or target_length <= 0:
        return None
    array = np.asarray(array, dtype=np.float64)
    if array.size == target_length:
        return array.astype(np.float32, copy=True)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(target_length, dtype=np.float32)
    if finite.sum() == 1:
        return np.full(target_length, array[finite][0], dtype=np.float32)
    source = np.linspace(0.0, 1.0, array.size)
    target = np.linspace(0.0, 1.0, target_length)
    return np.interp(target, source[finite], array[finite]).astype(np.float32)


class RegistrationData:
    """Lazy, read-only access to a Suite2p registered binary and metadata."""

    def __init__(self, plane: Dict[str, Any]):
        self.plane = dict(plane)
        folder = Path(str(self.plane.get("folder") or ""))
        ops_path = Path(str(self.plane.get("ops_path") or folder / "ops.npy"))
        if not ops_path.exists():
            raise FileNotFoundError(f"Suite2p ops.npy not found: {ops_path}")
        self.ops = load_suite2p_dict(ops_path)
        reg_path = self.ops.get("reg_file") or self.plane.get("reg_file") or folder / "data.bin"
        reg_path = Path(str(reg_path))
        if not reg_path.exists():
            fallback = folder / "data.bin"
            if fallback.exists():
                reg_path = fallback
            else:
                raise FileNotFoundError(f"Registered binary not found: {reg_path}")
        self.reg_path = reg_path
        self.ly = int(self.ops.get("Ly") or self.plane.get("Ly") or 0)
        self.lx = int(self.ops.get("Lx") or self.plane.get("Lx") or 0)
        if self.ly <= 0 or self.lx <= 0:
            raise ValueError("Suite2p Ly/Lx are missing or invalid")
        self.dtype = np.dtype(self.ops.get("datatype") or np.int16)
        frame_bytes = self.ly * self.lx * self.dtype.itemsize
        self.file_frames = os.path.getsize(self.reg_path) // frame_bytes
        requested = int(self.ops.get("nframes") or self.plane.get("nframes") or self.file_frames)
        self.nframes = min(max(requested, 0), self.file_frames)
        self.frame_count_note = None if requested == self.file_frames else (
            f"ops reports {requested:,} frames; binary contains {self.file_frames:,}; using {self.nframes:,}"
        )
        self._movie = np.memmap(self.reg_path, mode="r", dtype=self.dtype,
                                shape=(self.file_frames, self.ly, self.lx))
        reg_outputs = folder / "reg_outputs.npy"
        self.reg = load_suite2p_dict(reg_outputs) if reg_outputs.exists() else {}

    def read_frame(self, index: int) -> np.ndarray:
        if self.nframes <= 0:
            raise ValueError("Registered binary contains no usable frames")
        index = min(max(int(index), 0), self.nframes - 1)
        return np.asarray(self._movie[index])

    def metric(self, key: str) -> Optional[np.ndarray]:
        values = _as_array(self.reg.get(key))
        return None if values is None or values.ndim == 0 else values[:self.nframes]

    def pc_count(self) -> int:
        values = _as_array(self.reg.get("regPC"))
        return int(values.shape[1]) if values is not None and values.ndim == 4 and values.shape[0] >= 2 else 0

    def pc(self, index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        values = _as_array(self.reg.get("regPC"))
        if values is None or values.ndim != 4 or values.shape[0] < 2:
            return None
        index = min(max(int(index), 0), values.shape[1] - 1)
        return np.asarray(values[0, index]), np.asarray(values[1, index])

    def tpc(self, index: int) -> Optional[np.ndarray]:
        values = _as_array(self.reg.get("tPC"))
        if values is None or values.ndim != 2 or index < 0 or index >= values.shape[1]:
            return None
        return np.asarray(values[:, index])

    def reg_dx(self) -> Optional[np.ndarray]:
        return registration_pc_curves(self.reg.get("regDX", self.ops.get("regDX")))


class RegistrationViewer(QMainWindow):
    """Suite2p-style registration-PC view plus an independent binary viewer."""

    def __init__(self, plane: Dict[str, Any], plane_number: int = 0, parent=None):
        if not GUI_AVAILABLE:
            raise ImportError("PyQt5 and matplotlib Qt support are required for RegistrationViewer")
        super().__init__(parent)
        self.setWindowTitle(f"Metrics for registration - plane{plane_number}")
        self.resize(1300, 850)
        self.data = RegistrationData(plane)
        self.frame_index = 0
        self.pc_index = 0
        self.pc_animation_frame = 0
        self.binary_timer = QTimer(self)
        self.binary_timer.timeout.connect(self._advance_binary_frame)
        self.pc_timer = QTimer(self)
        self.pc_timer.timeout.connect(self._advance_pc_animation)
        self._build_ui()
        self._refresh_binary()
        self._refresh_pc()

    def _build_ui(self):
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_binary_tab(), "Registered Binary")
        self.tabs.addTab(self._build_pc_tab(), "Registration PCs")
        root_layout.addWidget(self.tabs)
        self.setCentralWidget(root)

    def _build_binary_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, max(0, self.data.nframes - 1))
        self.frame_spin.valueChanged.connect(self._set_binary_frame)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, max(0, self.data.nframes - 1))
        self.frame_slider.valueChanged.connect(self._set_binary_frame)
        self.binary_play_button = QPushButton("Play")
        self.binary_play_button.clicked.connect(self._toggle_binary_play)
        controls.addWidget(QLabel("Frame:"))
        controls.addWidget(self.frame_spin)
        controls.addWidget(self.frame_slider, 1)
        controls.addWidget(self.binary_play_button)
        layout.addLayout(controls)
        self.binary_status_label = QLabel()
        layout.addWidget(self.binary_status_label)
        self.binary_image_label = QLabel(alignment=Qt.AlignCenter)
        self.binary_image_label.setMinimumSize(700, 600)
        self.binary_canvas = FigureCanvas(Figure(figsize=(10, 2.8)))
        self.binary_trace_axes = self.binary_canvas.figure.subplots(3, 1, sharex=True)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.binary_image_label)
        splitter.addWidget(self.binary_canvas)
        splitter.setSizes([600, 240])
        layout.addWidget(splitter, 1)
        return tab

    def _build_pc_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.pc_spin = QSpinBox()
        self.pc_spin.setRange(1, max(1, self.data.pc_count()))
        self.pc_spin.setValue(1)
        self.pc_spin.valueChanged.connect(self._set_pc)
        self.pc_play_button = QPushButton("Play")
        self.pc_play_button.clicked.connect(self._toggle_pc_play)
        self.pc_values_label = QLabel()
        controls.addWidget(QLabel("PC:"))
        controls.addWidget(self.pc_spin)
        controls.addWidget(self.pc_play_button)
        controls.addWidget(self.pc_values_label, 1)
        layout.addLayout(controls)
        self.pc_canvas = FigureCanvas(Figure(figsize=(12, 7)))
        grid = self.pc_canvas.figure.add_gridspec(2, 3, height_ratios=(1, 2), hspace=0.35, wspace=0.15)
        self.pc_shift_axis = self.pc_canvas.figure.add_subplot(grid[0, 0])
        self.pc_trace_axis = self.pc_canvas.figure.add_subplot(grid[0, 1:])
        self.pc_trace_shift_axis = self.pc_trace_axis.twinx()
        self.pc_trace_axis.set_zorder(2)
        self.pc_trace_axis.patch.set_visible(False)
        self.pc_difference_axis = self.pc_canvas.figure.add_subplot(grid[1, 0])
        self.pc_merged_axis = self.pc_canvas.figure.add_subplot(grid[1, 1])
        self.pc_animation_axis = self.pc_canvas.figure.add_subplot(grid[1, 2])
        self.pc_image_axes = (self.pc_difference_axis, self.pc_merged_axis, self.pc_animation_axis)
        self._syncing_pc_image_view = False
        self._pc_pan_state = None
        for axis in self.pc_image_axes:
            axis.callbacks.connect("xlim_changed", self._sync_pc_image_view)
            axis.callbacks.connect("ylim_changed", self._sync_pc_image_view)
        self.pc_canvas.mpl_connect("scroll_event", self._zoom_pc_image_with_scroll)
        self.pc_canvas.mpl_connect("button_press_event", self._start_pc_image_pan)
        self.pc_canvas.mpl_connect("button_release_event", self._finish_pc_image_pan)
        self.pc_canvas.mpl_connect("motion_notify_event", self._move_pc_image_pan)
        self.pc_toolbar = NavigationToolbar(self.pc_canvas, tab)
        layout.addWidget(self.pc_toolbar)
        layout.addWidget(self.pc_canvas, 1)
        return tab

    def _set_binary_frame(self, index: int):
        self.frame_index = int(index)
        self.frame_spin.blockSignals(True)
        self.frame_slider.blockSignals(True)
        self.frame_spin.setValue(self.frame_index)
        self.frame_slider.setValue(self.frame_index)
        self.frame_spin.blockSignals(False)
        self.frame_slider.blockSignals(False)
        self._refresh_binary()

    def _advance_binary_frame(self):
        if self.data.nframes:
            self._set_binary_frame((self.frame_index + 1) % self.data.nframes)

    def _toggle_binary_play(self):
        if self.binary_timer.isActive():
            self.binary_timer.stop()
            self.binary_play_button.setText("Play")
        else:
            self.binary_timer.start(50)
            self.binary_play_button.setText("Pause")

    def _set_pc(self, value: int):
        self.pc_index = max(0, int(value) - 1)
        self.pc_animation_frame = 0
        self._refresh_pc()

    def _advance_pc_animation(self):
        self.pc_animation_frame = 1 - self.pc_animation_frame
        self._draw_pc_images()

    def _toggle_pc_play(self):
        if self.pc_timer.isActive():
            self.pc_timer.stop()
            self.pc_play_button.setText("Play")
        else:
            self.pc_timer.start(200)
            self.pc_play_button.setText("Pause")

    def _image_pixmap(self, image: np.ndarray) -> QPixmap:
        values = normalize_pc_image(image)
        scaled = np.rint(values * 255.0).astype(np.uint8)
        qimage = QImage(scaled.data, scaled.shape[1], scaled.shape[0], scaled.strides[0], QImage.Format_Grayscale8).copy()
        return QPixmap.fromImage(qimage)

    def _refresh_binary(self):
        if self.data.nframes:
            pixmap = self._image_pixmap(self.data.read_frame(self.frame_index))
            self.binary_image_label.setPixmap(pixmap.scaled(self.binary_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        region = ""
        if self.data.ops.get("xrange") is not None and self.data.ops.get("yrange") is not None:
            region = f" | valid region x={self.data.ops['xrange']}, y={self.data.ops['yrange']}"
        self.binary_status_label.setText((self.data.frame_count_note or f"Registered binary: {self.data.reg_path}") + region)
        self._draw_binary_traces()

    def _draw_binary_traces(self):
        labels = (("xoff", "x shift", "#4daf4a"),
                  ("yoff", "y shift", "#ff7f00"),
                  ("corrXY", "corrXY", "#377eb8"))
        has_trace = False
        for axis, (key, label, color) in zip(self.binary_trace_axes, labels):
            axis.clear()
            values = self.data.metric(key)
            if values is not None:
                values = np.asarray(values).reshape(-1)
                axis.plot(np.arange(values.size), values, color=color, linewidth=0.7)
                has_trace = True
            axis.axvline(self.frame_index, color="black", linewidth=0.8, alpha=0.7)
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
        bad = []
        for key in ("badframes", "badframes0"):
            values = self.data.metric(key)
            if values is not None:
                bad.append(np.flatnonzero(np.asarray(values, dtype=bool)))
        if bad:
            bad_indices = np.unique(np.concatenate(bad))
            self.binary_trace_axes[-1].scatter(
                bad_indices, np.zeros(bad_indices.size), marker="x", color="#d62728",
                label="bad frame", zorder=4,
            )
        self.binary_trace_axes[-1].set_xlabel("frame")
        if not has_trace:
            self.binary_trace_axes[0].text(
                0.5, 0.5, "Registration traces unavailable", ha="center", va="center",
                transform=self.binary_trace_axes[0].transAxes,
            )
        self.binary_canvas.figure.tight_layout()
        self.binary_canvas.draw_idle()

    def _refresh_pc(self):
        self.pc_animation_frame = 0
        self._draw_pc_plot()
        self._draw_pc_images()

    def _draw_pc_plot(self):
        curves = self.data.reg_dx()
        self.pc_shift_axis.clear()
        self.pc_trace_axis.clear()
        self.pc_trace_shift_axis.clear()
        self.pc_difference_axis.clear()
        self.pc_merged_axis.clear()
        self.pc_animation_axis.clear()
        self.pc_shift_axis.set_title("Registration shifts")
        self.pc_trace_axis.set_title(f"tPC {self.pc_index + 1}")
        self.pc_difference_axis.set_title("difference")
        self.pc_merged_axis.set_title("merged")
        self.pc_animation_axis.set_title("top")
        colors = ("#c8c8ff", "#ff6464", "#6432c8")
        labels = ("rigid", "nonrigid", "nonrigid max")
        if curves is not None:
            x = np.arange(1, curves.shape[0] + 1)
            for column, (color, label) in enumerate(zip(colors, labels)):
                self.pc_shift_axis.plot(x, curves[:, column], color=color, linewidth=1.0, label=label)
            selected = selected_pc_values(curves, self.pc_index)
            if selected is not None:
                self.pc_shift_axis.scatter([self.pc_index + 1] * 3, selected, color="white", edgecolor="black", zorder=4)
                self.pc_values_label.setText(" | ".join(f"{label}: {value:.3f}" for label, value in zip(labels, selected)))
            else:
                self.pc_values_label.setText("Registration shifts unavailable")
            self.pc_shift_axis.set_xlabel("PC #")
            self.pc_shift_axis.set_ylabel("pixel shift")
            self.pc_shift_axis.legend(loc="best", fontsize=8)
        else:
            self.pc_values_label.setText("Registration shifts unavailable")
        trace = self.data.tpc(self.pc_index)
        if trace is not None:
            self.pc_trace_axis.plot(np.arange(len(trace)), trace, color="#333333", linewidth=0.7)
            self.pc_trace_axis.set_xlabel("time")
            self.pc_trace_axis.set_ylabel("magnitude")
        else:
            self.pc_trace_axis.text(0.5, 0.5, "tPC unavailable", ha="center", va="center", transform=self.pc_trace_axis.transAxes)
        trace_length = len(trace) if trace is not None else 0
        shift_handles = []
        shift_labels = []
        for key, label, color in (("xoff", "x shift", "#4daf4a"), ("yoff", "y shift", "#ff7f00")):
            shift = resample_trace(self.data.metric(key), trace_length)
            if shift is not None:
                line, = self.pc_trace_shift_axis.plot(
                    np.arange(trace_length), shift, color=color, linestyle="--",
                    linewidth=0.8, label=label,
                )
                shift_handles.append(line)
                shift_labels.append(label)
        if shift_handles:
            self.pc_trace_shift_axis.set_ylabel("pixel shift")
            self.pc_trace_shift_axis.legend(shift_handles, shift_labels, loc="upper right", fontsize=8)
        self.pc_canvas.figure.tight_layout()

    def _pc_image_bounds(self):
        for axis in self.pc_image_axes:
            if axis.images:
                left, right, bottom, top = axis.images[0].get_extent()
                return (min(left, right), max(left, right)), (min(bottom, top), max(bottom, top))
        return None

    @staticmethod
    def _clip_pc_limit(limit, bounds):
        """Keep a limit within bounds without changing its axis orientation."""
        increasing = limit[1] >= limit[0]
        full_low, full_high = bounds
        low, high = sorted(limit)
        span = min(high - low, full_high - full_low)
        if span >= full_high - full_low:
            low, high = full_low, full_high
        else:
            center = (low + high) / 2.0
            low = max(full_low, min(center - span / 2.0, full_high - span))
            high = low + span
        return (low, high) if increasing else (high, low)

    def _sync_pc_image_view(self, source_axis):
        """Keep all PC image panels on the same bounded viewport."""
        if self._syncing_pc_image_view or source_axis not in self.pc_image_axes:
            return
        bounds = self._pc_image_bounds()
        if bounds is None:
            return
        self._syncing_pc_image_view = True
        try:
            xlim = self._clip_pc_limit(source_axis.get_xlim(), bounds[0])
            ylim = self._clip_pc_limit(source_axis.get_ylim(), bounds[1])
            source_axis.set_xlim(xlim)
            source_axis.set_ylim(ylim)
            for axis in self.pc_image_axes:
                if axis is not source_axis:
                    axis.set_xlim(xlim)
                    axis.set_ylim(ylim)
        finally:
            self._syncing_pc_image_view = False

    def _zoom_pc_image_with_scroll(self, event):
        """Zoom all PC image panels around the mouse position."""
        if event.inaxes not in self.pc_image_axes or event.xdata is None or event.ydata is None:
            return
        if event.step == 0:
            return
        factor = 1.0 / 1.2 if event.step > 0 else 1.2
        axis = event.inaxes
        x0, x1 = axis.get_xlim()
        y0, y1 = axis.get_ylim()
        new_xlim = (event.xdata - (event.xdata - x0) * factor,
                    event.xdata + (x1 - event.xdata) * factor)
        new_ylim = (event.ydata - (event.ydata - y0) * factor,
                    event.ydata + (y1 - event.ydata) * factor)
        bounds = self._pc_image_bounds()
        if bounds is not None:
            new_xlim = self._clip_pc_limit(new_xlim, bounds[0])
            new_ylim = self._clip_pc_limit(new_ylim, bounds[1])
        self._syncing_pc_image_view = True
        try:
            for image_axis in self.pc_image_axes:
                image_axis.set_xlim(new_xlim)
                image_axis.set_ylim(new_ylim)
        finally:
            self._syncing_pc_image_view = False
        self.pc_canvas.draw_idle()

    def _start_pc_image_pan(self, event):
        """Start a synchronized left-button drag on an image panel."""
        if event.button != 1 or event.inaxes not in self.pc_image_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._pc_pan_state = (event.inaxes, event.xdata, event.ydata,
                              event.inaxes.get_xlim(), event.inaxes.get_ylim())

    def _move_pc_image_pan(self, event):
        if self._pc_pan_state is None:
            return
        source_axis, start_x, start_y, start_xlim, start_ylim = self._pc_pan_state
        if event.inaxes is not source_axis or event.xdata is None or event.ydata is None:
            return
        dx = start_x - event.xdata
        dy = start_y - event.ydata
        new_xlim = (start_xlim[0] + dx, start_xlim[1] + dx)
        new_ylim = (start_ylim[0] + dy, start_ylim[1] + dy)
        bounds = self._pc_image_bounds()
        if bounds is not None:
            new_xlim = self._clip_pc_limit(new_xlim, bounds[0])
            new_ylim = self._clip_pc_limit(new_ylim, bounds[1])
        self._syncing_pc_image_view = True
        try:
            for axis in self.pc_image_axes:
                axis.set_xlim(new_xlim)
                axis.set_ylim(new_ylim)
        finally:
            self._syncing_pc_image_view = False
        self.pc_canvas.draw_idle()

    def _finish_pc_image_pan(self, event):
        if event.button == 1:
            self._pc_pan_state = None

    def _draw_pc_images(self):
        pc = self.data.pc(self.pc_index)
        if pc is None:
            for axis in (self.pc_difference_axis, self.pc_merged_axis, self.pc_animation_axis):
                axis.clear()
                axis.text(0.5, 0.5, "PC images unavailable", ha="center", va="center", transform=axis.transAxes)
            self.pc_canvas.draw_idle()
            return
        top, bottom = pc
        image_axes = self.pc_image_axes
        # Keep the users zoom/pan viewport when the animated image is redrawn.
        viewports = {
            axis: (axis.get_xlim(), axis.get_ylim())
            for axis in image_axes
            if axis.images
        }
        for axis in image_axes:
            axis.clear()
        difference, merged, normalized = compose_pc_images(top, bottom)
        animated = normalized[self.pc_animation_frame]
        self.pc_difference_axis.imshow(difference, cmap="gray", vmin=0, vmax=1)
        self.pc_merged_axis.imshow(merged, vmin=0, vmax=1)
        self.pc_animation_axis.imshow(animated, cmap="gray", vmin=0, vmax=1)
        self.pc_difference_axis.set_title("difference")
        self.pc_merged_axis.set_title("merged")
        self.pc_animation_axis.set_title("top" if self.pc_animation_frame == 0 else "bottom")
        for axis in image_axes:
            axis.set_xticks([])
            axis.set_yticks([])
            if axis in viewports:
                axis.set_xlim(viewports[axis][0])
                axis.set_ylim(viewports[axis][1])
        self.pc_canvas.draw_idle()

    def closeEvent(self, event):
        self.binary_timer.stop()
        self.pc_timer.stop()
        super().closeEvent(event)

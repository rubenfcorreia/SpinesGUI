#!/usr/bin/env python3
"""
Spines GUI
PyQt5 GUI for selecting, labeling, and extracting ROIs from Suite2p output.

Features:
  • Zoom in/out up to 500% using the mouse wheel and pan via left–click drag.
  • ROI drawing, editing, and deletion with a right–click menu.
  • A contrast slider allows adjusting the displayed mean image’s contrast without modifying the original meanImg.
  • ROIs.npy is stored in a "SpinesGUI" subfolder within the selected root folder.
  • "Extract ROIs" button:
       1. Copies ops.npy, data.bin, and data_chan2.bin (if available) from each original plane folder
          to the corresponding plane folder in SpinesGUI.
       2. Computes ROI masks (stat0) and ROI statistics (stat1) using a patched roi_stats.
       3. Loads the copied binary files using BinaryFile.
       4. Calls extraction_wrapper to extract ROI signals.
       5. Performs spike deconvolution:
             dF = F.copy() - ops["neucoeff"] * Fneu  
             dF = preprocess(F=dF, baseline=ops["baseline"],
                             win_baseline=ops["win_baseline"],
                             sig_baseline=ops["sig_baseline"],
                             fs=ops["fs"],
                             prctile_baseline=ops["prctile_baseline"])
             spks = oasis(F=dF, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"])
          and saves the result to spks.npy.
       6. Creates iscell.npy as a 2D array where each ROI is marked [1, 1].
       7. Builds ROIs_conversion.npy by adding both a "conversion" field ([plane, index])
          and a "conversion index" (a sequential integer assigned after sorting).
       8. Displays a conversion table dialog.
  • A ROIs table is also available.
  
Requirements:
  - PyQt5, numpy, matplotlib, suite2p (including suite2p.io.binary, suite2p.detection, suite2p.extraction,
    and suite2p.extraction.dcnv)
"""

import sys, os, shutil, numpy as np
from collections import OrderedDict
from matplotlib.path import Path

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPolygonF, QPen, QBrush, QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem,
                             QGraphicsEllipseItem, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
                             QFormLayout, QDialog, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMenu, QSlider)

# Import Suite2p functions and BinaryFile.
try:
    from suite2p.detection import roi_stats as original_roi_stats
except ImportError:
    print("Error: Could not import roi_stats from suite2p.detection.")
try:
    from suite2p.extraction import extraction_wrapper
except ImportError:
    print("Error: Could not import extraction_wrapper from suite2p.extraction.")
try:
    from suite2p.io.binary import BinaryFile
except ImportError:
    print("Error: Could not import BinaryFile from suite2p.io.binary.")
# Import oasis and preprocess for spike deconvolution.
try:
    from suite2p.extraction.dcnv import oasis, preprocess
except ImportError:
    print("Error: Could not import oasis and/or preprocess from suite2p.extraction.dcnv.")

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)


# --- Helper Functions ---
def median_pix(ypix, xpix):
    return [np.median(ypix), np.median(xpix)]

def norm_by_average(values: np.ndarray, estimator=np.mean, first_n: int = 100, offset: float = 0.) -> np.ndarray:
    return values / (estimator(values[:first_n]) + offset)


# --- Minimal ROI Class (Placeholder for roi_stats) ---
class ROI:
    def __init__(self, ypix, xpix, lam, med, do_crop):
        self.ypix = ypix
        self.xpix = xpix
        self.lam = lam
        self.med = med
        self.do_crop = do_crop
        # Dummy statistics for demonstration.
        self.mean_r_squared = 1.0
        self.mean_r_squared0 = 1.0
        self.mean_r_squared_compact = 1.0
        self.solidity = 1.0
        self.n_pixels = len(ypix)
        self.npix_soma = len(ypix)
        self.soma_crop = np.ones_like(ypix, dtype=bool)

    @staticmethod
    def get_overlap_count_image(rois, Ly, Lx):
        overlap = np.zeros((Ly, Lx))
        for roi in rois:
            overlap[roi.ypix, roi.xpix] += 1
        return overlap

    def get_overlap_image(self, overlap_image):
        return overlap_image[self.ypix, self.xpix] > 1

    def fit_ellipse(self, dy, dx):
        class DummyEllipse:
            pass
        de = DummyEllipse()
        de.radius = 5.0
        de.aspect_ratio = 1.0
        return de

    @staticmethod
    def filter_overlappers(rois, overlap_image, max_overlap):
        keep = []
        for roi in rois:
            ratio = np.mean(overlap_image[roi.ypix, roi.xpix] > 1)
            keep.append(ratio <= max_overlap)
        return keep


# --- Patched roi_stats Function ---
def patched_roi_stats(stat, Ly: int, Lx: int, aspect=None, diameter=None, max_overlap=None, do_crop=True):
    print("[DEBUG] Inside patched_roi_stats")
    if "med" not in stat[0]:
        for s in stat:
            s["med"] = median_pix(s["ypix"], s["xpix"])
    d0 = 10 if diameter is None or (isinstance(diameter, int) and diameter == 0) else diameter
    if aspect is not None:
        diameter_val = d0[0] if isinstance(d0, (list, np.ndarray)) else d0
        diameter_val = int(diameter_val)
        dy, dx = int(aspect * diameter_val), diameter_val
    else:
        if isinstance(d0, (list, np.ndarray)):
            dy, dx = int(d0[0]), int(d0[0])
        else:
            dy, dx = int(d0), int(d0)
    rois = [ROI(ypix=s["ypix"], xpix=s["xpix"], lam=s["lam"], med=s["med"], do_crop=do_crop)
            for s in stat]
    n_overlaps = ROI.get_overlap_count_image(rois=rois, Ly=Ly, Lx=Lx)
    for roi, s in zip(rois, stat):
        s["mrs"] = roi.mean_r_squared
        s["mrs0"] = roi.mean_r_squared0
        s["compact"] = roi.mean_r_squared_compact
        s["solidity"] = roi.solidity
        s["npix"] = roi.n_pixels
        s["npix_soma"] = roi.npix_soma
        s["soma_crop"] = roi.soma_crop
        s["overlap"] = roi.get_overlap_image(n_overlaps)
        ellipse = roi.fit_ellipse(dy, dx)
        s["radius"] = ellipse.radius
        s["aspect_ratio"] = ellipse.aspect_ratio
    mrs_normeds = norm_by_average(values=np.array([s["mrs"] for s in stat]),
                                  estimator=np.nanmedian, offset=1e-10, first_n=100)
    npix_normeds = norm_by_average(values=np.array([s["npix"] for s in stat]), first_n=100)
    npix_soma_normeds = norm_by_average(values=np.array([s["npix_soma"] for s in stat]), first_n=100)
    for s, mrs_normed, npix_normed, npix_soma_normed in zip(stat, mrs_normeds, npix_normeds, npix_soma_normeds):
        s["mrs"] = mrs_normed
        s["npix_norm_no_crop"] = npix_normeds
        s["npix_norm"] = npix_soma_normeds
        s["footprint"] = s.get("footprint", 0)
    if max_overlap is not None and max_overlap < 1.0:
        keep_rois = ROI.filter_overlappers(rois=rois, overlap_image=n_overlaps, max_overlap=max_overlap)
        stat = [s for s, keep in zip(stat, keep_rois) if keep]
        n_overlaps = ROI.get_overlap_count_image(rois=rois, Ly=Ly, Lx=Lx)
        rois = [ROI(ypix=s["ypix"], xpix=s["xpix"], lam=s["lam"], med=s["med"], do_crop=do_crop)
                for s in stat]
        for roi, s in zip(rois, stat):
            s["overlap"] = roi.get_overlap_image(n_overlaps)
    return stat

roi_stats = patched_roi_stats
print("[DEBUG] Using patched roi_stats:", roi_stats.__name__)


# --- ROIItem Class ---
class ROIItem(QGraphicsPolygonItem):
    def __init__(self, roi_id, polygon, roi_info, main_window, parent=None):
        super(ROIItem, self).__init__(polygon, parent)
        self.roi_id = roi_id
        self.roi_info = roi_info  # Expected keys: "roi-type", "plane", "ROI coordinates"
        self.main_window = main_window
        typ = self.roi_info["roi-type"][0]
        color = {0: Qt.blue, 1: Qt.red, 2: Qt.green}.get(typ, Qt.gray)
        self.setPen(QPen(color, 2))
        self.setBrush(QBrush(Qt.transparent))
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.setFlags(QGraphicsPolygonItem.ItemIsSelectable | QGraphicsPolygonItem.ItemIsMovable)
        self.dragging_vertex_index = None
        self.vertex_markers = []
        self.update_vertex_markers()

    def update_vertex_markers(self):
        for marker in self.vertex_markers:
            if self.scene() is not None:
                self.scene().removeItem(marker)
        self.vertex_markers = []
        radius = 2
        for pt in self.polygon():
            rect = QRectF(pt.x() - radius, pt.y() - radius, 2 * radius, 2 * radius)
            marker = QGraphicsEllipseItem(rect, self)
            marker.setBrush(QBrush(QColor("orange")))
            marker.setPen(QPen(Qt.black))
            marker.setZValue(3)
            self.vertex_markers.append(marker)

    def contextMenuEvent(self, event):
        menu = QMenu()
        edit_action = menu.addAction("Edit ROI")
        delete_action = menu.addAction("Delete ROI")
        action = menu.exec_(event.screenPos())
        if action == edit_action:
            self.edit_roi()
        elif action == delete_action:
            self.delete_roi()
        event.accept()

    def edit_roi(self):
        from PyQt5.QtWidgets import QInputDialog
        num, ok = QInputDialog.getInt(None, "Edit ROI",
                                      "Enter new number of sides (3-10):",
                                      min=3, max=10, value=len(self.roi_info.get("ROI coordinates", [])))
        if ok:
            confirm = QMessageBox.question(None, "Confirm Edit",
                                           "ROI shape will be changed to a regular polygon. Proceed?",
                                           QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                pts = self.roi_info.get("ROI coordinates", [])
                if pts.size == 0:
                    return
                xs = pts[:, 0]
                ys = pts[:, 1]
                center = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
                radius = min(xs.max() - xs.min(), ys.max() - ys.min()) / 2
                new_pts = np.array([(center[0] + radius * np.cos(2 * np.pi * i / num),
                                     center[1] + radius * np.sin(2 * np.pi * i / num))
                                    for i in range(num)])
                self.roi_info["ROI coordinates"] = new_pts
                new_poly = QPolygonF([QPointF(x, y) for x, y in new_pts])
                self.setPolygon(new_poly)
                self.update_vertex_markers()

    def delete_roi(self):
        assoc_ids = []
        typ, cellID, parentID, spineID = self.roi_info["roi-type"]
        if typ == 0:
            for rid, info in self.main_window.roi_data.items():
                if rid == self.roi_id:
                    continue
                rtype, cid, pid, sid = info["roi-type"]
                if cid == cellID:
                    assoc_ids.append(rid)
        elif typ == 1:
            for rid, info in self.main_window.roi_data.items():
                if rid == self.roi_id:
                    continue
                rtype, cid, pid, sid = info["roi-type"]
                if rtype == 2 and cid == cellID and pid == parentID:
                    assoc_ids.append(rid)
        if assoc_ids:
            confirm = QMessageBox.question(None, "Delete ROI",
                                           "Deleting this ROI will also delete its associated ROIs.\nProceed?",
                                           QMessageBox.Yes | QMessageBox.No)
            if confirm != QMessageBox.Yes:
                return
            for rid in assoc_ids:
                self.main_window.remove_roi(rid)
        self.main_window.remove_roi(self.roi_id)

    def mousePressEvent(self, event):
        pos = event.pos()
        poly = self.polygon()
        threshold = 10
        for i, pt in enumerate(poly):
            if (pt - pos).manhattanLength() < threshold:
                self.dragging_vertex_index = i
                self.setFlag(QGraphicsPolygonItem.ItemIsMovable, False)
                event.accept()
                return
        super(ROIItem, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging_vertex_index is not None:
            new_pos = event.pos()
            poly = self.polygon()
            poly[self.dragging_vertex_index] = new_pos
            self.setPolygon(poly)
            event.accept()
        else:
            super(ROIItem, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging_vertex_index is not None:
            poly = self.polygon()
            pts = np.array([[pt.x(), pt.y()] for pt in poly])
            self.roi_info["ROI coordinates"] = pts
            self.dragging_vertex_index = None
            self.setFlag(QGraphicsPolygonItem.ItemIsMovable, True)
            self.update_vertex_markers()
            event.accept()
        else:
            super(ROIItem, self).mouseReleaseEvent(event)


# --- CustomGraphicsView Class ---
class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super(CustomGraphicsView, self).__init__(parent)
        self.setMouseTracking(True)
        self.drawing_roi = False
        self.first_click_point = None
        self.temp_polygon_item = None
        self.parent_window = parent
        self.current_scale = 1.0
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        factor = 1.1 ** delta
        new_scale = self.current_scale * factor
        # Allow zooming up to 350%
        if new_scale < 1.0:
            factor = 1.0 / self.current_scale
            self.current_scale = 1.0
        elif new_scale > 5:
            factor = 5 / self.current_scale
            self.current_scale = 5
        else:
            self.current_scale = new_scale
        self.scale(factor, factor)
        event.accept()

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        if (self.parent_window.image_width is not None and self.parent_window.image_height is not None and
            0 <= scene_pos.x() <= self.parent_window.image_width and
            0 <= scene_pos.y() <= self.parent_window.image_height):
            text = "X : %d\nY : %d" % (int(scene_pos.x()), int(scene_pos.y()))
        else:
            text = "X : Out of range\nY : Out of range"
        self.parent_window.coord_label.setText(text)
        super(CustomGraphicsView, self).mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if self.drawing_roi and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if not self.parent_window.current_valid_rect.contains(scene_pos):
                QMessageBox.warning(self, "Out of Bounds", "Click must be within the valid ROI area.")
                return
            if self.parent_window.pending_roi_shape == "tracing":
                if not self.parent_window.tracing_vertices:
                    self.parent_window.tracing_vertices = [scene_pos]
                    poly = QPolygonF(self.parent_window.tracing_vertices)
                    self.parent_window.tracing_polygon_item = QGraphicsPolygonItem(poly)
                    self.parent_window.tracing_polygon_item.setPen(QPen(Qt.magenta, 2, Qt.DashLine))
                    self.scene().addItem(self.parent_window.tracing_polygon_item)
                    r = 3
                    # First vertex in red.
                    marker = QGraphicsEllipseItem(QRectF(scene_pos.x()-r, scene_pos.y()-r, 2*r, 2*r))
                    marker.setBrush(QBrush(QColor("red")))
                    marker.setPen(QPen(Qt.black))
                    marker.setZValue(4)
                    self.parent_window.tracing_markers = [marker]
                    self.scene().addItem(marker)
                else:
                    first = self.parent_window.tracing_vertices[0]
                    if (first - scene_pos).manhattanLength() < 10:
                        msgBox = QMessageBox()
                        msgBox.setWindowTitle("Finish Tracing?")
                        msgBox.setText("Do you wish to finish?")
                        finish_btn = msgBox.addButton("Finish", QMessageBox.AcceptRole)
                        continue_btn = msgBox.addButton("Continue", QMessageBox.ActionRole)
                        cancel_btn = msgBox.addButton("Cancel", QMessageBox.RejectRole)
                        msgBox.exec_()
                        if msgBox.clickedButton() == finish_btn:
                            self.parent_window.finish_tracing_roi()
                        elif msgBox.clickedButton() == continue_btn:
                            self.parent_window.tracing_vertices.append(scene_pos)
                            self.parent_window.update_tracing_display()
                        elif msgBox.clickedButton() == cancel_btn:
                            self.parent_window.cancel_tracing()
                        return
                    else:
                        self.parent_window.tracing_vertices.append(scene_pos)
                        self.parent_window.update_tracing_display()
                return
            else:
                if not self.parent_window.current_valid_rect.contains(scene_pos):
                    QMessageBox.warning(self, "Out of Bounds", "Click must be within the valid ROI area.")
                    return
                if self.first_click_point is None:
                    self.first_click_point = scene_pos
                    self.temp_polygon_item = QGraphicsPolygonItem()
                    self.temp_polygon_item.setPen(QPen(Qt.red, 2, Qt.DashLine))
                    self.scene().addItem(self.temp_polygon_item)
                    self.setDragMode(QGraphicsView.NoDrag)
                else:
                    self.parent_window.finish_roi_drawing(self.first_click_point, scene_pos)
                    self.first_click_point = None
                    if self.temp_polygon_item:
                        self.scene().removeItem(self.temp_polygon_item)
                        self.temp_polygon_item = None
                    self.drawing_roi = False
                    self.setDragMode(QGraphicsView.ScrollHandDrag)
        else:
            super(CustomGraphicsView, self).mousePressEvent(event)

    def keyPressEvent(self, event):
        if not self.drawing_roi:
            if event.key() == Qt.Key_Left:
                self.parent_window.change_plane(-1)
            elif event.key() == Qt.Key_Right:
                self.parent_window.change_plane(1)
            else:
                super(CustomGraphicsView, self).keyPressEvent(event)
        else:
            super(CustomGraphicsView, self).keyPressEvent(event)


# --- ROITableWindow and ConfirmROITableDialog Classes ---
class ROITableWindow(QDialog):
    def __init__(self, roi_data, parent=None):
        super(ROITableWindow, self).__init__(parent)
        self.setWindowTitle("ROIs Table")
        self.resize(800, 300)
        self.roi_data = roi_data
        self.main_window = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["ROI #", "ROI Type", "Cell ID", "Parent Dendrite ID",
                                               "Dendritic Spine ID", "Plane", "ROI Coordinates"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.populate_table()
        self.table.cellClicked.connect(self.row_clicked)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def populate_table(self):
        self.table.setRowCount(len(self.roi_data))
        for row, roi_id in enumerate(sorted(self.roi_data.keys())):
            info = self.roi_data[roi_id]
            typ, cellID, parentID, spineID = info["roi-type"]
            typ_str = {0: "Cell", 1: "Parent Dendrite", 2: "Dendritic Spine"}.get(typ, "Unknown")
            items = [
                QTableWidgetItem(str(roi_id)),
                QTableWidgetItem(typ_str),
                QTableWidgetItem(str(cellID)),
                QTableWidgetItem(str(parentID)),
                QTableWidgetItem(str(spineID)),
                QTableWidgetItem(str(info["plane"])),
                QTableWidgetItem(str(info["ROI coordinates"].tolist() if hasattr(info["ROI coordinates"], "tolist") else info["ROI coordinates"]))
            ]
            for col, item in enumerate(items):
                self.table.setItem(row, col, item)

    def row_clicked(self, row, col):
        roi_number = int(self.table.item(row, 0).text())
        self.main_window.highlight_roi(roi_number)


class ConfirmROITableDialog(QDialog):
    def __init__(self, roi_data, parent=None):
        super(ConfirmROITableDialog, self).__init__(parent)
        self.setWindowTitle("Confirm ROI Addition")
        self.resize(800, 300)
        self.roi_data = roi_data
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["ROI #", "ROI Type", "Cell ID", "Parent Dendrite ID",
                                               "Dendritic Spine ID", "Plane", "ROI Coordinates"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.populate_table()
        layout.addWidget(self.table)
        btn_layout = QHBoxLayout()
        self.continue_btn = QPushButton("Continue")
        self.cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(self.continue_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.continue_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def populate_table(self):
        self.table.setRowCount(len(self.roi_data))
        for row, roi_id in enumerate(sorted(self.roi_data.keys())):
            info = self.roi_data[roi_id]
            typ, cellID, parentID, spineID = info["roi-type"]
            typ_str = {0: "Cell", 1: "Parent Dendrite", 2: "Dendritic Spine"}.get(typ, "Unknown")
            items = [
                QTableWidgetItem(str(roi_id)),
                QTableWidgetItem(typ_str),
                QTableWidgetItem(str(cellID)),
                QTableWidgetItem(str(parentID)),
                QTableWidgetItem(str(spineID)),
                QTableWidgetItem(str(info["plane"])),
                QTableWidgetItem(str(info["ROI coordinates"].tolist() if hasattr(info["ROI coordinates"], "tolist") else info["ROI coordinates"]))
            ]
            for col, item in enumerate(items):
                self.table.setItem(row, col, item)


# --- ConversionTableDialog Class ---
class ConversionTableDialog(QDialog):
    def __init__(self, conversion_dict, parent=None):
        super(ConversionTableDialog, self).__init__(parent)
        self.setWindowTitle("ROIs Conversion Table")
        self.resize(900, 300)
        self.conversion_dict = conversion_dict
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.table = QTableWidget()
        # 9 columns: added one for "Conversion Index"
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(["ROI #", "ROI Type", "Cell ID", "Parent Dendrite ID",
                                               "Dendritic Spine ID", "Plane", "ROI Coordinates", "Conversion", "Conversion Index"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.populate_table()
        layout.addWidget(self.table)
        self.setLayout(layout)

    def populate_table(self):
        self.table.setRowCount(len(self.conversion_dict))
        for row, roi_id in enumerate(sorted(self.conversion_dict.keys())):
            info = self.conversion_dict[roi_id]
            typ, cellID, parentID, spineID = info["roi-type"]
            typ_str = {0: "Cell", 1: "Parent Dendrite", 2: "Dendritic Spine"}.get(typ, "Unknown")
            roi_coords = info.get("ROI coordinates", [])
            if hasattr(roi_coords, "tolist"):
                roi_coords_str = str(roi_coords.tolist())
            else:
                roi_coords_str = str(roi_coords)
            conversion = info.get("conversion", ["N/A", "N/A"])
            conv_index = info.get("conversion index", "N/A")
            items = [
                QTableWidgetItem(str(roi_id)),
                QTableWidgetItem(typ_str),
                QTableWidgetItem(str(cellID)),
                QTableWidgetItem(str(parentID)),
                QTableWidgetItem(str(spineID)),
                QTableWidgetItem(str(info["plane"])),
                QTableWidgetItem(roi_coords_str),
                QTableWidgetItem(str(conversion)),
                QTableWidgetItem(str(conv_index))
            ]
            for col, item in enumerate(items):
                self.table.setItem(row, col, item)


# --- ROITypeDialog Class ---
class ROITypeDialog(QDialog):
    def __init__(self, existing_cells, existing_parents, parent=None):
        super(ROITypeDialog, self).__init__(parent)
        self.setWindowTitle("Select ROI Type")
        self._selected_roi_type = None
        self._selected_assoc = None
        self.init_ui(existing_cells, existing_parents)

    def init_ui(self, existing_cells, existing_parents):
        layout = QFormLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItem("Cell (no association needed)", 0)
        self.type_combo.addItem("Parent Dendrite (associate with a Cell)", 1)
        self.type_combo.addItem("Dendritic Spine (associate with a Parent Dendrite)", 2)
        layout.addRow("ROI Type:", self.type_combo)
        self.assoc_combo = QComboBox()
        layout.addRow("Associate with:", self.assoc_combo)
        self.assoc_combo.hide()
        self.type_combo.currentIndexChanged.connect(lambda: self.update_association(existing_cells, existing_parents))
        try:
            self.assoc_combo.currentIndexChanged.disconnect()
        except Exception:
            pass
        self.assoc_combo.currentIndexChanged.connect(self.highlight_association)
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addRow(btn_layout)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        self.setLayout(layout)
        self.update_association(existing_cells, existing_parents)

    def update_association(self, existing_cells, existing_parents):
        roi_type = self.type_combo.currentData()
        self.assoc_combo.clear()
        if roi_type == 1:
            if not existing_cells:
                QMessageBox.warning(self, "No Cells", "No cell ROIs available.")
                self.reject()
                return
            sorted_cells = sorted(existing_cells)
            for key in sorted_cells:
                self.assoc_combo.addItem(f"Cell ROI #{key}", key)
            self.assoc_combo.show()
            self.assoc_combo.setCurrentIndex(len(sorted_cells) - 1)
            self.highlight_association()
        elif roi_type == 2:
            if not existing_parents:
                QMessageBox.warning(self, "No Parent Dendrites", "No parent dendrite ROIs available.")
                self.reject()
                return
            sorted_parents = sorted(existing_parents)
            for key in sorted_parents:
                self.assoc_combo.addItem(f"Parent Dendrite ROI #{key}", key)
            self.assoc_combo.show()
            self.assoc_combo.setCurrentIndex(len(sorted_parents) - 1)
            self.highlight_association()
        else:
            self.assoc_combo.hide()

    def highlight_association(self):
        if self.assoc_combo.isVisible():
            assoc_key = self.assoc_combo.currentData()
            if self.parent():
                self.parent().highlight_roi(assoc_key)

    def accept(self):
        try:
            self._selected_roi_type = int(self.type_combo.currentData())
        except Exception:
            self._selected_roi_type = 0
        if self.assoc_combo.isVisible():
            try:
                self._selected_assoc = int(self.assoc_combo.currentData())
            except Exception:
                self._selected_assoc = 0
        else:
            self._selected_assoc = 0
        if self.parent():
            self.parent().clear_highlight()
        super(ROITypeDialog, self).accept()

    def get_values(self):
        return self._selected_roi_type, self._selected_assoc


# --- ROIShapeDialog Class ---
class ROIShapeDialog(QDialog):
    def __init__(self, parent=None):
        super(ROIShapeDialog, self).__init__(parent)
        self.setWindowTitle("Select ROI Shape")
        self.shape = None
        self.num_sides = None
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()
        self.shape_combo = QComboBox()
        self.shape_combo.addItem("Rectangle", "rectangle")
        self.shape_combo.addItem("Other Polygon", "polygon")
        self.shape_combo.addItem("Tracing", "tracing")
        layout.addRow("ROI Shape:", self.shape_combo)
        self.sides_input = QLineEdit()
        self.sides_input.setPlaceholderText("Enter number of sides (3-10)")
        self.sides_input.setEnabled(self.shape_combo.currentData() == "polygon")
        layout.addRow("Polygon sides:", self.sides_input)
        self.shape_combo.currentIndexChanged.connect(lambda: self.sides_input.setEnabled(self.shape_combo.currentData() == "polygon"))
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addRow(btn_layout)
        self.ok_btn.clicked.connect(self.on_ok)
        self.cancel_btn.clicked.connect(self.reject)
        self.setLayout(layout)

    def on_ok(self):
        self.shape = self.shape_combo.currentData()
        if self.shape == "polygon":
            try:
                num = int(self.sides_input.text())
                if num < 3 or num > 10:
                    raise ValueError
                self.num_sides = num
            except Exception as e:
                QMessageBox.warning(self, "Invalid Input", "Please enter an integer between 3 and 10 for polygon sides.")
                return
        self.accept()

    def get_values(self):
        return self.shape, self.num_sides


# --- MainWindow Class ---
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Spines GUI")
        self.setGeometry(50, 50, 1200, 800)
        self.showFullScreen()
        self.root_folder = None
        self.plane_data = OrderedDict()  # keys: plane number; values: dict with meanImg, yrange, xrange, folder
        self.plane_order = []
        self.current_plane_index = 0
        self.current_valid_rect = None
        self.image_width = None
        self.image_height = None
        self.roi_data = {}    # ROI key -> ROI info
        self.roi_items = {}   # ROI key -> ROIItem
        self.next_roi_id = 0
        self.pending_roi_type = None  # Tuple: (roi_type, associated ROI key)
        self.pending_roi_shape = None  # "rectangle", "polygon", or "tracing"
        self.pending_polygon_sides = None
        self.tracing_vertices = []
        self.tracing_polygon_item = None
        self.tracing_markers = []
        self.current_meanImg = None  # Store original meanImg for contrast adjustments
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()
        self.graphics_scene = QGraphicsScene()
        self.view = CustomGraphicsView(self)
        self.view.setScene(self.graphics_scene)
        self.image_pixmap_item = None
        self.load_button = QPushButton("Load Suite2p")
        self.load_button.clicked.connect(self.load_suite2p_folder)
        self.add_roi_button = QPushButton("Add ROI")
        self.add_roi_button.clicked.connect(self.start_roi_addition)
        self.roi_table_button = QPushButton("ROIs Table")
        self.roi_table_button.clicked.connect(self.open_roi_table)
        self.extract_button = QPushButton("Extract ROIs")
        self.extract_button.clicked.connect(self.extract_rois)
        self.clear_rois_button = QPushButton("Clear All ROIs")
        self.clear_rois_button.clicked.connect(self.clear_all_rois)
        self.left_arrow_button = QPushButton("<")
        self.left_arrow_button.clicked.connect(lambda: self.change_plane(-1))
        self.right_arrow_button = QPushButton(">")
        self.right_arrow_button.clicked.connect(lambda: self.change_plane(1))
        self.current_plane_label = QLabel("Current plane: N/A")
        self.contrast_label = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(50)
        self.contrast_slider.setMaximum(200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        self.coord_label = QLabel("X : Out of range\nY : Out of range")
        self.coord_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        bottom_layout.addWidget(self.current_plane_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.contrast_label)
        bottom_layout.addWidget(self.contrast_slider)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.coord_label)
        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.add_roi_button)
        top_layout.addWidget(self.roi_table_button)
        top_layout.addWidget(self.extract_button)
        top_layout.addWidget(self.clear_rois_button)
        top_layout.addWidget(self.left_arrow_button)
        top_layout.addWidget(self.right_arrow_button)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.view)
        main_layout.addLayout(bottom_layout)
        central_widget.setLayout(main_layout)

    def update_contrast(self):
        if self.current_meanImg is None:
            return
        # Use the provided formula on the original image without modifying it.
        mimg = self.current_meanImg.astype(np.float32)
        mimg1 = np.percentile(mimg, 1)
        mimg99 = np.percentile(mimg, 99)
        mimg_disp = (mimg - mimg1) / (mimg99 - mimg1)
        mimg_disp = np.clip(mimg_disp, 0, 1)
        # Apply contrast adjustment from the slider (default factor 1.0 when slider is 100)
        factor = self.contrast_slider.value() / 100.0
        mimg_disp = 0.5 + factor * (mimg_disp - 0.5)
        mimg_disp = np.clip(mimg_disp, 0, 1)
        mimg_disp = (mimg_disp * 255).astype(np.uint8)
        height, width = mimg_disp.shape
        image = QImage(mimg_disp.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image)
        # Try updating the existing pixmap item; if not valid, create a new one.
        try:
            if self.image_pixmap_item is None or self.image_pixmap_item.scene() is None:
                raise RuntimeError("Pixmap item is not valid")
            else:
                self.image_pixmap_item.setPixmap(pixmap)
        except RuntimeError:
            self.image_pixmap_item = QGraphicsPixmapItem(pixmap)
            self.graphics_scene.addItem(self.image_pixmap_item)
        # Note: The yellow border is now added solely in update_plane_display().


    def load_suite2p_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Root Folder", os.getcwd())
        if not folder:
            return
        if self.root_folder is not None:
            confirm = QMessageBox.question(self, "Change Folder",
                                           "Loading a new folder will clear current data.\nProceed?",
                                           QMessageBox.Yes | QMessageBox.No)
            if confirm != QMessageBox.Yes:
                return
        self.root_folder = folder
        self.plane_data.clear()
        self.plane_order = []
        self.current_plane_index = 0
        self.roi_data.clear()
        self.next_roi_id = 0
        self.clear_scene()
        for entry in sorted(os.listdir(folder)):
            subfolder = os.path.join(folder, entry)
            if os.path.isdir(subfolder) and entry.startswith("plane") and "combined" not in entry.lower():
                try:
                    plane_num = int(entry.replace("plane", ""))
                except:
                    continue
                ops_file = os.path.join(subfolder, "ops.npy")
                if os.path.exists(ops_file):
                    try:
                        ops = np.load(ops_file, allow_pickle=True).item()
                        meanImg = ops.get("meanImg", None)
                        yrange = ops.get("yrange", None)
                        xrange_ = ops.get("xrange", None)
                        if meanImg is None or yrange is None or xrange_ is None:
                            continue
                        if np.isnan(meanImg).any():
                            continue
                        self.plane_data[plane_num] = {"meanImg": meanImg, "yrange": yrange, "xrange": xrange_, "folder": subfolder}
                        self.plane_order.append(plane_num)
                    except Exception as e:
                        print(f"[DEBUG] Error loading ops.npy in {subfolder}: {e}")
        self.plane_order = sorted(self.plane_order)
        if not self.plane_order:
            QMessageBox.warning(self, "Error", "No valid plane folders found.")
            return
        spines_gui_folder = os.path.join(self.root_folder, "SpinesGUI")
        if not os.path.exists(spines_gui_folder):
            os.makedirs(spines_gui_folder)
        rois_file = os.path.join(spines_gui_folder, "ROIs.npy")
        if os.path.exists(rois_file):
            try:
                loaded_rois = np.load(rois_file, allow_pickle=True).item()
                self.roi_data = {int(k): v for k, v in loaded_rois.items()}
                if self.roi_data:
                    self.next_roi_id = max(self.roi_data.keys()) + 1
            except Exception as e:
                print(f"[DEBUG] Error loading ROIs.npy: {e}")
        self.update_plane_display()

    def update_plane_display(self):
        if not self.plane_order:
            return
        # Clear the scene to refresh everything.
        self.graphics_scene.clear()
        plane_num = self.plane_order[self.current_plane_index]
        plane = self.plane_data[plane_num]
        self.current_meanImg = plane["meanImg"]  # Original image for contrast adjustments
        self.update_contrast()  # This now only updates the image
        yrange = plane["yrange"]
        xrange_ = plane["xrange"]
        height, width = self.current_meanImg.shape
        self.image_height = height
        self.image_width = width
        valid_rect = QRectF(xrange_[0], yrange[0], xrange_[1]-xrange_[0], yrange[1]-yrange[0])
        self.current_valid_rect = valid_rect
        # Add the valid region border (yellow) once.
        rect_item = self.graphics_scene.addRect(valid_rect, QPen(Qt.yellow, 2))
        rect_item.setZValue(1)
        self.current_plane_label.setText(f"Current plane: {plane_num}")
        # Add ROI items for the current plane.
        for roi_id, info in self.roi_data.items():
            if info["plane"] == plane_num:
                pts = info["ROI coordinates"]
                poly = QPolygonF([QPointF(x, y) for x, y in pts])
                roi_item = ROIItem(roi_id, poly, info, self)
                roi_item.setZValue(2)
                self.graphics_scene.addItem(roi_item)
                self.roi_items[roi_id] = roi_item
        self.view.fitInView(self.image_pixmap_item, Qt.KeepAspectRatio)


    def clear_scene(self):
        self.graphics_scene.clear()
        self.image_pixmap_item = None
        self.roi_items.clear()

    def change_plane(self, delta):
        if not self.plane_order:
            return
        new_index = self.current_plane_index + delta
        if new_index < 0 or new_index >= len(self.plane_order):
            return
        self.current_plane_index = new_index
        self.update_plane_display()

    def start_roi_addition(self):
        if not self.plane_order:
            QMessageBox.warning(self, "Error", "No plane loaded.")
            return
        existing_cells = [roi_id for roi_id, info in self.roi_data.items() if info["roi-type"][0] == 0]
        existing_parents = [roi_id for roi_id, info in self.roi_data.items() if info["roi-type"][0] == 1]
        type_dialog = ROITypeDialog(existing_cells, existing_parents, self)
        if type_dialog.exec_() != QDialog.Accepted:
            return
        roi_type_choice, assoc_key = type_dialog.get_values()
        self.pending_roi_type = (roi_type_choice, assoc_key)
        if self.pending_roi_type[0] == 2 and not existing_parents:
            QMessageBox.warning(self, "Error", "No Parent dendrite ROIs available for association.")
            return
        shape_dialog = ROIShapeDialog(self)
        if shape_dialog.exec_() != QDialog.Accepted:
            return
        shape, sides = shape_dialog.get_values()
        self.pending_roi_shape = shape
        self.pending_polygon_sides = sides
        if self.pending_roi_shape == "tracing":
            self.tracing_vertices = []
            if self.tracing_polygon_item:
                self.graphics_scene.removeItem(self.tracing_polygon_item)
                self.tracing_polygon_item = None
            if self.tracing_markers:
                for marker in self.tracing_markers:
                    self.graphics_scene.removeItem(marker)
                self.tracing_markers = []
        QMessageBox.information(self, "ROI Drawing",
                                "Click within the valid ROI area.\n"
                                "For rectangle, click two opposite corners.\n"
                                "For polygon, a regular polygon is inscribed.\n"
                                "For tracing, click to add vertices. The first vertex is highlighted in red.")
        self.view.drawing_roi = True
        self.view.setDragMode(QGraphicsView.NoDrag)

    def get_next_cell_id(self):
        cell_ids = [info["roi-type"][1] for info in self.roi_data.values() if info["roi-type"][0] == 0]
        return max(cell_ids) + 1 if cell_ids else 1

    def get_next_parent_id(self, cell_id):
        parent_ids = [info["roi-type"][2] for info in self.roi_data.values() if info["roi-type"][0] == 1 and info["roi-type"][1] == cell_id]
        return max(parent_ids) + 1 if parent_ids else 1

    def get_next_spine_id(self, cell_id, parent_id):
        spine_ids = [info["roi-type"][3] for info in self.roi_data.values() if info["roi-type"][0] == 2 and info["roi-type"][1] == cell_id and info["roi-type"][2] == parent_id]
        return max(spine_ids) + 1 if spine_ids else 1

    def finish_roi_drawing(self, point1, point2):
        x1, y1 = point1.x(), point1.y()
        x2, y2 = point2.x(), point2.y()
        if self.pending_roi_shape == "rectangle":
            pts = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        else:
            num_sides = self.pending_polygon_sides
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            center = ((left+right)/2, (top+bottom)/2)
            radius = min((right-left), (bottom-top)) / 2
            pts = np.array([(center[0] + radius * np.cos(2*np.pi*i/num_sides),
                             center[1] + radius * np.sin(2*np.pi*i/num_sides))
                            for i in range(num_sides)])
        self._create_roi_from_points(pts)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

    def finish_tracing_roi(self):
        pts = np.array([[pt.x(), pt.y()] for pt in self.tracing_vertices])
        if self.tracing_polygon_item:
            self.graphics_scene.removeItem(self.tracing_polygon_item)
            self.tracing_polygon_item = None
        if self.tracing_markers:
            for marker in self.tracing_markers:
                self.graphics_scene.removeItem(marker)
            self.tracing_markers = []
        self.tracing_vertices = []
        self.view.drawing_roi = False
        self._create_roi_from_points(pts)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

    def update_tracing_display(self):
        if self.tracing_polygon_item:
            self.graphics_scene.removeItem(self.tracing_polygon_item)
        poly = QPolygonF(self.tracing_vertices)
        self.tracing_polygon_item = QGraphicsPolygonItem(poly)
        self.tracing_polygon_item.setPen(QPen(Qt.magenta, 2, Qt.DashLine))
        self.graphics_scene.addItem(self.tracing_polygon_item)
        if self.tracing_markers:
            for marker in self.tracing_markers:
                self.graphics_scene.removeItem(marker)
        self.tracing_markers = []
        for i, pt in enumerate(self.tracing_vertices):
            r = 3
            marker = QGraphicsEllipseItem(QRectF(pt.x()-r, pt.y()-r, 2*r, 2*r))
            if i == 0:
                marker.setBrush(QBrush(QColor("red")))
            else:
                marker.setBrush(QBrush(Qt.yellow))
            marker.setPen(QPen(Qt.black))
            marker.setZValue(4)
            self.tracing_markers.append(marker)
            self.graphics_scene.addItem(marker)

    def cancel_tracing(self):
        if self.tracing_polygon_item:
            self.graphics_scene.removeItem(self.tracing_polygon_item)
            self.tracing_polygon_item = None
        if self.tracing_markers:
            for marker in self.tracing_markers:
                self.graphics_scene.removeItem(marker)
        self.tracing_markers = []
        self.tracing_vertices = []
        self.view.drawing_roi = False
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

    def _create_roi_from_points(self, pts):
        if self.pending_roi_type[0] == 0:
            cell_id = self.get_next_cell_id()
            roi_type_list = [0, cell_id, 0, 0]
        elif self.pending_roi_type[0] == 1:
            associated_cell_key = self.pending_roi_type[1]
            cell_roi = self.roi_data.get(associated_cell_key)
            if cell_roi is None:
                QMessageBox.warning(self, "Error", "Associated cell ROI not found.")
                return
            cell_id = cell_roi["roi-type"][1]
            parent_id = self.get_next_parent_id(cell_id)
            roi_type_list = [1, cell_id, parent_id, 0]
        elif self.pending_roi_type[0] == 2:
            associated_parent_key = self.pending_roi_type[1]
            parent_roi = self.roi_data.get(associated_parent_key)
            if parent_roi is None:
                QMessageBox.warning(self, "Error", "Associated parent ROI not found.")
                return
            if parent_roi["roi-type"][0] != 1:
                QMessageBox.warning(self, "Error", "Selected associated ROI is not a Parent dendrite.")
                return
            cell_id = parent_roi["roi-type"][1]
            parent_id = parent_roi["roi-type"][2]
            if parent_id == 0:
                QMessageBox.warning(self, "Error", "The selected parent ROI does not have a valid Parent dendrite ID.")
                return
            spine_id = self.get_next_spine_id(cell_id, parent_id)
            roi_type_list = [2, cell_id, parent_id, spine_id]
        else:
            roi_type_list = [self.pending_roi_type[0], 0, 0, 0]
        current_plane = self.plane_order[self.current_plane_index]
        roi_info = {"roi-type": roi_type_list, "plane": current_plane, "ROI coordinates": pts}
        polygon = QPolygonF([QPointF(x, y) for x, y in pts])
        roi_item = ROIItem(self.next_roi_id, polygon, roi_info, self)
        roi_item.setZValue(2)
        self.graphics_scene.addItem(roi_item)
        self.roi_items[self.next_roi_id] = roi_item
        self.roi_data[self.next_roi_id] = roi_info
        review_dialog = ConfirmROITableDialog(self.roi_data, self)
        if review_dialog.exec_() != QDialog.Accepted:
            self.graphics_scene.removeItem(roi_item)
            del self.roi_data[self.next_roi_id]
            return
        confirm = QMessageBox.question(self, "Confirm ROI",
                                       f"Did details get stored correctly for ROI #{self.next_roi_id}?",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.No:
            self.graphics_scene.removeItem(roi_item)
            del self.roi_data[self.next_roi_id]
        else:
            self.next_roi_id += 1
            self.save_rois()

    def save_rois(self):
        if self.root_folder is None:
            return
        spines_gui_folder = os.path.join(self.root_folder, "SpinesGUI")
        if not os.path.exists(spines_gui_folder):
            os.makedirs(spines_gui_folder)
        rois_file = os.path.join(spines_gui_folder, "ROIs.npy")
        try:
            np.save(rois_file, self.roi_data)
        except Exception as e:
            print(f"[DEBUG] Error saving ROIs: {e}")

    def open_roi_table(self):
        table_win = ROITableWindow(self.roi_data, self)
        table_win.exec_()

    def highlight_roi(self, roi_id):
        if roi_id is None:
            self.clear_highlight()
            return
        for r_id, item in self.roi_items.items():
            if r_id == roi_id:
                item.setPen(QPen(QColor("purple"), 3))
            else:
                typ = self.roi_data[r_id]["roi-type"][0]
                color = {0: Qt.blue, 1: Qt.red, 2: Qt.green}.get(typ, Qt.gray)
                item.setPen(QPen(color, 2))

    def clear_highlight(self):
        for r_id, item in self.roi_items.items():
            typ = self.roi_data[r_id]["roi-type"][0]
            color = {0: Qt.blue, 1: Qt.red, 2: Qt.green}.get(typ, Qt.gray)
            item.setPen(QPen(color, 2))

    def remove_roi(self, roi_id):
        if roi_id in self.roi_items:
            self.graphics_scene.removeItem(self.roi_items[roi_id])
            del self.roi_items[roi_id]
        if roi_id in self.roi_data:
            del self.roi_data[roi_id]
        self.save_rois()

    def clear_all_rois(self):
        confirm = QMessageBox.question(self, "Clear All ROIs",
                                       "Are you sure you want to clear all ROIs?",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            for item in list(self.roi_items.values()):
                self.graphics_scene.removeItem(item)
            self.roi_items.clear()
            self.roi_data.clear()
            self.next_roi_id = 0
            self.save_rois()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Exit Confirmation", "Do you wish to exit?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def extract_rois(self):
        print("[DEBUG] Starting extraction process")
        if self.root_folder is None:
            QMessageBox.warning(self, "Error", "No root folder loaded.")
            return
        spines_gui_folder = os.path.join(self.root_folder, "SpinesGUI")
        if not os.path.exists(spines_gui_folder):
            os.makedirs(spines_gui_folder)
        reextract = False
        for plane in self.plane_data.keys():
            plane_folder = os.path.join(spines_gui_folder, f"plane{plane}")
            stat1_file = os.path.join(plane_folder, "stat1.npy")
            if os.path.exists(stat1_file):
                reextract = True
                break
        if reextract:
            confirm = QMessageBox.question(self, "Re-run Extraction",
                                           "Are you sure? This will re-run extraction and delete previous extraction files.",
                                           QMessageBox.Yes | QMessageBox.No)
            if confirm != QMessageBox.Yes:
                return
            for plane in self.plane_data.keys():
                plane_folder = os.path.join(spines_gui_folder, f"plane{plane}")
                for fname in ["stat0.npy", "stat1.npy", "stat.npy", "F.npy", "Fneu.npy", "F_chan2.npy", "Fneu_chan2.npy", "spks.npy", "iscell.npy"]:
                    fpath = os.path.join(plane_folder, fname)
                    if os.path.exists(fpath):
                        try:
                            os.remove(fpath)
                            print(f"[DEBUG] Deleted {fpath}")
                        except Exception as e:
                            print(f"[DEBUG] Error deleting {fpath}: {e}")
        # Build conversion dictionary.
        conversion_dict = {}
        plane_groups = {}
        for key, roi in self.roi_data.items():
            p = roi["plane"]
            plane_groups.setdefault(p, []).append((key, roi))
        for plane, items in plane_groups.items():
            items_sorted = sorted(items, key=lambda x: x[0])
            for idx, (roi_key, roi) in enumerate(items_sorted):
                roi["conversion"] = [plane, idx]
                conversion_dict[roi_key] = roi
        # Create conversion index across all ROIs.
        sorted_conversion = sorted(conversion_dict.items(), key=lambda x: (x[1]["conversion"][0], x[1]["conversion"][1]))
        for new_index, (roi_key, roi) in enumerate(sorted_conversion):
            roi["conversion index"] = new_index

        rois_conv_file = os.path.join(spines_gui_folder, "ROIs_conversion.npy")
        try:
            np.save(rois_conv_file, conversion_dict)
            print(f"[DEBUG] Saved ROIs_conversion to {rois_conv_file}")
        except Exception as e:
            print(f"[DEBUG] Error saving ROIs_conversion: {e}")
            return

        # Process extraction for each plane.
        for plane in self.plane_data.keys():
            print(f"[DEBUG] Processing extraction for plane {plane}")
            plane_folder = os.path.join(spines_gui_folder, f"plane{plane}")
            if not os.path.exists(plane_folder):
                os.makedirs(plane_folder)
            # Copy ops.npy.
            ops_src = os.path.join(self.plane_data[plane]["folder"], "ops.npy")
            ops_dest = os.path.join(plane_folder, "ops.npy")
            try:
                shutil.copy(ops_src, ops_dest)
                print(f"[DEBUG] Copied ops.npy from {ops_src} to {ops_dest}")
            except Exception as e:
                print(f"[DEBUG] Error copying ops.npy for plane {plane}: {e}")
            # Copy data.bin.
            data_bin_src = os.path.join(self.plane_data[plane]["folder"], "data.bin")
            data_bin_dest = os.path.join(plane_folder, "data.bin")
            try:
                shutil.copy(data_bin_src, data_bin_dest)
                print(f"[DEBUG] Copied data.bin from {data_bin_src} to {data_bin_dest}")
            except Exception as e:
                print(f"[DEBUG] Error copying data.bin for plane {plane}: {e}")
                continue
            # Copy data_chan2.bin if exists.
            data_chan2_src = os.path.join(self.plane_data[plane]["folder"], "data_chan2.bin")
            data_chan2_dest = os.path.join(plane_folder, "data_chan2.bin")
            if os.path.exists(data_chan2_src):
                try:
                    shutil.copy(data_chan2_src, data_chan2_dest)
                    print(f"[DEBUG] Copied data_chan2.bin from {data_chan2_src} to {data_chan2_dest}")
                except Exception as e:
                    print(f"[DEBUG] Error copying data_chan2.bin for plane {plane}: {e}")
                    data_chan2_dest = None
            else:
                data_chan2_dest = None

            # Load copied ops.
            try:
                ops = np.load(ops_dest, allow_pickle=True).item()
                print(f"[DEBUG] Loaded copied ops.npy for plane {plane}")
            except Exception as e:
                print(f"[DEBUG] Error loading copied ops.npy for plane {plane}: {e}")
                continue

            # Get extraction parameters.
            Ly = ops.get("Ly")
            Lx = ops.get("Lx")
            aspect = ops.get("aspect", 1.0)
            if isinstance(aspect, (list, tuple, np.ndarray)):
                aspect = aspect[0]
            diameter = ops.get("diameter", 10)
            if isinstance(diameter, (list, tuple, np.ndarray)):
                diameter = diameter[0]
            max_overlap = ops.get("max_overlap", 1.0)
            if isinstance(max_overlap, (list, tuple, np.ndarray)):
                max_overlap = max_overlap[0]
            do_crop = ops.get("soma_crop", 1)
            if isinstance(do_crop, (list, tuple, np.ndarray)):
                do_crop = do_crop[0]
            print(f"[DEBUG] For plane {plane}, parameters: Ly={Ly}, Lx={Lx}, aspect={aspect}, diameter={diameter}, max_overlap={max_overlap}, do_crop={do_crop}")

            # Compute stat0.
            stat0 = {}
            roi_list = [(k, roi) for k, roi in self.roi_data.items() if roi["plane"] == plane]
            roi_list_sorted = sorted(roi_list, key=lambda x: x[0])
            for idx, (roi_key, roi) in enumerate(roi_list_sorted):
                vertices = np.array(roi["ROI coordinates"])
                if vertices.size == 0:
                    print(f"[DEBUG] ROI {roi_key} on plane {plane} has no vertices.")
                    continue
                x_min = int(np.floor(np.min(vertices[:, 0])))
                x_max = int(np.ceil(np.max(vertices[:, 0])))
                y_min = int(np.floor(np.min(vertices[:, 1])))
                y_max = int(np.ceil(np.max(vertices[:, 1])))
                xx, yy = np.meshgrid(np.arange(x_min, x_max+1), np.arange(y_min, y_max+1))
                points = np.vstack((xx.flatten(), yy.flatten())).T
                poly_path = Path(vertices)
                inside = poly_path.contains_points(points)
                inside = inside.reshape(yy.shape)
                ypix = np.where(inside)[0] + y_min
                xpix = np.where(inside)[1] + x_min
                lam = np.ones(ypix.shape)
                stat0[idx] = {"ypix": np.array(ypix), "xpix": np.array(xpix), "lam": np.array(lam)}
                print(f"[DEBUG] Plane {plane}, ROI index {idx}: computed mask with {len(ypix)} pixels.")
            stat0_file = os.path.join(plane_folder, "stat0.npy")
            try:
                np.save(stat0_file, stat0)
                print(f"[DEBUG] Saved stat0.npy for plane {plane} in {plane_folder}")
            except Exception as e:
                print(f"[DEBUG] Error saving stat0.npy for plane {plane}: {e}")
                continue
            stat0_list = list(stat0.values())

            # Compute stat1.
            try:
                print("[DEBUG] Calling roi_stats with patched roi_stats")
                stat1 = roi_stats(stat0_list, Ly, Lx, aspect=aspect, diameter=diameter,
                                  max_overlap=max_overlap, do_crop=do_crop)
                stat1_file = os.path.join(plane_folder, "stat1.npy")
                np.save(stat1_file, stat1)
                print(f"[DEBUG] Saved stat1.npy for plane {plane} in {plane_folder}")
            except Exception as e:
                print(f"[DEBUG] Error in roi_stats for plane {plane}: {e}")
                continue

            # Load binary files from copied files.
            try:
                f_reg_data = BinaryFile(Ly, Lx, data_bin_dest, n_frames=ops.get("nframes"), dtype=ops.get("datatype", "int16"))
                print(f"[DEBUG] Loaded BinaryFile for data.bin for plane {plane}")
            except Exception as e:
                print(f"[DEBUG] Error loading BinaryFile for data.bin for plane {plane}: {e}")
                continue
            if data_chan2_dest is not None:
                try:
                    f_reg_chan2_data = BinaryFile(Ly, Lx, data_chan2_dest, n_frames=ops.get("nframes"), dtype=ops.get("datatype", "int16"))
                    print(f"[DEBUG] Loaded BinaryFile for data_chan2.bin for plane {plane}")
                except Exception as e:
                    print(f"[DEBUG] Error loading BinaryFile for data_chan2.bin for plane {plane}: {e}")
                    f_reg_chan2_data = None
            else:
                f_reg_chan2_data = None

            # Run extraction_wrapper.
            try:
                print(f"[DEBUG] Calling extraction_wrapper for plane {plane}")
                outputs = extraction_wrapper(stat1, f_reg_data, f_reg_chan2_data,
                                             cell_masks=None, neuropil_masks=None, ops=ops)
                stat_out, F, Fneu, F_chan2, Fneu_chan2 = outputs
                np.save(os.path.join(plane_folder, "stat.npy"), stat_out)
                np.save(os.path.join(plane_folder, "F.npy"), F)
                np.save(os.path.join(plane_folder, "Fneu.npy"), Fneu)
                np.save(os.path.join(plane_folder, "F_chan2.npy"), F_chan2)
                np.save(os.path.join(plane_folder, "Fneu_chan2.npy"), Fneu_chan2)
                print(f"[DEBUG] Extraction complete for plane {plane}.")
            except Exception as e:
                print(f"[DEBUG] Error in extraction_wrapper for plane {plane}: {e}")
                continue

            # Spike deconvolution.
            try:
                print(f"[DEBUG] Running spike deconvolution for plane {plane}")
                dF = F.copy() - ops["neucoeff"] * Fneu
                dF = preprocess(F=dF, baseline=ops["baseline"],
                                win_baseline=ops["win_baseline"],
                                sig_baseline=ops["sig_baseline"],
                                fs=ops["fs"],
                                prctile_baseline=ops["prctile_baseline"])
                spks = oasis(F=dF, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"])
                spks_file = os.path.join(plane_folder, "spks.npy")
                np.save(spks_file, spks)
                print(f"[DEBUG] Saved spks.npy for plane {plane} in {plane_folder}")
            except Exception as e:
                print(f"[DEBUG] Error in spike deconvolution for plane {plane}: {e}")

            # Create iscell.npy.
            try:
                roi_ids = [roi_id for roi_id, info in self.roi_data.items() if info["plane"] == plane]
                iscell_arr = np.ones((len(roi_ids), 2), dtype=int)
                iscell_file = os.path.join(plane_folder, "iscell.npy")
                np.save(iscell_file, iscell_arr)
                print(f"[DEBUG] Saved iscell.npy for plane {plane} in {plane_folder}")
            except Exception as e:
                print(f"[DEBUG] Error creating iscell.npy for plane {plane}: {e}")

        QMessageBox.information(self, "Extraction Finished", "Extraction finished.")
        try:
            conv_dict = np.load(rois_conv_file, allow_pickle=True).item()
            print("[DEBUG] Loaded ROIs_conversion dictionary for display")
            conv_dialog = ConversionTableDialog(conv_dict, self)
            conv_dialog.exec_()
        except Exception as e:
            print(f"[DEBUG] Error showing conversion table: {e}")

    def highlight_roi(self, roi_id):
        if roi_id is None:
            self.clear_highlight()
            return
        for r_id, item in self.roi_items.items():
            if r_id == roi_id:
                item.setPen(QPen(QColor("purple"), 3))
            else:
                typ = self.roi_data[r_id]["roi-type"][0]
                color = {0: Qt.blue, 1: Qt.red, 2: Qt.green}.get(typ, Qt.gray)
                item.setPen(QPen(color, 2))

    def clear_highlight(self):
        for r_id, item in self.roi_items.items():
            typ = self.roi_data[r_id]["roi-type"][0]
            color = {0: Qt.blue, 1: Qt.red, 2: Qt.green}.get(typ, Qt.gray)
            item.setPen(QPen(color, 2))

    def remove_roi(self, roi_id):
        if roi_id in self.roi_items:
            self.graphics_scene.removeItem(self.roi_items[roi_id])
            del self.roi_items[roi_id]
        if roi_id in self.roi_data:
            del self.roi_data[roi_id]
        self.save_rois()

    def clear_all_rois(self):
        confirm = QMessageBox.question(self, "Clear All ROIs",
                                       "Are you sure you want to clear all ROIs?",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            for item in list(self.roi_items.values()):
                self.graphics_scene.removeItem(item)
            self.roi_items.clear()
            self.roi_data.clear()
            self.next_roi_id = 0
            self.save_rois()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Exit Confirmation", "Do you wish to exit?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())

#!/usr/bin/env python3
"""
Spines GUI
PyQt5 GUI for selecting, labeling, and extracting ROIs from Suite2p output.

Features:
  • Loads both the original mean image (meanImg) and an enhanced version (meanImgE) from ops.npy.
  • Displays a yellow valid ROI border (using yrange/xrange from ops.npy).
  • Full-screen view with mouse–wheel zooming (100%–500%) and panning.
  • Shows mouse coordinates in the bottom–right.
  • ROI drawing via rectangle, regular polygon, or tracing (with vertex markers of radius 1).
  • Right–click on ROIs to edit or delete (with associated ROI deletion).
  • ROIs are stored in a .npy file in a “SpinesGUI” subfolder.
  • “ROIs Table” shows ROI details in a table; table headers adjust based on the current mode.
  • “Extract ROIs” button runs an extraction process using suite2p functions.
  • Supports two modes:
       Normal Mode: ROI types – 0: Cell, 1: Parent Dendrite, 2: Dendritic Spine.
       Dendrites/Axons Mode: ROI types – 0: Parent Dendrite, 1: Dendritic Spine, 2: Parent Axon, 3: Axonal Bouton.
     Mode toggle (top–right) with confirmation; file names change accordingly.
  • A “View” selector (bottom) lets you choose between “Mean Image” and “Mean Image Enhanced.”
     Switching views preserves the current zoom/position.
  • All ROI outlines and tracing lines are drawn with 75% transparency.
  • Debug prints are included throughout.
  
Requirements:
  - PyQt5, numpy, matplotlib, suite2p
"""

import sys, io, os, shutil, numpy as np
from collections import OrderedDict
from matplotlib.path import Path

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPolygonF, QPen, QBrush, QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem,
                             QGraphicsEllipseItem, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
                             QFormLayout, QDialog, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMenu, QSlider, QButtonGroup, QRadioButton)

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

# --- Minimal ROI Class (for roi_stats) ---
class ROI:
    def __init__(self, ypix, xpix, lam, med, do_crop):
        self.ypix = ypix
        self.xpix = xpix
        self.lam = lam
        self.med = med
        self.do_crop = do_crop
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
    rois = [ROI(ypix=s["ypix"], xpix=s["xpix"], lam=s["lam"], med=s["med"], do_crop=do_crop) for s in stat]
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
    mrs_normeds = norm_by_average(values=np.array([s["mrs"] for s in stat]), estimator=np.nanmedian, offset=1e-10, first_n=100)
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
        rois = [ROI(ypix=s["ypix"], xpix=s["xpix"], lam=s["lam"], med=s["med"], do_crop=do_crop) for s in stat]
        for roi, s in zip(rois, stat):
            s["overlap"] = roi.get_overlap_image(n_overlaps)
    return stat

roi_stats = patched_roi_stats
print("[DEBUG] Using patched roi_stats:", roi_stats.__name__)

# --- ModeHelpers for Dendrites/Axons Mode ---
class ModeHelpers:
    @staticmethod
    def get_next_parent_dendrite_id(roi_data):
        ids = [info["roi-type"][1] for info in roi_data.values() if info["roi-type"][0] == 0]
        return max(ids)+1 if ids else 1
    @staticmethod
    def get_next_dendritic_spine_id(roi_data, parent_dendrite_id):
        ids = [info["roi-type"][2] for info in roi_data.values() if info["roi-type"][0] == 1 and info["roi-type"][1] == parent_dendrite_id]
        return max(ids)+1 if ids else 1
    @staticmethod
    def get_next_parent_axon_id(roi_data):
        ids = [info["roi-type"][3] for info in roi_data.values() if info["roi-type"][0] == 2]
        return max(ids)+1 if ids else 1
    @staticmethod
    def get_next_axonal_bouton_id(roi_data, parent_axon_id):
        ids = [info["roi-type"][4] for info in roi_data.values() if info["roi-type"][0] == 3 and info["roi-type"][3] == parent_axon_id]
        return max(ids)+1 if ids else 1

# --- ROIItem Class ---
class ROIItem(QGraphicsPolygonItem):
    def __init__(self, roi_id, polygon, roi_info, main_window, parent=None):
        super(ROIItem, self).__init__(polygon, parent)
        self.roi_id = roi_id
        self.roi_info = roi_info
        self.main_window = main_window
        if self.main_window.mode == "normal":
            base_color = {0: Qt.blue, 1: Qt.red, 2: Qt.green}.get(self.roi_info["roi-type"][0], Qt.gray)
        else:
            base_color = {0: Qt.blue, 1: Qt.red, 2: Qt.green, 3: Qt.magenta}.get(self.roi_info["roi-type"][0], Qt.gray)
        color = QColor(base_color)
        color.setAlpha(64)
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
        r = 1
        for pt in self.polygon():
            rect = QRectF(pt.x()-r, pt.y()-r, 2*r, 2*r)
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
        num, ok = QInputDialog.getInt(None, "Edit ROI", "Enter new number of sides (3-10):", min=3, max=10, value=len(self.roi_info.get("ROI coordinates", [])))
        if ok:
            confirm = QMessageBox.question(None, "Confirm Edit", "ROI shape will be changed to a regular polygon. Proceed?", QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                pts = self.roi_info.get("ROI coordinates", [])
                if pts.size == 0:
                    return
                xs = pts[:,0]
                ys = pts[:,1]
                center = ((xs.min()+xs.max())/2, (ys.min()+ys.max())/2)
                radius = min(xs.max()-xs.min(), ys.max()-ys.min())/2
                new_pts = np.array([(center[0] + radius*np.cos(2*np.pi*i/num),
                                     center[1] + radius*np.sin(2*np.pi*i/num))
                                    for i in range(num)])
                self.roi_info["ROI coordinates"] = new_pts
                new_poly = QPolygonF([QPointF(x,y) for x,y in new_pts])
                self.setPolygon(new_poly)
                self.update_vertex_markers()
    def delete_roi(self):
        assoc_ids = []
        typ = self.roi_info["roi-type"][0]
        if self.main_window.mode == "normal":
            cellID = self.roi_info["roi-type"][1]
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
                    if rtype == 2 and cid == cellID and pid == self.roi_info["roi-type"][2]:
                        assoc_ids.append(rid)
        else:
            if typ == 1:
                parent_dend_id = self.roi_info["roi-type"][1]
                for rid, info in self.main_window.roi_data.items():
                    if rid == self.roi_id:
                        continue
                    if info["roi-type"][0] == 1 and info["roi-type"][1] == parent_dend_id:
                        assoc_ids.append(rid)
            elif typ == 3:
                parent_axon_id = self.roi_info["roi-type"][3]
                for rid, info in self.main_window.roi_data.items():
                    if rid == self.roi_id:
                        continue
                    if info["roi-type"][0] == 3 and info["roi-type"][3] == parent_axon_id:
                        assoc_ids.append(rid)
        if assoc_ids:
            confirm = QMessageBox.question(None, "Delete ROI", "Deleting this ROI will also delete its associated ROIs.\nProceed?", QMessageBox.Yes | QMessageBox.No)
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
        delta = event.angleDelta().y()/120
        factor = 1.1**delta
        new_scale = self.current_scale * factor
        if new_scale < 1.0:
            factor = 1.0/self.current_scale
            self.current_scale = 1.0
        elif new_scale > 5.0:
            factor = 5.0/self.current_scale
            self.current_scale = 5.0
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
                    tracing_color = QColor(255, 0, 255, 64)
                    self.parent_window.tracing_polygon_item.setPen(QPen(tracing_color, 2, Qt.DashLine))
                    self.scene().addItem(self.parent_window.tracing_polygon_item)
                    r = 1
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
                    temp_color = QColor(255, 0, 0, 64)
                    self.temp_polygon_item.setPen(QPen(temp_color, 2, Qt.DashLine))
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

# --- ROITableWindow, ConfirmROITableDialog, ConversionTableDialog Classes ---
class ROITableWindow(QDialog):
    def __init__(self, roi_data, parent=None):
        super(ROITableWindow, self).__init__(parent)
        self.setWindowTitle("ROIs Table")
        self.resize(800,300)
        self.roi_data = roi_data
        self.main_window = parent
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout()
        self.table = QTableWidget()
        if self.main_window.mode == "dendrites_axons":
            columns = ["ROI #", "ROI Type", "Parent Dendrite ID", "Dendritic Spine ID", "Parent Axon ID", "Axonal Bouton ID", "Plane", "ROI Coordinates"]
        else:
            columns = ["ROI #", "ROI Type", "Cell ID", "Parent Dendrite ID", "Dendritic Spine ID", "Plane", "ROI Coordinates"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.populate_table()
        self.table.cellClicked.connect(self.row_clicked)
        layout.addWidget(self.table)
        self.setLayout(layout)
    def populate_table(self):
        self.table.setRowCount(len(self.roi_data))
        for row, roi_id in enumerate(sorted(self.roi_data.keys())):
            info = self.roi_data[roi_id]
            if self.main_window.mode == "dendrites_axons":
                typ, pd, ds, pa, ab = info["roi-type"]
                typ_str = {0:"Parent Dendrite", 1:"Dendritic Spine", 2:"Parent Axon", 3:"Axonal Bouton"}.get(typ, "Unknown")
                items = [
                    QTableWidgetItem(str(roi_id)),
                    QTableWidgetItem(typ_str),
                    QTableWidgetItem(str(pd)),
                    QTableWidgetItem(str(ds)),
                    QTableWidgetItem(str(pa)),
                    QTableWidgetItem(str(ab)),
                    QTableWidgetItem(str(info["plane"])),
                    QTableWidgetItem(str(info["ROI coordinates"].tolist() if hasattr(info["ROI coordinates"],"tolist") else info["ROI coordinates"]))
                ]
            else:
                typ, cellID, parentID, spineID = info["roi-type"]
                typ_str = {0:"Cell", 1:"Parent Dendrite", 2:"Dendritic Spine"}.get(typ, "Unknown")
                items = [
                    QTableWidgetItem(str(roi_id)),
                    QTableWidgetItem(typ_str),
                    QTableWidgetItem(str(cellID)),
                    QTableWidgetItem(str(parentID)),
                    QTableWidgetItem(str(spineID)),
                    QTableWidgetItem(str(info["plane"])),
                    QTableWidgetItem(str(info["ROI coordinates"].tolist() if hasattr(info["ROI coordinates"],"tolist") else info["ROI coordinates"]))
                ]
            for col, item in enumerate(items):
                self.table.setItem(row, col, item)
    def row_clicked(self, row, col):
        text = self.table.item(row, 0).text()
        try:
            roi_number = int(text)
        except ValueError:
            print(f"[DEBUG] Invalid ROI key: {text}")
            return
        self.main_window.highlight_roi(roi_number)

class ConfirmROITableDialog(QDialog):
    def __init__(self, roi_data, parent=None):
        super(ConfirmROITableDialog, self).__init__(parent)
        self.setWindowTitle("Confirm ROI Addition")
        self.resize(800,300)
        self.roi_data = roi_data
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout()
        self.table = QTableWidget()
        if self.parent().mode == "dendrites_axons":
            columns = ["ROI #", "ROI Type", "Parent Dendrite ID", "Dendritic Spine ID", "Parent Axon ID", "Axonal Bouton ID", "Plane", "ROI Coordinates"]
        else:
            columns = ["ROI #", "ROI Type", "Cell ID", "Parent Dendrite ID", "Dendritic Spine ID", "Plane", "ROI Coordinates"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
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
            if self.parent().mode == "dendrites_axons":
                typ, pd, ds, pa, ab = info["roi-type"]
                typ_str = {0:"Parent Dendrite", 1:"Dendritic Spine", 2:"Parent Axon", 3:"Axonal Bouton"}.get(typ, "Unknown")
                items = [
                    QTableWidgetItem(str(roi_id)),
                    QTableWidgetItem(typ_str),
                    QTableWidgetItem(str(pd)),
                    QTableWidgetItem(str(ds)),
                    QTableWidgetItem(str(pa)),
                    QTableWidgetItem(str(ab)),
                    QTableWidgetItem(str(info["plane"])),
                    QTableWidgetItem(str(info["ROI coordinates"].tolist() if hasattr(info["ROI coordinates"],"tolist") else info["ROI coordinates"]))
                ]
            else:
                typ, cellID, parentID, spineID = info["roi-type"]
                typ_str = {0:"Cell", 1:"Parent Dendrite", 2:"Dendritic Spine"}.get(typ, "Unknown")
                items = [
                    QTableWidgetItem(str(roi_id)),
                    QTableWidgetItem(typ_str),
                    QTableWidgetItem(str(cellID)),
                    QTableWidgetItem(str(parentID)),
                    QTableWidgetItem(str(spineID)),
                    QTableWidgetItem(str(info["plane"])),
                    QTableWidgetItem(str(info["ROI coordinates"].tolist() if hasattr(info["ROI coordinates"],"tolist") else info["ROI coordinates"]))
                ]
            for col, item in enumerate(items):
                self.table.setItem(row, col, item)

class ConversionTableDialog(QDialog):
    def __init__(self, conversion_dict, parent=None):
        super(ConversionTableDialog, self).__init__(parent)
        self.setWindowTitle("ROIs Conversion Table")
        self.resize(900,300)
        self.conversion_dict = conversion_dict
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout()
        self.table = QTableWidget()
        if self.parent().mode == "dendrites_axons":
            columns = ["ROI #", "ROI Type", "Parent Dendrite ID", "Dendritic Spine ID", "Parent Axon ID", "Axonal Bouton ID", "Plane", "ROI Coordinates", "Conversion", "Conversion Index"]
        else:
            columns = ["ROI #", "ROI Type", "Cell ID", "Parent Dendrite ID", "Dendritic Spine ID", "Plane", "ROI Coordinates", "Conversion", "Conversion Index"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.populate_table()
        layout.addWidget(self.table)
        self.setLayout(layout)
    def populate_table(self):
        self.table.setRowCount(len(self.conversion_dict))
        for row, roi_id in enumerate(sorted(self.conversion_dict.keys())):
            info = self.conversion_dict[roi_id]
            if self.parent().mode == "dendrites_axons":
                typ, pd, ds, pa, ab = info["roi-type"]
                typ_str = {0:"Parent Dendrite", 1:"Dendritic Spine", 2:"Parent Axon", 3:"Axonal Bouton"}.get(typ, "Unknown")
                items = [
                    QTableWidgetItem(str(roi_id)),
                    QTableWidgetItem(typ_str),
                    QTableWidgetItem(str(pd)),
                    QTableWidgetItem(str(ds)),
                    QTableWidgetItem(str(pa)),
                    QTableWidgetItem(str(ab)),
                    QTableWidgetItem(str(info["plane"])),
                    QTableWidgetItem(str(info["ROI coordinates"].tolist() if hasattr(info["ROI coordinates"],"tolist") else info["ROI coordinates"])),
                    QTableWidgetItem(str(info.get("conversion", ["N/A", "N/A"]))),
                    QTableWidgetItem(str(info.get("conversion index", "N/A")))
                ]
            else:
                typ, cellID, parentID, spineID = info["roi-type"]
                typ_str = {0:"Cell", 1:"Parent Dendrite", 2:"Dendritic Spine"}.get(typ, "Unknown")
                items = [
                    QTableWidgetItem(str(roi_id)),
                    QTableWidgetItem(typ_str),
                    QTableWidgetItem(str(cellID)),
                    QTableWidgetItem(str(parentID)),
                    QTableWidgetItem(str(spineID)),
                    QTableWidgetItem(str(info["plane"])),
                    QTableWidgetItem(str(info["ROI coordinates"].tolist() if hasattr(info["ROI coordinates"],"tolist") else info["ROI coordinates"])),
                    QTableWidgetItem(str(info.get("conversion", ["N/A", "N/A"]))),
                    QTableWidgetItem(str(info.get("conversion index", "N/A")))
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
        if self.parent().mode == "normal":
            self.type_combo.addItem("Cell (no association needed)", 0)
            self.type_combo.addItem("Parent Dendrite (associate with a Cell)", 1)
            self.type_combo.addItem("Dendritic Spine (associate with a Parent Dendrite)", 2)
        else:
            self.type_combo.addItem("Parent Dendrite", 0)
            self.type_combo.addItem("Dendritic Spine (associate with a Parent Dendrite)", 1)
            self.type_combo.addItem("Parent Axon", 2)
            self.type_combo.addItem("Axonal Bouton (associate with a Parent Axon)", 3)
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
        if self.parent().mode == "normal":
            if roi_type == 1:
                if not existing_cells:
                    QMessageBox.warning(self, "No Cells", "No cell ROIs available.")
                    self.reject()
                    return
                sorted_cells = sorted(existing_cells)
                for key in sorted_cells:
                    self.assoc_combo.addItem(f"Cell ROI #{key}", key)
                self.assoc_combo.show()
                self.assoc_combo.setCurrentIndex(len(sorted_cells)-1)
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
                self.assoc_combo.setCurrentIndex(len(sorted_parents)-1)
                self.highlight_association()
            else:
                self.assoc_combo.hide()
        else:
            if roi_type == 1:
                available = [roi_id for roi_id, info in self.parent().roi_data.items() 
                             if info["plane"] == self.parent().plane_order[self.parent().current_plane_index] 
                             and info["roi-type"][0] == 0]
                if not available:
                    QMessageBox.warning(self, "No Parent Dendrites", "No Parent Dendrite ROIs available.")
                    self.reject()
                    return
                for key in sorted(available):
                    self.assoc_combo.addItem(f"Parent Dendrite ROI #{key}", key)
                self.assoc_combo.show()
                self.assoc_combo.setCurrentIndex(len(available)-1)
                self.highlight_association()
            elif roi_type == 3:
                available = [roi_id for roi_id, info in self.parent().roi_data.items() 
                             if info["plane"] == self.parent().plane_order[self.parent().current_plane_index] 
                             and info["roi-type"][0] == 2]
                if not available:
                    QMessageBox.warning(self, "No Parent Axons", "No Parent Axon ROIs available.")
                    self.reject()
                    return
                for key in sorted(available):
                    self.assoc_combo.addItem(f"Parent Axon ROI #{key}", key)
                self.assoc_combo.show()
                self.assoc_combo.setCurrentIndex(len(available)-1)
                self.highlight_association()
            else:
                self.assoc_combo.hide()
    def highlight_association(self):
        if self.assoc_combo.isVisible():
            assoc_key = self.assoc_combo.currentData()
            if self.parent():
                if assoc_key in self.parent().roi_data:
                    self.parent().highlight_roi(assoc_key)
                else:
                    self.parent().clear_highlight()
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
        self.shape_combo.addItem("Tracing", "tracing")
        self.shape_combo.addItem("Rectangle", "rectangle")
        self.shape_combo.addItem("Other Polygon", "polygon")
        layout.addRow("ROI Shape:", self.shape_combo)
        self.sides_input = QLineEdit()
        self.sides_input.setPlaceholderText("Enter number of sides (3-10)")
        self.sides_input.setEnabled(self.shape_combo.currentData()=="polygon")
        layout.addRow("Polygon sides:", self.sides_input)
        self.shape_combo.currentIndexChanged.connect(lambda: self.sides_input.setEnabled(self.shape_combo.currentData()=="polygon"))
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
            except Exception:
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
        self.plane_data = OrderedDict()  # {plane: {"meanImg":..., "meanImgE":..., "yrange":..., "xrange":..., "folder":...}}
        self.plane_order = []
        self.current_plane_index = 0
        self.current_valid_rect = None
        self.image_width = None
        self.image_height = None
        self.roi_data = {}    # ROI key -> ROI info
        self.roi_items = {}   # ROI key -> ROIItem
        self.next_roi_id = 0
        self.pending_roi_type = None  # (roi_type, associated ROI key)
        self.pending_roi_shape = None  # "rectangle", "polygon", or "tracing"
        self.pending_polygon_sides = None
        self.tracing_vertices = []
        self.tracing_polygon_item = None
        self.tracing_markers = []
        self.current_meanImg = None  # The image currently displayed
        self.current_view = "Mean Image"  # or "Mean Image Enhanced"
        self.mode = "normal"  # or "dendrites_axons"
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
        self.mode_toggle = QPushButton("Normal Mode")
        self.mode_toggle.setCheckable(True)
        self.mode_toggle.clicked.connect(self.toggle_mode)
        self.current_plane_label = QLabel("Current plane: N/A")
        self.contrast_label = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(50)
        self.contrast_slider.setMaximum(200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        self.view_button_group = QButtonGroup(self)
        self.radio_mean = QRadioButton("Mean Image")
        self.radio_mean_enhanced = QRadioButton("Mean Image Enhanced")
        self.radio_mean.setChecked(True)  # Default selection.
        self.view_button_group.addButton(self.radio_mean, 0)
        self.view_button_group.addButton(self.radio_mean_enhanced, 1)
        self.radio_mean.toggled.connect(self.update_view)
        self.radio_mean_enhanced.toggled.connect(self.update_view)
        view_layout = QHBoxLayout()
        view_layout.addWidget(self.radio_mean)
        view_layout.addWidget(self.radio_mean_enhanced)
        view_layout.addStretch()
        self.coord_label = QLabel("X : Out of range\nY : Out of range")
        self.coord_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        bottom_layout.addWidget(self.current_plane_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.contrast_label)
        bottom_layout.addWidget(self.contrast_slider)
        bottom_layout.addLayout(view_layout)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.coord_label)
        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.add_roi_button)
        top_layout.addWidget(self.roi_table_button)
        top_layout.addWidget(self.extract_button)
        top_layout.addWidget(self.clear_rois_button)
        top_layout.addWidget(self.left_arrow_button)
        top_layout.addWidget(self.right_arrow_button)
        top_layout.addStretch()
        top_layout.addWidget(self.mode_toggle)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.view)
        main_layout.addLayout(bottom_layout)
        central_widget.setLayout(main_layout)
    def toggle_mode(self):
        if self.mode_toggle.isChecked():
            reply = QMessageBox.question(self, "Switch Mode", 
                "Would you like to switch to the dendrites/axons mode?\nWARNING: Use only if you’re working with DENDRITES or AXONS!!",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.mode = "dendrites_axons"
                self.mode_toggle.setText("Dendrites/Axons Mode")
            else:
                self.mode_toggle.setChecked(False)
        else:
            reply = QMessageBox.question(self, "Switch Mode", "Switch back to Normal Mode?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.mode = "normal"
                self.mode_toggle.setText("Normal Mode")
            else:
                self.mode_toggle.setChecked(True)
        self.load_existing_rois_mode()
    def load_existing_rois_mode(self):
        # This function checks the SpinesGUI folder for saved ROI files and
        # always informs the user which mode was detected (even if it matches the current mode).
        if not self.root_folder:
            return
        spines_gui_folder = os.path.join(self.root_folder, "SpinesGUI")
        if not os.path.exists(spines_gui_folder):
            return

        roi_filename_normal = "ROIs.npy"
        roi_filename_dendrites = "ROIs_dendrite_axon_mode.npy"

        rois_file_normal = os.path.join(spines_gui_folder, roi_filename_normal)
        rois_file_dendrites = os.path.join(spines_gui_folder, roi_filename_dendrites)

        # Decide which file to load based on the current mode and file existence.
        if self.mode == "normal":
            if os.path.exists(rois_file_normal):
                # Even if we're in normal mode, show an info message.
                QMessageBox.information(self, "Mode Detected", 
                    "Normal mode ROI file detected. Loading Normal mode ROIs.")
                rois_file = rois_file_normal
            elif os.path.exists(rois_file_dendrites):
                print("[DEBUG] Normal mode selected but dendrites/axons ROI file found; switching mode.", flush=True)
                QMessageBox.information(self, "Mode Detected", 
                    "Dendrites/Axons ROI file detected. Switching to Dendrites/Axons mode.")
                self.mode = "dendrites_axons"
                self.mode_toggle.setChecked(True)
                self.mode_toggle.setText("Dendrites/Axons Mode")
                rois_file = rois_file_dendrites
            else:
                rois_file = rois_file_normal  # No ROI file exists.
        else:  # self.mode == "dendrites_axons"
            if os.path.exists(rois_file_dendrites):
                QMessageBox.information(self, "Mode Detected", 
                    "Dendrites/Axons mode ROI file detected. Loading Dendrites/Axons mode ROIs.")
                rois_file = rois_file_dendrites
            elif os.path.exists(rois_file_normal):
                print("[DEBUG] Dendrites/Axons mode selected but normal ROI file found; switching mode.", flush=True)
                QMessageBox.information(self, "Mode Detected", 
                    "Normal mode ROI file detected. Switching to Normal mode.")
                self.mode = "normal"
                self.mode_toggle.setChecked(False)
                self.mode_toggle.setText("Normal Mode")
                rois_file = rois_file_normal
            else:
                rois_file = rois_file_dendrites  # No ROI file exists.

        # Attempt to load the ROI file.
        if os.path.exists(rois_file):
            try:
                loaded_rois = np.load(rois_file, allow_pickle=True).item()
                print("[DEBUG] Loaded ROIs file")
                # Ensure the ROI keys are integers.
                self.roi_data = {int(k): v for k, v in loaded_rois.items()}
                if self.roi_data:
                    self.next_roi_id = max(self.roi_data.keys()) + 1
            except Exception as e:
                print("[DEBUG] Error loading ROIs file:", e, flush=True)
        else:
            print("[DEBUG] ROI file not found at", rois_file, flush=True)

    def update_view(self):
        # Determine the current view based on which radio button is checked.
        if self.radio_mean.isChecked():
            self.current_view = "Mean Image"
        else:
            self.current_view = "Mean Image Enhanced"
        print(f"[DEBUG] Current view: {self.current_view}", flush=True)
        # Update self.current_meanImg based on the selected view and current plane.
        if self.plane_order:
            plane_num = self.plane_order[self.current_plane_index]
            plane = self.plane_data[plane_num]
            if self.current_view == "Mean Image Enhanced" and plane.get("meanImgE") is not None:
                self.current_meanImg = plane["meanImgE"]
            else:
                self.current_meanImg = plane["meanImg"]
        # Preserve the current transformation and update the contrast.
        current_transform = self.view.transform()
        self.update_contrast()
        self.view.setTransform(current_transform)

    def update_contrast(self):
        if self.current_meanImg is None:
            return
        mimg = self.current_meanImg.astype(np.float32)
        mimg1 = np.percentile(mimg, 1)
        mimg99 = np.percentile(mimg, 99)
        mimg_disp = (mimg - mimg1) / (mimg99 - mimg1)
        mimg_disp = np.clip(mimg_disp, 0, 1)
        factor = self.contrast_slider.value() / 100.0
        mimg_disp = 0.5 + factor * (mimg_disp - 0.5)
        mimg_disp = np.clip(mimg_disp, 0, 1)
        mimg_disp = (mimg_disp * 255).astype(np.uint8)
        height, width = mimg_disp.shape
        image = QImage(mimg_disp.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image)
        try:
            # Try to access the scene safely. If the item is deleted, this will raise a RuntimeError.
            if self.image_pixmap_item is None:
                raise RuntimeError("Pixmap item is None")
            scene = self.image_pixmap_item.scene()
            if scene is None:
                raise RuntimeError("Pixmap item has no scene")
            self.image_pixmap_item.setPixmap(pixmap)
        except RuntimeError:
            # If the pixmap item was deleted, re-create it.
            self.image_pixmap_item = QGraphicsPixmapItem(pixmap)
            self.image_pixmap_item.setZValue(0)  # Ensure it is at the back so that ROI items (with higher Z) show on top.
            self.graphics_scene.addItem(self.image_pixmap_item)
    def load_suite2p_folder(self):
        print("[DEBUG] Entering load_suite2p_folder()", flush=True)
        folder = QFileDialog.getExistingDirectory(self, "Select Root Folder", os.getcwd())
        if not folder:
            return
        if self.root_folder is not None:
            reply = QMessageBox.question(self, "Change Folder", "Loading a new folder will clear current data.\nProceed?", QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
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
                        meanImgE = ops.get("meanImgE", None)
                        yrange = ops.get("yrange", None)
                        xrange_ = ops.get("xrange", None)
                        if meanImg is None or yrange is None or xrange_ is None:
                            continue
                        if np.isnan(meanImg).any():
                            continue
                        self.plane_data[plane_num] = {"meanImg": meanImg, "meanImgE": meanImgE, "yrange": yrange, "xrange": xrange_, "folder": subfolder}
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
        roi_filename = "ROIs.npy" if self.mode=="normal" else "ROIs_dendrite_axon_mode.npy"
        rois_file = os.path.join(spines_gui_folder, roi_filename)
        if os.path.exists(rois_file):
            try:
                loaded_rois = np.load(rois_file, allow_pickle=True).item()
                print("[DEBUG] Loaded ROIs file content")
                self.roi_data = {int(k): v for k, v in loaded_rois.items()}
                if self.roi_data:
                    self.next_roi_id = max(self.roi_data.keys()) + 1
            except Exception as e:
                print(f"[DEBUG] Error loading ROIs file: {e}", flush=True)
        else:
            print("[DEBUG] ROI file not found at", rois_file, flush=True)
        self.load_existing_rois_mode()
        self.update_plane_display()
    def update_plane_display(self):
        if not self.plane_order:
            return
        self.graphics_scene.clear()
        self.roi_items.clear()
        plane_num = self.plane_order[self.current_plane_index]
        print(f"[DEBUG] Checking plane_num: {plane_num}", flush=True)
        plane = self.plane_data[plane_num]
        if self.current_view == "Mean Image Enhanced" and plane.get("meanImgE") is not None:
            self.current_meanImg = plane["meanImgE"]
        else:
            self.current_meanImg = plane["meanImg"]
        self.update_contrast()
        yrange = plane["yrange"]
        xrange_ = plane["xrange"]
        height, width = self.current_meanImg.shape
        self.image_height = height
        self.image_width = width
        valid_rect = QRectF(xrange_[0], yrange[0], xrange_[1]-xrange_[0], yrange[1]-yrange[0])
        self.current_valid_rect = valid_rect
        rect_item = self.graphics_scene.addRect(valid_rect, QPen(Qt.yellow, 2))
        rect_item.setZValue(1)
        self.current_plane_label.setText(f"Current plane: {plane_num}")
        for roi_id, info in self.roi_data.items():
            if info.get("plane") == plane_num:
                pts = info["ROI coordinates"]
                poly = QPolygonF([QPointF(x, y) for x, y in pts])
                roi_item = ROIItem(roi_id, poly, info, self)
                roi_item.setZValue(2)
                self.graphics_scene.addItem(roi_item)
                self.roi_items[roi_id] = roi_item
        # Fit view if needed (optional)
        if self.image_pixmap_item:
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
        if self.mode == "normal":
            existing_cells = [roi_id for roi_id, info in self.roi_data.items() if info["roi-type"][0] == 0]
            existing_parents = [roi_id for roi_id, info in self.roi_data.items() if info["roi-type"][0] == 1]
        else:
            existing_cells = []
            existing_parents = [roi_id for roi_id, info in self.roi_data.items() if info["plane"] == self.plane_order[self.current_plane_index] and info["roi-type"][0] == 0]
        type_dialog = ROITypeDialog(existing_cells, existing_parents, self)
        if type_dialog.exec_() != QDialog.Accepted:
            return
        roi_type_choice, assoc_key = type_dialog.get_values()
        self.pending_roi_type = (roi_type_choice, assoc_key)
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
        QMessageBox.information(self, "ROI Drawing", "Click within the valid ROI area to define your ROI.")
        self.view.drawing_roi = True
        self.view.setDragMode(QGraphicsView.NoDrag)
    def get_next_cell_id(self):
        cell_ids = [info["roi-type"][1] for info in self.roi_data.values() if info["roi-type"][0]==0]
        return max(cell_ids)+1 if cell_ids else 1
    def get_next_parent_id(self, cell_id):
        parent_ids = [info["roi-type"][2] for info in self.roi_data.values() if info["roi-type"][0]==1 and info["roi-type"][1]==cell_id]
        return max(parent_ids)+1 if parent_ids else 1
    def get_next_spine_id(self, cell_id, parent_id):
        spine_ids = [info["roi-type"][3] for info in self.roi_data.values() if info["roi-type"][0]==2 and info["roi-type"][1]==cell_id and info["roi-type"][2]==parent_id]
        return max(spine_ids)+1 if spine_ids else 1
    def get_next_parent_dendrite_id(self):
        return ModeHelpers.get_next_parent_dendrite_id(self.roi_data)
    def get_next_dendritic_spine_id(self, parent_dendrite_id):
        return ModeHelpers.get_next_dendritic_spine_id(self.roi_data, parent_dendrite_id)
    def get_next_parent_axon_id(self):
        return ModeHelpers.get_next_parent_axon_id(self.roi_data)
    def get_next_axonal_bouton_id(self, parent_axon_id):
        return ModeHelpers.get_next_axonal_bouton_id(self.roi_data, parent_axon_id)
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
            radius = min(right-left, bottom-top)/2
            pts = np.array([(center[0]+radius*np.cos(2*np.pi*i/num_sides),
                             center[1]+radius*np.sin(2*np.pi*i/num_sides))
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
        tracing_color = QColor(255, 0, 255, 64)
        self.tracing_polygon_item.setPen(QPen(tracing_color, 2, Qt.DashLine))
        self.graphics_scene.addItem(self.tracing_polygon_item)
        if self.tracing_markers:
            for marker in self.tracing_markers:
                self.graphics_scene.removeItem(marker)
        self.tracing_markers = []
        for i, pt in enumerate(self.tracing_vertices):
            r = 1
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
        current_plane = self.plane_order[self.current_plane_index]
        if self.mode == "normal":
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
                associated_cell_key = self.pending_roi_type[1]
                cell_roi = self.roi_data.get(associated_cell_key)
                if cell_roi is None:
                    QMessageBox.warning(self, "Error", "Associated cell ROI not found.")
                    return
                cell_id = cell_roi["roi-type"][1]
                parent_id = self.get_next_parent_id(cell_id)
                spine_id = self.get_next_spine_id(cell_id, parent_id)
                roi_type_list = [2, cell_id, parent_id, spine_id]
            else:
                roi_type_list = [self.pending_roi_type[0], 0, 0, 0]
        else:
            if self.pending_roi_type[0] == 0:
                pd_id = self.get_next_parent_dendrite_id()
                roi_type_list = [0, pd_id, 0, 0, 0]
            elif self.pending_roi_type[0] == 1:
                associated_parent_key = self.pending_roi_type[1]
                parent_roi = self.roi_data.get(associated_parent_key)
                if parent_roi is None or parent_roi["roi-type"][0] != 0:
                    QMessageBox.warning(self, "Error", "Selected associated ROI is not a Parent Dendrite.")
                    return
                pd_id = parent_roi["roi-type"][1]
                ds_id = self.get_next_dendritic_spine_id(pd_id)
                roi_type_list = [1, pd_id, ds_id, 0, 0]
            elif self.pending_roi_type[0] == 2:
                pa_id = self.get_next_parent_axon_id()
                roi_type_list = [2, 0, 0, pa_id, 0]
            elif self.pending_roi_type[0] == 3:
                associated_parent_key = self.pending_roi_type[1]
                parent_roi = self.roi_data.get(associated_parent_key)
                if parent_roi is None or parent_roi["roi-type"][0] != 2:
                    QMessageBox.warning(self, "Error", "Selected associated ROI is not a Parent Axon.")
                    return
                pa_id = parent_roi["roi-type"][3]
                ab_id = self.get_next_axonal_bouton_id(pa_id)
                roi_type_list = [3, 0, 0, pa_id, ab_id]
            else:
                roi_type_list = [self.pending_roi_type[0], 0, 0, 0, 0]
        roi_info = {"roi-type": roi_type_list, "plane": current_plane, "ROI coordinates": pts}
        poly = QPolygonF([QPointF(x, y) for x, y in pts])
        roi_item = ROIItem(self.next_roi_id, poly, roi_info, self)
        roi_item.setZValue(2)
        self.graphics_scene.addItem(roi_item)
        self.roi_items[self.next_roi_id] = roi_item
        self.roi_data[self.next_roi_id] = roi_info
        review_dialog = ConfirmROITableDialog(self.roi_data, self)
        if review_dialog.exec_() != QDialog.Accepted:
            self.graphics_scene.removeItem(roi_item)
            return
        confirm = QMessageBox.question(self, "Confirm ROI", f"Did details get stored correctly for ROI #{self.next_roi_id}?",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.No:
            self.graphics_scene.removeItem(roi_item)
        else:
            self.next_roi_id += 1
            self.save_rois()
    def save_rois(self):
        if self.root_folder is None:
            return
        spines_gui_folder = os.path.join(self.root_folder, "SpinesGUI")
        roi_filename = "ROIs.npy" if self.mode=="normal" else "ROIs_dendrite_axon_mode.npy"
        rois_file = os.path.join(spines_gui_folder, roi_filename)
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
                if self.mode=="normal":
                    base_color = {0: Qt.blue, 1: Qt.red, 2: Qt.green}.get(typ, Qt.gray)
                else:
                    base_color = {0: Qt.blue, 1: Qt.red, 2: Qt.green, 3: Qt.magenta}.get(typ, Qt.gray)
                color = QColor(base_color)
                color.setAlpha(64)
                item.setPen(QPen(color, 2))
    def clear_highlight(self):
        for r_id, item in self.roi_items.items():
            typ = self.roi_data[r_id]["roi-type"][0]
            if self.mode=="normal":
                base_color = {0: Qt.blue, 1: Qt.red, 2: Qt.green}.get(typ, Qt.gray)
            else:
                base_color = {0: Qt.blue, 1: Qt.red, 2: Qt.green, 3: Qt.magenta}.get(typ, Qt.gray)
            color = QColor(base_color)
            color.setAlpha(64)
            item.setPen(QPen(color, 2))
    def remove_roi(self, roi_id):
        if roi_id in self.roi_items:
            self.graphics_scene.removeItem(self.roi_items[roi_id])
            del self.roi_items[roi_id]
        if roi_id in self.roi_data:
            del self.roi_data[roi_id]
        self.save_rois()
    def clear_all_rois(self):
        confirm = QMessageBox.question(self, "Clear All ROIs", "Are you sure you want to clear all ROIs?",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            for item in list(self.roi_items.values()):
                self.graphics_scene.removeItem(item)
            self.roi_items.clear()
            self.roi_data.clear()
            self.next_roi_id = 0
            self.save_rois()
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Exit Confirmation", "Do you wish to exit?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    def extract_rois(self):
        print("[DEBUG] Starting extraction process", flush=True)
        if self.root_folder is None:
            QMessageBox.warning(self, "Error", "No root folder loaded.")
            return

        spines_gui_folder = os.path.join(self.root_folder, "SpinesGUI")
        if not os.path.exists(spines_gui_folder):
            os.makedirs(spines_gui_folder)

        # Define log and success file paths.
        log_file = os.path.join(spines_gui_folder, "extraction_log.txt")
        success_file = os.path.join(spines_gui_folder, "extraction_successfull.txt")

        # Open (or create) the extraction log file for appending.
        with open(log_file, "a") as log:
            log.write("Extraction process started\n")

        # Check if re-extraction is needed (based on presence of stat1 file).
        reextract = False
        for plane in self.plane_data.keys():
            plane_folder = os.path.join(spines_gui_folder, f"plane{plane}")
            stat1_filename = "stat1.npy" if self.mode == "normal" else "stat1_dendrite_axon_mode.npy"
            stat1_file = os.path.join(plane_folder, stat1_filename)
            if os.path.exists(stat1_file):
                reextract = True
                break

        # If reextracting, build the confirmation message based on whether the previous extraction was successful.
        if reextract:
            if os.path.exists(success_file):
                msg = ("Previous extraction was successful. "
                    "Do you want to re-run extraction? This will delete previous extraction files "
                    "(except data.bin, data_chan2.bin, and logs).")
            else:
                msg = ("Previous extraction was not successful or not completed. "
                    "Do you want to re-run extraction? This will delete previous extraction files "
                    "(except data.bin, data_chan2.bin, and logs).")
            confirm = QMessageBox.question(self, "Re-run Extraction", msg,
                                        QMessageBox.Yes | QMessageBox.No)
            if confirm != QMessageBox.Yes:
                return
            # Delete extraction files for each plane (except data.bin, data_chan2.bin, and logs).
            for plane in self.plane_data.keys():
                plane_folder = os.path.join(spines_gui_folder, f"plane{plane}")
                for fname in ["stat0.npy", "stat1.npy", "stat.npy", "F.npy", "Fneu.npy",
                            "F_chan2.npy", "Fneu_chan2.npy", "spks.npy", "iscell.npy",
                            "stat0_dendrite_axon_mode.npy", "stat1_dendrite_axon_mode.npy"]:
                    fpath = os.path.join(plane_folder, fname)
                    if os.path.exists(fpath):
                        try:
                            os.remove(fpath)
                            print(f"[DEBUG] Deleted {fpath}", flush=True)
                            with open(log_file, "a") as log:
                                log.write(f"Deleted {fpath}\n")
                        except Exception as e:
                            print(f"[DEBUG] Error deleting {fpath}: {e}", flush=True)
                            with open(log_file, "a") as log:
                                log.write(f"Error deleting {fpath}: {e}\n")
            # Also delete the log files themselves.
            for fname in ["extraction_log.txt", "extraction_successfull.txt"]:
                fpath = os.path.join(spines_gui_folder, fname)
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                        print(f"[DEBUG] Deleted log file {fpath}", flush=True)
                        with open(log_file, "a") as log:
                            log.write(f"Deleted log file {fpath}\n")
                    except Exception as e:
                        print(f"[DEBUG] Error deleting log file {fpath}: {e}", flush=True)
                        with open(log_file, "a") as log:
                            log.write(f"Error deleting log file {fpath}: {e}\n")

        # For each plane, ensure that binary files are present.
        for plane in self.plane_data.keys():
            print(f"[DEBUG] Processing extraction for plane {plane}", flush=True)
            with open(log_file, "a") as log:
                log.write(f"Processing extraction for plane {plane}\n")
            plane_folder = os.path.join(spines_gui_folder, f"plane{plane}")
            if not os.path.exists(plane_folder):
                os.makedirs(plane_folder)

            # Check data.bin:
            data_bin_src = os.path.join(self.plane_data[plane]["folder"], "data.bin")
            data_bin_dest = os.path.join(plane_folder, "data.bin")
            if not os.path.exists(data_bin_dest):
                try:
                    shutil.copy(data_bin_src, data_bin_dest)
                    print(f"[DEBUG] Copied data.bin from {data_bin_src} to {data_bin_dest}", flush=True)
                    with open(log_file, "a") as log:
                        log.write(f"Copied data.bin from {data_bin_src} to {data_bin_dest}\n")
                except Exception as e:
                    print(f"[DEBUG] Error copying data.bin for plane {plane}: {e}", flush=True)
                    with open(log_file, "a") as log:
                        log.write(f"Error copying data.bin for plane {plane}: {e}\n")
                    continue
            else:
                print(f"[DEBUG] data.bin already exists in {plane_folder}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"data.bin already exists in {plane_folder}\n")

            # Check data_chan2.bin if it exists.
            data_chan2_src = os.path.join(self.plane_data[plane]["folder"], "data_chan2.bin")
            data_chan2_dest = os.path.join(plane_folder, "data_chan2.bin")
            if os.path.exists(data_chan2_src):
                if not os.path.exists(data_chan2_dest):
                    try:
                        shutil.copy(data_chan2_src, data_chan2_dest)
                        print(f"[DEBUG] Copied data_chan2.bin from {data_chan2_src} to {data_chan2_dest}", flush=True)
                        with open(log_file, "a") as log:
                            log.write(f"Copied data_chan2.bin from {data_chan2_src} to {data_chan2_dest}\n")
                    except Exception as e:
                        print(f"[DEBUG] Error copying data_chan2.bin for plane {plane}: {e}", flush=True)
                        with open(log_file, "a") as log:
                            log.write(f"Error copying data_chan2.bin for plane {plane}: {e}\n")
                        data_chan2_dest = None
                else:
                    print(f"[DEBUG] data_chan2.bin already exists in {plane_folder}", flush=True)
                    with open(log_file, "a") as log:
                        log.write(f"data_chan2.bin already exists in {plane_folder}\n")
            else:
                data_chan2_dest = None

            # Load ops from the copied ops.npy.
            ops_dest = os.path.join(plane_folder, "ops.npy")
            try:
                ops = np.load(ops_dest, allow_pickle=True).item()
                print(f"[DEBUG] Loaded copied ops.npy for plane {plane}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Loaded copied ops.npy for plane {plane}\n")
            except Exception as e:
                print(f"[DEBUG] Error loading copied ops.npy for plane {plane}: {e}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Error loading copied ops.npy for plane {plane}: {e}\n")
                continue

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
            print(f"[DEBUG] For plane {plane}, parameters: Ly={Ly}, Lx={Lx}, aspect={aspect}, diameter={diameter}, max_overlap={max_overlap}, do_crop={do_crop}", flush=True)
            with open(log_file, "a") as log:
                log.write(f"For plane {plane}, parameters: Ly={Ly}, Lx={Lx}, aspect={aspect}, diameter={diameter}, max_overlap={max_overlap}, do_crop={do_crop}\n")

            # Build stat0 from ROI data.
            stat0 = {}
            roi_list = [(k, roi) for k, roi in self.roi_data.items() if roi["plane"] == plane]
            roi_list_sorted = sorted(roi_list, key=lambda x: x[0])
            for idx, (roi_key, roi) in enumerate(roi_list_sorted):
                vertices = np.array(roi["ROI coordinates"])
                if vertices.size == 0:
                    print(f"[DEBUG] ROI {roi_key} on plane {plane} has no vertices.", flush=True)
                    with open(log_file, "a") as log:
                        log.write(f"ROI {roi_key} on plane {plane} has no vertices.\n")
                    continue
                x_min = int(np.floor(np.min(vertices[:, 0])))
                x_max = int(np.ceil(np.max(vertices[:, 0])))
                y_min = int(np.floor(np.min(vertices[:, 1])))
                y_max = int(np.ceil(np.max(vertices[:, 1])))
                xx, yy = np.meshgrid(np.arange(x_min, x_max+1), np.arange(y_min, y_max+1))
                points = np.vstack((xx.flatten(), yy.flatten())).T
                from matplotlib.path import Path
                poly_path = Path(vertices)
                inside = poly_path.contains_points(points)
                inside = inside.reshape(yy.shape)
                ypix = np.where(inside)[0] + y_min
                xpix = np.where(inside)[1] + x_min
                lam = np.ones(ypix.shape)
                stat0[idx] = {"ypix": np.array(ypix), "xpix": np.array(xpix), "lam": np.array(lam)}
                print(f"[DEBUG] Plane {plane}, ROI index {idx}: computed mask with {len(ypix)} pixels.", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Plane {plane}, ROI index {idx}: computed mask with {len(ypix)} pixels.\n")
            stat0_file = os.path.join(plane_folder, "stat0.npy")
            try:
                np.save(stat0_file, stat0)
                print(f"[DEBUG] Saved stat0.npy for plane {plane} in {plane_folder}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Saved stat0.npy for plane {plane} in {plane_folder}\n")
            except Exception as e:
                print(f"[DEBUG] Error saving stat0.npy for plane {plane}: {e}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Error saving stat0.npy for plane {plane}: {e}\n")
                continue

            stat0_list = list(stat0.values())
            try:
                print("[DEBUG] Calling roi_stats with patched roi_stats", flush=True)
                stat1 = roi_stats(stat0_list, Ly, Lx, aspect=aspect, diameter=diameter, max_overlap=max_overlap, do_crop=do_crop)
                stat1_filename = "stat1.npy" if self.mode=="normal" else "stat1_dendrite_axon_mode.npy"
                stat1_file = os.path.join(plane_folder, stat1_filename)
                np.save(stat1_file, stat1)
                print(f"[DEBUG] Saved stat1 for plane {plane} in {plane_folder}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Saved stat1 for plane {plane} in {plane_folder}\n")
            except Exception as e:
                print(f"[DEBUG] Error in roi_stats for plane {plane}: {e}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Error in roi_stats for plane {plane}: {e}\n")
                continue

            try:
                f_reg_data = BinaryFile(Ly, Lx, data_bin_dest, n_frames=ops.get("nframes"), dtype=ops.get("datatype", "int16"))
                print(f"[DEBUG] Loaded BinaryFile for data.bin for plane {plane}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Loaded BinaryFile for data.bin for plane {plane}\n")
            except Exception as e:
                print(f"[DEBUG] Error loading BinaryFile for data.bin for plane {plane}: {e}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Error loading BinaryFile for data.bin for plane {plane}: {e}\n")
                continue

            if data_chan2_dest is not None:
                try:
                    f_reg_chan2_data = BinaryFile(Ly, Lx, data_chan2_dest, n_frames=ops.get("nframes"), dtype=ops.get("datatype", "int16"))
                    print(f"[DEBUG] Loaded BinaryFile for data_chan2.bin for plane {plane}", flush=True)
                    with open(log_file, "a") as log:
                        log.write(f"Loaded BinaryFile for data_chan2.bin for plane {plane}\n")
                except Exception as e:
                    print(f"[DEBUG] Error loading BinaryFile for data_chan2.bin for plane {plane}: {e}", flush=True)
                    with open(log_file, "a") as log:
                        log.write(f"Error loading BinaryFile for data_chan2.bin for plane {plane}: {e}\n")
                    f_reg_chan2_data = None
            else:
                f_reg_chan2_data = None

            try:
                print(f"[DEBUG] Calling extraction_wrapper for plane {plane}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Calling extraction_wrapper for plane {plane}\n")
                
                # Redirect stdout to capture what extraction_wrapper prints.
                import io, sys
                old_stdout = sys.stdout
                sys.stdout = mystdout = io.StringIO()
                try:
                    outputs = extraction_wrapper(stat1, f_reg_data, f_reg_chan2_data, cell_masks=None, neuropil_masks=None, ops=ops)
                finally:
                    sys.stdout = old_stdout  # Always restore stdout
                extraction_printed = mystdout.getvalue()
                with open(log_file, "a") as log:
                    log.write("Extraction wrapper printed:\n" + extraction_printed + "\n")
                
                stat_out, F, Fneu, F_chan2, Fneu_chan2 = outputs
                with open(log_file, "a") as log:
                    log.write(f"Extraction complete for plane {plane}.\n")
            except Exception as e:
                print(f"[DEBUG] Error in extraction_wrapper for plane {plane}: {e}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Error in extraction_wrapper for plane {plane}: {e}\n")
                continue


            try:
                print(f"[DEBUG] Running spike deconvolution for plane {plane}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Running spike deconvolution for plane {plane}\n")
                dF = F.copy() - ops["neucoeff"] * Fneu
                dF = preprocess(F=dF, baseline=ops["baseline"], win_baseline=ops["win_baseline"],
                                sig_baseline=ops["sig_baseline"], fs=ops["fs"],
                                prctile_baseline=ops["prctile_baseline"])
                spks = oasis(F=dF, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"])
                spks_file = os.path.join(plane_folder, "spks.npy")
                np.save(spks_file, spks)
                print(f"[DEBUG] Saved spks.npy for plane {plane} in {plane_folder}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Saved spks.npy for plane {plane} in {plane_folder}\n")
            except Exception as e:
                print(f"[DEBUG] Error in spike deconvolution for plane {plane}: {e}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Error in spike deconvolution for plane {plane}: {e}\n")
            try:
                roi_ids = [roi_id for roi_id, info in self.roi_data.items() if info["plane"] == plane]
                iscell_arr = np.ones((len(roi_ids), 2), dtype=int)
                iscell_file = os.path.join(plane_folder, "iscell.npy")
                np.save(iscell_file, iscell_arr)
                print(f"[DEBUG] Saved iscell.npy for plane {plane} in {plane_folder}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Saved iscell.npy for plane {plane} in {plane_folder}\n")
            except Exception as e:
                print(f"[DEBUG] Error creating iscell.npy for plane {plane}: {e}", flush=True)
                with open(log_file, "a") as log:
                    log.write(f"Error creating iscell.npy for plane {plane}: {e}\n")
        # After processing all planes, create a success file.
        try:
            with open(success_file, "w") as sf:
                sf.write("Extraction finished successfully.\n")
            print(f"[DEBUG] Created extraction success file at {success_file}", flush=True)
            with open(log_file, "a") as log:
                log.write(f"Created extraction success file at {success_file}\n")
        except Exception as e:
            print(f"[DEBUG] Error creating extraction success file: {e}", flush=True)
            with open(log_file, "a") as log:
                log.write(f"Error creating extraction success file: {e}\n")
    # --- Build and display conversion dictionary ---
        conversion_dict = {}
        plane_groups = {}
        # Build conversion dictionary from self.roi_data.
        for key, roi in self.roi_data.items():
            # In dendrites/axons mode, ensure the roi-type list has 5 elements.
            if self.mode == "dendrites_axons" and len(roi["roi-type"]) < 5:
                roi["roi-type"] = roi["roi-type"] + [0]
            p = roi["plane"]
            plane_groups.setdefault(p, []).append((key, roi))
        for plane, items in plane_groups.items():
            items_sorted = sorted(items, key=lambda x: x[0])
            for idx, (roi_key, roi) in enumerate(items_sorted):
                roi["conversion"] = [plane, idx]
                conversion_dict[roi_key] = roi
        # Sort and add conversion index.
        sorted_conversion = sorted(conversion_dict.items(),
                                key=lambda x: (x[1]["conversion"][0], x[1]["conversion"][1]))
        for new_index, (roi_key, roi) in enumerate(sorted_conversion):
            roi["conversion index"] = new_index

        # Determine file name based on mode.
        conv_filename = "ROIs_conversion.npy" if self.mode == "normal" else "ROIs_dendrite_axon_mode_conversion.npy"
        rois_conv_file = os.path.join(spines_gui_folder, conv_filename)
        try:
            np.save(rois_conv_file, conversion_dict)
            print(f"[DEBUG] Saved conversion dictionary to {rois_conv_file}", flush=True)
        except Exception as e:
            print(f"[DEBUG] Error saving conversion dictionary: {e}", flush=True)

        # Now load the conversion dictionary and display it.
        try:
            conv_dict_loaded = np.load(rois_conv_file, allow_pickle=True).item()
            print("[DEBUG] Loaded conversion dictionary for display:", conv_dict_loaded, flush=True)
            conv_dialog = ConversionTableDialog(conv_dict_loaded, self)
            conv_dialog.exec_()
        except Exception as e:
            print(f"[DEBUG] Error displaying conversion table: {e}", flush=True)

        QMessageBox.information(self, "Extraction Finished", "Extraction finished.")
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())

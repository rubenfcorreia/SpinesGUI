#!/usr/bin/env python3
"""
Spines GUI
PyQt5 GUI for selecting and storing ROIs from Suite2p output.
This version adds a new "Tracing" drawing mode.
• In Tracing mode you can click to add vertices one by one.
• All vertex markers (including the first vertex) are drawn with a radius of 3.
• When clicking near the first vertex (within 10 pixels), a dialog appears offering Finish, Continue, or Cancel.
  - Finish: the traced polygon is finalized, temporary tracing items are removed, and the ROI is created.
  - Continue: the vertex is added and you continue tracing.
  - Cancel: the entire ROI addition is canceled.
• Associated ROI information in the "Select ROI Type" dialog is refreshed and saved when "OK" is pressed.
• For Parent dendrite ROIs (type 1), the associated Cell ROI’s cell ID is taken from the stored ROIs.
• For Dendritic spine ROIs (type 2), the associated Parent dendrite ROI (type 1) is required and its cell ID and parent dendrite ID are used.
Extra debug prints are provided.
"""

import sys, os, numpy as np
from collections import OrderedDict
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPolygonF, QPen, QBrush, QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem,
                             QGraphicsEllipseItem, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
                             QFormLayout, QDialog, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
                             QHeaderView)

# Enable high DPI scaling.
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)


# ------------------------------
# ROIItem: interactive ROI polygon with vertex editing.
# ------------------------------
class ROIItem(QGraphicsPolygonItem):
    def __init__(self, roi_id, polygon, roi_info, main_window, parent=None):
        super(ROIItem, self).__init__(polygon, parent)
        self.roi_id = roi_id
        self.roi_info = roi_info  # Format: {"roi-type": [type, cellID, parentID, spineID], "plane": <plane>, "ROI coordinates": [...]}
        self.main_window = main_window
        typ = self.roi_info["roi-type"][0]
        color = {0: Qt.blue, 1: Qt.red, 2: Qt.green}.get(typ, Qt.gray)
        self.setPen(QPen(color, 2))
        self.setBrush(QBrush(Qt.transparent))
        self.setFlags(QGraphicsPolygonItem.ItemIsSelectable | QGraphicsPolygonItem.ItemIsMovable)
        self.dragging_vertex_index = None
        self.vertex_markers = []
        self.update_vertex_markers()

    def update_vertex_markers(self):
        # All markers now use radius = 3.
        for marker in self.vertex_markers:
            if self.scene() is not None:
                self.scene().removeItem(marker)
        self.vertex_markers = []
        radius = 3
        for pt in self.polygon():
            rect = QRectF(pt.x()-radius, pt.y()-radius, 2*radius, 2*radius)
            marker = QGraphicsEllipseItem(rect, self)
            marker.setBrush(QBrush(QColor("orange")))
            marker.setPen(QPen(Qt.black))
            marker.setZValue(3)
            self.vertex_markers.append(marker)

    def contextMenuEvent(self, event):
        menu = QMessageBox()
        menu.setWindowTitle("ROI Options")
        menu.setText("Choose an action for ROI #%d:" % self.roi_id)
        edit_btn = menu.addButton("Edit ROI", QMessageBox.ActionRole)
        delete_btn = menu.addButton("Delete ROI", QMessageBox.ActionRole)
        cancel_btn = menu.addButton("Cancel", QMessageBox.RejectRole)
        menu.exec_()
        if menu.clickedButton() == edit_btn:
            self.edit_roi()
        elif menu.clickedButton() == delete_btn:
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
                if not pts:
                    return
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                center = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2)
                radius = min(max(xs)-min(xs), max(ys)-min(ys)) / 2
                new_pts = [(center[0] + radius * np.cos(2*np.pi*i/num),
                            center[1] + radius * np.sin(2*np.pi*i/num)) for i in range(num)]
                self.roi_info["ROI coordinates"] = new_pts
                new_poly = QPolygonF([QPointF(x, y) for (x, y) in new_pts])
                self.setPolygon(new_poly)
                self.update_vertex_markers()

    def delete_roi(self):
        assoc_ids = []
        typ, cellID, parentID, spineID = self.roi_info["roi-type"]
        if typ == 0:
            for rid, info in self.main_window.roi_data.items():
                if info["roi-type"][0] == 0 and rid != self.roi_id:
                    assoc_ids.append(rid)
        elif typ == 1:
            for rid, info in self.main_window.roi_data.items():
                if info["roi-type"][0] == 2 and info["roi-type"][1] == cellID and info["roi-type"][2] == parentID:
                    assoc_ids.append(rid)
        if assoc_ids:
            confirm = QMessageBox.question(None, "Delete ROI",
                                           "Deleting this ROI will delete its associated ROIs.\nProceed?",
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
            self.roi_info["ROI coordinates"] = [(pt.x(), pt.y()) for pt in poly]
            self.dragging_vertex_index = None
            self.setFlag(QGraphicsPolygonItem.ItemIsMovable, True)
            self.update_vertex_markers()
            event.accept()
        else:
            super(ROIItem, self).mouseReleaseEvent(event)


# ------------------------------
# CustomGraphicsView: now supports "Tracing" mode.
# ------------------------------
class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super(CustomGraphicsView, self).__init__(parent)
        self.setMouseTracking(True)
        self.drawing_roi = False
        self.first_click_point = None
        self.temp_polygon_item = None
        self.parent_window = parent

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        if self.parent_window and self.parent_window.current_valid_rect:
            if self.parent_window.current_valid_rect.contains(scene_pos):
                text = "X : %d\nY : %d" % (int(scene_pos.x()), int(scene_pos.y()))
            else:
                text = "X : Out of range\nY : Out of range"
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
            # If in tracing mode:
            if self.parent_window.pending_roi_shape == "tracing":
                if not self.parent_window.tracing_vertices:
                    self.parent_window.tracing_vertices = [scene_pos]
                    poly = QPolygonF(self.parent_window.tracing_vertices)
                    self.parent_window.tracing_polygon_item = QGraphicsPolygonItem(poly)
                    self.parent_window.tracing_polygon_item.setPen(QPen(Qt.magenta, 2, Qt.DashLine))
                    self.scene().addItem(self.parent_window.tracing_polygon_item)
                    # First vertex marker with radius 3.
                    r = 3
                    marker = QGraphicsEllipseItem(QRectF(scene_pos.x()-r, scene_pos.y()-r, 2*r, 2*r))
                    marker.setBrush(QBrush(Qt.yellow))
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
                # Non-tracing mode.
                if not self.parent_window.current_valid_rect.contains(scene_pos):
                    QMessageBox.warning(self, "Out of Bounds", "Click must be within the valid ROI area.")
                    return
                if self.first_click_point is None:
                    self.first_click_point = scene_pos
                    self.temp_polygon_item = QGraphicsPolygonItem()
                    self.temp_polygon_item.setPen(QPen(Qt.red, 2, Qt.DashLine))
                    self.scene().addItem(self.temp_polygon_item)
                else:
                    self.parent_window.finish_roi_drawing(self.first_click_point, scene_pos)
                    self.first_click_point = None
                    if self.temp_polygon_item:
                        self.scene().removeItem(self.temp_polygon_item)
                        self.temp_polygon_item = None
                    self.drawing_roi = False
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


# ------------------------------
# ROITableWindow and ConfirmROITableDialog remain unchanged.
# ------------------------------
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
        self.table.setHorizontalHeaderLabels(["ROI #", "ROI Type", "Cell ID", "Parent Dendrite ID", "Dendritic Spine ID", "Plane", "ROI Coordinates"])
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
            typ_str = {0:"Cell", 1:"Parent Dendrite", 2:"Dendritic Spine"}.get(typ, "Unknown")
            items = [QTableWidgetItem(str(roi_id)),
                     QTableWidgetItem(typ_str),
                     QTableWidgetItem(str(cellID)),
                     QTableWidgetItem(str(parentID)),
                     QTableWidgetItem(str(spineID)),
                     QTableWidgetItem(str(info["plane"])),
                     QTableWidgetItem(str(info["ROI coordinates"]))]
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
        self.table.setHorizontalHeaderLabels(["ROI #", "ROI Type", "Cell ID", "Parent Dendrite ID", "Dendritic Spine ID", "Plane", "ROI Coordinates"])
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
            typ_str = {0:"Cell", 1:"Parent Dendrite", 2:"Dendritic Spine"}.get(typ, "Unknown")
            items = [QTableWidgetItem(str(roi_id)),
                     QTableWidgetItem(typ_str),
                     QTableWidgetItem(str(cellID)),
                     QTableWidgetItem(str(parentID)),
                     QTableWidgetItem(str(spineID)),
                     QTableWidgetItem(str(info["plane"])),
                     QTableWidgetItem(str(info["ROI coordinates"]))]
            for col, item in enumerate(items):
                self.table.setItem(row, col, item)


# ------------------------------
# ROITypeDialog: now saves the current selection upon accept.
# ------------------------------
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
        except Exception as e:
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
                self.assoc_combo.addItem("Cell ROI #%d" % key, key)
            self.assoc_combo.show()
            self.assoc_combo.setCurrentIndex(len(sorted_cells) - 1)
            print("[DEBUG] ROITypeDialog: Available Cell ROI keys for Parent dendrite:", sorted_cells)
            print("[DEBUG] ROITypeDialog: Current associated cell ROI key (pre-accept):", self.assoc_combo.currentData())
            self.highlight_association()
        elif roi_type == 2:
            if not existing_parents:
                QMessageBox.warning(self, "No Parent Dendrites", "No parent dendrite ROIs available.")
                self.reject()
                return
            sorted_parents = sorted(existing_parents)
            for key in sorted_parents:
                self.assoc_combo.addItem("Parent Dendrite ROI #%d" % key, key)
            self.assoc_combo.show()
            self.assoc_combo.setCurrentIndex(len(sorted_parents) - 1)
            print("[DEBUG] ROITypeDialog: Available Parent ROI keys for Spine:", sorted_parents)
            print("[DEBUG] ROITypeDialog: Current associated parent ROI key (pre-accept):", self.assoc_combo.currentData())
            self.highlight_association()
        else:
            self.assoc_combo.hide()
    def highlight_association(self):
        if self.assoc_combo.isVisible():
            assoc_key = self.assoc_combo.currentData()
            print("[DEBUG] ROITypeDialog.highlight_association() - current association:", assoc_key)
            if self.parent():
                self.parent().highlight_roi(assoc_key)
    def accept(self):
        try:
            self._selected_roi_type = int(self.type_combo.currentData())
        except Exception as e:
            print("[DEBUG] ROITypeDialog.accept() error converting type:", e)
            self._selected_roi_type = 0
        if self.assoc_combo.isVisible():
            try:
                self._selected_assoc = int(self.assoc_combo.currentData())
            except Exception as e:
                print("[DEBUG] ROITypeDialog.accept() error converting association:", e)
                self._selected_assoc = 0
        else:
            self._selected_assoc = 0
        print("[DEBUG] ROITypeDialog.accept() saved: roi_type =", self._selected_roi_type, "association =", self._selected_assoc)
        if self.parent():
            self.parent().clear_highlight()
        super(ROITypeDialog, self).accept()
    def get_values(self):
        print("[DEBUG] ROITypeDialog.get_values() returns: roi_type =", self._selected_roi_type, "association =", self._selected_assoc)
        return self._selected_roi_type, self._selected_assoc


# ------------------------------
# ROIShapeDialog: now includes "Tracing" option.
# ------------------------------
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
            except Exception as e:
                QMessageBox.warning(self, "Invalid Input", "Please enter an integer between 3 and 10 for polygon sides.")
                print("[DEBUG] ROIShapeDialog.on_ok error:", e)
                return
        self.accept()
    def get_values(self):
        return self.shape, self.num_sides


# ------------------------------
# MainWindow (Spines GUI)
# ------------------------------
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
        self.roi_data = {}    # ROI key -> ROI info
        self.roi_items = {}   # ROI key -> ROIItem
        self.next_roi_id = 0
        self.pending_roi_type = None  # Tuple: (roi_type, associated ROI key) from ROITypeDialog
        self.pending_roi_shape = None  # "rectangle", "polygon", or "tracing"
        self.pending_polygon_sides = None
        # For tracing mode:
        self.tracing_vertices = []
        self.tracing_polygon_item = None
        self.tracing_markers = []
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
        self.clear_rois_button = QPushButton("Clear All ROIs")
        self.clear_rois_button.clicked.connect(self.clear_all_rois)
        self.left_arrow_button = QPushButton("<")
        self.left_arrow_button.clicked.connect(lambda: self.change_plane(-1))
        self.right_arrow_button = QPushButton(">")
        self.right_arrow_button.clicked.connect(lambda: self.change_plane(1))
        self.current_plane_label = QLabel("Current plane: N/A")
        self.coord_label = QLabel("X : Out of range\nY : Out of range")
        self.coord_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.add_roi_button)
        top_layout.addWidget(self.left_arrow_button)
        top_layout.addWidget(self.right_arrow_button)
        top_layout.addWidget(self.roi_table_button)
        top_layout.addWidget(self.clear_rois_button)
        bottom_layout.addWidget(self.current_plane_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.coord_label)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.view)
        main_layout.addLayout(bottom_layout)
        central_widget.setLayout(main_layout)

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
        print("[DEBUG] Selected root folder:", folder)
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
                            print("[DEBUG] ops.npy in %s missing required keys" % subfolder)
                            continue
                        if np.isnan(meanImg).any():
                            print("[DEBUG] meanImg in %s contains NaNs" % subfolder)
                            continue
                        self.plane_data[plane_num] = {"meanImg": meanImg, "yrange": yrange, "xrange": xrange_, "folder": subfolder}
                        self.plane_order.append(plane_num)
                        print("[DEBUG] Loaded plane %d from %s" % (plane_num, subfolder))
                    except Exception as e:
                        print("[DEBUG] Error loading ops.npy in %s: %s" % (subfolder, str(e)))
        self.plane_order = sorted(self.plane_order)
        if not self.plane_order:
            QMessageBox.warning(self, "Error", "No valid plane folders found.")
            return
        rois_file = os.path.join(self.root_folder, "ROIs.npy")
        if os.path.exists(rois_file):
            try:
                loaded_rois = np.load(rois_file, allow_pickle=True).item()
                self.roi_data = {int(k): v for k, v in loaded_rois.items()}
                if self.roi_data:
                    self.next_roi_id = max(self.roi_data.keys()) + 1
                print("[DEBUG] Loaded existing ROIs from ROIs.npy")
            except Exception as e:
                print("[DEBUG] Error loading ROIs.npy:", e)
        self.update_plane_display()

    def update_plane_display(self):
        if not self.plane_order:
            return
        plane_num = self.plane_order[self.current_plane_index]
        plane = self.plane_data[plane_num]
        meanImg = plane["meanImg"]
        yrange = plane["yrange"]
        xrange_ = plane["xrange"]
        print("[DEBUG] Displaying plane %d: meanImg shape: %s, yrange: %s, xrange: %s" % (plane_num, meanImg.shape, str(yrange), str(xrange_)))
        mi = meanImg.astype(np.float32)
        mi = (mi - mi.min()) / (mi.max() - mi.min() + 1e-8) * 255
        mi = mi.astype(np.uint8)
        height, width = mi.shape
        image = QImage(mi.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image)
        self.image_pixmap = pixmap
        self.graphics_scene.clear()
        self.image_pixmap_item = QGraphicsPixmapItem(pixmap)
        self.graphics_scene.addItem(self.image_pixmap_item)
        valid_rect = QRectF(xrange_[0], yrange[0], xrange_[1]-xrange_[0], yrange[1]-yrange[0])
        self.current_valid_rect = valid_rect
        rect_item = self.graphics_scene.addRect(valid_rect, QPen(Qt.yellow, 2))
        rect_item.setZValue(1)
        self.current_plane_label.setText("Current plane: %d" % plane_num)
        self.roi_items = {}
        for roi_id, info in self.roi_data.items():
            if info["plane"] == plane_num:
                pts = info["ROI coordinates"]
                poly = QPolygonF([QPointF(x, y) for (x, y) in pts])
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
        print("[DEBUG] In start_roi_addition: pending_roi_type =", self.pending_roi_type)
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
                                "For tracing, click to add vertices. The first vertex is highlighted (radius 3). Click near it to finish tracing.")
        self.view.drawing_roi = True

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
        # For rectangle or fixed regular polygon.
        x1, y1 = point1.x(), point1.y()
        x2, y2 = point2.x(), point2.y()
        if self.pending_roi_shape == "rectangle":
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            num_sides = self.pending_polygon_sides
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            center = ((left+right)/2, (top+bottom)/2)
            radius = min((right-left), (bottom-top)) / 2
            pts = [(center[0] + radius * np.cos(2*np.pi*i/num_sides),
                    center[1] + radius * np.sin(2*np.pi*i/num_sides)) for i in range(num_sides)]
        self._create_roi_from_points(pts)

    def finish_tracing_roi(self):
        pts = [(pt.x(), pt.y()) for pt in self.tracing_vertices]
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

    def _create_roi_from_points(self, pts):
        if self.pending_roi_type[0] == 0:
            cell_id = self.get_next_cell_id()
            roi_type_list = [0, cell_id, 0, 0]
        elif self.pending_roi_type[0] == 1:
            associated_cell_key = self.pending_roi_type[1]
            print("[DEBUG] Adding Parent dendrite; associated cell ROI key:", associated_cell_key)
            cell_roi = self.roi_data.get(associated_cell_key)
            if cell_roi is None:
                print("[DEBUG] Error: Associated cell ROI not found.")
                QMessageBox.warning(self, "Error", "Associated cell ROI not found.")
                return
            cell_id = cell_roi["roi-type"][1]
            parent_id = self.get_next_parent_id(cell_id)
            print("[DEBUG] Retrieved for Parent dendrite: cell_id =", cell_id, "parent_id =", parent_id)
            roi_type_list = [1, cell_id, parent_id, 0]
        elif self.pending_roi_type[0] == 2:
            associated_parent_key = self.pending_roi_type[1]
            print("[DEBUG] Adding Dendritic Spine; associated parent ROI key (from dialog):", associated_parent_key)
            parent_roi = self.roi_data.get(associated_parent_key)
            if parent_roi is None:
                print("[DEBUG] Error: Associated parent ROI not found.")
                QMessageBox.warning(self, "Error", "Associated parent ROI not found.")
                return
            if parent_roi["roi-type"][0] != 1:
                print("[DEBUG] Error: Selected associated ROI is not a Parent dendrite.")
                QMessageBox.warning(self, "Error", "Selected associated ROI is not a Parent dendrite.")
                return
            cell_id = parent_roi["roi-type"][1]
            parent_id = parent_roi["roi-type"][2]
            print("[DEBUG] Retrieved from parent ROI: cell_id =", cell_id, "parent_id =", parent_id)
            if parent_id == 0:
                print("[DEBUG] Error: Associated parent dendrite ID is 0. Aborting spine addition.")
                QMessageBox.warning(self, "Error", "The selected parent ROI does not have a valid Parent dendrite ID.")
                return
            spine_id = self.get_next_spine_id(cell_id, parent_id)
            roi_type_list = [2, cell_id, parent_id, spine_id]
        else:
            roi_type_list = [self.pending_roi_type[0], 0, 0, 0]
        current_plane = self.plane_order[self.current_plane_index]
        roi_info = {"roi-type": roi_type_list, "plane": current_plane, "ROI coordinates": pts}
        polygon = QPolygonF([QPointF(x, y) for (x, y) in pts])
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
                                       "Did details get stored correctly for ROI #%d?" % self.next_roi_id,
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.No:
            self.graphics_scene.removeItem(roi_item)
            del self.roi_data[self.next_roi_id]
        else:
            self.next_roi_id += 1
            self.save_rois()

    def update_tracing_display(self):
        if self.tracing_vertices:
            poly = QPolygonF(self.tracing_vertices)
            if self.tracing_polygon_item is None:
                self.tracing_polygon_item = QGraphicsPolygonItem(poly)
                self.tracing_polygon_item.setPen(QPen(Qt.magenta, 2, Qt.DashLine))
                self.graphics_scene.addItem(self.tracing_polygon_item)
            else:
                self.tracing_polygon_item.setPolygon(poly)
            if self.tracing_markers:
                for marker in self.tracing_markers:
                    self.graphics_scene.removeItem(marker)
            self.tracing_markers = []
            for i, pt in enumerate(self.tracing_vertices):
                r = 3  # All vertices now have radius 3.
                # Optionally, you can differentiate the first vertex by color.
                col = Qt.yellow if i == 0 else Qt.cyan
                marker = QGraphicsEllipseItem(QRectF(pt.x()-r, pt.y()-r, 2*r, 2*r))
                marker.setBrush(QBrush(col))
                marker.setPen(QPen(Qt.black))
                marker.setZValue(4)
                self.graphics_scene.addItem(marker)
                self.tracing_markers.append(marker)

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

    def save_rois(self):
        if self.root_folder is None:
            return
        rois_file = os.path.join(self.root_folder, "ROIs.npy")
        try:
            np.save(rois_file, self.roi_data)
            print("[DEBUG] Saved ROIs to", rois_file)
        except Exception as e:
            print("[DEBUG] Error saving ROIs:", e)

    def open_roi_table(self):
        table_win = ROITableWindow(self.roi_data, parent=self)
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


# ------------------------------
# Main application execution.
# ------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())

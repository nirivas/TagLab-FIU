# TagLab
# A semi-automatic segmentation tool
#
# Project > Georeference Map: set the georeference of a map from a file georeferenced in ArcGIS Pro, or from typed
# coordinates of points of the map, and share it with the maps on the same pixel grid (see source/Georeference.py).

import logging
import math
import os

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QEvent, QPointF, QTimer
from PyQt5.QtGui import QColor, QFont, QPainterPath, QPen, QBrush
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox, QPushButton, \
    QLineEdit, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QDoubleSpinBox, QFileDialog, QMessageBox, \
    QGraphicsItem, QGraphicsPathItem, QGraphicsSimpleTextItem, QCheckBox, QSplitter, QSizePolicy, QAbstractItemView, \
    QGroupBox

from source import Georeference as GR
from source.QtImageViewer import QtImageViewer

logfile = logging.getLogger("tool-logger")

# coordinate systems offered (EPSG code, name); None = another code typed by the user
INPUT_SYSTEMS = [(4326, "Latitude / longitude (WGS 84, GPS)"),
                 (32620, "WGS 84 / UTM zone 20N (m)"),
                 (32619, "WGS 84 / UTM zone 19N (m)"),
                 (6566, "NAD83(2011) / Puerto Rico and Virgin Is. (m)"),
                 (None, "Other EPSG code...")]
OUTPUT_AUTO = "auto"
OUTPUT_SYSTEMS = [(OUTPUT_AUTO, "Automatic (UTM zone of the points)"),
                  (32620, "WGS 84 / UTM zone 20N (m)"),
                  (32619, "WGS 84 / UTM zone 19N (m)"),
                  (6566, "NAD83(2011) / Puerto Rico and Virgin Is. (m)"),
                  (None, "Other EPSG code...")]
FITS = [(GR.FIT_RIGID, "Keep TagLab's pixel size (rotation and shift)"),
        (GR.FIT_SIMILARITY, "Fit the scale too (2 or more points)"),
        (GR.FIT_AFFINE, "Affine: also stretch and shear (3 or more points)")]

MARKER_SIZE = 9          # screen pixels
PICK_DISTANCE = 10       # screen pixels: a press closer than this to a point moves it
CLICK_TOLERANCE = 4      # screen pixels: a press that moves less than this is a click (adds a point)

COL_N, COL_X, COL_Y, COL_C1, COL_C2, COL_ERR, COL_DEL = range(7)


class PointMarker(QGraphicsPathItem):
    """
    A numbered control point drawn at a constant size on screen.
    """

    def __init__(self, number):
        s = MARKER_SIZE
        path = QPainterPath()
        path.moveTo(-s, 0)
        path.lineTo(s, 0)
        path.moveTo(0, -s)
        path.lineTo(0, s)
        path.addEllipse(QPointF(0, 0), s * 0.6, s * 0.6)
        super(PointMarker, self).__init__(path)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self.setZValue(5)
        self.setPen(QPen(QColor(255, 220, 0), 2))
        self.label = QGraphicsSimpleTextItem(str(number), self)
        self.label.setFont(QFont("Roboto", 10, QFont.Bold))
        self.label.setBrush(QBrush(QColor(255, 220, 0)))
        self.label.setPos(s * 0.7, -s * 2.2)

    def setNumber(self, number):
        self.label.setText(str(number))


class QtSetGeoreferenceWidget(QWidget):
    """
    The window of Project > Georeference Map. taglab is the main TagLab widget (runInBackground is used for the long
    jobs). The signal applied(list of images) is emitted after the georeference of maps has been set or removed.
    """

    applied = pyqtSignal(list)
    closed = pyqtSignal()

    def __init__(self, project, taglab, image=None, parent=None):
        super(QtSetGeoreferenceWidget, self).__init__(parent)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Georeference Map")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.project = project
        self.taglab = taglab
        self.image = None
        self.result = None           # the georeference computed (dictionary, see Georeference.fromFile / fromCoordinates)
        self.result_tab = None       # the tab that computed it
        self.checks = {}             # image id -> gridCheck result
        self.points = []             # [[x, y], [c1, c2]]: TagLab pixel coordinates, typed coordinates (None = not typed)
        self.bad_text = {}           # (row, 0 | 1) -> text that is not a coordinate
        self.markers = []
        self.viewer_image = None     # the image shown in the viewer
        self.drag = None             # index of the point being moved
        self.press = None            # screen position of the last press
        self.filling = False

        # --- the map
        self.mapCombo = QComboBox()
        for im in project.images:
            self.mapCombo.addItem(im.name)
        self.lblStatus = QLabel()
        self.lblStatus.setWordWrap(True)
        top = QHBoxLayout()
        top.addWidget(QLabel("Map:"))
        top.addWidget(self.mapCombo, 1)

        # --- tab 1: from a file
        fileTab = QWidget()
        self.lblFileHint = QLabel(
            "In ArcGIS Pro, add the map's image to a map whose coordinate system is projected (in metres, e.g. "
            "WGS 84 / UTM zone 20N), open <b>Imagery &gt; Georeference</b> and add control points. Then:<br>"
            "- <b>Save as New</b> (recommended): ArcGIS writes a new georeferenced raster. Choose it here: TagLab "
            "recognises the map in it.<br>"
            "- <b>Save</b>: ArcGIS writes side files (.aux.xml, world file) next to the image. Choose the map's own "
            "image file here.<br>The image files of the project are never changed.")
        self.lblFileHint.setWordWrap(True)
        self.lblFileHint.setTextFormat(Qt.RichText)
        self.editFile = QLineEdit()
        self.editFile.setPlaceholderText("Georeferenced raster (.tif) or the map's own image file")
        self.btnBrowse = QPushButton("Browse...")
        self.btnOwnFile = QPushButton("The map's own file")
        self.btnOwnFile.setToolTip("Use the map's image file (after Save in ArcGIS Pro)")
        self.lblEpsgFile = QLabel("EPSG code (the file has no coordinate system):")
        self.editEpsgFile = QLineEdit()
        self.editEpsgFile.setPlaceholderText("e.g. 32620")
        self.editEpsgFile.setMaximumWidth(120)
        self.btnRead = QPushButton("Read the georeference")
        row = QHBoxLayout()
        row.addWidget(self.editFile, 1)
        row.addWidget(self.btnBrowse)
        row.addWidget(self.btnOwnFile)
        row2 = QHBoxLayout()
        row2.addWidget(self.lblEpsgFile)
        row2.addWidget(self.editEpsgFile)
        row2.addStretch(1)
        row2.addWidget(self.btnRead)
        lay = QVBoxLayout(fileTab)
        lay.addWidget(self.lblFileHint)
        lay.addLayout(row)
        lay.addLayout(row2)
        lay.addStretch(1)

        # --- tab 2: from coordinates
        coordTab = QWidget()
        self.viewer = QtImageViewer()
        self.viewer.setMinimumSize(420, 380)
        self.viewer.viewport().installEventFilter(self)
        hint = QLabel("Click on the map to add a point, drag a point to move it; wheel to zoom, drag to pan. Type the "
                      "coordinates of each point in the table (decimal degrees, or degrees minutes seconds such as "
                      "18°19'12.3\"N).")
        hint.setWordWrap(True)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["#", "Map x (px)", "Map y (px)", "Latitude", "Longitude", "Error (cm)",
                                              ""])
        self.table.verticalHeader().hide()
        hh = self.table.horizontalHeader()
        for c in (COL_N, COL_ERR, COL_DEL):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        for c in (COL_X, COL_Y, COL_C1, COL_C2):
            hh.setSectionResizeMode(c, QHeaderView.Stretch)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.inputCombo = QComboBox()
        for code, name in INPUT_SYSTEMS:
            self.inputCombo.addItem(name, code)
        self.editEpsgIn = QLineEdit()
        self.editEpsgIn.setPlaceholderText("EPSG")
        self.editEpsgIn.setMaximumWidth(80)
        self.outputCombo = QComboBox()
        for code, name in OUTPUT_SYSTEMS:
            self.outputCombo.addItem(name, code)
        self.editEpsgOut = QLineEdit()
        self.editEpsgOut.setPlaceholderText("EPSG")
        self.editEpsgOut.setMaximumWidth(80)
        self.lblOutput = QLabel()
        self.fitCombo = QComboBox()
        for code, name in FITS:
            self.fitCombo.addItem(name, code)
        self.spinAngle = QDoubleSpinBox()
        self.spinAngle.setRange(0.0, 359.99)
        self.spinAngle.setDecimals(2)
        self.spinAngle.setSuffix("°")
        self.spinAngle.setToolTip("With a single point: the direction the top of the map points to, in degrees "
                                  "clockwise from north (0 = north up)")
        self.lblAngle = QLabel("Top of the map points to (one point only):")
        grid = QGridLayout()
        grid.addWidget(QLabel("Coordinates typed in:"), 0, 0)
        grid.addWidget(self.inputCombo, 0, 1)
        grid.addWidget(self.editEpsgIn, 0, 2)
        grid.addWidget(QLabel("Georeference the map in:"), 1, 0)
        grid.addWidget(self.outputCombo, 1, 1)
        grid.addWidget(self.editEpsgOut, 1, 2)
        grid.addWidget(self.lblOutput, 2, 1, 1, 2)
        grid.addWidget(QLabel("Fit:"), 3, 0)
        grid.addWidget(self.fitCombo, 3, 1, 1, 2)
        grid.addWidget(self.lblAngle, 4, 0)
        grid.addWidget(self.spinAngle, 4, 1)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(hint)
        rl.addWidget(self.table, 1)
        rl.addLayout(grid)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.viewer)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        cl = QVBoxLayout(coordTab)
        cl.addWidget(split)

        self.tabs = QTabWidget()
        self.tabs.addTab(fileTab, "From an ArcGIS file")
        self.tabs.addTab(coordTab, "From coordinates")

        # --- result
        self.lblResult = QLabel("No georeference computed yet.")
        self.lblResult.setWordWrap(True)
        self.lblResult.setTextFormat(Qt.RichText)
        self.lblResult.setTextInteractionFlags(Qt.TextSelectableByMouse)
        resBox = QGroupBox("Result")
        QVBoxLayout(resBox).addWidget(self.lblResult)

        # --- apply to
        self.applyTable = QTableWidget(0, 4)
        self.applyTable.setHorizontalHeaderLabels(["Apply", "Map", "Same pixel grid?", "Georeferenced now"])
        self.applyTable.verticalHeader().hide()
        ah = self.applyTable.horizontalHeader()
        ah.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        ah.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        ah.setSectionResizeMode(2, QHeaderView.Stretch)
        ah.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.applyTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.applyTable.setMaximumHeight(150)
        self.btnCheck = QPushButton("Check the other maps")
        self.btnCheck.setToolTip("Find which maps with the same pixel size are on the same pixel grid (aligned with "
                                 "the Alignment Tool before TagLab recorded it)")
        applyBox = QGroupBox("Also apply to the maps on the same pixel grid")
        al = QVBoxLayout(applyBox)
        al.addWidget(self.applyTable)
        r3 = QHBoxLayout()
        r3.addStretch(1)
        r3.addWidget(self.btnCheck)
        al.addLayout(r3)

        # --- buttons
        self.btnRemove = QPushButton("Remove TagLab's georeference")
        self.btnApply = QPushButton("Apply")
        self.btnApply.setDefault(True)
        self.btnClose = QPushButton("Close")
        buttons = QHBoxLayout()
        buttons.addWidget(self.btnRemove)
        buttons.addStretch(1)
        buttons.addWidget(self.btnApply)
        buttons.addWidget(self.btnClose)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.lblStatus)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(resBox)
        layout.addWidget(applyBox)
        layout.addLayout(buttons)
        self.resize(1100, 860)

        self.fitTimer = QTimer(self)
        self.fitTimer.setSingleShot(True)
        self.fitTimer.setInterval(150)
        self.fitTimer.timeout.connect(self.fitCoordinates)

        self.mapCombo.currentIndexChanged.connect(self.mapChanged)
        self.tabs.currentChanged.connect(self.tabChanged)
        self.btnBrowse.clicked.connect(self.browseFile)
        self.btnOwnFile.clicked.connect(self.useOwnFile)
        self.btnRead.clicked.connect(self.readFile)
        self.editFile.textChanged.connect(self.fileChanged)
        self.table.itemChanged.connect(self.tableEdited)
        self.inputCombo.currentIndexChanged.connect(self.systemsChanged)
        self.outputCombo.currentIndexChanged.connect(self.systemsChanged)
        self.editEpsgIn.textChanged.connect(self.systemsChanged)
        self.editEpsgOut.textChanged.connect(self.systemsChanged)
        self.fitCombo.currentIndexChanged.connect(self.scheduleFit)
        self.spinAngle.valueChanged.connect(self.scheduleFit)
        self.btnCheck.clicked.connect(self.checkOtherMaps)
        self.btnApply.clicked.connect(self.apply)
        self.btnRemove.clicked.connect(self.removeGeoreference)
        self.btnClose.clicked.connect(self.close)

        index = project.images.index(image) if image in project.images else 0
        self.mapCombo.setCurrentIndex(index)
        self.mapChanged(index)

    # ------------------------------------------------------------------------------------------------- the map

    @pyqtSlot(int)
    def mapChanged(self, index):
        if index < 0 or index >= len(self.project.images):
            return
        self.image = self.project.images[index]
        self.checks = {}
        self.setResult(None, None)
        self.editFile.blockSignals(True)
        self.editFile.setText("")
        self.editFile.blockSignals(False)
        self.lblEpsgFile.hide()
        self.editEpsgFile.hide()
        self.editEpsgFile.setText("")
        self.loadPoints()
        self.updateStatus()
        self.fillApplyTable()
        if self.tabs.currentIndex() == 1:
            self.showMapInViewer()

    def updateStatus(self):
        g = GR.get(self.image)
        if g is None:
            text = "<b>{}</b> is not georeferenced.".format(self.image.name)
        else:
            if GR.isSetInTagLab(self.image):
                entry = self.image.metadata[GR.KEY]
                how = {"sidecar": "from ArcGIS side files", "control points": "from control points of a file",
                       "matched": "matched to a georeferenced raster",
                       "coordinates": "from typed coordinates"}.get(entry.get("method"), entry.get("method", ""))
                src = entry.get("from_map")
                if src and src != self.image.id:
                    how += ", shared from '{}'".format(self.mapName(src))
                text = "<b>{}</b> is georeferenced in TagLab ({}{}).".format(
                    self.image.name, how, ", " + entry["date"] if entry.get("date") else "")
            else:
                text = "<b>{}</b> is georeferenced by its image file.".format(self.image.name)
            text += " " + GR.crsName(g.crs) + "."
        self.lblStatus.setText(text)
        self.btnRemove.setEnabled(GR.isSetInTagLab(self.image))

    def mapName(self, image_id):
        for im in self.project.images:
            if im.id == image_id:
                return im.name
        return image_id

    @pyqtSlot(int)
    def tabChanged(self, index):
        if index == 1:
            self.showMapInViewer()
            self.scheduleFit()
        elif self.result_tab == 1:
            self.setResult(None, None)

    def showMapInViewer(self):
        if self.viewer_image is self.image:
            return
        ch = self.image.getRGBChannel() or self.image.getDEMChannel()
        if ch is None:
            return
        qimage = ch.qimage if ch.qimage is not None else ch.loadData(self.taglab.taglab_dir)
        self.viewer.setImg(qimage)
        self.viewer_image = self.image
        self.drawMarkers()

    # ------------------------------------------------------------------------------------------------- tab 1

    @pyqtSlot()
    def browseFile(self):
        folder = os.path.dirname(GR.mapFile(self.image) or "")
        filename, _ = QFileDialog.getOpenFileName(self, "Georeferenced raster", folder,
                                                  "Rasters (*.tif *.tiff *.png *.jpg *.img *.vrt);;All files (*)")
        if filename:
            self.editFile.setText(filename)

    @pyqtSlot()
    def useOwnFile(self):
        f = GR.mapFile(self.image)
        if f:
            self.editFile.setText(os.path.abspath(os.path.join(self.taglab.taglab_dir, f)))

    @pyqtSlot(str)
    def fileChanged(self, text):
        if self.result_tab == 0:
            self.setResult(None, None)
        self.lblEpsgFile.hide()
        self.editEpsgFile.hide()

    @pyqtSlot()
    def readFile(self):
        filename = self.editFile.text().strip()
        if not filename:
            return
        epsg = None
        if self.editEpsgFile.isVisible() and self.editEpsgFile.text().strip():
            try:
                epsg = int(self.editEpsgFile.text().strip())
            except ValueError:
                self.setError("The EPSG code must be a number (e.g. 32620).")
                return
        image = self.image
        try:
            info = GR.readFile(filename, epsg)
        except Exception as e:
            self.setError(str(e))
            return
        if info["crs"] is None:
            self.lblEpsgFile.show()
            self.editEpsgFile.show()
            self.editEpsgFile.setFocus()
            self.setError("The file has a georeference but no coordinate system: type its EPSG code and read again.")
            return

        def job(progress, stop):
            return GR.fromFile(image, filename, epsg, progress=lambda p, msg="": progress(p))

        try:
            result, stopped = self.taglab.runInBackground("Reading the georeference..", job)
        except Exception as e:
            self.setError(str(e))
            return
        if stopped or result is None:
            return
        self.setResult(result, 0)

    # ------------------------------------------------------------------------------------------------- tab 2

    def loadPoints(self):
        """
        The control points of the map (if its georeference was typed), else none.
        """
        self.points = []
        self.bad_text = {}
        entry = self.image.metadata.get(GR.KEY) if self.image.metadata else None
        self.filling = True
        try:
            if entry and entry.get("method") == "coordinates" and entry.get("from_map", self.image.id) == self.image.id:
                for cp in entry.get("control_points", []):
                    self.points.append([list(cp["px"]), list(cp["input"])])
                if entry.get("control_points"):
                    self.setSystem(self.inputCombo, self.editEpsgIn, entry["control_points"][0].get("input_crs"))
                out = entry.get("output_crs")
                if out:
                    self.setSystem(self.outputCombo, self.editEpsgOut, out)
                fit = entry.get("fit")
                for i in range(self.fitCombo.count()):
                    if self.fitCombo.itemData(i) == fit:
                        self.fitCombo.setCurrentIndex(i)
                if "angle" in entry:
                    self.spinAngle.setValue(float(entry["angle"]) % 360.0)
        finally:
            self.filling = False
        self.updateSystemWidgets()
        self.fillTable()
        self.drawMarkers()

    def setSystem(self, combo, edit, text):
        try:
            code = int(str(text).split(":")[-1])
        except (ValueError, TypeError):
            return
        for i in range(combo.count()):
            if combo.itemData(i) == code:
                combo.setCurrentIndex(i)
                return
        for i in range(combo.count()):
            if combo.itemData(i) is None:
                combo.setCurrentIndex(i)
                edit.setText(str(code))
                return

    def inputSystem(self):
        code = self.inputCombo.currentData()
        if code is None:
            code = int(self.editEpsgIn.text().strip())
        return GR.CRS.from_epsg(code)

    def outputSystem(self, in_crs=None):
        code = self.outputCombo.currentData()
        if code == OUTPUT_AUTO:
            in_crs = in_crs or self.inputSystem()
            if not in_crs.is_geographic:
                return in_crs
            typed = [p[1] for p in self.points if p[1][0] is not None and p[1][1] is not None]
            if not typed:
                return None
            lat = float(np.mean([t[0] for t in typed]))
            lon = float(np.mean([t[1] for t in typed]))
            if in_crs.to_epsg() != 4326:
                lat, lon = GR.transformPoints([[lat, lon]], in_crs, 4326)[0]
            return GR.CRS.from_epsg(GR.utmZone(lon, lat))
        if code is None:
            code = int(self.editEpsgOut.text().strip())
        return GR.CRS.from_epsg(code)

    def updateSystemWidgets(self):
        self.editEpsgIn.setVisible(self.inputCombo.currentData() is None)
        self.editEpsgOut.setVisible(self.outputCombo.currentData() is None)
        geographic = True
        try:
            geographic = self.inputSystem().is_geographic
        except Exception:
            pass
        names = ("Latitude", "Longitude") if geographic else ("X (easting)", "Y (northing)")
        self.table.setHorizontalHeaderItem(COL_C1, QTableWidgetItem(names[0]))
        self.table.setHorizontalHeaderItem(COL_C2, QTableWidgetItem(names[1]))
        try:
            out = self.outputSystem()
            self.lblOutput.setText("" if out is None else "= " + GR.crsName(out))
        except Exception:
            self.lblOutput.setText("")
        one = len(self.points) == 1 and self.fitCombo.currentData() == GR.FIT_RIGID
        self.spinAngle.setEnabled(one)
        self.lblAngle.setEnabled(one)

    @pyqtSlot()
    def systemsChanged(self):
        if self.filling:
            return
        self.updateSystemWidgets()
        self.scheduleFit()

    def fillTable(self, errors=None):
        self.filling = True
        try:
            self.table.setRowCount(len(self.points))
            geographic = True
            try:
                geographic = self.inputSystem().is_geographic
            except Exception:
                pass
            for i, (px, c) in enumerate(self.points):
                values = [str(i + 1), "{:.1f}".format(px[0]), "{:.1f}".format(px[1])]
                for k in (0, 1):
                    if c[k] is None:
                        values.append(self.bad_text.get((i, k), ""))
                    elif geographic:
                        values.append("{:.8f}".format(c[k]).rstrip("0").rstrip("."))
                    else:
                        values.append("{:.3f}".format(c[k]))
                err = ""
                if errors is not None and i < len(errors) and errors[i] is not None:
                    err = "{:.1f}".format(errors[i] * 100.0)
                values.append(err)
                for col, v in enumerate(values):
                    item = self.table.item(i, col)
                    if item is None:
                        item = QTableWidgetItem()
                        self.table.setItem(i, col, item)
                    item.setText(v)
                    if col in (COL_C1, COL_C2):
                        bad = (i, col - COL_C1) in self.bad_text
                        item.setBackground(QBrush(QColor(255, 200, 200)) if bad else QBrush())
                        item.setToolTip("Not a coordinate" if bad else "")
                    editable = col in (COL_X, COL_Y, COL_C1, COL_C2)
                    flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
                    if editable:
                        flags |= Qt.ItemIsEditable
                    item.setFlags(flags)
                if self.table.cellWidget(i, COL_DEL) is None:
                    btn = QPushButton("×")
                    btn.setToolTip("Remove this point")
                    btn.setMaximumWidth(28)
                    btn.clicked.connect(self.removePointClicked)
                    self.table.setCellWidget(i, COL_DEL, btn)
        finally:
            self.filling = False

    @pyqtSlot()
    def removePointClicked(self):
        btn = self.sender()
        for i in range(self.table.rowCount()):
            if self.table.cellWidget(i, COL_DEL) is btn:
                self.removePoint(i)
                return

    def removePoint(self, i):
        del self.points[i]
        self.bad_text = {(r if r < i else r - 1, k): v for (r, k), v in self.bad_text.items() if r != i}
        self.table.removeRow(i)
        self.fillTable()
        self.drawMarkers()
        self.updateSystemWidgets()
        self.scheduleFit()

    def addPoint(self, x, y):
        x = min(max(float(x), 0.0), self.image.width - 1.0)
        y = min(max(float(y), 0.0), self.image.height - 1.0)
        self.points.append([[round(x, 1), round(y, 1)], [None, None]])
        self.fillTable()
        self.drawMarkers()
        self.updateSystemWidgets()
        row = len(self.points) - 1
        self.table.setCurrentCell(row, COL_C1)
        self.table.scrollToBottom()
        self.scheduleFit()

    @pyqtSlot(QTableWidgetItem)
    def tableEdited(self, item):
        if self.filling:
            return
        i, col = item.row(), item.column()
        if i >= len(self.points):
            return
        text = item.text().strip()
        if col in (COL_X, COL_Y):
            try:
                v = float(text.replace(",", "."))
                limit = (self.image.width if col == COL_X else self.image.height) - 1.0
                self.points[i][0][col - COL_X] = min(max(v, 0.0), limit)
            except ValueError:
                pass
            self.fillTable()
            self.drawMarkers()
        else:
            c = self.points[i][1]
            k = col - COL_C1
            self.bad_text.pop((i, k), None)
            if text == "":
                c[k] = None
            else:
                try:
                    c[k] = GR.parseCoordinate(text)
                except ValueError:
                    c[k] = None
                    self.bad_text[(i, k)] = text
            self.fillTable()
        self.updateSystemWidgets()
        self.scheduleFit()

    def drawMarkers(self):
        for m in self.markers:
            if m.scene() is not None:
                m.scene().removeItem(m)
        self.markers = []
        if self.viewer_image is not self.image:
            return
        for i, (px, c) in enumerate(self.points):
            m = PointMarker(i + 1)
            m.setPos(px[0] + 0.5, px[1] + 0.5)      # pixel centre (the scene has the pixel corners at integers)
            self.viewer.scene.addItem(m)
            self.markers.append(m)

    def pointNear(self, view_pos):
        best, best_d = None, PICK_DISTANCE
        for i, (px, c) in enumerate(self.points):
            p = self.viewer.mapFromScene(QPointF(px[0] + 0.5, px[1] + 0.5))
            d = math.hypot(p.x() - view_pos.x(), p.y() - view_pos.y())
            if d <= best_d:
                best, best_d = i, d
        return best

    def eventFilter(self, obj, event):
        if obj is self.viewer.viewport() and self.viewer_image is self.image:
            t = event.type()
            if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                i = self.pointNear(event.pos())
                if i is not None:
                    self.drag = i
                    return True
                self.press = event.pos()
            elif t == QEvent.MouseMove and self.drag is not None:
                s = self.viewer.mapToScene(event.pos())
                px = self.points[self.drag][0]
                px[0] = min(max(s.x() - 0.5, 0.0), self.image.width - 1.0)
                px[1] = min(max(s.y() - 0.5, 0.0), self.image.height - 1.0)
                self.markers[self.drag].setPos(px[0] + 0.5, px[1] + 0.5)
                return True
            elif t == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self.drag is not None:
                    px = self.points[self.drag][0]
                    px[0], px[1] = round(px[0], 1), round(px[1], 1)
                    self.drag = None
                    self.fillTable()
                    self.drawMarkers()
                    self.scheduleFit()
                    return True
                if self.press is not None:
                    d = event.pos() - self.press
                    self.press = None
                    if abs(d.x()) + abs(d.y()) <= CLICK_TOLERANCE:
                        s = self.viewer.mapToScene(event.pos())
                        if 0 <= s.x() < self.image.width and 0 <= s.y() < self.image.height:
                            self.addPoint(s.x() - 0.5, s.y() - 0.5)
        return super(QtSetGeoreferenceWidget, self).eventFilter(obj, event)

    @pyqtSlot()
    def scheduleFit(self):
        if self.filling:
            return
        if self.tabs.currentIndex() == 1:
            self.updateSystemWidgets()
            self.fitTimer.start()

    @pyqtSlot()
    def fitCoordinates(self):
        """
        Fit the georeference to the points with both coordinates typed; the residuals go in the table.
        """
        if self.tabs.currentIndex() != 1:
            return
        complete = [(i, p) for i, p in enumerate(self.points) if p[1][0] is not None and p[1][1] is not None]
        if not complete:
            self.fillTable()
            self.setResult(None, None, "Add points on the map and type their coordinates.")
            return
        try:
            in_crs = self.inputSystem()
            out_crs = self.outputSystem(in_crs)
        except Exception:
            self.setError("Type a valid EPSG code.")
            return
        if in_crs.is_geographic:
            bad = [i + 1 for i, p in complete if not (-90 <= p[1][0] <= 90 and -180 <= p[1][1] <= 180)]
            if bad:
                self.setError("Point {}: latitude must be between -90 and 90 and longitude between -180 and 180 "
                              "(latitude first).".format(", ".join(map(str, bad))))
                return
        fit = self.fitCombo.currentData()
        try:
            r = GR.fromCoordinates(self.image, [(tuple(p[0]), tuple(p[1])) for i, p in complete], in_crs, out_crs,
                                   fit, self.spinAngle.value())
        except Exception as e:
            self.fillTable()
            self.setError(str(e))
            return
        errors = [None] * len(self.points)
        for (i, p), e in zip(complete, r["errors_m"]):
            errors[i] = e
        self.fillTable(errors)
        r["method"] = "coordinates"
        r["source"] = ""
        r["rmse_mm"] = r["rmse_m"] * 1000.0 if len(complete) > {GR.FIT_RIGID: 1, GR.FIT_SIMILARITY: 2,
                                                                 GR.FIT_AFFINE: 3}[fit] else None
        r["extra"] = {"fit": fit, "output_crs": "EPSG:{}".format(out_crs.to_epsg()) if out_crs.to_epsg() else
                      out_crs.to_wkt(),
                      "control_points": [{"px": [float(p[0][0]), float(p[0][1])],
                                          "input": [float(p[1][0]), float(p[1][1])],
                                          "input_crs": "EPSG:{}".format(in_crs.to_epsg())}
                                         for i, p in complete]}
        if fit == GR.FIT_RIGID and len(complete) == 1:
            r["extra"]["angle"] = float(self.spinAngle.value())
        r["points"] = len(complete)
        self.setResult(r, 1)

    # ------------------------------------------------------------------------------------------------- result

    def setError(self, text):
        self.result = None
        self.result_tab = None
        self.lblResult.setText("<span style='color:#e04040'>{}</span>".format(self.escape(text)))
        self.btnApply.setEnabled(False)

    @staticmethod
    def escape(text):
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

    def setResult(self, result, tab, text=None):
        self.result = result
        self.result_tab = tab if result is not None else None
        self.btnApply.setEnabled(result is not None)
        if result is None:
            self.lblResult.setText(text or "No georeference computed yet.")
            return
        desc = GR.describe(result["crs"], result["transform"], (self.image.width, self.image.height),
                           self.image.pixelSize())
        kind = result["method"]
        if kind == "sidecar":
            method = "the side files of the map's image (ArcGIS Save)"
        elif kind == "control points":
            method = "the control points of the file"
        elif kind == "matched":
            method = "the map matched to the georeferenced raster ({} matching points)".format(result.get("markers"))
        elif kind == "coordinates":
            fit = (result.get("extra") or {}).get("fit", GR.FIT_RIGID)
            fit_text = dict(FITS).get(fit, fit)
            fit_text = fit_text[:1].lower() + fit_text[1:]      # "keep TagLab's pixel size ..."
            method = "{} typed point{} ({})".format(result.get("points"), "s" if result.get("points", 0) != 1 else "",
                                                    fit_text)
        else:
            method = kind
        lines = ["<b>From</b> " + self.escape(method)]
        if result.get("rmse_mm") is not None:
            lines.append("<b>Accuracy</b> (RMSE of the points): {:.1f} mm".format(result["rmse_mm"]))
        lines.append("<b>Coordinate system</b>: " + self.escape(desc["crs"]))
        lines.append("<b>Pixel size</b> from the georeference: {:.4f} mm (TagLab's: {:.4f} mm)".format(
            desc["pixel_mm"], self.image.pixelSize()))
        lines.append("<b>Top of the map</b> points to {:.1f}° (clockwise from north)".format(desc["top_bearing"]))
        if desc["stretch"] > 1e-4:
            lines.append("<b>Stretch / shear</b>: {:.2f}%".format(100.0 * desc["stretch"]))
        warnings = list(result.get("warnings", [])) + desc["warnings"]
        for w in warnings:
            lines.append("<span style='color:#e08000'>⚠ {}</span>".format(self.escape(w)))
        self.lblResult.setText("<br>".join(lines))

    # ------------------------------------------------------------------------------------------------- apply to

    def fillApplyTable(self):
        self.applyTable.setRowCount(0)
        self.applyRows = []
        for c in GR.coRegistered(self.project, self.image):
            im = c["image"]
            status, offset, enabled, ticked = c["status"], c["offset"], False, False
            if status == "same grid":
                text = "yes (aligned with the Alignment Tool), offset {:+.0f}, {:+.0f} px".format(*offset) \
                    if offset != (0.0, 0.0) else "yes (aligned with the Alignment Tool)"
                enabled, ticked, from_check = True, True, False
            elif status == "to check":
                chk = self.checks.get(im.id)
                from_check = True
                if chk is None:
                    text = "not recorded: press Check the other maps"
                elif chk["ok"]:
                    offset = chk["offset"]
                    text = "yes (checked: lines up within {:.1f} mm){}".format(
                        chk["residual_mm"], ", offset {:+.0f}, {:+.0f} px".format(*offset)
                        if offset != (0.0, 0.0) else "")
                    enabled, ticked = True, True
                else:
                    text = "no ({})".format(chk.get("reason", ""))
            else:
                text = "no (different pixel size)"
                from_check = False
            row = self.applyTable.rowCount()
            self.applyTable.insertRow(row)
            box = QCheckBox()
            box.setChecked(ticked)
            box.setEnabled(enabled)
            cell = QWidget()
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(6, 0, 6, 0)
            cl.addWidget(box)
            self.applyTable.setCellWidget(row, 0, cell)
            self.applyTable.setItem(row, 1, QTableWidgetItem(im.name))
            self.applyTable.setItem(row, 2, QTableWidgetItem(text))
            now = "in TagLab" if GR.isSetInTagLab(im) else ("by its file" if GR.isGeoreferenced(im) else "no")
            self.applyTable.setItem(row, 3, QTableWidgetItem(now))
            self.applyRows.append({"image": im, "box": box, "offset": offset, "from_check": from_check,
                                   "enabled": enabled})
        self.btnCheck.setEnabled(any(c["status"] == "to check" and c["image"].id not in self.checks
                                     for c in GR.coRegistered(self.project, self.image)))

    @pyqtSlot()
    def checkOtherMaps(self):
        todo = [c["image"] for c in GR.coRegistered(self.project, self.image)
                if c["status"] == "to check" and c["image"].id not in self.checks]
        if not todo:
            return
        source = self.image

        def job(progress, stop):
            out = {}
            for k, im in enumerate(todo):
                if stop():
                    break
                out[im.id] = GR.gridCheck(source, im, progress=lambda p, msg="", k=k: progress(
                    100.0 * (k + p / 100.0) / len(todo)), stop=stop)
            return out

        try:
            result, stopped = self.taglab.runInBackground("Comparing the maps..", job)
        except Exception as e:
            QMessageBox.warning(self, "Georeference Map", "The maps could not be compared:\n" + str(e))
            return
        for key, value in (result or {}).items():
            if value.get("reason") != "stopped":
                self.checks[key] = value
        self.fillApplyTable()

    def targets(self):
        return [(r["image"], r["offset"], r["from_check"]) for r in self.applyRows
                if r["enabled"] and r["box"].isChecked() and r["offset"] is not None]

    # ------------------------------------------------------------------------------------------------- apply / remove

    @pyqtSlot()
    def apply(self):
        r = self.result
        if r is None:
            return
        targets = self.targets()
        maps = [self.image] + [t[0] for t in targets]
        new_t = {self.image.id: r["transform"]}
        for im, off, _ in targets:
            new_t[im.id] = GR.shifted(r["transform"], off[0], off[1])
        replaced = []
        for im in maps:
            g = GR.get(im)
            if g is not None and (g.crs != r["crs"] or not g.transform.almost_equals(new_t[im.id], 1e-9)):
                replaced.append(im.name + (" (set in TagLab)" if GR.isSetInTagLab(im) else " (its image file)"))
        if replaced:
            answer = QMessageBox.question(
                self, "Georeference Map", "These maps already have a different georeference:\n\n- " +
                "\n- ".join(replaced) + "\n\nReplace it? (TagLab's georeference is used instead of the one of the "
                "image file, which is not changed.)", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        changed = GR.applyToGroup(self.project, self.image, r["crs"], r["transform"], r["method"],
                                  r.get("source", ""), r.get("rmse_mm"), r.get("extra"), targets)
        desc = GR.describe(r["crs"], r["transform"], (self.image.width, self.image.height), self.image.pixelSize())
        logfile.info("[GEOREF] '{}' georeferenced ({}{}; {}; pixel {:.4f} mm, top {:.1f} deg){}.".format(
            self.image.id, r["method"], ", rmse {:.1f} mm".format(r["rmse_mm"]) if r.get("rmse_mm") is not None
            else "", desc["crs"], desc["pixel_mm"], desc["top_bearing"],
            "; also applied to " + ", ".join("'{}' ({:+.0f}, {:+.0f} px)".format(im.id, off[0], off[1])
                                             for im, off, _ in targets) if targets else ""))
        self.updateStatus()
        self.fillApplyTable()
        self.applied.emit(changed)
        QMessageBox.information(self, "Georeference Map", "The georeference has been set on: " +
                                ", ".join("'" + im.name + "'" for im in changed) +
                                ".\n\nSave the project to keep it (File > Save).")

    @pyqtSlot()
    def removeGeoreference(self):
        maps = [self.image] + [r["image"] for r in self.applyRows if r["enabled"] and r["box"].isChecked()]
        maps = [im for im in maps if GR.isSetInTagLab(im)]
        if not maps:
            return
        answer = QMessageBox.question(
            self, "Georeference Map", "Remove TagLab's georeference of " + ", ".join("'" + im.name + "'"
                                                                                    for im in maps) +
            "? Maps whose image file is georeferenced go back to the file's georeference.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        for im in maps:
            GR.clear(im)
        logfile.info("[GEOREF] TagLab's georeference removed from " + ", ".join("'" + im.id + "'" for im in maps))
        self.loadPoints()
        self.setResult(None, None)
        self.updateStatus()
        self.fillApplyTable()
        self.applied.emit(maps)

    def closeEvent(self, event):
        self.closed.emit()
        super(QtSetGeoreferenceWidget, self).closeEvent(event)

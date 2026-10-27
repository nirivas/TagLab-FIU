import math
import os
from typing import Optional, List

import numpy as np
import cv2

from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot, QPointF
from PyQt5.QtGui import QImage, QPen, QFont, QColor, QPainter, QPainterPath, QKeyEvent, QCloseEvent, QMouseEvent, \
    QWheelEvent
from PyQt5.QtWidgets import QWidget, QHeaderView, QSizePolicy, QVBoxLayout, QLabel, QTableWidget, QTableView, \
    QHBoxLayout, QTableWidgetItem, QComboBox, QSlider, QApplication, QCheckBox, QPushButton, QMessageBox, \
    QGraphicsItem, QGraphicsPathItem, QGraphicsTextItem, QProgressDialog

from source.QtImageViewer import QtImageViewer
from source import Alignment as AL


def qimageToArray(qimg: QImage):
    """
    Zero-copy (h, w, 4) BGRA view of a 32-bit QImage. Returns (qimage, array): the returned QImage must be kept
    alive while the array is used.
    """
    if qimg.format() not in (QImage.Format_RGB32, QImage.Format_ARGB32, QImage.Format_ARGB32_Premultiplied):
        qimg = qimg.convertToFormat(QImage.Format_RGB32)
    ptr = qimg.constBits()
    ptr.setsize(qimg.height() * qimg.bytesPerLine())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(qimg.height(), qimg.bytesPerLine())
    return qimg, arr[:, :qimg.width() * 4].reshape(qimg.height(), qimg.width(), 4)


class ImagePyramid(object):
    """
    Full resolution image (zero-copy view of the QImage) plus a few reduced versions, used to render quickly
    any part of the map at any zoom level.
    """

    def __init__(self, qimage: QImage):
        self.qimage, base = qimageToArray(qimage)
        self.height, self.width = base.shape[0], base.shape[1]
        self.levels = [(1.0, 1.0, base)]
        small = cv2.resize(base, (max(1, self.width // 4), max(1, self.height // 4)), interpolation=cv2.INTER_AREA)
        self.levels.append((small.shape[1] / self.width, small.shape[0] / self.height, small))
        while max(small.shape[:2]) > 512:
            small = cv2.resize(small, (max(1, small.shape[1] // 2), max(1, small.shape[0] // 2)),
                               interpolation=cv2.INTER_AREA)
            self.levels.append((small.shape[1] / self.width, small.shape[0] / self.height, small))

    def render(self, V, w, h):
        """
        Render the image transformed by V (full-resolution pixels -> screen pixels) on a w x h buffer (BGRA).
        """
        z = math.hypot(V[0, 0], V[1, 0])
        sx, sy, arr = self.levels[0]
        for (lx, ly, a) in reversed(self.levels):
            if lx >= z and ly >= z:
                sx, sy, arr = lx, ly, a
                break
        Vl = V.copy()
        Vl[:, 0] = V[:, 0] / sx
        Vl[:, 1] = V[:, 1] / sy
        flags = cv2.INTER_NEAREST if z / sx > 2.0 else cv2.INTER_LINEAR
        return AL.warpArray(arr, Vl, w, h, border_value=0, flags=flags)


class AlignmentPreviewPane(QWidget):
    """
    Preview of the alignment: the reference image and the image to align (transformed with the current
    transformation) are rendered on the CPU, only for the visible area, and combined (blend, difference or
    checkerboard). Left-drag pans, the wheel zooms.
    """

    viewChanged = pyqtSignal(float, float, float)  # center x, center y (reference pixels), zoom

    BLEND = 0
    DIFFERENCE = 1
    CHECKERBOARD = 2

    BACKGROUND = (40, 40, 40)

    def __init__(self, mode, parent=None):
        super(AlignmentPreviewPane, self).__init__(parent)
        self.setMinimumSize(250, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mode = mode
        self.alpha = 0.5
        self.threshold = 32
        self.checkerSize = 64
        self.ref: Optional[ImagePyramid] = None
        self.mov: Optional[ImagePyramid] = None
        self.M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.cx = 0.0
        self.cy = 0.0
        self.zoom = 1.0
        self.minZoom = 0.01
        self.refPoints = np.zeros((0, 2))
        self.movPoints = np.zeros((0, 2))
        self.pointIds = []
        self.showPoints = True
        self.lastPos = None
        self.setCursor(Qt.OpenHandCursor)

    def setImages(self, ref: ImagePyramid, mov: ImagePyramid) -> None:
        self.ref = ref
        self.mov = mov
        self.fitView()

    def setTransform(self, M) -> None:
        self.M = np.asarray(M, dtype=np.float64)
        self.update()

    def setPoints(self, refPoints, movPoints, ids) -> None:
        """
        :param: refPoints the markers on the reference image (pixels)
        :param: movPoints the markers on the image to align (pixels of the image to align)
        """
        self.refPoints = np.asarray(refPoints, dtype=np.float64).reshape(-1, 2)
        self.movPoints = np.asarray(movPoints, dtype=np.float64).reshape(-1, 2)
        self.pointIds = list(ids)
        self.update()

    def setMode(self, mode) -> None:
        self.mode = mode
        self.update()

    def setAlpha(self, alpha: float) -> None:
        self.alpha = min(max(alpha, 0.0), 1.0)
        self.update()

    def setThreshold(self, thr: int) -> None:
        self.threshold = thr
        self.update()

    def setCheckerSize(self, size: int) -> None:
        self.checkerSize = size
        self.update()

    def setPointsVisibility(self, visible: bool) -> None:
        self.showPoints = visible
        self.update()

    def fitView(self) -> None:
        if self.ref is None:
            return
        w = max(1, self.width())
        h = max(1, self.height())
        self.zoom = 0.95 * min(w / self.ref.width, h / self.ref.height)
        self.minZoom = self.zoom * 0.5
        self.cx = self.ref.width / 2.0
        self.cy = self.ref.height / 2.0
        self.update()

    def setView(self, cx: float, cy: float, zoom: float) -> None:
        self.cx = cx
        self.cy = cy
        self.zoom = zoom
        self.update()

    def viewMatrix(self):
        z = self.zoom
        return np.array([[z, 0.0, self.width() / 2.0 - self.cx * z], [0.0, z, self.height() / 2.0 - self.cy * z]])

    def compose(self):
        """
        Combine the two rendered images according to the current mode. Returns a (h, w, 4) BGRA buffer.
        """
        w = self.width()
        h = self.height()
        V = self.viewMatrix()
        ref = self.ref.render(V, w, h)
        mov = self.mov.render(AL.composeMatrix(V, self.M), w, h)

        # valid pixels: inside the image and not black (no data)
        rv = (ref[:, :, 3] > 0) & (ref[:, :, :3].max(axis=2) > 0)
        mv = (mov[:, :, 3] > 0) & (mov[:, :, :3].max(axis=2) > 0)

        out = np.empty((h, w, 4), dtype=np.uint8)
        out[:, :, 0] = self.BACKGROUND[2]
        out[:, :, 1] = self.BACKGROUND[1]
        out[:, :, 2] = self.BACKGROUND[0]
        both = rv & mv
        onlyR = rv & ~mv
        onlyM = mv & ~rv

        if self.mode == AlignmentPreviewPane.BLEND:
            mix = cv2.addWeighted(ref, 1.0 - self.alpha, mov, self.alpha, 0.0)
            out[both] = mix[both]
            out[onlyR] = ref[onlyR]
            out[onlyM] = mov[onlyM]
        elif self.mode == AlignmentPreviewPane.DIFFERENCE:
            gr = cv2.cvtColor(ref, cv2.COLOR_BGRA2GRAY)
            gm = cv2.cvtColor(mov, cv2.COLOR_BGRA2GRAY)
            d = cv2.absdiff(gr, gm)
            d[d <= self.threshold] = 0
            out[both, :3] = d[both][:, None]
            out[onlyR, :3] = (gr[onlyR] // 2)[:, None]
            out[onlyM, :3] = (gm[onlyM] // 2)[:, None]
        else:
            # tiles anchored to the map (they do not move when panning)
            n = max(4, int(self.checkerSize))
            offx = int(round(self.cx * self.zoom - w / 2.0))
            offy = int(round(self.cy * self.zoom - h / 2.0))
            tx = ((np.arange(w) + offx) // n) % 2
            ty = ((np.arange(h) + offy) // n) % 2
            useMov = (ty[:, None] ^ tx[None, :]).astype(bool)
            pickM = mv & (useMov | ~rv)
            pickR = rv & ~pickM
            out[pickR] = ref[pickR]
            out[pickM] = mov[pickM]

        out[:, :, 3] = 255
        return out

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(*self.BACKGROUND))
        if self.ref is None or self.mov is None or self.width() < 2 or self.height() < 2:
            p.end()
            return

        out = self.compose()
        img = QImage(out.data, out.shape[1], out.shape[0], out.strides[0], QImage.Format_RGB32)
        p.drawImage(0, 0, img)

        if self.showPoints and len(self.refPoints) > 0:
            V = self.viewMatrix()
            r = AL.applyMatrix(V, self.refPoints)
            m = AL.applyMatrix(AL.composeMatrix(V, self.M), self.movPoints)
            p.setRenderHint(QPainter.Antialiasing)
            p.setFont(QFont("Roboto", 9, QFont.Bold))
            for (a, b, ident) in zip(r, m, self.pointIds):
                p.setPen(QPen(QColor(255, 255, 0), 1))
                p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))
                p.setPen(QPen(Qt.black, 1))
                p.setBrush(QColor(255, 255, 255))
                p.drawEllipse(QPointF(a[0], a[1]), 4, 4)
                p.setBrush(QColor(0, 255, 255))
                p.drawEllipse(QPointF(b[0], b[1]), 4, 4)
                p.setPen(Qt.white)
                p.drawText(QPointF(a[0] + 6, a[1] - 6), str(ident))
        p.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.lastPos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.lastPos is None:
            return
        d = event.pos() - self.lastPos
        self.lastPos = event.pos()
        self.cx -= d.x() / self.zoom
        self.cy -= d.y() / self.zoom
        self.update()
        self.viewChanged.emit(self.cx, self.cy, self.zoom)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.lastPos = None
            self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        f = math.pow(2.0, event.angleDelta().y() / 240.0)
        newZoom = min(max(self.zoom * f, self.minZoom), 32.0)
        # keep the point under the cursor fixed
        px = event.pos().x() - self.width() / 2.0
        py = event.pos().y() - self.height() / 2.0
        mx = self.cx + px / self.zoom
        my = self.cy + py / self.zoom
        self.zoom = newZoom
        self.cx = mx - px / self.zoom
        self.cy = my - py / self.zoom
        self.update()
        self.viewChanged.emit(self.cx, self.cy, self.zoom)


class MarkerObjData:
    """
    Marker data class: a pair of corresponding points, one on the reference image (left) and one on the image to
    align (right). Positions are in millimeters.
    """

    # Types
    SOFT_MARKER = 0
    HARD_MARKER = 1

    # Weights
    SOFT_MARKER_W = 1
    HARD_MARKER_W = 3

    # Draw properties (screen pixels)
    MARKER_SIZE = 8
    MARKER_WIDTH = 3
    PICK_RADIUS = 10
    HARD_MARKER_COLOR = QColor(Qt.red)
    SOFT_MARKER_COLOR = QColor(Qt.yellow)
    DISABLED_MARKER_COLOR = QColor(150, 150, 150)
    MARKER_HOVER_COLOR = QColor(Qt.white)

    def __init__(self, identifier: int, lpos: Optional[QPointF], rpos: Optional[QPointF], typ=SOFT_MARKER,
                 enabled=True):
        self.identifier = identifier
        self.lViewPos: Optional[QPointF] = lpos
        self.rViewPos: Optional[QPointF] = rpos
        self.typ = typ
        self.enabled = enabled
        self.errorx = None
        self.errory = None
        self.error = None
        self.items = [None, None]   # graphics items in the left and in the right view

    @property
    def weight(self) -> int:
        return MarkerObjData.HARD_MARKER_W if self.typ == MarkerObjData.HARD_MARKER else MarkerObjData.SOFT_MARKER_W

    def color(self) -> QColor:
        if not self.enabled:
            return MarkerObjData.DISABLED_MARKER_COLOR
        return MarkerObjData.HARD_MARKER_COLOR if self.typ == MarkerObjData.HARD_MARKER else MarkerObjData.SOFT_MARKER_COLOR

    def toggleType(self) -> None:
        self.typ = MarkerObjData.SOFT_MARKER if self.typ == MarkerObjData.HARD_MARKER else MarkerObjData.HARD_MARKER

    def isComplete(self) -> bool:
        return self.lViewPos is not None and self.rViewPos is not None

    def pos(self, isLeft: bool) -> Optional[QPointF]:
        return self.lViewPos if isLeft else self.rViewPos

    def setPos(self, pos: QPointF, isLeft: bool) -> None:
        if isLeft:
            self.lViewPos = QPointF(pos.x(), pos.y())
        else:
            self.rViewPos = QPointF(pos.x(), pos.y())


class QtAlignmentToolWidget(QWidget):
    """
    A custom widget that show two images and, with the help of the user, can align the right image to the left one.
    The user places markers (pairs of corresponding points, also found automatically) and the rigid (or
    similarity) transformation is estimated by least squares. A preview page shows the result before confirming.
    """

    closed = pyqtSignal()

    ROT_STEP = 0.01      # degrees per slider tick
    TRA_STEP = 0.1       # millimeters per slider tick

    def __init__(self, taglab_dir, project, parent=None):
        super(QtAlignmentToolWidget, self).__init__(parent)

        self.setStyleSheet("background-color: rgb(40,40,40); color: white")
        self.setAttribute(Qt.WA_DeleteOnClose)

        # ==============================================================

        self.taglab_dir = taglab_dir
        self.project = project
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.setMinimumWidth(1200)
        self.setMinimumHeight(600)
        self.setWindowTitle("Alignment Tool")
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowTitleHint)
        self.syncEnabled = True
        self.isPreviewMode = False
        self.pxSizeL = 1.0
        self.pxSizeR = 1.0
        self.markers: List[MarkerObjData] = []
        self.hoveringMarker = None      # (index, isLeft)
        self.draggingMarker = None      # (index, isLeft)
        self.lastMousePos = None
        self.fitRigid = None
        self.fitScaled = None
        self.baseMatrix = None          # fitted transformation (moving mm -> reference mm)
        self.matrix = None              # fitted transformation + manual adjustments
        self.dAngle = 0.0
        self.dTx = 0.0
        self.dTy = 0.0
        self.pyramids = {}
        self.updatingTable = False

        # ==============================================================
        # Top buttons
        # ==============================================================

        self.syncCheck = QCheckBox("Sync")
        self.syncCheck.setChecked(True)
        self.syncCheck.setFocusPolicy(Qt.NoFocus)
        self.syncCheck.setMaximumWidth(80)
        self.syncCheck.stateChanged[int].connect(self.toggleSync)

        self.autoMarkersButton = QPushButton("Auto Markers")
        self.autoMarkersButton.setFixedWidth(150)
        self.autoMarkersButton.setFixedHeight(20)
        self.autoMarkersButton.setToolTip("Find corresponding points automatically (feature matching)")
        self.autoMarkersButton.clicked.connect(self.onAutoMarkersRequested)

        self.previewButton = QPushButton("Preview Alignment")
        self.previewButton.setFixedWidth(200)
        self.previewButton.setFixedHeight(20)
        self.previewButton.clicked.connect(self.onPreviewRequested)

        self.backToEditButton = QPushButton("Back to Edit")
        self.backToEditButton.setFixedWidth(150)
        self.backToEditButton.setFixedHeight(20)
        self.backToEditButton.clicked.connect(self.onBackToEditRequested)

        self.clearMarkersButton = QPushButton("Clear Markers")
        self.clearMarkersButton.setFixedWidth(150)
        self.clearMarkersButton.setFixedHeight(20)
        self.clearMarkersButton.clicked.connect(self.onClearMarkersRequested)

        self.applyScaleCheck = QCheckBox("Apply scale")
        self.applyScaleCheck.setChecked(False)
        self.applyScaleCheck.setFocusPolicy(Qt.NoFocus)
        self.applyScaleCheck.setToolTip("Rescale the image to align to match the reference (changes its measured areas)")
        self.applyScaleCheck.stateChanged[int].connect(self.toggleApplyScale)

        self.showMarkersCheck = QCheckBox("Show Markers")
        self.showMarkersCheck.setChecked(True)
        self.showMarkersCheck.setFocusPolicy(Qt.NoFocus)
        self.showMarkersCheck.stateChanged[int].connect(self.toggleMarkersVisibility)

        self.resetTransfButton = QPushButton("Reset Adjustments")
        self.resetTransfButton.setFixedWidth(200)
        self.resetTransfButton.setFixedHeight(20)
        self.resetTransfButton.clicked.connect(self.onResetTransformations)

        self.confirmAlignmentButton = QPushButton("Confirm")
        self.confirmAlignmentButton.setFixedWidth(100)
        self.confirmAlignmentButton.setFixedHeight(20)
        self.confirmAlignmentButton.clicked.connect(self.onConfirmAlignment)

        self.helpLabel = QLabel("SHIFT + Left click: add marker (or toggle soft/hard)   Right drag: move marker   "
                                "DEL: delete the marker under the mouse")
        self.helpLabel.setStyleSheet("color: rgb(170,170,170)")

        self.buttons = QHBoxLayout()
        self.buttons.addWidget(self.syncCheck)
        self.buttons.addWidget(self.backToEditButton)
        self.buttons.addWidget(self.clearMarkersButton)
        self.buttons.addWidget(self.autoMarkersButton)
        self.buttons.addWidget(self.resetTransfButton)
        self.buttons.addWidget(self.showMarkersCheck)
        self.buttons.addWidget(self.applyScaleCheck)
        self.buttons.addStretch()
        self.buttons.addWidget(self.helpLabel)
        self.buttons.addStretch()
        self.buttons.addWidget(self.previewButton)
        self.buttons.addWidget(self.confirmAlignmentButton)

        # ==============================================================
        # Manual adjustments (preview)
        # ==============================================================

        def makeSlider(minimum, maximum, slot):
            slider = QSlider(Qt.Horizontal)
            slider.setFocusPolicy(Qt.StrongFocus)
            slider.setMinimum(minimum)
            slider.setMaximum(maximum)
            slider.setValue(0)
            slider.setMinimumWidth(50)
            slider.valueChanged.connect(slot)
            return slider

        def makeButton(text, slot):
            button = QPushButton(text)
            button.setFixedWidth(20)
            button.setFixedHeight(20)
            button.clicked.connect(slot)
            return button

        self.xSliderLabel = QLabel()
        self.xSliderLabel.setMinimumWidth(130)
        self.xSlider = makeSlider(-1000, 1000, self.xOffsetChanges)
        self.ySliderLabel = QLabel()
        self.ySliderLabel.setMinimumWidth(130)
        self.ySlider = makeSlider(-1000, 1000, self.yOffsetChanges)
        self.rSliderLabel = QLabel()
        self.rSliderLabel.setMinimumWidth(130)
        self.rSlider = makeSlider(-500, 500, self.rotationAngleChanges)
        self.moveRightButton = makeButton("+", lambda: self.xSlider.setValue(self.xSlider.value() + 1))
        self.moveLeftButton = makeButton("-", lambda: self.xSlider.setValue(self.xSlider.value() - 1))
        self.moveDownButton = makeButton("+", lambda: self.ySlider.setValue(self.ySlider.value() + 1))
        self.moveUpButton = makeButton("-", lambda: self.ySlider.setValue(self.ySlider.value() - 1))
        self.rotateLeftButton = makeButton("+", lambda: self.rSlider.setValue(self.rSlider.value() + 1))
        self.rotateRightButton = makeButton("-", lambda: self.rSlider.setValue(self.rSlider.value() - 1))

        self.layoutSliders = QVBoxLayout()
        for (label, slider, b1, b2) in [(self.xSliderLabel, self.xSlider, self.moveRightButton, self.moveLeftButton),
                                        (self.ySliderLabel, self.ySlider, self.moveDownButton, self.moveUpButton),
                                        (self.rSliderLabel, self.rSlider, self.rotateLeftButton, self.rotateRightButton)]:
            row = QHBoxLayout()
            row.addWidget(label)
            row.addWidget(slider)
            row.addWidget(b1)
            row.addWidget(b2)
            self.layoutSliders.addLayout(row)
        self.__updateSliderLabels()

        # ==============================================================
        # Markers table and fit statistics
        # ==============================================================

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Use", "Id", "Type", "X err (mm)", "Y err (mm)", "Dist err (mm)"])
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setMaximumHeight(170)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setStyleSheet("QTableCornerButton::section { background-color: rgb(40,40,40); }"
                                 "QHeaderView::section { background-color: rgb(40,40,40); }")
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for c in range(1, 6):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
        self.table.itemChanged[QTableWidgetItem].connect(self.onTableItemChanged)
        self.table.cellClicked[int, int].connect(self.onTableCellClicked)

        self.statsLabel = QLabel()
        self.statsLabel.setMinimumWidth(320)
        self.statsLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.statsLabel.setWordWrap(True)

        self.layoutTop = QHBoxLayout()
        self.layoutTop.addLayout(self.layoutSliders)
        self.layoutTop.addSpacing(30)
        self.layoutTop.addWidget(self.table)
        self.layoutTop.addSpacing(30)
        self.layoutTop.addWidget(self.statsLabel)
        self.layoutTop.setStretchFactor(self.table, 3)
        self.layoutTop.setStretchFactor(self.statsLabel, 1)

        # ==============================================================
        # Edit views
        # ==============================================================

        self.leftComboboxLabel = QLabel("Reference Image")
        self.leftCombobox = QComboBox()
        self.rightComboboxLabel = QLabel("Image to align")
        self.rightCombobox = QComboBox()
        for image in self.project.images:
            self.leftCombobox.addItem(image.name)
            self.rightCombobox.addItem(image.name)
        self.leftCombobox.setCurrentIndex(0)
        self.rightCombobox.setCurrentIndex(1)
        self.leftCombobox.currentIndexChanged.connect(self.leftImageChanges)
        self.rightCombobox.currentIndexChanged.connect(self.rightImageChanges)

        self.leftImgViewer = QtImageViewer()
        self.rightImgViewer = QtImageViewer()
        for (viewer, isLeft) in [(self.leftImgViewer, True), (self.rightImgViewer, False)]:
            viewer.setOpacity(1)
            viewer.mouseDown.connect(lambda event, isLeft=isLeft: self.__onMouseDown(event, isLeft))
            viewer.mouseUp.connect(lambda event, isLeft=isLeft: self.__onMouseUp(event, isLeft))
            viewer.mouseMove.connect(lambda event, isLeft=isLeft: self.__onMouseMove(event, isLeft))
            viewer.mouseOut.connect(lambda isLeft=isLeft: self.__onMouseOut(isLeft))
        self.leftImgViewer.viewHasChanged[float, float, float].connect(self.leftImgViewerParamsChanges)
        self.rightImgViewer.viewHasChanged[float, float, float].connect(self.rightImgViewerParamsChanges)

        layout8 = QHBoxLayout()
        layout8.addWidget(self.leftComboboxLabel)
        layout8.addWidget(self.leftCombobox)
        layout8.setStretchFactor(self.leftComboboxLabel, 1)
        layout8.setStretchFactor(self.leftCombobox, 1)
        leftLayout = QVBoxLayout()
        leftLayout.addLayout(layout8)
        leftLayout.addWidget(self.leftImgViewer)

        layout9 = QHBoxLayout()
        layout9.addWidget(self.rightComboboxLabel)
        layout9.addWidget(self.rightCombobox)
        layout9.setStretchFactor(self.rightComboboxLabel, 1)
        layout9.setStretchFactor(self.rightCombobox, 1)
        rightLayout = QVBoxLayout()
        rightLayout.addLayout(layout9)
        rightLayout.addWidget(self.rightImgViewer)

        self.editLayout = QHBoxLayout()
        self.editLayout.addLayout(leftLayout)
        self.editLayout.setStretchFactor(leftLayout, 1)
        self.editLayout.addLayout(rightLayout)
        self.editLayout.setStretchFactor(rightLayout, 1)

        # ==============================================================
        # Preview
        # ==============================================================

        self.leftPreviewViewer = AlignmentPreviewPane(AlignmentPreviewPane.BLEND)
        self.rightPreviewViewer = AlignmentPreviewPane(AlignmentPreviewPane.DIFFERENCE)
        self.leftPreviewViewer.viewChanged[float, float, float].connect(self.rightPreviewViewer.setView)
        self.rightPreviewViewer.viewChanged[float, float, float].connect(self.leftPreviewViewer.setView)

        self.leftModeCombobox = QComboBox()
        self.leftModeCombobox.addItem("Blend")
        self.leftModeCombobox.addItem("Checkerboard")
        self.leftModeCombobox.currentIndexChanged[int].connect(self.leftPreviewModeChanges)

        self.alphaSliderLabel = QLabel("Alpha: 50")
        self.alphaSliderLabel.setMinimumWidth(110)
        self.alphaSlider = makeSlider(0, 100, self.previewAlphaValueChanges)
        self.alphaSlider.setValue(50)

        self.thresholdSliderLabel = QLabel("Threshold: 32")
        self.thresholdSliderLabel.setMinimumWidth(110)
        self.thresholdSlider = makeSlider(0, 255, self.thresholdValueChanges)
        self.thresholdSlider.setValue(32)

        layoutAS = QHBoxLayout()
        layoutAS.addWidget(self.leftModeCombobox)
        layoutAS.addWidget(self.alphaSliderLabel)
        layoutAS.addWidget(self.alphaSlider)

        layoutDIFF = QHBoxLayout()
        layoutDIFF.addWidget(QLabel("Difference"))
        layoutDIFF.addWidget(self.thresholdSliderLabel)
        layoutDIFF.addWidget(self.thresholdSlider)

        self.previewLayoutL = QVBoxLayout()
        self.previewLayoutL.addLayout(layoutAS)
        self.previewLayoutL.addWidget(self.leftPreviewViewer)
        self.previewLayoutR = QVBoxLayout()
        self.previewLayoutR.addLayout(layoutDIFF)
        self.previewLayoutR.addWidget(self.rightPreviewViewer)
        self.previewLayout = QHBoxLayout()
        self.previewLayout.addLayout(self.previewLayoutL)
        self.previewLayout.addLayout(self.previewLayoutR)
        self.previewOnlyWidgets = [self.leftModeCombobox, self.alphaSliderLabel, self.alphaSlider,
                                   self.thresholdSliderLabel, self.thresholdSlider, self.leftPreviewViewer,
                                   self.rightPreviewViewer]
        for i in range(layoutDIFF.count()):
            if layoutDIFF.itemAt(i).widget() is not None:
                self.previewOnlyWidgets.append(layoutDIFF.itemAt(i).widget())

        # ==============================================================
        # Layout
        # ==============================================================

        content = QVBoxLayout()
        content.addLayout(self.buttons)
        content.addLayout(self.layoutTop)
        content.addLayout(self.editLayout)
        content.addLayout(self.previewLayout)
        self.setLayout(content)

        self.__updateImgViewers()
        self.__togglePreviewMode(False)

    # ==================================================================================================================
    # Window events
    # ==================================================================================================================

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closed.emit()
        super(QtAlignmentToolWidget, self).closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Delete and self.hoveringMarker is not None and not self.isPreviewMode:
            i, _ = self.hoveringMarker
            self.hoveringMarker = None
            self.__deleteMarker(i)
            self.__markersChanged()

        if event.modifiers() & Qt.ShiftModifier:
            self.rightImgViewer.disablePan()
            self.leftImgViewer.disablePan()

        super(QtAlignmentToolWidget, self).keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if not (event.modifiers() & Qt.ShiftModifier):
            self.rightImgViewer.enablePan()
            self.leftImgViewer.enablePan()
        super(QtAlignmentToolWidget, self).keyReleaseEvent(event)

    # ==================================================================================================================
    # Images
    # ==================================================================================================================

    def __images(self):
        return self.project.images[self.leftCombobox.currentIndex()], self.project.images[self.rightCombobox.currentIndex()]

    @pyqtSlot(int)
    def leftImageChanges(self, index: int) -> None:
        N = len(self.project.images)
        if index == -1 or index >= N:
            return
        if index == self.rightCombobox.currentIndex():
            self.rightCombobox.blockSignals(True)
            self.rightCombobox.setCurrentIndex((index + 1) % N)
            self.rightCombobox.blockSignals(False)
        self.__updateImgViewers()

    @pyqtSlot(int)
    def rightImageChanges(self, index: int) -> None:
        N = len(self.project.images)
        if index == -1 or index >= N:
            return
        if index == self.leftCombobox.currentIndex():
            self.leftCombobox.blockSignals(True)
            self.leftCombobox.setCurrentIndex((index + 1) % N)
            self.leftCombobox.blockSignals(False)
        self.__updateImgViewers()

    def __rgbChannel(self, image):
        channel = image.getRGBChannel()
        if channel is None:
            channel = image.channels[0]
        if channel.qimage is None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                channel.loadData(self.taglab_dir)
            finally:
                QApplication.restoreOverrideCursor()
        return channel

    def __updateImgViewers(self) -> None:
        """
        Show the selected images and restore the markers saved for this pair of images.
        """
        image1, image2 = self.__images()
        self.pxSizeL = image1.pixelSize()
        self.pxSizeR = image2.pixelSize()
        channel1 = self.__rgbChannel(image1)
        channel2 = self.__rgbChannel(image2)

        self.__removeAllMarkerItems()
        self.markers = []
        self.hoveringMarker = None
        self.draggingMarker = None

        self.leftImgViewer.setImg(channel1.qimage)
        self.leftImgViewer.px_to_mm = self.pxSizeL
        self.rightImgViewer.setImg(channel2.qimage)
        self.rightImgViewer.px_to_mm = self.pxSizeR

        self.onResetTransformations()

        saved = self.project.retrieveMarkersOrEmpty(refImgId=image1.id, coregImgId=image2.id)
        for (i, m) in enumerate(saved):
            marker = MarkerObjData(i + 1, QPointF(m['refPos'][0], m['refPos'][1]),
                                   QPointF(m['coregPos'][0], m['coregPos'][1]), m['type'], m.get('enabled', True))
            self.markers.append(marker)

        self.applyScaleCheck.blockSignals(True)
        self.applyScaleCheck.setChecked(self.project.retrieveAlignmentApplyScale(image1.id, image2.id))
        self.applyScaleCheck.blockSignals(False)

        self.__refit()

    # ==================================================================================================================
    # Sync / visibility / options
    # ==================================================================================================================

    @pyqtSlot(float, float, float)
    def leftImgViewerParamsChanges(self, posx: float, posy: float, zoom: float) -> None:
        if self.syncEnabled:
            self.rightImgViewer.setViewParameters(posx, posy, zoom)

    @pyqtSlot(float, float, float)
    def rightImgViewerParamsChanges(self, posx: float, posy: float, zoom: float) -> None:
        if self.syncEnabled:
            self.leftImgViewer.setViewParameters(posx, posy, zoom)

    @pyqtSlot(int)
    def toggleSync(self, value: int) -> None:
        self.syncEnabled = value != 0

    @pyqtSlot(int)
    def toggleMarkersVisibility(self, value: int) -> None:
        self.leftPreviewViewer.setPointsVisibility(value != 0)
        self.rightPreviewViewer.setPointsVisibility(value != 0)

    @pyqtSlot(int)
    def toggleApplyScale(self, value: int) -> None:
        self.__refit()
        self.__storeMarkers()

    # ==================================================================================================================
    # Preview
    # ==================================================================================================================

    def __pyramid(self, qimage: QImage) -> ImagePyramid:
        key = qimage.cacheKey()
        if key not in self.pyramids:
            self.pyramids[key] = ImagePyramid(qimage)
        return self.pyramids[key]

    def __previewMatrix(self):
        """
        Current transformation in pixels (image to align -> reference image).
        """
        return AL.pixelMatrix(self.matrix, self.pxSizeL, self.pxSizeR)

    def __updatePreview(self) -> None:
        if not self.isPreviewMode or self.matrix is None:
            return
        M = self.__previewMatrix()
        complete = [m for m in self.markers if m.isComplete() and m.enabled]
        refPts = [[m.lViewPos.x() / self.pxSizeL, m.lViewPos.y() / self.pxSizeL] for m in complete]
        movPts = [[m.rViewPos.x() / self.pxSizeR, m.rViewPos.y() / self.pxSizeR] for m in complete]
        ids = [m.identifier for m in complete]
        for pane in [self.leftPreviewViewer, self.rightPreviewViewer]:
            pane.setTransform(M)
            pane.setPoints(refPts, movPts, ids)

    @pyqtSlot()
    def onPreviewRequested(self) -> None:
        if self.matrix is None:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("Alignment Tool")
            msgBox.setText("At least 3 complete markers are required (place them with SHIFT + Left click on both "
                           "images, or use Auto Markers).")
            msgBox.exec()
            return
        image1, image2 = self.__images()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ref = self.__pyramid(self.__rgbChannel(image1).qimage)
            mov = self.__pyramid(self.__rgbChannel(image2).qimage)
        finally:
            QApplication.restoreOverrideCursor()
        self.__togglePreviewMode(True)
        QApplication.processEvents()
        self.leftPreviewViewer.setImages(ref, mov)
        self.rightPreviewViewer.setImages(ref, mov)
        self.__updatePreview()

    @pyqtSlot()
    def onBackToEditRequested(self) -> None:
        self.__togglePreviewMode(False)

    @pyqtSlot(int)
    def leftPreviewModeChanges(self, index: int) -> None:
        if index == 0:
            self.leftPreviewViewer.setMode(AlignmentPreviewPane.BLEND)
            self.alphaSliderLabel.setText("Alpha: " + str(self.alphaSlider.value()))
            self.alphaSlider.setRange(0, 100)
            self.alphaSlider.setValue(50)
        else:
            self.leftPreviewViewer.setMode(AlignmentPreviewPane.CHECKERBOARD)
            self.alphaSlider.setRange(8, 512)
            self.alphaSlider.setValue(64)

    @pyqtSlot(int)
    def previewAlphaValueChanges(self, value: int) -> None:
        if self.leftModeCombobox.currentIndex() == 0:
            self.alphaSliderLabel.setText("Alpha: " + str(value))
            self.leftPreviewViewer.setAlpha(value / 100.0)
        else:
            self.alphaSliderLabel.setText("Tile: " + str(value) + " px")
            self.leftPreviewViewer.setCheckerSize(value)

    @pyqtSlot(int)
    def thresholdValueChanges(self, value: int) -> None:
        self.thresholdSliderLabel.setText("Threshold: " + str(value))
        self.rightPreviewViewer.setThreshold(value)

    def __togglePreviewMode(self, isPreviewMode: bool) -> None:
        self.isPreviewMode = isPreviewMode
        for w in self.previewOnlyWidgets + [self.backToEditButton, self.resetTransfButton, self.showMarkersCheck,
                                             self.confirmAlignmentButton, self.xSliderLabel, self.xSlider,
                                             self.moveLeftButton, self.moveRightButton, self.ySliderLabel, self.ySlider,
                                             self.moveUpButton, self.moveDownButton, self.rSliderLabel, self.rSlider,
                                             self.rotateLeftButton, self.rotateRightButton]:
            w.setVisible(isPreviewMode)
        for w in [self.leftImgViewer, self.rightImgViewer, self.clearMarkersButton, self.syncCheck,
                  self.previewButton, self.autoMarkersButton, self.helpLabel, self.leftComboboxLabel,
                  self.leftCombobox, self.rightComboboxLabel, self.rightCombobox]:
            w.setVisible(not isPreviewMode)

    # ==================================================================================================================
    # Manual adjustments
    # ==================================================================================================================

    def __updateSliderLabels(self) -> None:
        self.xSliderLabel.setText("Adjust Tx: {:+.1f} mm".format(self.dTx))
        self.ySliderLabel.setText("Adjust Ty: {:+.1f} mm".format(self.dTy))
        self.rSliderLabel.setText("Adjust R: {:+.2f}°".format(self.dAngle))

    @pyqtSlot(int)
    def xOffsetChanges(self, value: int) -> None:
        self.dTx = value * QtAlignmentToolWidget.TRA_STEP
        self.__adjustmentsChanged()

    @pyqtSlot(int)
    def yOffsetChanges(self, value: int) -> None:
        self.dTy = value * QtAlignmentToolWidget.TRA_STEP
        self.__adjustmentsChanged()

    @pyqtSlot(int)
    def rotationAngleChanges(self, value: int) -> None:
        self.dAngle = value * QtAlignmentToolWidget.ROT_STEP
        self.__adjustmentsChanged()

    def __adjustmentsChanged(self) -> None:
        self.__updateSliderLabels()
        self.__applyAdjustments()
        self.__updateErrors()
        self.__updatePreview()

    @pyqtSlot()
    def onResetTransformations(self) -> None:
        for slider in [self.xSlider, self.ySlider, self.rSlider]:
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        self.dTx = self.dTy = self.dAngle = 0.0
        self.__adjustmentsChanged()

    def __applyAdjustments(self) -> None:
        if self.baseMatrix is None:
            self.matrix = None
            return
        # the rotation adjustment is around the center of the image to align
        image1, image2 = self.__images()
        center = [image2.width * self.pxSizeR / 2.0, image2.height * self.pxSizeR / 2.0] \
            if image2.width is not None else [0.0, 0.0]
        pivot = AL.applyMatrix(self.baseMatrix, [center])[0]
        self.matrix = AL.adjustMatrix(self.baseMatrix, self.dAngle, self.dTx, self.dTy, pivot)

    # ==================================================================================================================
    # Fit
    # ==================================================================================================================

    def __usableMarkers(self):
        return [m for m in self.markers if m.isComplete() and m.enabled]

    def __refit(self) -> None:
        """
        Estimate the transformation from the enabled complete markers and update errors, table and labels.
        """
        usable = self.__usableMarkers()
        if len(usable) >= 3:
            ref = [[m.lViewPos.x(), m.lViewPos.y()] for m in usable]
            mov = [[m.rViewPos.x(), m.rViewPos.y()] for m in usable]
            weights = [m.weight for m in usable]
            self.fitRigid = AL.fitTransform(ref, mov, weights, with_scale=False)
            self.fitScaled = AL.fitTransform(ref, mov, weights, with_scale=True)
            fit = self.fitScaled if self.applyScaleCheck.isChecked() else self.fitRigid
            self.baseMatrix = fit['matrix']
        else:
            self.fitRigid = self.fitScaled = None
            self.baseMatrix = None
        self.__applyAdjustments()
        self.__updateErrors()
        self.__updatePreview()

    def __updateErrors(self) -> None:
        for m in self.markers:
            if self.matrix is not None and m.isComplete():
                p = AL.applyMatrix(self.matrix, [[m.rViewPos.x(), m.rViewPos.y()]])[0]
                m.errorx = p[0] - m.lViewPos.x()
                m.errory = p[1] - m.lViewPos.y()
                m.error = math.hypot(m.errorx, m.errory)
            else:
                m.errorx = m.errory = m.error = None
        self.__drawMarkers()
        self.__fillTable()
        self.__updateStats()

    def __outlierThreshold(self) -> Optional[float]:
        errors = [m.error for m in self.__usableMarkers() if m.error is not None]
        if len(errors) < 4:
            return None
        rmse = math.sqrt(np.mean(np.square(errors)))
        return max(3.0 * float(np.median(errors)), 2.0 * rmse, 1.0)

    def __updateStats(self) -> None:
        usable = self.__usableMarkers()
        complete = [m for m in self.markers if m.isComplete()]
        if self.matrix is None:
            self.statsLabel.setText("Markers: {:d} complete, {:d} used.\nAt least 3 are needed to compute the "
                                    "alignment.".format(len(complete), len(usable)))
            return
        errors = np.array([m.error for m in usable])
        rmse = math.sqrt(np.mean(errors ** 2))
        angle, scale, _, _ = AL.matrixToParams(self.matrix)
        fitted_scale = self.fitScaled['scale'] if self.fitScaled is not None else 1.0
        txt = "Markers used: {:d} of {:d}\n".format(len(usable), len(complete))
        txt += "RMSE: {:.2f} mm   max: {:.2f} mm\n".format(rmse, errors.max())
        txt += "Rotation: {:.3f}°\n".format(angle)
        txt += "Fitted scale: {:.4f} ({:+.2f} %) - {}".format(
            fitted_scale, (fitted_scale - 1.0) * 100.0, "applied" if self.applyScaleCheck.isChecked() else "not applied")
        thr = self.__outlierThreshold()
        if thr is not None:
            n_out = len([m for m in usable if m.error > thr])
            if n_out > 0:
                txt += "\n{:d} marker(s) with a large error (red rows)".format(n_out)
        self.statsLabel.setText(txt)

    # ==================================================================================================================
    # Markers table
    # ==================================================================================================================

    def __fillTable(self) -> None:
        self.updatingTable = True
        self.table.blockSignals(True)
        complete = [m for m in self.markers if m.isComplete()]
        self.table.setRowCount(len(complete))
        thr = self.__outlierThreshold()
        for (row, m) in enumerate(complete):
            chbx = QTableWidgetItem()
            chbx.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chbx.setCheckState(Qt.Checked if m.enabled else Qt.Unchecked)
            chbx.setData(Qt.UserRole, m.identifier)
            self.table.setItem(row, 0, chbx)
            values = [str(m.identifier), "hard" if m.typ == MarkerObjData.HARD_MARKER else "soft"]
            for v in [m.errorx, m.errory, m.error]:
                values.append("" if v is None else "{:.2f}".format(v))
            for (c, v) in enumerate(values):
                item = QTableWidgetItem(v)
                if thr is not None and m.enabled and m.error is not None and m.error > thr:
                    item.setBackground(QColor(120, 30, 30))
                self.table.setItem(row, c + 1, item)
        self.table.blockSignals(False)
        self.updatingTable = False

    def __markerById(self, identifier) -> Optional[MarkerObjData]:
        for m in self.markers:
            if m.identifier == identifier:
                return m
        return None

    @pyqtSlot(QTableWidgetItem)
    def onTableItemChanged(self, item: QTableWidgetItem) -> None:
        if self.updatingTable or item.column() != 0:
            return
        marker = self.__markerById(item.data(Qt.UserRole))
        if marker is None:
            return
        enable = item.checkState() == Qt.Checked
        if not enable and len(self.__usableMarkers()) <= 3:
            box = QMessageBox()
            box.setWindowTitle("Alignment Tool")
            box.setText("At least 3 markers must be used to compute the alignment.")
            box.exec()
            self.__fillTable()
            return
        marker.enabled = enable
        self.__markersChanged()

    @pyqtSlot(int, int)
    def onTableCellClicked(self, row: int, column: int) -> None:
        if column == 0 or self.isPreviewMode:
            return
        item = self.table.item(row, 0)
        marker = self.__markerById(item.data(Qt.UserRole)) if item is not None else None
        if marker is None:
            return
        for (viewer, pos, px) in [(self.leftImgViewer, marker.lViewPos, self.pxSizeL),
                                  (self.rightImgViewer, marker.rViewPos, self.pxSizeR)]:
            if viewer.imgwidth > 0:
                viewer.blockSignals(True)
                viewer.centerOn(pos.x() / px, pos.y() / px)
                viewer.blockSignals(False)

    # ==================================================================================================================
    # Markers editing
    # ==================================================================================================================

    def __viewer(self, isLeft: bool) -> QtImageViewer:
        return self.leftImgViewer if isLeft else self.rightImgViewer

    def __pxSize(self, isLeft: bool) -> float:
        return self.pxSizeL if isLeft else self.pxSizeR

    def __mapToViewer(self, pos, isLeft: bool) -> QPointF:
        viewer = self.__viewer(isLeft)
        tmp = viewer.mapToScene(pos)
        return QPointF(min(max(tmp.x(), 0), viewer.imgwidth), min(max(tmp.y(), 0), viewer.imgheight))

    def __findMarkerAt(self, pos: QPointF, isLeft: bool) -> Optional[int]:
        """
        Index of the marker within PICK_RADIUS screen pixels from pos (scene coordinates), or None.
        """
        viewer = self.__viewer(isLeft)
        px = self.__pxSize(isLeft)
        best = None
        bestDist = MarkerObjData.PICK_RADIUS
        for (i, m) in enumerate(self.markers):
            mp = m.pos(isLeft)
            if mp is None:
                continue
            d = math.hypot(mp.x() / px - pos.x(), mp.y() / px - pos.y()) * viewer.zoom_factor
            if d <= bestDist:
                best = i
                bestDist = d
        return best

    def __nextIdentifier(self) -> int:
        return max([m.identifier for m in self.markers], default=0) + 1

    def __addMarkerSide(self, posMm: QPointF, isLeft: bool) -> None:
        """
        Add a point on one side: it completes the first marker that misses this side, otherwise a new marker is
        created.
        """
        for m in self.markers:
            if m.pos(isLeft) is None:
                m.setPos(posMm, isLeft)
                return
        m = MarkerObjData(self.__nextIdentifier(), None, None)
        m.setPos(posMm, isLeft)
        self.markers.append(m)

    def __deleteMarker(self, i: int) -> None:
        self.__removeMarkerItems(self.markers[i])
        del self.markers[i]

    @pyqtSlot()
    def onClearMarkersRequested(self) -> None:
        if len(self.markers) == 0:
            return
        reply = QMessageBox.question(self, "Alignment Tool", "Delete all the {:d} markers?".format(len(self.markers)),
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.__removeAllMarkerItems()
        self.markers = []
        self.hoveringMarker = None
        self.__markersChanged()

    def __markersChanged(self) -> None:
        self.__refit()
        self.__storeMarkers()

    def __onMouseDown(self, event: QMouseEvent, isLeft: bool) -> None:
        pos = self.__mapToViewer(event.pos(), isLeft)
        self.lastMousePos = pos
        hovering = self.__findMarkerAt(pos, isLeft)

        if event.modifiers() & Qt.ShiftModifier and event.button() == Qt.LeftButton:
            if hovering is None:
                px = self.__pxSize(isLeft)
                self.__addMarkerSide(QPointF(pos.x() * px, pos.y() * px), isLeft)
            else:
                self.markers[hovering].toggleType()
            self.__markersChanged()

        if event.button() == Qt.RightButton and hovering is not None:
            self.draggingMarker = (hovering, isLeft)

    def __onMouseUp(self, event: QMouseEvent, isLeft: bool) -> None:
        if self.draggingMarker is not None:
            self.draggingMarker = None
            self.__markersChanged()

    def __onMouseMove(self, event: QMouseEvent, isLeft: bool) -> None:
        pos = self.__mapToViewer(event.pos(), isLeft)

        if self.draggingMarker is not None and (event.buttons() & Qt.RightButton):
            i, side = self.draggingMarker
            if side == isLeft and self.lastMousePos is not None:
                px = self.__pxSize(isLeft)
                m = self.markers[i]
                old = m.pos(isLeft)
                m.setPos(QPointF(old.x() + (pos.x() - self.lastMousePos.x()) * px,
                                 old.y() + (pos.y() - self.lastMousePos.y()) * px), isLeft)
                self.lastMousePos = pos
                self.__drawMarker(m)
            return

        self.lastMousePos = pos
        hovering = self.__findMarkerAt(pos, isLeft)
        newHover = (hovering, isLeft) if hovering is not None else None
        if newHover != self.hoveringMarker:
            old = self.hoveringMarker
            self.hoveringMarker = newHover
            for h in [old, newHover]:
                if h is not None and h[0] < len(self.markers):
                    self.__drawMarker(self.markers[h[0]])

    def __onMouseOut(self, isLeft: bool) -> None:
        if self.draggingMarker is not None:
            self.draggingMarker = None
            self.__markersChanged()
        if self.hoveringMarker is not None:
            old = self.hoveringMarker
            self.hoveringMarker = None
            if old[0] < len(self.markers):
                self.__drawMarker(self.markers[old[0]])

    # ==================================================================================================================
    # Markers drawing (constant size on screen)
    # ==================================================================================================================

    def __createMarkerItem(self) -> QGraphicsPathItem:
        s = MarkerObjData.MARKER_SIZE
        path = QPainterPath()
        path.moveTo(-s, -s)
        path.lineTo(s, s)
        path.moveTo(-s, s)
        path.lineTo(s, -s)
        item = QGraphicsPathItem(path)
        item.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        item.setZValue(5)
        item.label = QGraphicsTextItem(item)
        item.label.setFont(QFont("Roboto", 10, QFont.Bold))
        item.label.setDefaultTextColor(Qt.black)
        item.errLabel = QGraphicsTextItem(item)
        item.errLabel.setFont(QFont("Roboto", 9))
        item.errLabel.setDefaultTextColor(Qt.black)
        return item

    def __drawMarker(self, m: MarkerObjData) -> None:
        index = self.markers.index(m) if m in self.markers else -1
        for side, isLeft in [(0, True), (1, False)]:
            pos = m.pos(isLeft)
            viewer = self.__viewer(isLeft)
            if pos is None:
                if m.items[side] is not None:
                    viewer.scene.removeItem(m.items[side])
                    m.items[side] = None
                continue
            if m.items[side] is None:
                m.items[side] = self.__createMarkerItem()
                viewer.scene.addItem(m.items[side])
            item = m.items[side]
            px = self.__pxSize(isLeft)
            item.setPos(pos.x() / px, pos.y() / px)
            hovered = self.hoveringMarker == (index, isLeft)
            color = MarkerObjData.MARKER_HOVER_COLOR if hovered else m.color()
            pen = QPen(color, MarkerObjData.MARKER_WIDTH + (2 if hovered else 0))
            pen.setCosmetic(True)
            item.setPen(pen)
            bg = m.color().name()
            item.label.setHtml('<div style="background:' + bg + ';">&nbsp;' + str(m.identifier) + '&nbsp;</div>')
            item.label.adjustSize()
            item.label.setPos(-item.label.boundingRect().width() / 2.0,
                              -MarkerObjData.MARKER_SIZE - item.label.boundingRect().height())
            if m.error is not None:
                item.errLabel.setHtml('<div style="background:' + bg + ';">&nbsp;{:.1f} mm&nbsp;</div>'.format(m.error))
                item.errLabel.setVisible(True)
                item.errLabel.adjustSize()
                item.errLabel.setPos(-item.errLabel.boundingRect().width() / 2.0, MarkerObjData.MARKER_SIZE)
            else:
                item.errLabel.setVisible(False)

    def __drawMarkers(self) -> None:
        for m in self.markers:
            self.__drawMarker(m)

    def __removeMarkerItems(self, m: MarkerObjData) -> None:
        for side, viewer in [(0, self.leftImgViewer), (1, self.rightImgViewer)]:
            if m.items[side] is not None:
                viewer.scene.removeItem(m.items[side])
                m.items[side] = None

    def __removeAllMarkerItems(self) -> None:
        for m in self.markers:
            self.__removeMarkerItems(m)

    # ==================================================================================================================
    # Markers persistence
    # ==================================================================================================================

    def __storeMarkers(self) -> None:
        image1, image2 = self.__images()
        complete = [m for m in self.markers if m.isComplete()]
        self.project.addOrUpdateMarkers(
            refImgId=image1.id,
            refMarkers=[[m.lViewPos.x(), m.lViewPos.y()] for m in complete],
            coregImgId=image2.id,
            coregMarkers=[[m.rViewPos.x(), m.rViewPos.y()] for m in complete],
            markersTypes=[m.typ for m in complete],
            markersEnabled=[m.enabled for m in complete],
            applyScale=self.applyScaleCheck.isChecked())

    # ==================================================================================================================
    # Automatic markers
    # ==================================================================================================================

    @pyqtSlot()
    def onAutoMarkersRequested(self) -> None:
        image1, image2 = self.__images()
        ch1 = image1.getRGBChannel()
        ch2 = image2.getRGBChannel()
        if ch1 is None or ch2 is None:
            box = QMessageBox()
            box.setText("Both images need an RGB channel.")
            box.exec()
            return

        progress = QProgressDialog("Searching corresponding points..", None, 0, 100, self)
        progress.setWindowTitle("Auto Markers")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        def report(perc, msg):
            progress.setLabelText(msg)
            progress.setValue(int(perc))
            QApplication.processEvents()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pairs, info = AL.autoMarkers(os.path.join(self.taglab_dir, ch1.filename),
                                         os.path.join(self.taglab_dir, ch2.filename),
                                         self.pxSizeL, self.pxSizeR, max_markers=30, progress=report)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            progress.close()
            box = QMessageBox()
            box.setWindowTitle("Auto Markers")
            box.setText("Corresponding points could not be found automatically:\n" + str(e) +
                        "\n\nPlace a few markers by hand (SHIFT + Left click).")
            box.exec()
            return
        QApplication.restoreOverrideCursor()
        progress.close()

        for (refPos, movPos) in pairs:
            m = MarkerObjData(self.__nextIdentifier(), QPointF(refPos[0], refPos[1]), QPointF(movPos[0], movPos[1]))
            self.markers.append(m)
        self.__markersChanged()

        box = QMessageBox()
        box.setWindowTitle("Auto Markers")
        box.setText("{:d} markers have been added ({:d} feature matches, {:d} consistent).\n"
                    "Check the table: markers with a large error are shown in red, uncheck or delete them.".format(
                        info['markers'], info['matches'], info['inliers']))
        box.exec()

    # ==================================================================================================================
    # Confirm
    # ==================================================================================================================

    @pyqtSlot()
    def onConfirmAlignment(self) -> None:
        if self.matrix is None:
            return

        index1 = self.leftCombobox.currentIndex()
        index2 = self.rightCombobox.currentIndex()
        image1 = self.project.images[index1]
        image2 = self.project.images[index2]
        ref_w, ref_h = image1.width, image1.height
        mov_w, mov_h = image2.width, image2.height

        M0 = AL.pixelMatrix(self.matrix, self.pxSizeL, self.pxSizeR)
        canvas = AL.unionCanvas(M0, mov_w, mov_h, ref_w, ref_h)
        left, top = canvas['left'], canvas['top']
        needRef = left > 0 or top > 0
        appliedScale = AL.matrixToParams(self.matrix)[1] * 1.0
        scaleApplied = self.applyScaleCheck.isChecked()

        suffix2 = AL.uniqueSuffix(self.project, image2, "_coreg", self.taglab_dir)
        suffix1 = AL.uniqueSuffix(self.project, image1, "_ref", self.taglab_dir) if needRef else None

        txt = "The image '{}' will be aligned to '{}' and added to the project as '{}' ({} x {} pixels, " \
              "pixel size {:.4f} mm).".format(image2.name, image1.name, image2.id + suffix2, canvas['width'],
                                              canvas['height'], self.pxSizeL)
        if needRef:
            txt += "\n\nThe canvas extends beyond the left/top side of the reference, so a padded copy of the " \
                   "reference, '{}', is also created (shifted by {} px right, {} px down).".format(
                        image1.id + suffix1, left, top)
        txt += "\n\nScale: " + ("the fitted scale is applied (measured areas of the aligned image change)."
                                if scaleApplied else "not applied.")
        txt += "\nRegions and points are transformed. Annotation grids and shape layers are not copied."
        txt += "\n\nProceed?"
        reply = QMessageBox.question(self, "Alignment Tool", txt, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        M = AL.pixelMatrix(self.matrix, self.pxSizeL, self.pxSizeR, (left, top))
        crs, geoTransform = AL.georeferenceOfCanvas(image1.georef_filename, left, top)

        nimages = 2 if needRef else 1
        progress = QProgressDialog("Aligning..", None, 0, 100, self)
        progress.setWindowTitle("Alignment Tool")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        def report(k):
            def cb(perc, msg):
                progress.setLabelText(msg)
                progress.setValue(int((k * 100.0 + perc) / nimages))
                QApplication.processEvents()
            return cb

        QApplication.setOverrideCursor(Qt.WaitCursor)
        created = []
        try:
            cpy2 = AL.createTransformedImage(image2, suffix2, M, canvas['width'], canvas['height'], crs, geoTransform,
                                             self.pxSizeL, self.taglab_dir,
                                             appliedScale if scaleApplied else 1.0, report(0))
            created.append(cpy2)
            if needRef:
                T = np.array([[1.0, 0.0, float(left)], [0.0, 1.0, float(top)]])
                cpy1 = AL.createTransformedImage(image1, suffix1, T, canvas['width'], canvas['height'], crs,
                                                 geoTransform, self.pxSizeL, self.taglab_dir, 1.0, report(1))
                created.append(cpy1)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            progress.close()
            box = QMessageBox()
            box.setWindowTitle("Alignment Tool")
            box.setText("The alignment could not be completed:\n" + str(e))
            box.exec()
            return
        QApplication.restoreOverrideCursor()
        progress.close()

        for im in created:
            self.project.addNewImage(im, sort=False)
        self.__storeMarkers()
        self.project.orderImagesByAcquisitionDate()

        txt = "Added to the project: " + ", ".join("'" + im.id + "'" for im in created) + ".\n" \
              "The new image files are next to the original ones.\n\nRemember to save the project (File > Save)."
        if needRef and self.project.working_area is not None:
            txt += "\n\nNote: the working area refers to the original reference; for the new images it must be " \
                   "shifted by {} px right and {} px down.".format(left, top)
        box = QMessageBox()
        box.setWindowTitle("Alignment Tool")
        box.setText(txt)
        box.exec()
        self.close()

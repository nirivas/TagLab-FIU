import logging
import math
import os
from typing import Optional, List

import numpy as np
import cv2

from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot, QPointF, QPoint, QRectF, QTimer
from PyQt5.QtGui import QImage, QPen, QFont, QColor, QPainter, QPainterPath, QKeyEvent, QCloseEvent, QMouseEvent, \
    QWheelEvent
from PyQt5.QtWidgets import QWidget, QHeaderView, QSizePolicy, QVBoxLayout, QLabel, QTableWidget, QTableView, \
    QHBoxLayout, QTableWidgetItem, QComboBox, QSlider, QApplication, QCheckBox, QPushButton, QMessageBox, \
    QGraphicsItem, QGraphicsPathItem, QGraphicsTextItem, QProgressDialog

from source.QtImageViewer import QtImageViewer
from source import Alignment as AL
from source.DemMetrics import DemReader

logfile = logging.getLogger("tool-logger")


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
    An image (zero-copy view of a QImage, or an array: BGRA, or float heights with NaN for no data) plus a few
    reduced versions, used to render quickly any part of the map at any zoom level. The image may itself be a
    reduced version of the map (e.g. the DEM shading): full_size is then the size of the map in pixels (width,
    height), and render() always works in map pixels.
    """

    def __init__(self, image, full_size=None):
        if isinstance(image, QImage):
            self.qimage, base = qimageToArray(image)
        else:
            self.qimage, base = None, np.ascontiguousarray(image)
        bh, bw = base.shape[0], base.shape[1]
        self.width, self.height = (bw, bh) if full_size is None else (int(full_size[0]), int(full_size[1]))
        self.levels = [(bw / self.width, bh / self.height, base)]
        small = cv2.resize(base, (max(1, bw // 4), max(1, bh // 4)), interpolation=cv2.INTER_AREA)
        self.levels.append((small.shape[1] / self.width, small.shape[0] / self.height, small))
        while max(small.shape[:2]) > 512:
            small = cv2.resize(small, (max(1, small.shape[1] // 2), max(1, small.shape[0] // 2)),
                               interpolation=cv2.INTER_AREA)
            self.levels.append((small.shape[1] / self.width, small.shape[0] / self.height, small))

    def render(self, V, w, h, border_value=0):
        """
        Render the image transformed by V (map pixels -> screen pixels) on a w x h buffer.
        """
        z = math.hypot(V[0, 0], V[1, 0])
        sx, sy, arr = self.levels[0]
        for (lx, ly, a) in reversed(self.levels):
            if lx >= z and ly >= z:
                sx, sy, arr = lx, ly, a
                break
        # pixel i of a level covers the map pixels [i / s, (i + 1) / s): its centre is at (i + 0.5) / s - 0.5
        L = np.array([[1.0 / sx, 0.0, 0.5 / sx - 0.5], [0.0, 1.0 / sy, 0.5 / sy - 0.5]])
        flags = cv2.INTER_NEAREST if z / sx > 2.0 else cv2.INTER_LINEAR
        return AL.warpArray(arr, AL.composeMatrix(V, L), w, h, border_value=border_value, flags=flags)


class AlignmentPreviewPane(QWidget):
    """
    Preview of the alignment: the reference image and the image to align (transformed with the current
    transformation) are rendered on the CPU, only for the visible area, and combined: blend, red / cyan overlay
    (grey where the two maps match, coloured edges where they don't), checkerboard, swipe (the reference left of a
    divider that follows the mouse), flicker (the two maps in turn) or difference. The height difference mode shows
    the heights of the aligned DEM minus the reference DEM (blue lower, red higher).
    Left-drag pans, the wheel zooms.
    """

    viewChanged = pyqtSignal(float, float, float)  # center x, center y (reference pixels), zoom

    BLEND = 0
    DIFFERENCE = 1
    CHECKERBOARD = 2
    SWIPE = 3
    FLICKER = 4
    HEIGHT = 5
    OVERLAY = 6

    BACKGROUND = (40, 40, 40)
    OVERLAY_STD = 48.0      # red / cyan overlay: standard deviation of the stretched grey levels at 100% contrast
    MAX_ZOOM = 32.0

    def __init__(self, mode, parent=None):
        super(AlignmentPreviewPane, self).__init__(parent)
        self.setMinimumSize(250, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mode = mode
        self.alpha = 0.5
        self.threshold = 32
        self.checkerSize = 64
        self.flickerRate = 2            # changes per second
        self.flickerMoving = False      # flicker: the image to align is shown now
        self.swipeX = None              # swipe: screen x of the divider (None: the middle)
        self.overlayContrast = 1.0      # red / cyan overlay: contrast of the two maps
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
        self.renderCache = None         # (key, rendered reference, rendered image to align)
        self.qualityCells = []          # alignment check (AL.residualGrid), drawn over the images
        self.qualityStale = False
        self.qualityVisible = True
        self.qualityHighlight = None    # index of the cell shown by "Worst"
        self.heightDiff: Optional[ImagePyramid] = None  # heights of the aligned DEM minus the reference (m, NaN)
        self.heightRange = 0.05         # height difference shown at full colour (m)
        self.heightCache = None
        self.setCursor(Qt.OpenHandCursor)
        self.setMouseTracking(True)

        self.flickerTimer = QTimer(self)
        self.flickerTimer.timeout.connect(self.flick)

    def setImages(self, ref: ImagePyramid, mov: ImagePyramid) -> None:
        self.ref = ref
        self.mov = mov
        self.renderCache = None
        self.fitView()
        self.updateTimer()

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
        self.updateTimer()
        self.update()

    def setParameter(self, mode, value) -> None:
        """
        The slider value of a mode: alpha (%), contrast (%), tile size (px), flicker rate (changes per second),
        threshold, height range (cm).
        """
        if mode == AlignmentPreviewPane.BLEND:
            self.alpha = min(max(value / 100.0, 0.0), 1.0)
        elif mode == AlignmentPreviewPane.OVERLAY:
            self.overlayContrast = max(10, int(value)) / 100.0
        elif mode == AlignmentPreviewPane.CHECKERBOARD:
            self.checkerSize = int(value)
        elif mode == AlignmentPreviewPane.FLICKER:
            self.flickerRate = max(1, int(value))
            self.updateTimer()
        elif mode == AlignmentPreviewPane.DIFFERENCE:
            self.threshold = int(value)
        elif mode == AlignmentPreviewPane.HEIGHT:
            self.heightRange = max(1, int(value)) / 100.0
        self.update()

    def setHeightDifference(self, diff: Optional[ImagePyramid]) -> None:
        """
        :param: diff heights of the aligned DEM minus the reference DEM (m, NaN = no data), on the reference map
        """
        self.heightDiff = diff
        self.heightCache = None
        self.update()

    def setAlpha(self, alpha: float) -> None:
        self.setParameter(AlignmentPreviewPane.BLEND, 100.0 * alpha)

    def setThreshold(self, thr: int) -> None:
        self.setParameter(AlignmentPreviewPane.DIFFERENCE, thr)

    def setCheckerSize(self, size: int) -> None:
        self.setParameter(AlignmentPreviewPane.CHECKERBOARD, size)

    def setPointsVisibility(self, visible: bool) -> None:
        self.showPoints = visible
        self.update()

    def setQuality(self, cells, stale=False, highlight=None) -> None:
        self.qualityCells = cells
        self.qualityStale = stale
        self.qualityHighlight = highlight
        self.update()

    def setQualityVisibility(self, visible: bool) -> None:
        self.qualityVisible = visible
        self.update()

    # flicker ---------------------------------------------------------------------------------------------------------

    def updateTimer(self) -> None:
        if self.mode == AlignmentPreviewPane.FLICKER and self.isVisible() and self.ref is not None:
            interval = int(round(1000.0 / self.flickerRate))
            if not self.flickerTimer.isActive() or self.flickerTimer.interval() != interval:
                self.flickerTimer.start(interval)
        else:
            self.flickerTimer.stop()

    @pyqtSlot()
    def flick(self) -> None:
        self.flickerMoving = not self.flickerMoving
        self.update()

    def showEvent(self, event) -> None:
        super(AlignmentPreviewPane, self).showEvent(event)
        self.updateTimer()

    def hideEvent(self, event) -> None:
        super(AlignmentPreviewPane, self).hideEvent(event)
        self.updateTimer()

    # view ------------------------------------------------------------------------------------------------------------

    def fitView(self, notify=False) -> None:
        if self.ref is None:
            return
        w = max(1, self.width())
        h = max(1, self.height())
        self.zoom = 0.95 * min(w / self.ref.width, h / self.ref.height)
        self.minZoom = self.zoom * 0.5
        self.cx = self.ref.width / 2.0
        self.cy = self.ref.height / 2.0
        self.update()
        if notify:
            self.viewChanged.emit(self.cx, self.cy, self.zoom)

    def setView(self, cx: float, cy: float, zoom: float) -> None:
        self.cx = cx
        self.cy = cy
        self.zoom = zoom
        self.update()

    def setZoom(self, zoom: float, notify=True) -> None:
        """
        Zoom around the center of the view (zoom = screen pixels per reference pixel).
        """
        self.zoom = min(max(zoom, self.minZoom), AlignmentPreviewPane.MAX_ZOOM)
        self.update()
        if notify:
            self.viewChanged.emit(self.cx, self.cy, self.zoom)

    def centerOn(self, x: float, y: float, zoom: Optional[float] = None) -> None:
        self.cx = x
        self.cy = y
        if zoom is not None:
            self.zoom = min(max(zoom, self.minZoom), AlignmentPreviewPane.MAX_ZOOM)
        self.update()
        self.viewChanged.emit(self.cx, self.cy, self.zoom)

    def viewMatrix(self):
        z = self.zoom
        return np.array([[z, 0.0, self.width() / 2.0 - self.cx * z], [0.0, z, self.height() / 2.0 - self.cy * z]])

    # rendering -------------------------------------------------------------------------------------------------------

    def renderImages(self):
        """
        The reference and the image to align rendered for the current view (BGRA); kept until the view or the
        transformation changes (the swipe divider and the flicker only change how they are combined).
        """
        w = self.width()
        h = self.height()
        V = self.viewMatrix()
        key = (w, h, V.tobytes(), self.M.tobytes(), id(self.ref), id(self.mov))
        if self.renderCache is None or self.renderCache[0] != key:
            ref = self.ref.render(V, w, h)
            mov = self.mov.render(AL.composeMatrix(V, self.M), w, h)
            # grey images (DEM shading)
            if ref.ndim == 2:
                ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGRA)
            if mov.ndim == 2:
                mov = cv2.cvtColor(mov, cv2.COLOR_GRAY2BGRA)
            self.renderCache = (key, ref, mov)
        return self.renderCache[1], self.renderCache[2]

    def renderHeight(self):
        """
        The height difference rendered for the current view (m, NaN = no data); kept until the view changes.
        """
        w = self.width()
        h = self.height()
        V = self.viewMatrix()
        key = (w, h, V.tobytes(), id(self.heightDiff))
        if self.heightCache is None or self.heightCache[0] != key:
            self.heightCache = (key, self.heightDiff.render(V, w, h, border_value=np.nan))
        return self.heightCache[1]

    def composeHeight(self, out):
        """
        Height difference: white = same height, red = the aligned map is higher, blue = lower (full colour at
        +/- heightRange).
        """
        if self.heightDiff is None:
            return out
        d = self.renderHeight()
        valid = np.isfinite(d)
        t = np.clip(d[valid] / self.heightRange, -1.0, 1.0)
        fade = (255.0 * (1.0 - np.abs(t))).astype(np.uint8)
        full = np.full(t.shape, 255, dtype=np.uint8)
        out[valid, 0] = np.where(t < 0, full, fade)     # blue
        out[valid, 1] = fade
        out[valid, 2] = np.where(t > 0, full, fade)     # red
        return out

    def compose(self):
        """
        Combine the two rendered images according to the current mode. Returns a (h, w, 4) BGRA buffer.
        """
        w = self.width()
        h = self.height()
        if self.mode == AlignmentPreviewPane.HEIGHT:
            out = np.empty((h, w, 4), dtype=np.uint8)
            out[:, :, 0] = self.BACKGROUND[2]
            out[:, :, 1] = self.BACKGROUND[1]
            out[:, :, 2] = self.BACKGROUND[0]
            out = self.composeHeight(out)
            out[:, :, 3] = 255
            return out

        ref, mov = self.renderImages()

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
        elif self.mode == AlignmentPreviewPane.OVERLAY:
            # reference -> red, aligned map -> green + blue (cyan): equal grey levels give grey
            gr = self.stretch(cv2.cvtColor(ref, cv2.COLOR_BGRA2GRAY), both if both.any() else rv)
            gm = self.stretch(cv2.cvtColor(mov, cv2.COLOR_BGRA2GRAY), both if both.any() else mv)
            out[rv | mv, :3] = 0
            out[rv, 2] = gr[rv]
            out[mv, 1] = gm[mv]
            out[mv, 0] = gm[mv]
        elif self.mode == AlignmentPreviewPane.DIFFERENCE:
            gr = cv2.cvtColor(ref, cv2.COLOR_BGRA2GRAY)
            gm = cv2.cvtColor(mov, cv2.COLOR_BGRA2GRAY)
            d = cv2.absdiff(gr, gm)
            d[d <= self.threshold] = 0
            out[both, :3] = d[both][:, None]
            out[onlyR, :3] = (gr[onlyR] // 2)[:, None]
            out[onlyM, :3] = (gm[onlyM] // 2)[:, None]
        elif self.mode == AlignmentPreviewPane.SWIPE:
            x = self.swipePosition()
            left = np.zeros((h, w), dtype=bool)
            left[:, :x] = True
            pickR = rv & left
            pickM = mv & ~left
            out[pickR] = ref[pickR]
            out[pickM] = mov[pickM]
        elif self.mode == AlignmentPreviewPane.FLICKER:
            if self.flickerMoving:
                out[mv] = mov[mv]
            else:
                out[rv] = ref[rv]
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

    def stretch(self, gray, mask):
        """
        Grey levels stretched to mean 128 and the same spread for both maps (measured on mask), so that a
        difference of brightness or contrast between the two maps does not tint the red / cyan overlay.
        """
        sel = gray[mask] if mask.any() else gray
        mean = float(sel.mean())
        std = max(float(sel.std()), 1.0)
        gain = self.OVERLAY_STD * self.overlayContrast / std
        return np.clip(128.0 + (gray.astype(np.float32) - mean) * gain, 0, 255).astype(np.uint8)

    def swipePosition(self) -> int:
        x = self.width() // 2 if self.swipeX is None else self.swipeX
        return int(min(max(x, 0), self.width()))

    def drawCaption(self, p: QPainter, text: str, y: int = 8) -> None:
        p.setFont(QFont("Roboto", 10, QFont.Bold))
        rect = p.fontMetrics().boundingRect(text).adjusted(-6, -3, 6, 3)
        rect.moveTopLeft(QPoint(8, y))
        p.fillRect(rect, QColor(0, 0, 0, 170))
        p.setPen(Qt.white)
        p.drawText(rect, Qt.AlignCenter, text)

    QUALITY_COLORS = {AL.QUALITY_GOOD: (60, 200, 80), AL.QUALITY_FAIR: (240, 200, 40),
                      AL.QUALITY_POOR: (235, 60, 50), AL.QUALITY_NONE: (150, 150, 150)}

    def drawQuality(self, p: QPainter) -> None:
        """
        The cells of the alignment check: coloured by the residual (green / yellow / red, grey = not measurable),
        with an arrow in the direction of the residual (its length grows with it).
        """
        V = self.viewMatrix()
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(QFont("Roboto", 8))
        bounds = QRectF(self.rect())
        for k, cell in enumerate(self.qualityCells):
            x0, y0, x1, y1 = cell['rect']
            (sx0, sy0), (sx1, sy1) = AL.applyMatrix(V, [[x0, y0], [x1, y1]]) + 0.5
            rect = QRectF(QPointF(sx0, sy0), QPointF(sx1, sy1))
            if not rect.intersects(bounds):
                continue
            color = QColor(*self.QUALITY_COLORS[cell['status']])
            fill = QColor(color)
            fill.setAlpha(25 if self.qualityStale else 55)
            p.fillRect(rect.adjusted(1, 1, -1, -1), fill)
            if k == self.qualityHighlight:
                pen = QPen(Qt.white, 3)
            else:
                pen = QPen(color, 1, Qt.DashLine if self.qualityStale else Qt.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect.adjusted(1, 1, -1, -1))

            if cell['mm'] is None:
                continue
            cx, cy = 0.5 * (sx0 + sx1), 0.5 * (sy0 + sy1)
            if cell['mm'] >= 0.05:
                length = min(0.45 * min(rect.width(), rect.height()), 6.0 + 8.0 * cell['mm'])
                ux, uy = cell['dx_mm'] / cell['mm'], cell['dy_mm'] / cell['mm']
                tip = QPointF(cx + ux * length, cy + uy * length)
                p.setPen(QPen(color, 2))
                p.drawLine(QPointF(cx, cy), tip)
                for side in (1.0, -1.0):
                    hx = -ux * 5.0 + side * -uy * 3.5
                    hy = -uy * 5.0 + side * ux * 3.5
                    p.drawLine(tip, QPointF(tip.x() + hx, tip.y() + hy))
            if rect.width() > 70 and rect.height() > 40:
                p.setPen(Qt.white)
                p.drawText(QPointF(sx0 + 5, sy1 - 5), "{:.1f} mm".format(cell['mm']))

        if self.qualityStale:
            self.drawCaption(p, "Alignment check out of date", y=self.height() - 30)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(*self.BACKGROUND))
        if self.ref is None or self.mov is None or self.width() < 2 or self.height() < 2:
            p.end()
            return

        out = self.compose()
        img = QImage(out.data, out.shape[1], out.shape[0], out.strides[0], QImage.Format_RGB32)
        p.drawImage(0, 0, img)

        if self.mode == AlignmentPreviewPane.SWIPE:
            x = self.swipePosition()
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawLine(x, 0, x, self.height())
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.drawLine(x - 1, 0, x - 1, self.height())
            self.drawCaption(p, "Reference | Aligned")
        elif self.mode == AlignmentPreviewPane.FLICKER:
            self.drawCaption(p, "Aligned" if self.flickerMoving else "Reference")
        elif self.mode == AlignmentPreviewPane.OVERLAY:
            self.drawCaption(p, "Red = reference, cyan = aligned")
        elif self.mode == AlignmentPreviewPane.HEIGHT:
            if self.heightDiff is None:
                self.drawCaption(p, "Height difference: both maps need a DEM")
            else:
                cm = self.heightRange * 100.0
                self.drawCaption(p, "Aligned - reference heights: blue -{:g} cm, white 0, red +{:g} cm".format(cm, cm))

        if self.qualityVisible and len(self.qualityCells) > 0:
            self.drawQuality(p)

        if self.showPoints and len(self.refPoints) > 0:
            # screen pixel j is drawn at [j, j+1): pixel centers are at +0.5
            V = self.viewMatrix()
            r = AL.applyMatrix(V, self.refPoints) + 0.5
            m = AL.applyMatrix(AL.composeMatrix(V, self.M), self.movPoints) + 0.5
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

    # mouse -----------------------------------------------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.lastPos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.mode == AlignmentPreviewPane.SWIPE:
            self.swipeX = event.pos().x()
            self.update()
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
        newZoom = min(max(self.zoom * f, self.minZoom), AlignmentPreviewPane.MAX_ZOOM)
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


class PreviewModeControls(QWidget):
    """
    The mode list of a preview pane and the slider of its current mode (alpha, contrast, tile size, flicker rate,
    threshold, height range); each mode keeps its own slider value.
    """

    NAMES = {AlignmentPreviewPane.BLEND: "Blend",
             AlignmentPreviewPane.OVERLAY: "Red / cyan overlay",
             AlignmentPreviewPane.CHECKERBOARD: "Checkerboard",
             AlignmentPreviewPane.SWIPE: "Swipe",
             AlignmentPreviewPane.FLICKER: "Flicker",
             AlignmentPreviewPane.DIFFERENCE: "Difference",
             AlignmentPreviewPane.HEIGHT: "Height difference"}

    TOOLTIPS = {AlignmentPreviewPane.BLEND: "The two maps on top of each other (alpha: weight of the aligned map)",
                AlignmentPreviewPane.OVERLAY: "The reference in red, the aligned map in cyan, on top of each other: "
                                              "grey where they match; a shift shows as red and cyan edges side by "
                                              "side. The slider sets the contrast",
                AlignmentPreviewPane.CHECKERBOARD: "Tiles of the two maps in turn: features must continue across "
                                                   "the tile borders",
                AlignmentPreviewPane.SWIPE: "The reference left of the divider, the aligned map right of it: move "
                                            "the mouse to move the divider",
                AlignmentPreviewPane.FLICKER: "The two maps in turn: a shift shows up as a jump",
                AlignmentPreviewPane.DIFFERENCE: "Gray-level difference (below the threshold: black)",
                AlignmentPreviewPane.HEIGHT: "Heights of the aligned DEM minus the reference DEM (after the height "
                                             "correction, if checked): blue lower, red higher; the slider sets the "
                                             "range in cm"}

    # mode: (minimum, maximum, default, label)
    SLIDERS = {AlignmentPreviewPane.BLEND: (0, 100, 50, "Alpha: {:d}"),
               AlignmentPreviewPane.OVERLAY: (50, 300, 100, "Contrast: {:d}%"),
               AlignmentPreviewPane.CHECKERBOARD: (8, 512, 64, "Tile: {:d} px"),
               AlignmentPreviewPane.FLICKER: (1, 8, 2, "Rate: {:d}/s"),
               AlignmentPreviewPane.DIFFERENCE: (0, 255, 32, "Threshold: {:d}"),
               AlignmentPreviewPane.HEIGHT: (1, 50, 5, "Range: +/-{:d} cm")}

    HINTS = {AlignmentPreviewPane.SWIPE: "the divider follows the mouse"}

    def __init__(self, pane: AlignmentPreviewPane, modes, mode, parent=None):
        super(PreviewModeControls, self).__init__(parent)
        self.pane = pane
        self.values = {m: spec[2] for (m, spec) in self.SLIDERS.items()}

        self.combo = QComboBox()
        self.combo.setFocusPolicy(Qt.NoFocus)
        self.label = QLabel()
        self.label.setMinimumWidth(110)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumWidth(80)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.hint = QLabel()
        self.hint.setStyleSheet("color: rgb(170,170,170)")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.hint)
        layout.setStretchFactor(self.slider, 1)
        self.setLayout(layout)

        for (m, value) in self.values.items():
            pane.setParameter(m, value)
        self.setModes(modes, mode)
        self.combo.currentIndexChanged[int].connect(self.modeChanged)
        self.slider.valueChanged[int].connect(self.valueChanged)

    def mode(self):
        return self.combo.currentData()

    def setModes(self, modes, mode=None) -> None:
        """
        Fill the mode list (the current mode is kept if still listed, else the given or the first one).
        """
        current = self.mode() if mode is None else mode
        self.combo.blockSignals(True)
        self.combo.clear()
        for m in modes:
            self.combo.addItem(self.NAMES[m], m)
            self.combo.setItemData(self.combo.count() - 1, self.TOOLTIPS.get(m, ""), Qt.ToolTipRole)
        index = self.combo.findData(current)
        self.combo.setCurrentIndex(index if index >= 0 else 0)
        self.combo.blockSignals(False)
        self.modeChanged()

    @pyqtSlot(int)
    def modeChanged(self, index=None) -> None:
        mode = self.mode()
        self.pane.setMode(mode)
        spec = self.SLIDERS.get(mode)
        self.slider.blockSignals(True)
        if spec is not None:
            self.slider.setRange(spec[0], spec[1])
            self.slider.setValue(self.values[mode])
            self.label.setText(spec[3].format(self.values[mode]))
        self.slider.blockSignals(False)
        self.label.setVisible(spec is not None)
        self.slider.setVisible(spec is not None)
        self.hint.setText(self.HINTS.get(mode, ""))
        self.hint.setVisible(mode in self.HINTS)
        self.combo.setToolTip(self.TOOLTIPS.get(mode, ""))

    @pyqtSlot(int)
    def valueChanged(self, value: int) -> None:
        mode = self.mode()
        spec = self.SLIDERS.get(mode)
        if spec is None:
            return
        self.values[mode] = value
        self.label.setText(spec[3].format(value))
        self.pane.setParameter(mode, value)


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

    PREVIEW_MODES = [AlignmentPreviewPane.BLEND, AlignmentPreviewPane.OVERLAY, AlignmentPreviewPane.CHECKERBOARD,
                     AlignmentPreviewPane.SWIPE, AlignmentPreviewPane.FLICKER, AlignmentPreviewPane.DIFFERENCE]
    # the DEM shading of the map to align is lit from the same direction as the reference (in the reference frame):
    # it is computed again if the rotation changes by more than this (degrees)
    SHADE_TOLERANCE = 5.0

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
        self.quality = None             # last alignment check: cells, stats, pixel matrix and images measured
        self.worstOrder = []            # measured cells, largest residual first ("Worst")
        self.worstIndex = -1
        self.hasDems = False            # both maps have a DEM: DEM shading, matching on the DEM, heights
        self.shadePyramids = {}         # DEM shading of the maps of the current pair: image id -> (azimuth, pyramid)
        self.heightData = None          # the two DEMs reduced to compare the heights: (reference, map to align)
        self.heightPlane = None         # plane fitted to the height difference (AL.fitHeightPlane) ...
        self.heightPlaneM = None        # ... with this pixel matrix
        self.heightTimer = QTimer(self)
        self.heightTimer.setSingleShot(True)
        self.heightTimer.setInterval(300)
        self.heightTimer.timeout.connect(self.__updateHeights)

        # result of Confirm, used by TagLab when the tool closes: (original map, aligned copy, pixel matrix) to put in
        # place of the original maps, the shift of the working area (reference extended) and the final message
        self.replacements = []
        self.workingAreaShift = None
        self.resultMessage = ""
        self.resultLog = ""

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

        self.matchOnLabel = QLabel("Match on:")
        self.matchCombo = QComboBox()
        self.matchCombo.setFocusPolicy(Qt.NoFocus)
        self.matchCombo.addItem("Colour", "colour")
        self.matchCombo.addItem("DEM relief", "dem")
        self.matchCombo.addItem("Colour + DEM", "both")
        self.matchCombo.setToolTip("What Auto Markers compares: the colours, the relief of the DEMs (not affected by "
                                   "colour changes such as bleaching, algae or light), or both")

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
        self.buttons.addWidget(self.matchOnLabel)
        self.buttons.addWidget(self.matchCombo)
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
        self.leftPreviewViewer.viewChanged[float, float, float].connect(self.previewViewChanged)
        self.rightPreviewViewer.viewChanged[float, float, float].connect(self.previewViewChanged)

        # each pane has its own mode (Height difference is added when both maps have a DEM)
        self.leftPreviewControls = PreviewModeControls(self.leftPreviewViewer, self.PREVIEW_MODES,
                                                       AlignmentPreviewPane.BLEND)
        self.rightPreviewControls = PreviewModeControls(self.rightPreviewViewer, self.PREVIEW_MODES,
                                                        AlignmentPreviewPane.DIFFERENCE)

        # zoom
        self.fitButton = QPushButton("Fit")
        self.fitButton.setFixedWidth(60)
        self.fitButton.setFixedHeight(20)
        self.fitButton.setToolTip("Show the whole reference map")
        self.fitButton.clicked.connect(self.onFitPreview)
        self.oneToOneButton = QPushButton("1:1")
        self.oneToOneButton.setFixedWidth(60)
        self.oneToOneButton.setFixedHeight(20)
        self.oneToOneButton.setToolTip("One pixel of the reference map per screen pixel")
        self.oneToOneButton.clicked.connect(self.onOneToOnePreview)
        self.zoomLabel = QLabel()
        self.zoomLabel.setToolTip("Wheel or +/- keys: zoom; left drag: pan")

        # alignment check (quality map)
        self.checkAlignmentButton = QPushButton("Check alignment")
        self.checkAlignmentButton.setFixedWidth(130)
        self.checkAlignmentButton.setFixedHeight(20)
        self.checkAlignmentButton.setToolTip("Measure the shift left between the two maps on a grid of patches "
                                             "(phase correlation): green <= {:g} mm, yellow <= {:g} mm, red more, "
                                             "grey not measurable".format(AL.QUALITY_GOOD_MM, AL.QUALITY_FAIR_MM))
        self.checkAlignmentButton.clicked.connect(self.onCheckAlignment)
        self.worstButton = QPushButton("Worst ▶")
        self.worstButton.setFixedWidth(80)
        self.worstButton.setFixedHeight(20)
        self.worstButton.setToolTip("Zoom on the cell with the largest shift left (press again for the next one)")
        self.worstButton.setEnabled(False)
        self.worstButton.clicked.connect(self.onWorstCell)
        self.qualityCheck = QCheckBox("Quality map")
        self.qualityCheck.setChecked(True)
        self.qualityCheck.setFocusPolicy(Qt.NoFocus)
        self.qualityCheck.stateChanged[int].connect(self.toggleQualityVisibility)
        self.qualityLabel = QLabel()
        self.qualityLabel.setWordWrap(True)

        self.previewBar = QHBoxLayout()
        self.previewBar.addWidget(self.fitButton)
        self.previewBar.addWidget(self.oneToOneButton)
        self.previewBar.addWidget(self.zoomLabel)
        self.previewBar.addSpacing(30)
        self.previewBar.addWidget(self.checkAlignmentButton)
        self.previewBar.addWidget(self.worstButton)
        self.previewBar.addWidget(self.qualityCheck)
        self.previewBar.addWidget(self.qualityLabel)
        self.previewBar.setStretchFactor(self.qualityLabel, 1)

        # DEM (both maps have one)
        self.showLabel = QLabel("Show:")
        self.showCombo = QComboBox()
        self.showCombo.setFocusPolicy(Qt.NoFocus)
        self.showCombo.addItem("Colour", "colour")
        self.showCombo.addItem("DEM shading", "dem")
        self.showCombo.setToolTip("Show the colours, or the shading of the two DEMs (light from the top left of the "
                                  "reference); Check alignment measures what is shown")
        self.showCombo.currentIndexChanged[int].connect(self.onShowChanged)
        self.matchHeightsCheck = QCheckBox("Match DEM heights to the reference")
        self.matchHeightsCheck.setChecked(True)
        self.matchHeightsCheck.setFocusPolicy(Qt.NoFocus)
        self.matchHeightsCheck.setToolTip("Remove the height offset and tilt between the two DEMs (a plane fitted to "
                                          "the parts that did not change) when the aligned DEM is written")
        self.matchHeightsCheck.stateChanged[int].connect(self.toggleMatchHeights)
        self.heightLabel = QLabel()
        self.heightLabel.setWordWrap(True)

        self.demBar = QHBoxLayout()
        self.demBar.addWidget(self.showLabel)
        self.demBar.addWidget(self.showCombo)
        self.demBar.addSpacing(30)
        self.demBar.addWidget(self.matchHeightsCheck)
        self.demBar.addWidget(self.heightLabel)
        self.demBar.setStretchFactor(self.heightLabel, 1)
        self.demPreviewWidgets = [self.showLabel, self.showCombo, self.matchHeightsCheck, self.heightLabel]
        self.demEditWidgets = [self.matchOnLabel, self.matchCombo]

        self.previewLayoutL = QVBoxLayout()
        self.previewLayoutL.addWidget(self.leftPreviewControls)
        self.previewLayoutL.addWidget(self.leftPreviewViewer)
        self.previewLayoutR = QVBoxLayout()
        self.previewLayoutR.addWidget(self.rightPreviewControls)
        self.previewLayoutR.addWidget(self.rightPreviewViewer)
        self.previewLayout = QHBoxLayout()
        self.previewLayout.addLayout(self.previewLayoutL)
        self.previewLayout.addLayout(self.previewLayoutR)
        self.previewOnlyWidgets = [self.leftPreviewControls, self.rightPreviewControls, self.leftPreviewViewer,
                                   self.rightPreviewViewer, self.fitButton, self.oneToOneButton, self.zoomLabel,
                                   self.checkAlignmentButton, self.worstButton, self.qualityCheck, self.qualityLabel]

        # ==============================================================
        # Layout
        # ==============================================================

        content = QVBoxLayout()
        content.addLayout(self.buttons)
        content.addLayout(self.layoutTop)
        content.addLayout(self.editLayout)
        content.addLayout(self.previewBar)
        content.addLayout(self.demBar)
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

        if self.isPreviewMode and event.key() in (Qt.Key_Plus, Qt.Key_Equal, Qt.Key_Minus):
            f = 0.5 if event.key() == Qt.Key_Minus else 2.0
            self.leftPreviewViewer.setZoom(self.leftPreviewViewer.zoom * f)
            event.accept()
            return

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
        # the fit of the previous pair must not be used with the markers of this one (before __refit)
        self.fitRigid = self.fitScaled = None
        self.baseMatrix = None
        self.matrix = None
        self.__resetQuality()
        self.hasDems = image1.hasDEM() and image2.hasDEM()
        self.__resetDem()

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
        refPts = AL.mmToPixels([[m.lViewPos.x(), m.lViewPos.y()] for m in complete], self.pxSizeL)
        movPts = AL.mmToPixels([[m.rViewPos.x(), m.rViewPos.y()] for m in complete], self.pxSizeR)
        ids = [m.identifier for m in complete]
        for pane in [self.leftPreviewViewer, self.rightPreviewViewer]:
            pane.setTransform(M)
            pane.setPoints(refPts, movPts, ids)
        if self.quality is not None:
            self.__showQuality()
        if self.heightData is not None:
            self.heightTimer.start()

    @pyqtSlot()
    def onPreviewRequested(self) -> None:
        if self.matrix is None:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("Alignment Tool")
            msgBox.setText("At least 3 complete markers are required (place them with SHIFT + Left click on both "
                           "images, or use Auto Markers).")
            msgBox.exec()
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ref, mov = self.__previewPyramids()
            if self.hasDems and self.heightData is None:
                self.__loadHeights()
        finally:
            QApplication.restoreOverrideCursor()
        self.__togglePreviewMode(True)
        QApplication.processEvents()
        self.__setPreviewImages(ref, mov, keepView=False)
        self.__updatePreview()
        if self.heightData is not None:
            self.heightTimer.stop()
            self.__updateHeights()
        self.previewViewChanged()

    def __setPreviewImages(self, ref: ImagePyramid, mov: ImagePyramid, keepView=True) -> None:
        left = self.leftPreviewViewer
        view = (left.cx, left.cy, left.zoom) if keepView and left.ref is not None else None
        for pane in [left, self.rightPreviewViewer]:
            pane.setImages(ref, mov)
        if view is not None:
            left.setView(*view)
        self.rightPreviewViewer.setView(left.cx, left.cy, left.zoom)

    def __previewPyramids(self):
        """
        The images shown by the preview: the colours, or the shading of the DEMs (the map to align is lit from the
        same direction as the reference, in the frame of the reference).
        """
        image1, image2 = self.__images()
        if self.hasDems and self.showCombo.currentData() == "dem":
            angle = AL.matrixToParams(self.matrix)[0] if self.matrix is not None else 0.0
            return self.__shadePyramid(image1, AL.SHADE_AZIMUTH), \
                self.__shadePyramid(image2, AL.SHADE_AZIMUTH - angle)
        return self.__pyramid(self.__rgbChannel(image1).qimage), self.__pyramid(self.__rgbChannel(image2).qimage)

    def __shadePyramid(self, image, azimuth: float) -> ImagePyramid:
        """
        Shading of the DEM of the map lit from azimuth, computed on the DEM reduced to at most AL.SHADE_MAX_SIDE
        pixels a side; computed again only if the light changes by more than SHADE_TOLERANCE.
        """
        cached = self.shadePyramids.get(image.id)
        if cached is not None and abs((cached[0] - azimuth + 180.0) % 360.0 - 180.0) <= self.SHADE_TOLERANCE:
            return cached[1]
        reader = DemReader(os.path.join(self.taglab_dir, image.getDEMChannel().filename), image.pixelSize())
        try:
            W, H = reader.dataset.width, reader.dataset.height
            shape = AL.reducedShape(W, H, AL.SHADE_MAX_SIDE)
            z = AL.readDem(reader.dataset, shape)
            shade = AL.hillshade(z, reader.dx * W / shape[1], reader.dy * H / shape[0], azimuth)
        finally:
            reader.close()
        pyramid = ImagePyramid(shade, full_size=(image.width, image.height))
        self.shadePyramids[image.id] = (azimuth, pyramid)
        return pyramid

    @pyqtSlot(int)
    def onShowChanged(self, index: int) -> None:
        if not self.isPreviewMode:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ref, mov = self.__previewPyramids()
        finally:
            QApplication.restoreOverrideCursor()
        self.__setPreviewImages(ref, mov)

    # ==================================================================================================================
    # DEM heights
    # ==================================================================================================================

    def __resetDem(self) -> None:
        """
        A new pair of maps: no DEM shading or heights yet; the preview modes that need a DEM on both maps.
        """
        self.shadePyramids = {}
        self.heightData = None
        self.heightPlane = None
        self.heightPlaneM = None
        self.heightTimer.stop()
        self.heightLabel.setText("")
        self.showCombo.blockSignals(True)
        self.showCombo.setCurrentIndex(0)
        self.showCombo.blockSignals(False)
        modes = self.PREVIEW_MODES + ([AlignmentPreviewPane.HEIGHT] if self.hasDems else [])
        for controls in [self.leftPreviewControls, self.rightPreviewControls]:
            controls.setModes(modes)
            controls.pane.setHeightDifference(None)
        for w in self.demEditWidgets:
            w.setVisible(not self.isPreviewMode and self.hasDems)

    def __loadHeights(self) -> None:
        """
        Read the two DEMs reduced to at most AL.HEIGHT_MAX_SIDE pixels a side, to compare their heights.
        """
        data = []
        for image in self.__images():
            reader = DemReader(os.path.join(self.taglab_dir, image.getDEMChannel().filename), image.pixelSize())
            try:
                ds = reader.dataset
                data.append(AL.readDem(ds, AL.reducedShape(ds.width, ds.height, AL.HEIGHT_MAX_SIDE)))
            finally:
                reader.close()
        self.heightData = (data[0], data[1])

    def __updateHeights(self) -> None:
        """
        Compare the heights of the two DEMs with the current transformation: the plane fitted to their difference
        (offset, tilt, spread) and the height difference shown by the preview (after the correction, if checked).
        """
        if self.heightData is None or self.matrix is None:
            return
        image1, image2 = self.__images()
        M = self.__previewMatrix()
        ref_size = (image1.width, image1.height)
        diff = AL.heightDifference(self.heightData[0], self.heightData[1], M, ref_size, (image2.width, image2.height))
        plane = AL.fitHeightPlane(diff, ref_size, self.pxSizeL)
        self.heightPlane = plane
        self.heightPlaneM = M.copy()
        txt = AL.heightPlaneText(plane)
        if plane is not None:
            if self.matchHeightsCheck.isChecked():
                diff = diff - AL.planeOnGrid(plane, diff.shape, ref_size)
                txt += " Removed from the aligned DEM when you confirm."
            else:
                txt += " Not removed (the box is unchecked)."
        pyramid = ImagePyramid(diff, full_size=ref_size)
        for pane in [self.leftPreviewViewer, self.rightPreviewViewer]:
            pane.setHeightDifference(pyramid)
        self.heightLabel.setText(txt)

    def __currentHeightPlane(self):
        """
        The height plane for the current transformation (None: no DEMs, or they do not overlap enough).
        """
        if not self.hasDems or self.matrix is None:
            return None
        if self.heightData is None:
            self.__loadHeights()
        M = self.__previewMatrix()
        if self.heightPlaneM is None or not np.allclose(self.heightPlaneM, M, rtol=0.0, atol=1e-9):
            self.heightTimer.stop()
            self.__updateHeights()
        return self.heightPlane

    @pyqtSlot(int)
    def toggleMatchHeights(self, value: int) -> None:
        if self.heightData is not None and self.isPreviewMode:
            self.__updateHeights()

    @pyqtSlot()
    def onBackToEditRequested(self) -> None:
        self.__togglePreviewMode(False)

    @pyqtSlot()
    def onFitPreview(self) -> None:
        self.leftPreviewViewer.fitView(notify=True)

    @pyqtSlot()
    def onOneToOnePreview(self) -> None:
        self.leftPreviewViewer.setZoom(1.0)

    @pyqtSlot()
    @pyqtSlot(float, float, float)
    def previewViewChanged(self, cx=None, cy=None, zoom=None) -> None:
        """
        Show the zoom (screen pixels per pixel of the reference map) and the size of a screen pixel on the map.
        """
        zoom = self.leftPreviewViewer.zoom
        self.zoomLabel.setText("Zoom {:.2f}x   ({:.3g} mm per screen pixel)".format(zoom, self.pxSizeL / zoom))

    # ==================================================================================================================
    # Alignment check (quality map)
    # ==================================================================================================================

    @pyqtSlot()
    def onCheckAlignment(self) -> None:
        """
        Measure the shift left between the two maps with the current transformation (AL.residualGrid, on the
        images shown by the preview) and show it over both panes.
        """
        pane = self.leftPreviewViewer
        if self.matrix is None or pane.ref is None or pane.mov is None:
            return
        M = self.__previewMatrix()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            cells, stats = AL.residualGrid(pane.ref.levels[0][2], pane.mov.levels[0][2], M, self.pxSizeL,
                                           (pane.ref.width, pane.ref.height), (pane.mov.width, pane.mov.height))
        finally:
            QApplication.restoreOverrideCursor()

        source = "DEM shading" if self.hasDems and self.showCombo.currentData() == "dem" else None
        self.quality = {'cells': cells, 'stats': stats, 'M': M.copy(), 'source': source}
        measured = [k for (k, c) in enumerate(cells) if c['mm'] is not None]
        self.worstOrder = sorted(measured, key=lambda k: -cells[k]['mm'])
        self.worstIndex = -1
        self.worstButton.setEnabled(len(self.worstOrder) > 0)
        self.qualityCheck.setChecked(True)
        self.__showQuality()

        image1, image2 = self.__images()
        logfile.info("[ALIGNMENT] '{}' on '{}': {}".format(image2.id, image1.id, AL.qualityText(stats, source)))

    @pyqtSlot()
    def onWorstCell(self) -> None:
        """
        Zoom the preview on the next cell with the largest residual.
        """
        if self.quality is None or len(self.worstOrder) == 0:
            return
        self.worstIndex = (self.worstIndex + 1) % len(self.worstOrder)
        k = self.worstOrder[self.worstIndex]
        cell = self.quality['cells'][k]
        x0, y0, x1, y1 = cell['patch'] if cell['patch'] is not None else cell['rect']
        pane = self.leftPreviewViewer
        zoom = 0.6 * min(pane.width(), pane.height()) / max(x1 - x0, y1 - y0, 1.0)
        self.__showQuality()
        pane.centerOn(0.5 * (x0 + x1), 0.5 * (y0 + y1), zoom)

    @pyqtSlot(int)
    def toggleQualityVisibility(self, value: int) -> None:
        self.leftPreviewViewer.setQualityVisibility(value != 0)
        self.rightPreviewViewer.setQualityVisibility(value != 0)

    def __qualityIsCurrent(self) -> bool:
        """
        True if the last alignment check was made with the current transformation.
        """
        return self.quality is not None and self.matrix is not None and \
            np.allclose(self.quality['M'], self.__previewMatrix(), rtol=0.0, atol=1e-9)

    def __qualitySummary(self) -> str:
        if self.quality is None:
            return "Alignment check: not done."
        txt = AL.qualityText(self.quality['stats'], self.quality['source'])
        if not self.__qualityIsCurrent():
            txt += " (measured before the last change of the transformation)"
        return txt

    def __showQuality(self) -> None:
        """
        Show the last alignment check over both panes (marked out of date if the transformation has changed).
        """
        if self.quality is None:
            for pane in [self.leftPreviewViewer, self.rightPreviewViewer]:
                pane.setQuality([])
            self.qualityLabel.setText("")
            return
        stale = not self.__qualityIsCurrent()
        highlight = self.worstOrder[self.worstIndex] if self.worstIndex >= 0 else None
        for pane in [self.leftPreviewViewer, self.rightPreviewViewer]:
            pane.setQuality(self.quality['cells'], stale, highlight)
        txt = AL.qualityText(self.quality['stats'], self.quality['source'])
        if stale:
            txt = "OUT OF DATE (the transformation has changed; press Check alignment again). " + txt
        self.qualityLabel.setText(txt)

    def __resetQuality(self) -> None:
        self.quality = None
        self.worstOrder = []
        self.worstIndex = -1
        self.worstButton.setEnabled(False)
        self.__showQuality()

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
        for w in self.demPreviewWidgets:
            w.setVisible(isPreviewMode and self.hasDems)
        for w in self.demEditWidgets:
            w.setVisible(not isPreviewMode and self.hasDems)

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
        if self.matrix is None or len(usable) == 0:
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
        mode = self.matchCombo.currentData() if self.hasDems else "colour"
        ch1 = image1.getRGBChannel()
        ch2 = image2.getRGBChannel()
        if mode != "dem" and (ch1 is None or ch2 is None):
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
        refs, movs, keep = [], [], []
        try:
            if mode in ("colour", "both"):
                # the colours come from the maps already loaded (only windows are converted)
                for (image, channel, sources) in [(image1, ch1, refs), (image2, ch2, movs)]:
                    qimage, array = qimageToArray(self.__rgbChannel(image).qimage)
                    keep.append(qimage)
                    sources.append(AL.ColourSource(os.path.join(self.taglab_dir, channel.filename), array))
            if mode in ("dem", "both"):
                for (image, px, sources) in [(image1, self.pxSizeL, refs), (image2, self.pxSizeR, movs)]:
                    sources.append(AL.DemReliefSource(os.path.join(self.taglab_dir, image.getDEMChannel().filename),
                                                      px))
            pairs, info = AL.autoMarkers(refs, movs, self.pxSizeL, self.pxSizeR, max_markers=30, progress=report)
        except Exception as e:
            for source in refs + movs:
                source.close()
            QApplication.restoreOverrideCursor()
            progress.close()
            box = QMessageBox()
            box.setWindowTitle("Auto Markers")
            box.setText("Corresponding points could not be found automatically:\n" + str(e) +
                        "\n\nPlace a few markers by hand (SHIFT + Left click).")
            box.exec()
            return
        for source in refs + movs:
            source.close()
        QApplication.restoreOverrideCursor()
        progress.close()

        for (refPos, movPos) in pairs:
            m = MarkerObjData(self.__nextIdentifier(), QPointF(refPos[0], refPos[1]), QPointF(movPos[0], movPos[1]))
            self.markers.append(m)
        self.__markersChanged()

        found = ""
        if len(info['sources']) > 1:
            found = " (" + ", ".join("{}: {:d}".format(name, n) for (name, n) in info['sources'].items()) + ")"
        elif mode == "dem":
            found = " (on the DEM relief)"
        box = QMessageBox()
        box.setWindowTitle("Auto Markers")
        box.setText("{:d} markers have been added{} ({:d} feature matches, {:d} consistent).\n"
                    "Check the table: markers with a large error are shown in red, uncheck or delete them.".format(
                        info['markers'], found, info['matches'], info['inliers']))
        box.exec()
        logfile.info("[ALIGNMENT] Auto Markers on '{}' / '{}' ({}): {:d} markers{}, {:d} matches, {:d} consistent."
                     .format(image1.id, image2.id, mode, info['markers'], found, info['matches'], info['inliers']))

    # ==================================================================================================================
    # Confirm
    # ==================================================================================================================

    def askConfirmation(self, text, extendChoice):
        """
        Ask to proceed. With extendChoice the reference has to be extended to the left/top: the answer is "extend",
        "crop" or None (cancel); otherwise it is "yes" or None.
        """
        box = QMessageBox(self)
        box.setWindowTitle("Alignment Tool")
        box.setText(text)
        if extendChoice:
            extend = box.addButton("Extend the reference", QMessageBox.AcceptRole)
            crop = box.addButton("Crop to the reference", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(extend)
            box.exec()
            if box.clickedButton() is extend:
                return "extend"
            if box.clickedButton() is crop:
                return "crop"
            return None
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        return "yes" if box.exec() == QMessageBox.Yes else None

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
        needRef = canvas['left'] > 0 or canvas['top'] > 0
        appliedScale = AL.matrixToParams(self.matrix)[1] * 1.0
        scaleApplied = self.applyScaleCheck.isChecked()

        suffix2 = AL.uniqueSuffix(self.project, image2, "_coreg", self.taglab_dir)
        suffix1 = AL.uniqueSuffix(self.project, image1, "_ref", self.taglab_dir) if needRef else None

        txt = "'{}' will be aligned to '{}'. The aligned map replaces '{}' in the project (same name); its image " \
              "files are new files with the suffix '{}' ({} x {} pixels, pixel size {:.4f} mm). The original files " \
              "are not changed.".format(image2.name, image1.name, image2.name, suffix2,
                                        canvas['width'], canvas['height'], self.pxSizeL)
        txt += "\n\nScale: " + ("the fitted scale is applied (measured areas of the aligned map change)."
                                if scaleApplied else "not applied.")
        txt += "\nRegions and points are transformed, and the matches with the other years are kept."
        txt += "\n\n" + self.__qualitySummary()

        # heights of the DEM
        plane = None
        heightsTxt = None
        if self.hasDems:
            if self.matchHeightsCheck.isChecked():
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    plane = self.__currentHeightPlane()
                finally:
                    QApplication.restoreOverrideCursor()
                if plane is None:
                    heightsTxt = "DEM heights: not changed. " + AL.heightPlaneText(None)
                else:
                    heightsTxt = "DEM h" + AL.heightPlaneText(plane)[1:] + " This offset and tilt are removed from the " \
                                 "aligned DEM; the DEM metrics and surface areas of its regions are cleared (compute " \
                                 "them again from the DEM menu)."
            else:
                heightsTxt = "DEM heights: not changed (Match DEM heights is unchecked)."
            txt += "\n\n" + heightsTxt
        lost = AL.notCarriedOver(image2)
        if len(lost) > 0:
            txt += "\nNot carried over to the aligned map: " + ", ".join(lost) + "."

        if needRef:
            others = [im.name for im in self.project.images if im is not image1 and im is not image2]
            txt += "\n\nThe aligned map extends beyond the left/top edge of the reference ({} px to the left, " \
                   "{} px above). You can:".format(canvas['left'], canvas['top'])
            txt += "\n\n- Extend the reference: '{}' is also replaced by a copy extended by that amount (an exact " \
                   "shift: its regions move with it; new files with the suffix '{}'). The working area is shifted " \
                   "too.".format(image1.name, suffix1)
            lost1 = AL.notCarriedOver(image1)
            if len(lost1) > 0:
                txt += " Not carried over to the extended reference: " + ", ".join(lost1) + "."
            if len(others) > 0:
                txt += "\n  The other maps of the project ({}) are not shifted: if they were aligned to '{}', " \
                       "they will be offset by the same amount and must be aligned again.".format(
                            ", ".join("'" + name + "'" for name in others), image1.name)
            txt += "\n\n- Crop to the reference: the aligned map is cut at the left/top edge of the reference, " \
                   "nothing else changes."
        else:
            txt += "\n\nProceed?"

        choice = self.askConfirmation(txt, needRef)
        if choice is None:
            return
        extend = choice == "extend"
        if needRef and not extend:
            # cropped at the left/top edge of the reference: the canvas only grows to the right and down
            canvas = {'left': 0, 'top': 0, 'right': canvas['right'], 'bottom': canvas['bottom'],
                      'width': ref_w + canvas['right'], 'height': ref_h + canvas['bottom']}
        left, top = canvas['left'], canvas['top']

        M = AL.pixelMatrix(self.matrix, self.pxSizeL, self.pxSizeR, (left, top))
        crs, geoTransform = AL.georeferenceOfCanvas(image1.georef_filename, left, top)

        nimages = 2 if extend else 1
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
        replacements = []
        try:
            cpy2 = AL.createTransformedImage(image2, suffix2, M, canvas['width'], canvas['height'], crs, geoTransform,
                                             self.pxSizeL, self.taglab_dir,
                                             appliedScale if scaleApplied else 1.0, report(0),
                                             height_plane=AL.shiftedPlane(plane, left, top) if plane is not None
                                             else None)
            replacements.append((image2, cpy2, M))
            if extend:
                T = np.array([[1.0, 0.0, float(left)], [0.0, 1.0, float(top)]])
                cpy1 = AL.createTransformedImage(image1, suffix1, T, canvas['width'], canvas['height'], crs,
                                                 geoTransform, self.pxSizeL, self.taglab_dir, 1.0, report(1))
                replacements.append((image1, cpy1, T))
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

        # the markers are stored for the original maps: TagLab moves them into the frames of the copies
        self.__storeMarkers()
        self.replacements = replacements
        self.workingAreaShift = (left, top) if extend else None

        files = ", ".join(os.path.basename(ch.filename) for (old, new, _) in replacements for ch in new.channels)
        txt = "'{}' has been aligned to '{}'.".format(image2.name, image1.name)
        if extend:
            txt += " '{}' has been extended by {} px to the left and {} px above.".format(image1.name, left, top)
        txt += "\n\nThe project now uses the new image files ({}). The original files are unchanged (they are no " \
               "longer part of the project once you save it).".format(files)
        if extend and self.project.working_area is not None:
            txt += "\n\nThe working area has been shifted by {} px right and {} px down.".format(left, top)
        if plane is not None:
            txt += "\n\nThe heights of the aligned DEM now match the reference (offset {:+.1f} cm and tilt {:.2f}° " \
                   "removed). Compute the DEM metrics and surface areas of its regions again (DEM menu).".format(
                       plane['offset'] * 100.0, plane['tilt'])
        txt += "\n\nRemember to save the project (File > Save)."
        self.resultMessage = txt
        errors = [m.error for m in self.markers if m.isComplete() and m.enabled and m.error is not None]
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if len(errors) > 0 else float('nan')
        self.resultLog = "'{}' aligned to '{}' (markers rmse {:.3f} mm, {}{}), files: {}. {}".format(
            image2.id, image1.id, rmse,
            "scale applied" if scaleApplied else "no scale",
            ", reference extended by ({}, {}) px".format(left, top) if extend else
            (", cropped to the reference" if needRef else ""), files, self.__qualitySummary())
        if heightsTxt is not None:
            self.resultLog += " " + heightsTxt
        self.close()

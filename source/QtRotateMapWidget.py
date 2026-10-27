# TagLab
# A semi-automatic segmentation tool
#
# Dialog to create a rotated copy of a map (all the channels and the annotations).

import os

import numpy as np
import cv2
import rasterio as rio
from rasterio.enums import Resampling

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QRadioButton, QButtonGroup, \
    QDoubleSpinBox, QPushButton, QProgressDialog, QApplication, QMessageBox

from source import Alignment as AL


class QtRotateMapWidget(QDialog):
    """
    Create a rotated copy of a map: 90, 180, 270 degrees (exact, no resampling) or any angle.
    The new map is stored in self.newImage when the dialog is accepted.
    """

    THUMB_SIZE = 400

    def __init__(self, taglab_dir, project, current_index=0, parent=None):
        super(QtRotateMapWidget, self).__init__(parent)

        self.setStyleSheet("background-color: rgb(40,40,40); color: white")
        self.setWindowTitle("Rotate Map")

        self.taglab_dir = taglab_dir
        self.project = project
        self.newImage = None
        self.thumbnails = {}

        self.comboMap = QComboBox()
        for image in self.project.images:
            self.comboMap.addItem(image.name)
        self.comboMap.setCurrentIndex(max(0, min(current_index, len(self.project.images) - 1)))
        self.comboMap.currentIndexChanged.connect(self.updatePreview)

        self.radio90 = QRadioButton("90° clockwise")
        self.radio180 = QRadioButton("180°")
        self.radio270 = QRadioButton("90° counter-clockwise")
        self.radioCustom = QRadioButton("Custom angle (clockwise):")
        self.radio90.setChecked(True)
        self.group = QButtonGroup(self)
        for rb in [self.radio90, self.radio180, self.radio270, self.radioCustom]:
            self.group.addButton(rb)
            rb.toggled.connect(self.updatePreview)

        self.spinAngle = QDoubleSpinBox()
        self.spinAngle.setRange(-180.0, 180.0)
        self.spinAngle.setDecimals(1)
        self.spinAngle.setSingleStep(0.1)
        self.spinAngle.setSuffix(" °")
        self.spinAngle.setValue(0.0)
        self.spinAngle.valueChanged.connect(self.onAngleEdited)

        self.thumbLabel = QLabel()
        self.thumbLabel.setFixedSize(QtRotateMapWidget.THUMB_SIZE, QtRotateMapWidget.THUMB_SIZE)
        self.thumbLabel.setAlignment(Qt.AlignCenter)
        self.thumbLabel.setStyleSheet("background-color: rgb(25,25,25)")

        self.infoLabel = QLabel()
        self.infoLabel.setWordWrap(True)

        self.btnRotate = QPushButton("Rotate")
        self.btnRotate.clicked.connect(self.rotate)
        self.btnCancel = QPushButton("Cancel")
        self.btnCancel.clicked.connect(self.reject)

        layoutMap = QHBoxLayout()
        layoutMap.addWidget(QLabel("Map:"))
        layoutMap.addWidget(self.comboMap)

        layoutCustom = QHBoxLayout()
        layoutCustom.addWidget(self.radioCustom)
        layoutCustom.addWidget(self.spinAngle)

        layoutOptions = QVBoxLayout()
        layoutOptions.addLayout(layoutMap)
        layoutOptions.addSpacing(10)
        layoutOptions.addWidget(self.radio90)
        layoutOptions.addWidget(self.radio180)
        layoutOptions.addWidget(self.radio270)
        layoutOptions.addLayout(layoutCustom)
        layoutOptions.addSpacing(10)
        layoutOptions.addWidget(self.infoLabel)
        layoutOptions.addStretch()

        layoutMain = QHBoxLayout()
        layoutMain.addLayout(layoutOptions)
        layoutMain.addWidget(self.thumbLabel)

        layoutButtons = QHBoxLayout()
        layoutButtons.addStretch()
        layoutButtons.addWidget(self.btnRotate)
        layoutButtons.addWidget(self.btnCancel)

        layout = QVBoxLayout()
        layout.addLayout(layoutMain)
        layout.addLayout(layoutButtons)
        self.setLayout(layout)

        self.updatePreview()

    def selectedImage(self):
        return self.project.images[self.comboMap.currentIndex()]

    def angle(self) -> float:
        """
        Rotation angle in degrees, positive = clockwise.
        """
        if self.radio90.isChecked():
            return 90.0
        if self.radio180.isChecked():
            return 180.0
        if self.radio270.isChecked():
            return 270.0
        return float(self.spinAngle.value())

    def suffix(self) -> str:
        a = self.angle()
        if self.radioCustom.isChecked():
            txt = "{:.1f}".format(a).rstrip("0").rstrip(".")
            return "_rot" + txt
        return "_rot{:d}".format(int(round(a)))

    @pyqtSlot(float)
    def onAngleEdited(self, value) -> None:
        if not self.radioCustom.isChecked():
            self.radioCustom.setChecked(True)
        self.updatePreview()

    def __thumbnail(self, image):
        """
        Small BGRA version of the RGB channel of the image (longest side <= THUMB_SIZE).
        """
        if image.id in self.thumbnails:
            return self.thumbnails[image.id]
        channel = image.getRGBChannel()
        if channel is None and len(image.channels) > 0:
            channel = image.channels[0]
        if channel is None:
            return None
        s = QtRotateMapWidget.THUMB_SIZE
        if channel.qimage is not None:
            q = channel.qimage.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            q = q.convertToFormat(QImage.Format_RGB32)
            ptr = q.constBits()
            ptr.setsize(q.height() * q.bytesPerLine())
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(q.height(), q.bytesPerLine())
            thumb = arr[:, :q.width() * 4].reshape(q.height(), q.width(), 4).copy()
        else:
            with rio.open(os.path.join(self.taglab_dir, channel.filename)) as src:
                f = max(src.width, src.height) / float(s)
                h = max(1, int(round(src.height / f)))
                w = max(1, int(round(src.width / f)))
                nb = min(src.count, 3)
                data = src.read(list(range(1, nb + 1)), out_shape=(nb, h, w), resampling=Resampling.average)
            if nb == 1:
                data = np.repeat(data, 3, axis=0)
            thumb = np.zeros((h, w, 4), dtype=np.uint8)
            thumb[:, :, 0] = data[2]
            thumb[:, :, 1] = data[1]
            thumb[:, :, 2] = data[0]
            thumb[:, :, 3] = 255
        self.thumbnails[image.id] = thumb
        return thumb

    @pyqtSlot()
    def updatePreview(self) -> None:
        if len(self.project.images) == 0:
            return
        image = self.selectedImage()
        angle = self.angle()

        if image.width is not None and image.height is not None:
            M, out_w, out_h, exact = AL.rotationMatrix(image.width, image.height, angle)
            txt = "New map: '{}'\nSize: {} x {} pixels\n".format(image.name + self.suffix(), out_w, out_h)
            txt += "Exact rotation (no resampling)." if exact else "The map is resampled (bilinear interpolation)."
            txt += "\n\nAll the channels (RGB, DEM) and the regions/points are rotated. The original map is not changed."
            self.infoLabel.setText(txt)

        try:
            thumb = self.__thumbnail(image)
        except Exception:
            thumb = None
        if thumb is None:
            self.thumbLabel.clear()
            return

        Mt, tw, th, _ = AL.rotationMatrix(thumb.shape[1], thumb.shape[0], angle)
        rot = np.ascontiguousarray(cv2.warpAffine(thumb, Mt, (tw, th), flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(25, 25, 25, 255)))
        qimg = QImage(rot.data, tw, th, rot.strides[0], QImage.Format_RGB32)
        s = QtRotateMapWidget.THUMB_SIZE
        self.thumbLabel.setPixmap(QPixmap.fromImage(qimg).scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot()
    def rotate(self) -> None:
        """
        Create the rotated copy of the selected map (stored in self.newImage) and close the dialog.
        """
        image = self.selectedImage()
        if image.width is None or image.height is None or len(image.channels) == 0:
            box = QMessageBox()
            box.setText("This map has no image.")
            box.exec()
            return

        angle = self.angle()
        M, out_w, out_h, exact = AL.rotationMatrix(image.width, image.height, angle)
        crs, transform = AL.transformedGeoreference(image.georef_filename, M)
        suffix = AL.uniqueSuffix(self.project, image, self.suffix(), self.taglab_dir)

        progress = QProgressDialog("Rotating..", None, 0, 100, self)
        progress.setWindowTitle("Rotate Map")
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
            self.newImage = AL.createTransformedImage(image, suffix, M, out_w, out_h, crs, transform,
                                                      image.pixelSize(), self.taglab_dir, 1.0, report)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            progress.close()
            box = QMessageBox()
            box.setWindowTitle("Rotate Map")
            box.setText("The map could not be rotated:\n" + str(e))
            box.exec()
            return
        QApplication.restoreOverrideCursor()
        progress.close()
        self.accept()

# TagLab
# A semi-automatic segmentation tool
#
# Dialog of Compare > Carry Regions Over: choose the annotated map, the new map of the same plot, the classes and
# the options. The work is done by TagLab.runCarryOver (source/CarryOver.py).

import torch

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox, QCheckBox, \
    QListWidget, QListWidgetItem, QPushButton


class QtCarryOverWidget(QDialog):

    def __init__(self, project, source_index=0, target_index=1, parent=None):
        super(QtCarryOverWidget, self).__init__(parent)

        self.setStyleSheet("background-color: rgb(40,40,40); color: white")
        self.setWindowTitle("Carry Regions Over")
        self.setMinimumWidth(560)

        self.project = project

        self.comboSource = QComboBox()
        self.comboTarget = QComboBox()
        for image in self.project.images:
            self.comboSource.addItem(image.name)
            self.comboTarget.addItem(image.name)
        n = len(self.project.images)
        self.comboSource.setCurrentIndex(max(0, min(source_index, n - 1)))
        self.comboTarget.setCurrentIndex(max(0, min(target_index, n - 1)))
        self.comboSource.currentIndexChanged.connect(self.updateSource)
        self.comboTarget.currentIndexChanged.connect(self.updateInfo)

        self.listClasses = QListWidget()
        self.listClasses.setMinimumHeight(140)
        self.listClasses.itemChanged.connect(self.updateInfo)

        self.chkRefit = QCheckBox("Fit the outlines to the new map (RITM)")
        self.chkRefit.setChecked(True)
        self.chkRefit.setToolTip("Each outline is adjusted to the new image with RITM, starting from the old one.\n"
                                 "Off: the outlines are copied unchanged (only moved).")
        self.chkRefit.toggled.connect(self.updateInfo)

        self.chkDem = QCheckBox("Use the elevation change (DEM)")
        self.chkDem.setToolTip("A colony that lost most of its height above the surrounding substrate is flagged\n"
                               "as possibly gone. Needs a DEM in both maps.")

        explanation = QLabel(
            "Each region of the first map is placed at the same position on the new map, its outline is fitted to "
            "the new image, and TagLab checks whether the colony is still there (structure, colour, texture and, "
            "with a DEM, height).\n\n"
            "The maps must already be aligned (for example with the Alignment Tool). Doubtful colonies are marked "
            "\"check\" or \"possibly gone\" in the region attribute \"carryover\" and listed for review; nothing is "
            "deleted.")
        explanation.setWordWrap(True)

        self.infoLabel = QLabel()
        self.infoLabel.setWordWrap(True)

        self.btnRun = QPushButton("Run")
        self.btnRun.clicked.connect(self.accept)
        self.btnCancel = QPushButton("Cancel")
        self.btnCancel.clicked.connect(self.reject)

        layoutMaps = QGridLayout()
        layoutMaps.addWidget(QLabel("From map (annotated):"), 0, 0)
        layoutMaps.addWidget(self.comboSource, 0, 1)
        layoutMaps.addWidget(QLabel("To map (new timepoint):"), 1, 0)
        layoutMaps.addWidget(self.comboTarget, 1, 1)

        layoutButtons = QHBoxLayout()
        layoutButtons.addStretch()
        layoutButtons.addWidget(self.btnRun)
        layoutButtons.addWidget(self.btnCancel)

        layout = QVBoxLayout()
        layout.addWidget(explanation)
        layout.addSpacing(8)
        layout.addLayout(layoutMaps)
        layout.addSpacing(8)
        layout.addWidget(QLabel("Classes to carry over:"))
        layout.addWidget(self.listClasses)
        layout.addWidget(self.chkRefit)
        layout.addWidget(self.chkDem)
        layout.addSpacing(8)
        layout.addWidget(self.infoLabel)
        layout.addLayout(layoutButtons)
        self.setLayout(layout)

        self.updateSource()

    def sourceIndex(self):
        return self.comboSource.currentIndex()

    def targetIndex(self):
        return self.comboTarget.currentIndex()

    def classes(self):
        selected = []
        for i in range(self.listClasses.count()):
            item = self.listClasses.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected

    def refit(self):
        return self.chkRefit.isChecked()

    def useDem(self):
        return self.chkDem.isEnabled() and self.chkDem.isChecked()

    def selectedBlobs(self):
        classes = set(self.classes())
        image = self.project.images[self.sourceIndex()]
        return [blob for blob in image.annotations.seg_blobs if blob.class_name in classes]

    @pyqtSlot()
    def updateSource(self):

        image = self.project.images[self.sourceIndex()]
        counts = {}
        for blob in image.annotations.seg_blobs:
            counts[blob.class_name] = counts.get(blob.class_name, 0) + 1

        self.listClasses.blockSignals(True)
        self.listClasses.clear()
        for class_name in sorted(counts):
            item = QListWidgetItem("{:s} ({:d})".format(class_name, counts[class_name]))
            item.setData(Qt.UserRole, class_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if class_name == "Empty" else Qt.Checked)
            self.listClasses.addItem(item)
        self.listClasses.blockSignals(False)

        self.updateInfo()

    @pyqtSlot()
    def updateInfo(self):

        source = self.project.images[self.sourceIndex()]
        target = self.project.images[self.targetIndex()]

        both_dem = source.hasDEM() and target.hasDEM()
        self.chkDem.setEnabled(both_dem)
        self.chkDem.setChecked(both_dem)

        lines = []
        ok = True
        if not both_dem:
            lines.append("The elevation change is not used: it needs a DEM in both maps.")
        if self.sourceIndex() == self.targetIndex():
            lines.append("<font color='orange'>Choose two different maps.</font>")
            ok = False

        n = len(self.selectedBlobs())
        if n == 0:
            lines.append("<font color='orange'>No regions to carry over: tick at least one class.</font>")
            ok = False

        px1 = source.pixelSize()
        px2 = target.pixelSize()
        lines.append("Pixel size: {:.3f} mm (from) / {:.3f} mm (to).".format(px1, px2))
        if abs(px1 - px2) > 0.01 * px2:
            lines.append("<font color='orange'>The pixel sizes differ: the regions are scaled accordingly. Check "
                         "them in the map settings if the maps should have the same pixel size.</font>")

        if n > 0:
            per_colony = 0.5 if self.refit() else 0.1
            if torch.cuda.is_available():
                per_colony = 0.15 if self.refit() else 0.1
            minutes = n * per_colony / 60.0
            lines.append("{:d} regions; about {:s}.".format(n, "{:.0f} min".format(minutes) if minutes >= 1.5
                                                              else "a minute or less"))

        self.infoLabel.setText("<br>".join(lines))
        self.btnRun.setEnabled(ok)

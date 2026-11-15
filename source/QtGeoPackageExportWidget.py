"""
File > Export > Export Regions As GeoPackage (ArcGIS) (Round 16): the options of GeoExport.exportGeoPackage.
"""

import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QCheckBox,
                             QLineEdit, QPushButton, QFileDialog, QGroupBox, QMessageBox)

from source import Georeference as GR


class QtGeoPackageExportWidget(QDialog):

    def __init__(self, project, taglab_dir, active_image=None, parent=None):
        super(QtGeoPackageExportWidget, self).__init__(parent)
        self.project = project
        self.setWindowTitle("Export to GeoPackage (ArcGIS Pro, QGIS)")
        self.setMinimumWidth(560)

        intro = QLabel("A GeoPackage (.gpkg) opens directly in ArcGIS Pro (Add Data, or the TagLab toolbox) and "
                       "in QGIS. Each map gets a layer of regions in its georeference, with the area, perimeter, "
                       "class, colony and every region attribute. Regions of classes hidden in the Labels panel "
                       "are left out, as in the shapefile export.")
        intro.setWordWrap(True)

        # maps
        self.listMaps = QListWidget()
        self.listMaps.setMinimumHeight(110)
        for image in project.images:
            georef = GR.get(image)
            item = QListWidgetItem()
            if georef is None:
                item.setText("{}  (not georeferenced: Project > Georeference Map)".format(image.name))
                item.setFlags(Qt.NoItemFlags)
                item.setCheckState(Qt.Unchecked)
            else:
                item.setText("{}  ({})".format(image.name, GR.crsName(georef.crs)))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, image.id)
            self.listMaps.addItem(item)
        self.listMaps.itemChanged.connect(self.updateButtons)
        boxMaps = QGroupBox("Maps")
        lm = QVBoxLayout()
        lm.addWidget(self.listMaps)
        boxMaps.setLayout(lm)

        # contents
        # short texts: a checkbox doesn't wrap, and the minimum width set above stops the layout from widening the
        # window, so a long text would be cut; the details are in the tooltips
        self.chkPoints = QCheckBox("Point annotations (one layer per map)")
        self.chkMatches = QCheckBox("Comparison tables (as in Export Matches)")
        self.chkMatches.setToolTip("One table per pair of maps: the matches between their regions, with the fate "
                                   "and the area change, as in Compare > Export Matches.")
        self.chkColonies = QCheckBox("Colonies table (as in the genet CSV)")
        self.chkColonies.setToolTip("One row per region per map, with its colony and genet, as in Compare > Export "
                                    "Genet Data as CSV.")
        self.chkViews = QCheckBox("Ortho and DEM views (.vrt files; nothing is copied)")
        self.chkViews.setToolTip("Small .vrt files next to the GeoPackage that point to the map's images and give "
                                 "them TagLab's georeference, so ArcGIS shows the ortho and the DEM in place.")
        for chk in (self.chkPoints, self.chkMatches, self.chkColonies, self.chkViews):
            chk.setChecked(True)
        self.chkWorkingArea = QCheckBox("Only the regions and points inside the working area")
        if project.working_area is not None:
            self.chkWorkingArea.setChecked(True)
        else:
            self.chkWorkingArea.setChecked(False)
            self.chkWorkingArea.setEnabled(False)
            self.chkWorkingArea.setToolTip("No working area is set (Project > Set Working Area).")
        boxContents = QGroupBox("Include")
        lc = QVBoxLayout()
        for chk in (self.chkPoints, self.chkMatches, self.chkColonies, self.chkViews, self.chkWorkingArea):
            lc.addWidget(chk)
        boxContents.setLayout(lc)

        # file
        if project.filename:
            default = os.path.splitext(project.filename)[0] + ".gpkg"
        else:
            default = os.path.join(taglab_dir, "taglab_export.gpkg")
        self.editFile = QLineEdit(default)
        self.editFile.textChanged.connect(self.updateButtons)
        self.btnBrowse = QPushButton("Browse...")
        self.btnBrowse.clicked.connect(self.browse)
        lf = QHBoxLayout()
        lf.addWidget(QLabel("File:"))
        lf.addWidget(self.editFile, 1)
        lf.addWidget(self.btnBrowse)

        self.btnExport = QPushButton("Export")
        self.btnExport.setDefault(True)
        self.btnExport.clicked.connect(self.onExport)
        self.btnCancel = QPushButton("Cancel")
        self.btnCancel.clicked.connect(self.reject)
        lb = QHBoxLayout()
        lb.addStretch(1)
        lb.addWidget(self.btnExport)
        lb.addWidget(self.btnCancel)

        layout = QVBoxLayout()
        layout.addWidget(intro)
        layout.addWidget(boxMaps)
        layout.addWidget(boxContents)
        layout.addLayout(lf)
        layout.addLayout(lb)
        self.setLayout(layout)
        self.updateButtons()

    def selectedImages(self):
        ids = set()
        for k in range(self.listMaps.count()):
            item = self.listMaps.item(k)
            if item.checkState() == Qt.Checked:
                ids.add(item.data(Qt.UserRole))
        return [im for im in self.project.images if im.id in ids]

    def options(self):
        return {"points": self.chkPoints.isChecked(), "matches": self.chkMatches.isChecked(),
                "colonies": self.chkColonies.isChecked(), "views": self.chkViews.isChecked(),
                "working_area": self.chkWorkingArea.isChecked() and self.chkWorkingArea.isEnabled()}

    def filename(self):
        name = self.editFile.text().strip()
        if name and not name.lower().endswith(".gpkg"):
            name += ".gpkg"
        return name

    def updateButtons(self, *args):
        self.btnExport.setEnabled(bool(self.selectedImages()) and bool(self.filename()))

    def browse(self):
        name, _ = QFileDialog.getSaveFileName(self, "Export to GeoPackage", self.filename(), "GeoPackage (*.gpkg)")
        if name:
            self.editFile.setText(name)

    def onExport(self):
        name = self.filename()
        folder = os.path.dirname(os.path.abspath(name))
        if not os.path.isdir(folder):
            QMessageBox.warning(self, self.windowTitle(), "The folder {} doesn't exist.".format(folder))
            return
        if os.path.exists(name):
            answer = QMessageBox.question(self, self.windowTitle(), "{} already exists. Replace it?".format(name),
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        self.editFile.setText(name)
        self.accept()

from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QWidget, QFileDialog, QApplication, QTableWidget, QTableWidgetItem, \
    QHeaderView, QAbstractItemView
from PyQt5.QtWidgets import QComboBox, QLineEdit, QLabel, QPushButton, QHBoxLayout, QVBoxLayout

import os

# project.metadata key of the CoralNet code of each class (remembered for the next export)
CODES_KEY = "coralnet_codes"


class QtExportCoralNetDataWidget(QWidget):
    """
    Points > Export Tiled Data for CoralNet: tiles of the active map (working area, if set) and the CoralNet
    annotation CSV of the points with a class, for training a CoralNet source (Annotation.exportCoralNetData).
    """
    closewidget = pyqtSignal()
    validchoices = pyqtSignal()

    def __init__(self, parent=None):
        super(QtExportCoralNetDataWidget, self).__init__(parent)

        # Parameters
        self.output_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.output_dir = f"{self.output_dir}\\temp"
        self.tile_size = 2048

        # Style for the widget
        self.setStyleSheet("background-color: rgb(40,40,40); color: white")

        # Directory Path
        layout_dir = QHBoxLayout()
        self.lblDir = QLabel("Output Directory: ")
        self.editDir = QLineEdit()
        self.editDir.setText(f"{self.output_dir}")
        self.btnBrowseDir = QPushButton("Browse")
        self.btnBrowseDir.clicked.connect(self.get_directory)
        layout_dir.addWidget(self.lblDir)
        layout_dir.addWidget(self.editDir)
        layout_dir.addWidget(self.btnBrowseDir)

        # Tile Size
        layout_tile = QHBoxLayout()
        self.lblTileSize = QLabel("Tile Size (px): ")
        self.editTileSize = QLineEdit()
        self.editTileSize.setText(f"{self.tile_size}")
        layout_tile.addWidget(self.lblTileSize)
        layout_tile.addWidget(self.editTileSize)

        # CoralNet code of each class
        self.lblCodes = QLabel("CoralNet label code of each class (the short codes of your source's label set).\n"
                               "Only the points with a class are exported: Empty points are skipped.")
        self.tableCodes = QTableWidget()
        self.tableCodes.setColumnCount(3)
        self.tableCodes.setHorizontalHeaderLabels(["Class", "Points", "CoralNet code"])
        self.tableCodes.verticalHeader().hide()
        self.tableCodes.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tableCodes.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tableCodes.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tableCodes.setStyleSheet("QHeaderView::section { background-color: rgb(40,40,40) }")
        self.tableCodes.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.tableCodes.setMinimumHeight(160)
        self.fillCodes()

        layout_info = QVBoxLayout()
        layout_info.setAlignment(Qt.AlignLeft)
        layout_info.addLayout(layout_dir)
        layout_info.addLayout(layout_tile)
        layout_info.addSpacing(10)
        layout_info.addWidget(self.lblCodes)
        layout_info.addWidget(self.tableCodes)

        # Buttons
        layout_buttons = QHBoxLayout()
        self.btnCancel = QPushButton("Cancel")
        self.btnCancel.clicked.connect(self.close)
        self.btnOK = QPushButton("Apply")
        self.btnOK.clicked.connect(self.apply)
        layout_buttons.setAlignment(Qt.AlignRight)
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.btnCancel)
        layout_buttons.addWidget(self.btnOK)

        # Final layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout_info)
        main_layout.addSpacing(20)
        main_layout.addLayout(layout_buttons)
        self.setLayout(main_layout)

        self.setWindowTitle("Export CoralNet Data")
        self.setWindowFlags(Qt.Window |
                            Qt.CustomizeWindowHint |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowTitleHint)

    def pointsPerClass(self):
        """
        Number of points with a class (not Empty) of the active map, inside the working area if one is set.
        """
        counts = {}
        viewer = self.parent().activeviewer
        if viewer is None or viewer.image is None:
            return counts

        working_area = self.parent().project.working_area
        if working_area:
            top, left, width, height = [int(v) for v in working_area]
        else:
            top, left, width, height = 0, 0, viewer.image.width, viewer.image.height

        for point in viewer.annotations.annpoints:
            if point.class_name != "Empty" and left <= point.coordx < left + width and top <= point.coordy < top + height:
                counts[point.class_name] = counts.get(point.class_name, 0) + 1
        return counts

    def fillCodes(self):
        """
        One row per class among the points to export, with the code used last time (or the class name).
        """
        codes = self.parent().project.metadata.get(CODES_KEY, {})
        counts = self.pointsPerClass()

        self.tableCodes.setRowCount(len(counts))
        for row, class_name in enumerate(sorted(counts)):
            item = QTableWidgetItem(class_name)
            item.setFlags(Qt.ItemIsEnabled)
            self.tableCodes.setItem(row, 0, item)
            item = QTableWidgetItem(str(counts[class_name]))
            item.setFlags(Qt.ItemIsEnabled)
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.tableCodes.setItem(row, 1, item)
            self.tableCodes.setItem(row, 2, QTableWidgetItem(codes.get(class_name, class_name)))

    def codes(self):
        """
        {class: code} from the table (codes without leading/trailing spaces).
        """
        codes = {}
        for row in range(self.tableCodes.rowCount()):
            class_name = self.tableCodes.item(row, 0).text()
            item = self.tableCodes.item(row, 2)
            codes[class_name] = item.text().strip() if item is not None else ""
        return codes

    @pyqtSlot()
    def get_directory(self):
        """

        """
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.editDir.setText(directory)

    @pyqtSlot()
    def apply(self):
        """
        Exports the tiles and the CoralNet annotation CSV; the codes are stored in the project.
        """
        try:
            tile_size = int(self.editTileSize.text())
        except ValueError:
            tile_size = 0
        # Check that tile sizes are correct
        if tile_size < 224 or tile_size > 8000:
            QMessageBox.warning(self, "Export CoralNet Data", "The tile size must be a whole number from 224 to 8000.")
            return

        if self.tableCodes.rowCount() == 0:
            QMessageBox.warning(self, "Export CoralNet Data",
                                "There are no points with a class to export on this map (inside the working area, "
                                "if one is set).\n\nPoints > Label Points from Regions gives the points the class of "
                                "the region under them.")
            return

        codes = self.codes()
        missing = [class_name for class_name, code in codes.items() if code == ""]
        if missing:
            QMessageBox.warning(self, "Export CoralNet Data", "Missing CoralNet code for: " + ", ".join(missing))
            return

        self.tile_size = tile_size
        self.output_dir = self.editDir.text()

        # remembered for the next export (saved with the project)
        project = self.parent().project
        stored = dict(project.metadata.get(CODES_KEY, {}))
        stored.update(codes)
        project.metadata[CODES_KEY] = stored

        box = QMessageBox()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Get the channel and annotations
            channel = self.parent().activeviewer.image.getRGBChannel()
            annotations = self.parent().activeviewer.annotations

            # Get the working area (if none, whole ortho is used)
            working_area = project.working_area

            # Export the data
            output_dir, csv_file = annotations.exportCoralNetData(self.output_dir, channel, annotations, working_area,
                                                                  tile_size=self.tile_size, training=True,
                                                                  codes=codes, always_tile=True)
            tiles = len(os.listdir(os.path.join(output_dir, "tiles")))
            points = sum(int(self.tableCodes.item(row, 1).text()) for row in range(self.tableCodes.rowCount()))

            self.close()
            box.setText(f"Exported {points} points on {tiles} tiles to:\n{output_dir}\n\n"
                        f"- tiles: the images to upload to your CoralNet source\n"
                        f"- coralnet_annotations.csv: the point annotations to upload after the images\n"
                        f"- label_codes.csv: the classes and their CoralNet codes\n\n"
                        f"The source's label set must contain all the codes. "
                        f"Save the project to remember the codes.")

        except Exception as e:
            box.setText(f"Failed to export data to CoralNet format! {e}")

        QApplication.restoreOverrideCursor()
        box.exec()

    def closeEvent(self, event):
        """

        """
        super().closeEvent(event)

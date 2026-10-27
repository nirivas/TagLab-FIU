from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QCheckBox, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, \
    QComboBox, QSpinBox, QDoubleSpinBox

from source import PointsFromRegions


class QtPointsFromRegionsWidget(QWidget):
    """
    Options of Points > Label Points from Regions (see source/PointsFromRegions.py); TagLab.labelPointsFromRegions
    does the work when Apply is clicked.
    """

    closewidget = pyqtSignal()

    LEAVE_EMPTY = "Leave them Empty (not exported)"

    def __init__(self, project, working_area, parent=None):
        super(QtPointsFromRegionsWidget, self).__init__(parent)

        self.setStyleSheet(":enabled {background-color: rgb(40,40,40); color: white} :disabled {color: rgb(110,110,110)}")
        spin_style = "background-color: rgb(55,55,55); color: rgb(255,255,255); border: 1px solid rgb(90,90,90)"

        # points already on the map
        self.chk_label = QCheckBox("Give the points already on the map the class of the region under them")
        self.chk_label.setChecked(True)
        self.chk_only_empty = QCheckBox("Only the points still Empty (keep the classes given by hand or by CoralNet)")
        self.chk_only_empty.setChecked(True)
        self.chk_label.toggled.connect(self.chk_only_empty.setEnabled)

        layout_label = QVBoxLayout()
        layout_label.addWidget(self.chk_label)
        layout_only = QHBoxLayout()
        layout_only.addSpacing(20)
        layout_only.addWidget(self.chk_only_empty)
        layout_label.addLayout(layout_only)
        group_label = QGroupBox("Points on the map")
        group_label.setLayout(layout_label)

        # points added inside the regions
        self.chk_add = QCheckBox("Add a point in every region without one")
        self.chk_add.setChecked(True)
        self.spin_per_class = QSpinBox()
        self.spin_per_class.setRange(0, 100000)
        self.spin_per_class.setValue(PointsFromRegions.DEFAULT_PER_CLASS)
        self.spin_per_class.setStyleSheet(spin_style)
        layout_per_class = QHBoxLayout()
        layout_per_class.addSpacing(20)
        layout_per_class.addWidget(QLabel("then more, until every class has at least"))
        layout_per_class.addWidget(self.spin_per_class)
        layout_per_class.addWidget(QLabel("points"))
        layout_per_class.addStretch()
        self.chk_add.toggled.connect(self.spin_per_class.setEnabled)

        layout_add = QVBoxLayout()
        layout_add.addWidget(self.chk_add)
        layout_add.addLayout(layout_per_class)
        group_add = QGroupBox("Points added inside the regions")
        group_add.setLayout(layout_add)

        # edge distance and points on no region
        self.spin_edge = QDoubleSpinBox()
        self.spin_edge.setRange(0.0, 100.0)
        self.spin_edge.setSingleStep(0.5)
        self.spin_edge.setDecimals(1)
        self.spin_edge.setValue(PointsFromRegions.DEFAULT_EDGE_MM)
        self.spin_edge.setStyleSheet(spin_style)
        layout_edge = QHBoxLayout()
        layout_edge.addWidget(QLabel("Edge distance:"))
        layout_edge.addWidget(self.spin_edge)
        layout_edge.addWidget(QLabel("mm (closer to another class or to an area without regions: left Empty)"))
        layout_edge.addStretch()

        self.combo_outside = QComboBox()
        self.combo_outside.setStyleSheet(spin_style)
        self.combo_outside.addItem(self.LEAVE_EMPTY)
        for name in sorted(project.labels.keys()):
            if name != PointsFromRegions.EMPTY:
                self.combo_outside.addItem(name)
        layout_outside = QHBoxLayout()
        layout_outside.addWidget(QLabel("Points on no region:"))
        layout_outside.addWidget(self.combo_outside)
        layout_outside.addStretch()

        txt = "Only the points and regions inside the working area are used." if working_area is not None \
            else "No working area: the whole map is used."
        lbl_area = QLabel(txt + "\nYou will see what will change before it is applied. It cannot be undone with Ctrl+Z.")

        # buttons
        self.btnCancel = QPushButton("Cancel")
        self.btnCancel.clicked.connect(self.close)
        self.btnApply = QPushButton("Apply")
        layout_buttons = QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.btnCancel)
        layout_buttons.addWidget(self.btnApply)

        layout = QVBoxLayout()
        layout.addWidget(group_label)
        layout.addWidget(group_add)
        layout.addLayout(layout_edge)
        layout.addLayout(layout_outside)
        layout.addSpacing(10)
        layout.addWidget(lbl_area)
        layout.addSpacing(10)
        layout.addLayout(layout_buttons)
        self.setLayout(layout)

        self.setWindowTitle("Label Points from Regions")
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowTitleHint)

    def options(self):
        outside = self.combo_outside.currentText()
        return {"label": self.chk_label.isChecked(),
                "only_empty": self.chk_only_empty.isChecked(),
                "add": self.chk_add.isChecked(),
                "per_class": self.spin_per_class.value(),
                "edge_mm": self.spin_edge.value(),
                "outside_class": None if outside == self.LEAVE_EMPTY else outside}

    def closeEvent(self, event):
        self.closewidget.emit()
        super(QtPointsFromRegionsWidget, self).closeEvent(event)

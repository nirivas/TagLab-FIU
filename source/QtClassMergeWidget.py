from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QComboBox, QPushButton, QLabel, \
    QHBoxLayout, QVBoxLayout, QHeaderView, QColorDialog, QAbstractItemView

from source import NewDataset


class QtClassMergeWidget(QDialog):
    """
    Export New Training Dataset > Merge classes...: the name each class is exported with. Classes with the same name
    are merged in the label tiles and "Background" leaves a class out; the regions keep their own classes. The
    result is merges() (stored by TagLab.openClassMerge, used by TagLab.exportNewDataset).
    """

    COL_CLASS, COL_REGIONS, COL_NAME, COL_COLOR = range(4)

    def __init__(self, labels, counts, merges=None, parent=None):
        """
        labels: the project's labels; counts: {class: regions} of the classes on the map;
        merges: {"map": {class: exported name}, "colors": {new name: [r, g, b]}} chosen before.
        """
        super(QtClassMergeWidget, self).__init__(parent)

        self.labels = labels
        self.classes = sorted(counts.keys())
        merges = merges if merges else {}
        stored_map = merges.get("map", {})
        # colours of the new names; the ones chosen by hand (or stored) are kept while the name is not used
        self.colors = {name: list(color) for name, color in merges.get("colors", {}).items()}
        self.picked = set(self.colors.keys())

        self.setStyleSheet("background-color: rgb(40,40,40); color: white")
        edit_style = "background-color: rgb(55,55,55); border: 1px solid rgb(90,90,90)"

        lblInfo = QLabel("Classes with the same name in \"Export as\" are merged in the exported tiles; "
                         "\"Background\" leaves a class out. The regions keep their own classes.\n"
                         "New names are added to the project's labels when the dataset is exported, so that "
                         "Train Your Network and the classifier find them.")
        lblInfo.setWordWrap(True)

        self.table = QTableWidget(len(self.classes), 4)
        self.table.setHorizontalHeaderLabels(["Class", "Regions", "Export as", "Colour"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setStyleSheet("QHeaderView::section {background-color: rgb(55,55,55); color: white; "
                                 "border: 1px solid rgb(90,90,90)}")

        self.combos = []
        self.color_buttons = []
        for row, name in enumerate(self.classes):
            item = QTableWidgetItem(name)
            if name in labels:
                item.setIcon(self.swatch(labels[name].fill))
            item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, self.COL_CLASS, item)

            item = QTableWidgetItem(str(counts[name]))
            item.setFlags(Qt.ItemIsEnabled)
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, self.COL_REGIONS, item)

            combo = QComboBox()
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.setStyleSheet(edit_style)
            combo.setMinimumWidth(200)
            self.combos.append(combo)
            self.table.setCellWidget(row, self.COL_NAME, combo)

            button = QPushButton()
            button.setFixedSize(52, 20)
            button.clicked.connect(lambda checked=False, r=row: self.chooseColor(r))
            self.color_buttons.append(button)
            self.table.setCellWidget(row, self.COL_COLOR, button)

        for row, combo in enumerate(self.combos):
            combo.setEditText(stored_map.get(self.classes[row], self.classes[row]))
        self.refreshChoices()
        for combo in self.combos:
            combo.currentTextChanged.connect(self.updateState)
            combo.activated.connect(self.refreshChoicesLater)
            combo.lineEdit().editingFinished.connect(self.refreshChoicesLater)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.COL_CLASS, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_REGIONS, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_COLOR, QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(min(420, 30 + 30 * max(1, len(self.classes))))

        self.lblSummary = QLabel()
        self.lblSummary.setWordWrap(True)
        self.lblErrors = QLabel()
        self.lblErrors.setWordWrap(True)
        self.lblErrors.setStyleSheet("color: rgb(255,110,110)")

        self.btnDropParts = QPushButton("Drop the (…) part of the names")
        self.btnDropParts.setToolTip("\"Vase (Boat)\" and \"Vase (Scuba)\" are both exported as \"Vase\", and so on")
        self.btnDropParts.clicked.connect(self.dropParts)
        self.btnReset = QPushButton("Reset")
        self.btnReset.setToolTip("Export every class as it is")
        self.btnReset.clicked.connect(self.reset)
        self.btnCancel = QPushButton("Cancel")
        self.btnCancel.clicked.connect(self.reject)
        self.btnOk = QPushButton("OK")
        self.btnOk.setDefault(True)
        self.btnOk.clicked.connect(self.accept)

        layoutButtons = QHBoxLayout()
        layoutButtons.addWidget(self.btnDropParts)
        layoutButtons.addWidget(self.btnReset)
        layoutButtons.addStretch()
        layoutButtons.addWidget(self.btnCancel)
        layoutButtons.addWidget(self.btnOk)

        layout = QVBoxLayout()
        layout.addWidget(lblInfo)
        layout.addWidget(self.table)
        layout.addWidget(self.lblSummary)
        layout.addWidget(self.lblErrors)
        layout.addLayout(layoutButtons)
        self.setLayout(layout)

        self.setWindowTitle("Merge Classes in the Exported Dataset")
        self.setMinimumWidth(640)
        self.updateState()

    @staticmethod
    def swatch(color):
        pixmap = QPixmap(14, 14)
        pixmap.fill(QColor(int(color[0]), int(color[1]), int(color[2])))
        return QIcon(pixmap)

    def targets(self):
        """
        The "Export as" name of every class (in the order of self.classes).
        """
        return [combo.currentText().strip() for combo in self.combos]

    def classMap(self):
        return {name: target for name, target in zip(self.classes, self.targets()) if target != name}

    def isNewName(self, name):
        return name != "" and name not in (NewDataset.BACKGROUND, "Empty") and name not in self.labels

    def choices(self, row):
        """
        Names offered for a class: its own name, its name without the (...) part, the names in use, the classes of
        the map, Background.
        """
        own = self.classes[row]
        names = [own, NewDataset.baseName(own)] + sorted(set(self.targets()) | set(self.classes))
        names.append(NewDataset.BACKGROUND)
        unique = []
        for name in names:
            if name != "" and name not in unique:
                unique.append(name)
        return unique

    def refreshChoicesLater(self):
        # not inside the combo's own signal
        QTimer.singleShot(0, self.refreshChoices)

    def refreshChoices(self):
        for row, combo in enumerate(self.combos):
            text = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self.choices(row))
            combo.setEditText(text)
            combo.blockSignals(False)

    def setTargets(self, names):
        for combo, name in zip(self.combos, names):
            combo.blockSignals(True)
            combo.setEditText(name)
            combo.blockSignals(False)
        self.refreshChoices()
        self.updateState()

    def dropParts(self):
        self.setTargets([NewDataset.baseName(name) for name in self.classes])

    def reset(self):
        self.colors = {}
        self.picked = set()
        self.setTargets(self.classes)

    def chooseColor(self, row):
        name = self.targets()[row]
        if not self.isNewName(name):
            return
        initial = QColor(*self.colors.get(name, [128, 128, 128]))
        color = QColorDialog.getColor(initial, self, "Colour of \"" + name + "\"")
        if color.isValid():
            self.colors[name] = [color.red(), color.green(), color.blue()]
            self.picked.add(name)
            self.updateState()

    def updateState(self):
        class_map = self.classMap()
        targets = self.targets()
        in_use = set(targets)
        new_names = sorted(name for name in in_use if self.isNewName(name))

        # a new name gets the average colour of its classes (unique, not black); unused suggestions are forgotten
        for name in list(self.colors.keys()):
            if name not in in_use and name not in self.picked:
                del self.colors[name]
        for name in new_names:
            if name not in self.colors:
                sources = [self.labels[key].fill for key, target in zip(self.classes, targets)
                           if target == name and key in self.labels]
                used = [label.fill for label in self.labels.values()] + \
                       [self.colors[other] for other in new_names if other in self.colors]
                self.colors[name] = NewDataset.suggestColor(sources, used)

        for button, name in zip(self.color_buttons, targets):
            if self.isNewName(name):
                self.setButtonColor(button, self.colors[name], True, "Click to choose the colour of \"" + name + "\"")
            elif name in self.labels:
                self.setButtonColor(button, self.labels[name].fill, False, "The colour of the label \"" + name + "\"")
            else:
                self.setButtonColor(button, None, False, "Left out: painted as background" if name else "")

        self.lblSummary.setText(NewDataset.mergeSummary(self.classes, class_map))
        errors = NewDataset.mergeErrors(self.labels, class_map, self.colors)
        self.lblErrors.setText("\n".join(errors))
        self.lblErrors.setVisible(len(errors) > 0)
        self.btnOk.setEnabled(len(errors) == 0)

    @staticmethod
    def setButtonColor(button, color, enabled, tooltip):
        if color is None:
            button.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")
        else:
            button.setStyleSheet("background-color: rgb({:d},{:d},{:d}); border: 1px solid rgb(90,90,90)".format(
                int(color[0]), int(color[1]), int(color[2])))
        button.setEnabled(enabled)
        button.setToolTip(tooltip)

    def merges(self):
        """
        {"map": {class: exported name}, "colors": {new name: [r, g, b]}}; classes exported with their own name are
        left out.
        """
        class_map = self.classMap()
        names = sorted(set(class_map.values()))
        return {"map": class_map,
                "colors": {name: list(self.colors[name]) for name in names if self.isNewName(name) and name in self.colors}}

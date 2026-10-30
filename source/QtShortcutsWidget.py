# TagLab
# A semi-automatic segmentation tool
#
# File > Keyboard and Mouse Shortcuts...: the editor of the keys and mouse buttons of source/Shortcuts.py. It edits a
# copy of the shortcuts in use; TagLab saves and applies it when the dialog is accepted (TagLab.openShortcutsEditor).

import html

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QBrush, QColor, QFont
from PyQt5.QtWidgets import QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton, \
    QTreeWidget, QTreeWidgetItem, QHeaderView, QComboBox, QKeySequenceEdit

from source import Shortcuts


class KeyCell(QWidget):
    """
    The key of one action: click and press the new key (only the first key is kept); x clears it.
    """

    def __init__(self, name, dialog):
        super(KeyCell, self).__init__()
        self.name = name
        self.dialog = dialog

        self.edit = QKeySequenceEdit()
        self.edit.setToolTip("Click, then press the new key (with CTRL / ALT / SHIFT if wanted).")
        self.edit.keySequenceChanged.connect(self.changed)
        self.edit.editingFinished.connect(self.finished)
        self.btnClear = QToolButton()
        self.btnClear.setText("x")
        self.btnClear.setToolTip("No key")
        self.btnClear.clicked.connect(self.clear)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.edit)
        layout.addWidget(self.btnClear)
        self.setLayout(layout)

    def showKey(self, code):
        self.edit.blockSignals(True)
        self.edit.setKeySequence(QKeySequence(code) if code != 0 else QKeySequence())
        self.edit.blockSignals(False)

    def changed(self, sequence):
        code = int(sequence[0]) if sequence.count() > 0 else 0
        self.dialog.setKey(self.name, code)

    def finished(self):
        # keep only the first key (and no modifiers on the Move Region keys)
        self.showKey(self.dialog.bindings.keys[self.name])

    def clear(self):
        self.dialog.setKey(self.name, 0)
        self.showKey(0)


class QtShortcutsWidget(QDialog):

    def __init__(self, bindings, parent=None):
        super(QtShortcutsWidget, self).__init__(parent)

        self.setStyleSheet("background-color: rgb(40,40,40); color: white")
        self.setWindowTitle("Keyboard and Mouse Shortcuts")
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowTitleHint)
        self.setMinimumSize(640, 480)
        self.resize(780, 720)

        self.bindings = bindings.copy()   # the edited copy (read by TagLab when the dialog is accepted)
        self.keyCells = {}
        self.mouseCombos = {}
        self.items = {}

        info = QLabel("Click a key and press the new key; the x button removes it. A mouse button runs the action when "
                      "it is pressed on a map (or on the review panel) without SHIFT / CTRL / ALT, and only when the "
                      "action can be used: otherwise it does what it did before (the right button opens the region "
                      "menu). The Move Region keys move by 10 pixels or turn by 90 degrees with SHIFT. Some mice send "
                      "their thumb buttons as the keys Back / Forward: they can be set in the Key column.")
        info.setWordWrap(True)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Action", "Key", "Mouse button"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QTreeWidget.NoSelection)
        self.tree.setFocusPolicy(Qt.NoFocus)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.tree.header().setSectionResizeMode(2, QHeaderView.Fixed)
        self.tree.header().setStretchLastSection(False)
        self.tree.setColumnWidth(1, 190)
        self.tree.setColumnWidth(2, 190)
        self.tree.setStyleSheet("QHeaderView::section { background-color: rgb(55,55,55); color: white }")

        bold = QFont()
        bold.setBold(True)
        for group in Shortcuts.GROUPS:
            group_item = QTreeWidgetItem([group])
            group_item.setFont(0, bold)
            group_item.setFlags(Qt.ItemIsEnabled)
            self.tree.addTopLevelItem(group_item)
            group_item.setFirstColumnSpanned(True)
            for name in Shortcuts.IDS:
                if Shortcuts.GROUP[name] != group:
                    continue
                item = QTreeWidgetItem([Shortcuts.LABEL[name]])
                item.setFlags(Qt.ItemIsEnabled)
                self.tree.addTopLevelItem(item)
                self.items[name] = item
                if Shortcuts.KIND[name] == Shortcuts.FIXED:
                    label = QLabel(Shortcuts.DEFAULT_KEY[name] + " (fixed)")
                    label.setToolTip("Every tool's instructions name this key: it can't be changed, but it can also get "
                                     "a mouse button.")
                    self.tree.setItemWidget(item, 1, label)
                else:
                    cell = KeyCell(name, self)
                    self.keyCells[name] = cell
                    self.tree.setItemWidget(item, 1, cell)
                combo = QComboBox()
                combo.addItem("(none)", "")
                for mouse, button, text in Shortcuts.MOUSE_BUTTONS:
                    combo.addItem(text[0].upper() + text[1:], mouse)
                combo.currentIndexChanged.connect(lambda index, n=name, c=combo: self.setMouse(n, c.currentData()))
                self.mouseCombos[name] = combo
                self.tree.setItemWidget(item, 2, combo)

        self.messageLabel = QLabel()
        self.messageLabel.setWordWrap(True)
        self.messageLabel.setTextFormat(Qt.RichText)

        self.btnDefaults = QPushButton("Restore defaults")
        self.btnDefaults.clicked.connect(self.restoreDefaults)
        self.btnCancel = QPushButton("Cancel")
        self.btnCancel.clicked.connect(self.reject)
        self.btnOk = QPushButton("OK")
        self.btnOk.clicked.connect(self.accept)

        layoutButtons = QHBoxLayout()
        layoutButtons.addWidget(self.btnDefaults)
        layoutButtons.addStretch()
        layoutButtons.addWidget(self.btnCancel)
        layoutButtons.addWidget(self.btnOk)

        layout = QVBoxLayout()
        layout.addWidget(info)
        layout.addWidget(self.tree)
        layout.addWidget(self.messageLabel)
        layout.addLayout(layoutButtons)
        self.setLayout(layout)

        self.fill()

    def fill(self):
        """Show the edited shortcuts."""
        for name, cell in self.keyCells.items():
            cell.showKey(self.bindings.keys[name])
        for name, combo in self.mouseCombos.items():
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(self.bindings.mouse[name])))
            combo.blockSignals(False)
        self.check()

    def setKey(self, name, code):
        self.bindings.setKey(name, code)
        self.check()

    def setMouse(self, name, mouse):
        self.bindings.setMouse(name, mouse)
        self.check()

    def restoreDefaults(self):
        self.bindings.resetDefaults()
        self.fill()

    def check(self):
        """Mark the conflicts in red; OK is off while there are some."""
        errors, notes, bad = self.bindings.conflicts()
        for name, item in self.items.items():
            item.setForeground(0, QBrush(QColor(255, 110, 90) if name in bad else QColor(255, 255, 255)))
        lines = ["<span style='color: rgb(255,110,90)'>" + html.escape(text) + "</span>" for text in errors]
        lines += ["<span style='color: rgb(170,170,170)'>" + html.escape(text) + "</span>" for text in notes]
        self.messageLabel.setText("<br>".join(lines))
        self.btnOk.setEnabled(len(errors) == 0)
        return errors

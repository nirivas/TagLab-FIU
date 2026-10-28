# TagLab
# A semi-automatic segmentation tool
#
# Review panel of the carried-over colonies (Compare > Review Carried-over Colonies). It lists the regions of the
# new map marked "possibly gone" or "check" (region attribute "carryover"), most likely gone first. TagLab shows each
# one in split screen (the old and the new outline selected); the user keeps it, edits it, or removes it.

from PyQt5.QtCore import Qt, QEvent, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget, \
    QTableWidgetItem, QHeaderView, QAbstractItemView

from source import CarryOver


class QtCarryOverReview(QWidget):

    showColony = pyqtSignal(int, int)     # source region id, target region id
    keepColony = pyqtSignal(int)          # target region id
    removeColony = pyqtSignal(int)        # target region id
    moveColony = pyqtSignal(int)          # target region id
    refitColony = pyqtSignal(int)         # target region id
    closed = pyqtSignal()

    COLUMNS = ["Status", "Score", "Why", "Old id", "New id"]
    STATUS_ORDER = {CarryOver.STATUS_GONE: 0, CarryOver.STATUS_CHECK: 1, CarryOver.STATUS_KEPT: 2}

    def __init__(self, source_image, target_image, reasons=None, parent=None):
        super(QtCarryOverReview, self).__init__(parent)

        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setStyleSheet("background-color: rgb(40,40,40); color: white")
        self.setWindowTitle("Review carried-over colonies: " + source_image.name + " -> " + target_image.name)
        self.setMinimumSize(560, 420)
        self.resize(900, 480)

        self.source_image = source_image
        self.target_image = target_image
        self.reasons = reasons if reasons is not None else {}   # target id -> list of reasons (this session only)
        self.removed = set()
        self.ids = []          # target ids of the listed regions, in review order
        self.current = -1

        self.summaryLabel = QLabel()
        self.summaryLabel.setWordWrap(True)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setStyleSheet("QHeaderView::section { background-color: rgb(55,55,55); color: white }")
        self.table.cellClicked.connect(self.goTo)

        helpLabel = QLabel("Each colony is shown in both maps with its old and new outline selected. Edit it with "
                           "the usual tools if needed (or change its class, e.g. to a dead-coral class), then keep "
                           "or remove it. If it landed in the wrong place, Move it (SHIFT + drag or the arrow keys) "
                           "and Re-fit its outline. Keys: N next, P previous, K keep, Delete remove.")
        helpLabel.setWordWrap(True)

        self.btnPrevious = QPushButton("< Previous (P)")
        self.btnPrevious.clicked.connect(self.previous)
        self.btnNext = QPushButton("Next (N) >")
        self.btnNext.clicked.connect(self.next)
        self.btnKeep = QPushButton("Keep (K)")
        self.btnKeep.setToolTip("The colony is still there: its status becomes \"kept\".")
        self.btnKeep.clicked.connect(self.keepCurrent)
        self.btnRemove = QPushButton("Remove (Del)")
        self.btnRemove.setToolTip("The colony is gone: the region is deleted from the new map and it becomes "
                                  "\"dead\" in the comparison table.")
        self.btnRemove.clicked.connect(self.removeCurrent)
        self.btnMove = QPushButton("Move (drag / arrows)")
        self.btnMove.setToolTip("Select the colony and activate the Move Region tool: SHIFT + drag it, or move it with "
                                "the arrow keys (SHIFT + arrow: 10 pixels). Its status does not change: press K "
                                "to keep it.")
        self.btnMove.clicked.connect(self.moveCurrent)
        self.btnRefit = QPushButton("Re-fit outline")
        self.btnRefit.setToolTip("Fit the outline to the new map with RITM (e.g. after moving it). Its status does "
                                 "not change: press K to keep it.")
        self.btnRefit.clicked.connect(self.refitCurrent)
        self.btnClose = QPushButton("Close")
        self.btnClose.clicked.connect(self.close)

        layoutButtons = QHBoxLayout()
        layoutButtons.addWidget(self.btnPrevious)
        layoutButtons.addWidget(self.btnNext)
        layoutButtons.addSpacing(20)
        layoutButtons.addWidget(self.btnKeep)
        layoutButtons.addWidget(self.btnRemove)
        layoutButtons.addSpacing(20)
        layoutButtons.addWidget(self.btnMove)
        layoutButtons.addWidget(self.btnRefit)
        layoutButtons.addStretch()
        layoutButtons.addWidget(self.btnClose)

        layout = QVBoxLayout()
        layout.addWidget(self.summaryLabel)
        layout.addWidget(self.table)
        layout.addWidget(helpLabel)
        layout.addLayout(layoutButtons)
        self.setLayout(layout)

        # the same keys when this panel has the focus (TagLab handles them when the main window has it); the table
        # would use the letters to jump to a row
        self.table.installEventFilter(self)

        self.rebuild()

    def rebuild(self):
        """
        List again the flagged regions of the new map (the review order is fixed here).
        """
        rows = []
        for blob in self.target_image.annotations.seg_blobs:
            status = blob.data.get(CarryOver.ATTR_STATUS)
            if status in self.STATUS_ORDER:
                score = blob.data.get(CarryOver.ATTR_SCORE)
                rows.append((self.STATUS_ORDER[status], -1.0 if score is None else score, blob.id))
        rows.sort()
        self.ids = [r[2] for r in rows]
        self.current = -1
        self.refresh()

    def refresh(self):
        """
        Update the statuses shown (after keep, remove, or an edit) and the counts.
        """
        blobs = {blob.id: blob for blob in self.target_image.annotations.seg_blobs}

        counts = {}
        for blob in blobs.values():
            status = blob.data.get(CarryOver.ATTR_STATUS)
            if status is not None:
                counts[status] = counts.get(status, 0) + 1

        self.table.setRowCount(len(self.ids))
        for row, tid in enumerate(self.ids):
            blob = blobs.get(tid)
            if blob is None:
                values = ["removed" if tid in self.removed else "not on the map", "", "", "", str(tid)]
                color = QColor(120, 120, 120)
            else:
                score = blob.data.get(CarryOver.ATTR_SCORE)
                source_id = blob.data.get(CarryOver.ATTR_FROM)
                status = blob.data.get(CarryOver.ATTR_STATUS, "")
                values = [status, "" if score is None else "{:.2f}".format(score),
                          ", ".join(self.reasons.get(tid, [])), "" if source_id is None else str(source_id), str(tid)]
                if status == CarryOver.STATUS_GONE:
                    color = QColor(255, 120, 100)
                elif status == CarryOver.STATUS_CHECK:
                    color = QColor(255, 200, 80)
                else:
                    color = QColor(140, 220, 140)
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setForeground(QBrush(color))
                if col == 2:
                    item.setToolTip(value)
                self.table.setItem(row, col, item)

        removed = len([tid for tid in self.removed if tid not in blobs])
        self.summaryLabel.setText(
            "Possibly gone: {:d}   Check: {:d}   Kept: {:d}   Removed: {:d}   OK (not listed): {:d}".format(
                counts.get(CarryOver.STATUS_GONE, 0), counts.get(CarryOver.STATUS_CHECK, 0),
                counts.get(CarryOver.STATUS_KEPT, 0), removed, counts.get(CarryOver.STATUS_OK, 0)))

        if 0 <= self.current < len(self.ids):
            self.table.selectRow(self.current)

    def currentTargetId(self):
        if 0 <= self.current < len(self.ids):
            return self.ids[self.current]
        return None

    def currentBlob(self):
        tid = self.currentTargetId()
        if tid is None:
            return None
        return self.target_image.annotations.blobById(tid)

    @pyqtSlot(int, int)
    def goTo(self, row, column=0):
        if row < 0 or row >= len(self.ids):
            return
        self.current = row
        self.table.selectRow(row)
        blob = self.currentBlob()
        if blob is not None:
            source_id = blob.data.get(CarryOver.ATTR_FROM)
            self.showColony.emit(-1 if source_id is None else int(source_id), blob.id)

    @pyqtSlot()
    def next(self):
        # skip the regions no longer on the map
        row = self.current + 1
        while row < len(self.ids) and self.target_image.annotations.blobById(self.ids[row]) is None:
            row += 1
        if row < len(self.ids):
            self.goTo(row)

    @pyqtSlot()
    def previous(self):
        row = self.current - 1
        while row >= 0 and self.target_image.annotations.blobById(self.ids[row]) is None:
            row -= 1
        if row >= 0:
            self.goTo(row)

    @pyqtSlot()
    def keepCurrent(self):
        blob = self.currentBlob()
        if blob is not None:
            self.keepColony.emit(blob.id)
            self.refresh()
            self.next()

    @pyqtSlot()
    def removeCurrent(self):
        blob = self.currentBlob()
        if blob is not None:
            self.removeColony.emit(blob.id)
            if self.target_image.annotations.blobById(blob.id) is None:
                self.removed.add(blob.id)
            self.refresh()
            self.next()

    @pyqtSlot()
    def moveCurrent(self):
        blob = self.currentBlob()
        if blob is not None:
            self.moveColony.emit(blob.id)

    @pyqtSlot()
    def refitCurrent(self):
        blob = self.currentBlob()
        if blob is not None:
            self.refitColony.emit(blob.id)

    @staticmethod
    def plainKey(event):
        return not int(event.modifiers()) & int(Qt.ControlModifier | Qt.AltModifier | Qt.ShiftModifier)

    def eventFilter(self, obj, event):
        if obj is self.table and event.type() == QEvent.KeyPress and self.plainKey(event):
            if self.handleKey(event.key()):
                return True
        return super(QtCarryOverReview, self).eventFilter(obj, event)

    def keyPressEvent(self, event):
        if not (self.plainKey(event) and self.handleKey(event.key())):
            super(QtCarryOverReview, self).keyPressEvent(event)

    def handleKey(self, key):
        """
        The review keys (also used by TagLab while this panel is open). Returns True if the key was used.
        """
        if key == Qt.Key_N:
            self.next()
        elif key == Qt.Key_P:
            self.previous()
        elif key == Qt.Key_K:
            self.keepCurrent()
        elif key == Qt.Key_Delete:
            self.removeCurrent()
        else:
            return False
        return True

    def closeEvent(self, event):
        self.closed.emit()
        super(QtCarryOverReview, self).closeEvent(event)

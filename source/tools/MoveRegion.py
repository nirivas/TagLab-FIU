import html
import math

from PyQt5.QtCore import Qt

from source.tools.Tool import Tool
from source import Shortcuts


class MoveRegion(Tool):
    """
    Move the selected regions to another place of the map: SHIFT + drag (a region not selected is selected first),
    or the arrow keys (see QtImageViewerPlus.nudgeSelected); , and . turn them (QtImageViewerPlus.turnSelected).
    The regions move by whole pixels, keeping their ids (so their matches with the other maps).
    SHIFT + click (without dragging) adds a positive point and SHIFT + right click a negative point that correct the
    outline of the selected region with RITM, as the positive/negative clicks tool does (the corrector, a Ritm
    tool); CTRL + Z removes the last point. SPACE confirms the points or, without points, re-fits the outlines of the
    selected regions (TagLab.refitSelectedRegions).
    """

    DRAG_START = 4     # screen pixels the mouse must move before a press becomes a drag (else it is a click)

    def __init__(self, viewerplus, corrector=None):
        super(MoveRegion, self).__init__(viewerplus)

        self.corrector = corrector
        self.press = None       # (x, y, region pressed, whether it was selected already)
        self.moved = False      # the mouse moved since the press (so the release is not a click)
        self.drag = None        # (x, y where the drag started, regions dragged)
        self.offset = (0, 0)    # current shift of the dragged regions (pixels)

        self.tool_message = self.messageText()

    @staticmethod
    def messageText():
        """The instructions, with the keys set in File > Keyboard and Mouse Shortcuts."""
        bindings = Shortcuts.bindings
        move_keys = bindings.moveKeysText()
        move_keys = html.escape(move_keys[0].upper() + move_keys[1:])
        message = "<p><i>Move regions to the right place and correct their outline</i></p>"
        message += "<p>- SHIFT + LMB drag: move the region (or all the selected regions)</p>"
        message += "<p>- " + move_keys + ": move the selected regions by 1 pixel (with SHIFT: 10 pixels)</p>"
        message += "<p>- " + html.escape(bindings.turnKeysText()) + " : turn the selected regions by 5 degrees " \
                   "(SHIFT: 90 degrees)</p>"
        message += "<p>- SHIFT + LMB click: positive point (part of the region), SHIFT + RMB: negative point " \
                   "(not part of it); RITM corrects the outline of the selected region. CTRL + Z removes the last " \
                   "point. Move and turn first: they are off while there are points.</p>"
        message += "<p>- SPACEBAR: confirm the points; without points, re-fit the outline of the selected regions " \
                   "to the image (RITM + colour)</p>"
        message += "<p>- ESC: cancel the move or the points in progress</p>"
        return f'<div style="text-align: left;">{message}</div>'

    def activate(self):
        self.tool_message = self.messageText()
        self.viewerplus.showMessage(self.tool_message)

    def deactivate(self):
        self.reset()
        self.viewerplus.clearMessage()

    def hasPoints(self):
        return self.corrector is not None and self.corrector.hasPoints()

    def leftPressed(self, x, y, mods=None):

        viewer = self.viewerplus
        self.drag = None
        self.moved = False
        if self.hasPoints():
            self.press = (x, y, None, True)
            return

        blob = viewer.annotations.clickedBlob(x, y)
        was_selected = blob is not None and blob in viewer.selected_blobs
        if blob is not None and not was_selected:
            viewer.resetSelection()
            viewer.addToSelectedList(blob)
        self.press = (x, y, blob, was_selected)
        self.offset = (0, 0)

    def rightPressed(self, x, y, mods=None):

        if mods is not None and mods & Qt.ShiftModifier:
            self.addPoint(x, y, positive=False)

    def mouseMove(self, x, y, mods=None):

        if self.press is None:
            return

        if self.drag is None:
            x0, y0, blob, was_selected = self.press
            if math.hypot(x - x0, y - y0) * self.viewerplus.zoom_factor < self.DRAG_START:
                return
            self.moved = True       # not a click any more
            if blob is None or self.hasPoints():
                return
            self.drag = (x0, y0, list(self.viewerplus.selected_blobs))

        x0, y0, blobs = self.drag
        self.offset = self.viewerplus.clampShift(blobs, int(round(x - x0)), int(round(y - y0)))
        self.showOffset(blobs, self.offset)

    def leftReleased(self, x, y):

        press = self.press
        self.press = None

        if self.drag is not None:
            blobs = self.drag[2]
            dx, dy = self.offset
            self.drag = None
            self.offset = (0, 0)
            self.showOffset(blobs, (0, 0))
            if dx != 0 or dy != 0:
                self.viewerplus.moveBlobs(blobs, dx, dy)
            return

        if press is None or self.moved:
            return
        x0, y0, blob, was_selected = press
        if blob is not None and not was_selected:
            # the click only selected the region
            return
        self.addPoint(x0, y0, positive=True)

    def addPoint(self, x, y, positive):
        """
        A positive or negative point correcting the outline of the selected region (the first point needs exactly
        one selected region; the corrector hides it and shows its corrected outline).
        """
        if self.corrector is None:
            return
        target = None
        if not self.hasPoints():
            if len(self.viewerplus.selected_blobs) != 1:
                return
            target = self.viewerplus.selected_blobs[0]
        if positive:
            self.corrector.leftPressed(x, y, Qt.ShiftModifier)
        else:
            self.corrector.rightPressed(x, y, Qt.ShiftModifier)
        if target is not None and not self.hasPoints() and \
                self.viewerplus.annotations.blobById(target.id) is target:
            # the first point failed (e.g. the view is too large): the region stays selected
            self.viewerplus.resetSelection()
            self.viewerplus.addToSelectedList(target)

    def undo_click(self):
        """
        Remove the last point; when none is left, the region gets back its outline and is selected again.
        """
        target = self.corrector.blob_to_correct
        self.corrector.undo_click()
        if not self.hasPoints() and target is not None and \
                self.viewerplus.annotations.blobById(target.id) is target:
            self.viewerplus.resetSelection()
            self.viewerplus.addToSelectedList(target)

    def confirmPoints(self):
        """
        Replace the region by its outline corrected with the points (one undo step); it stays selected.
        """
        if self.hasPoints():
            self.corrector.apply(keep_selected=True)

    def showOffset(self, blobs, offset):
        """
        Draw the dragged regions shifted by offset (only the drawing: the regions change when the drag ends).
        """
        dx, dy = offset
        for blob in blobs:
            if blob.qpath_gitem is not None:
                blob.qpath_gitem.setPos(dx, dy)
            if blob.id_item is not None:
                blob.id_item.setPos(blob.centroid[0] + dx, blob.centroid[1] + dy)
        self.viewerplus.scene.invalidate()

    def reset(self):
        if self.drag is not None:
            self.showOffset(self.drag[2], (0, 0))
        self.press = None
        self.moved = False
        self.drag = None
        self.offset = (0, 0)
        if self.hasPoints() and self.corrector is not None:
            self.corrector.reset()

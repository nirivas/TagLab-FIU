from source.tools.Tool import Tool


class MoveRegion(Tool):
    """
    Move the selected regions to another place of the map: SHIFT + drag (a region not selected is selected first),
    or the arrow keys (see QtImageViewerPlus.nudgeSelected). The regions move by whole pixels, keeping their ids
    (so their matches with the other maps). SPACE re-fits the outlines of the selected regions with RITM
    (TagLab.refitSelectedRegions).
    """

    def __init__(self, viewerplus):
        super(MoveRegion, self).__init__(viewerplus)

        self.drag = None        # (x, y where the drag started, regions dragged)
        self.offset = (0, 0)    # current shift of the dragged regions (pixels)

        message = "<p><i>Move regions to the right place</i></p>"
        message += "<p>- SHIFT + LMB drag: move the region (or all the selected regions)</p>"
        message += "<p>- Arrow keys: move the selected regions by 1 pixel (SHIFT + arrow: 10 pixels)</p>"
        message += "<p>- SPACEBAR: re-fit the outline of the selected regions to the image (RITM)</p>"
        message += "<p>- ESC: cancel the move in progress</p>"
        self.tool_message = f'<div style="text-align: left;">{message}</div>'

    def activate(self):
        self.viewerplus.showMessage(self.tool_message)

    def deactivate(self):
        self.reset()
        self.viewerplus.clearMessage()

    def leftPressed(self, x, y, mods=None):

        viewer = self.viewerplus
        blob = viewer.annotations.clickedBlob(x, y)
        if blob is None:
            return

        if blob not in viewer.selected_blobs:
            viewer.resetSelection()
            viewer.addToSelectedList(blob)

        self.drag = (x, y, list(viewer.selected_blobs))
        self.offset = (0, 0)

    def mouseMove(self, x, y, mods=None):

        if self.drag is None:
            return

        x0, y0, blobs = self.drag
        self.offset = self.viewerplus.clampShift(blobs, int(round(x - x0)), int(round(y - y0)))
        self.showOffset(blobs, self.offset)

    def leftReleased(self, x, y):

        if self.drag is None:
            return

        blobs = self.drag[2]
        dx, dy = self.offset
        self.drag = None
        self.offset = (0, 0)
        self.showOffset(blobs, (0, 0))
        if dx != 0 or dy != 0:
            self.viewerplus.moveBlobs(blobs, dx, dy)

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
        self.drag = None
        self.offset = (0, 0)

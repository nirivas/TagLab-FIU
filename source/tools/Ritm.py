from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QPen, QBrush

from source.tools.Tool import Tool
from source.Mask import paintMask, jointBox, jointMask, replaceMask, checkIntersection, intersectMask
from source.genutils import qimageToNumpyArray
from source.genutils import cropQImage, maskToQImage, floatmapToQImage

import os
import cv2
import numpy as np

import torch

from models.isegm.inference import clicker
from models.isegm.inference.predictors import get_predictor
from models.isegm.inference import utils as ritmutils

from source.tools import utils

class Ritm(Tool):

    def __init__(self, viewerplus, corrective_points):
        super(Ritm, self).__init__(viewerplus)

        self.points = corrective_points
        self.ritm_net = None
        self.MAX_POINTS = 30

        self.clicker = clicker.Clicker() #handles clicked point (original code of ritm)
        self.predictor = None
        self.predictor_params = {'brs_mode': 'NoBRS'}
        self.init_mask = None
        self.device = None
        self.current_blobs = []
        self.blob_to_correct = None  # selected blob
        self.correction_click = None  # automatic positive click (row, column in the work area) inside it, see segment
        self.work_area_bbox = [0, 0, 0, 0]
        self.work_area_mask = None   # to not overlap old segmented regions
        self.work_area_item = None
        self.states = []

        #message for message_widget window
        message = "<p><i>Segment by using positive/negative points</i></p>"
        message += "<p>Zoom in to frame the work-area containing the target instance.<br/> Try keeping this work-area small, to save memory and execution time.</p>"
        message += "<p>- SHIFT + LMB to add a point inside the instance (positive)<br/>\
                    - SHIFT + RMB to add a point outside the instance (negative)<br/>\
                    - CTRL + Z to remove last point</p>"
        message += "<p>SPACEBAR to confirm segmentation</p>"

        self.tool_message = f'<div style="text-align: left;">{message}</div>'

    def activate(self):
        self.viewerplus.showMessage(self.tool_message)

    def deactivate(self):
        self.viewerplus.clearMessage()

    def checkPointPosition(self, x, y):

        if self.work_area_bbox[2]==0 and self.work_area_bbox[3]==0:
            return True
        if x <= self.work_area_bbox[1] or x>= self.work_area_bbox[1]+self.work_area_bbox[2]:
            return False
        if y <= self.work_area_bbox[0] or y>= self.work_area_bbox[0] + self.work_area_bbox[3]:
            return False

        return True


    def leftPressed(self, x, y, mods):

        points = self.points.positive_points
        if len(points) < self.MAX_POINTS and self.checkPointPosition(x,y) is True and mods == Qt.ShiftModifier:
            self.points.addPoint(x, y, positive=True)
            message = "[TOOL][RITM] New positive point added (" + str(len(points)) + ")"
            self.log.emit(message)

            # apply segmentation
            self.segment()


    def rightPressed(self, x, y, mods):

        points = self.points.negative_points
        if len(points) < self.MAX_POINTS and self.checkPointPosition(x,y) is True and mods == Qt.ShiftModifier:
            self.points.addPoint(x, y, positive=False)
            message = "[TOOL][RITM] New negative point added (" + str(len(points)) + ")"
            self.log.emit(message)

            # apply segmentation
            self.segment()

    def hasPoints(self):
        return self.points.nclicks() > 0

    def undo_click(self):
        self.points.removeLastPoint()
        nclicks = self.points.nclicks()
        if nclicks == 0:
            # reset ALL
            self.reset()
        else:
            prev_state = self.states.pop()
            self.predictor.set_states(prev_state)
            self.segment(save_status=False)


    def initializeWorkArea(self):
        # if image_crop is too big it must be rescaled otherwise the image is too big and the ritm goes out of memory

        rect_map = self.viewerplus.viewportToScene()
        self.work_area_bbox = [round(rect_map.top()), round(rect_map.left()),
                               round(rect_map.width()), round(rect_map.height())]


        image_crop = cropQImage(self.viewerplus.img_map, self.work_area_bbox)
        input_image = qimageToNumpyArray(image_crop)

        # check input image size to prevent crash
        megapixels = (input_image.shape[0] * input_image.shape[1]) / (1024.0 * 1024.0)
        if megapixels > 9.0:
            box = QMessageBox()
            str = "The input size is too big ({:.1f} MPixels). Try to reduce the viewing area by zooming in.".format(megapixels)
            box.setText(str)
            box.exec()
            return False

        # CUDA memory to prevent crash
        oom = False
        try:
            self.predictor.set_input_image(input_image)
        except RuntimeError:  # Out of memory
            oom = True

        if oom:
            box = QMessageBox()
            box.setText("CUDA out of memory. Try to reduce the viewing area by zooming in.")
            box.exec()
            return False

        self.createWorkAreaMask()
        brush = QBrush(Qt.NoBrush)
        pen = QPen(Qt.DashLine)
        pen.setWidth(2)
        pen.setColor(Qt.white)
        pen.setCosmetic(True)
        x = self.work_area_bbox[1]
        y = self.work_area_bbox[0]
        w = self.work_area_bbox[2]
        h = self.work_area_bbox[3]
        self.work_area_item = self.viewerplus.scene.addRect(x, y, w, h, pen, brush)
        self.work_area_item.setZValue(3)
        #self.workingAreaIsActive.emit()

        return True

    def createWorkAreaMask(self):

        w = self.work_area_bbox[2]
        h = self.work_area_bbox[3]
        self.work_area_mask = np.zeros((h,w), dtype=np.int32)
        for blob in self.viewerplus.image.annotations.seg_blobs:
            if checkIntersection(self.work_area_bbox, blob.bbox):
                mask = blob.getMask()
                paintMask(self.work_area_mask, self.work_area_bbox, mask, blob.bbox, 1)

    def intersectionWithExistingBlobs(self, blob):
        bigmask = self.work_area_mask.copy()
        pixels_before = np.count_nonzero(bigmask)
        mask = blob.getMask()
        pixels = np.count_nonzero(mask)
        paintMask(bigmask, self.work_area_bbox, mask, blob.bbox, 0)
        pixels_after = np.count_nonzero(bigmask)
        perc_intersect = ((pixels_before - pixels_after) * 100.0) / pixels

        return perc_intersect

    def prepareInput(self):

        nclicks = self.points.nclicks()
        validArea = True

        if nclicks == 1 and self.work_area_bbox[2] == 0 and self.work_area_bbox[3] == 0:
            # the work area is assigned as the input image of the network
            validArea = self.initializeWorkArea()

        # init mask
        if nclicks == 1 and len(self.viewerplus.selected_blobs) > 0:
            if self.blob_to_correct is None:
                self.blob_to_correct = self.viewerplus.selected_blobs[0]
                self.viewerplus.resetSelection()
                self.viewerplus.undrawBlob(self.blob_to_correct) #removeBlob(self.blob_to_correct)
                if self.work_area_mask is not None:
                    paintMask(self.work_area_mask, self.work_area_bbox, self.blob_to_correct.getMask(),
                            self.blob_to_correct.bbox, 0)

            mask = self.blob_to_correct.getMask()

            init_mask = np.zeros((self.work_area_bbox[3], self.work_area_bbox[2]), dtype=np.int32)
            paintMask(init_mask, self.work_area_bbox, mask, self.blob_to_correct.bbox, 1)
            self.correction_click = self.mostInteriorPoint(init_mask)
            self.init_mask = torch.from_numpy(init_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            self.init_mask = self.init_mask.to(self.device)
        else:
            self.init_mask = None

        # create clicks (see segment for the automatic click of a correction)
        self.clicker.reset_clicks()
        if self.blob_to_correct is not None and self.correction_click is not None:
            self.clicker.add_click(clicker.Click(is_positive=True, coords=self.correction_click))
        for point in self.points.positive_points:
            x = point[0] - self.work_area_bbox[1]
            y = point[1] - self.work_area_bbox[0]
            click = clicker.Click(is_positive=True, coords=(y, x))
            self.clicker.add_click(click)

        for point in self.points.negative_points:
            x = point[0] - self.work_area_bbox[1]
            y = point[1] - self.work_area_bbox[0]
            click = clicker.Click(is_positive=False, coords=(y, x))
            self.clicker.add_click(click)

        return validArea

    def segment(self, save_status=True):

        self.infoMessage.emit("Segmentation is ongoing..")
        self.log.emit("[TOOL][RITM] Segmentation begins..")

        QApplication.setOverrideCursor(Qt.WaitCursor)
        if not self.loadNetwork():
            return

        QApplication.restoreOverrideCursor()

        if self.prepareInput() is True:

            if save_status:
                self.states.append(self.predictor.get_states())

            oom = False
            try:
                if self.init_mask is not None and self.correction_click is not None and save_status:
                    # correcting a region: RITM first gets back its outline from an automatic click at its most
                    # interior point, then the points correct that. Started from the outline alone, RITM changed it
                    # little at the first point. On 297 sponge outlines made too big, too small, or turned and
                    # shifted, the median IoU with the hand-made outline after 1 / 2 / 3 points was 0.61 / 0.82 / 0.90
                    # that way and 0.72 / 0.87 / 0.92 this way (>= 0.85 after 3 points: 79% -> 90%); it never made
                    # an outline worse. Without the old outline (clicks only) the first point gave 0.88, but made 15%
                    # of the outlines worse than the old one.
                    first = clicker.Clicker()
                    first.add_click(clicker.Click(is_positive=True, coords=self.correction_click))
                    self.predictor.get_prediction(first, prev_mask=self.init_mask)
                    pred = self.predictor.get_prediction(self.clicker)
                else:
                    pred = self.predictor.get_prediction(self.clicker, prev_mask=self.init_mask)
            except RuntimeError:  # Out of memory
                oom = True

            if oom:
                self.reset()
                box = QMessageBox()
                box.setText("CUDA out of memory. Try to reduce the viewing area by zooming in.")
                box.exec()
            else:
                segm_mask = pred > 0.5
                segm_mask = segm_mask.astype(np.int32)
                offsetx= self.work_area_bbox[1]
                offsety=self.work_area_bbox[0]

                # this handle corrections by fusing the new segm_mask with the one of the object to correct
                if self.blob_to_correct is not None:
                    bbox_to_correct = self.blob_to_correct.bbox
                    mask_to_correct = self.blob_to_correct.getMask()
                    joint_mask = jointMask(bbox_to_correct, self.work_area_bbox)
                    paintMask(joint_mask[0], joint_mask[1], mask_to_correct, bbox_to_correct, 1)
                    replaceMask(joint_mask[0], joint_mask[1], segm_mask, self.work_area_bbox)
                    offsetx= joint_mask[1][1]
                    offsety= joint_mask[1][0]
                    segm_mask = joint_mask[0]

                segm_mask = segm_mask*255
                torch.cuda.empty_cache()

                utils.undrawAllBlobs(self.current_blobs, self.viewerplus.scene)
                self.current_blobs = []

                blobs = self.viewerplus.annotations.blobsFromMask(segm_mask, offsetx, offsety, 10)

                for blob in blobs:
                    if self.intersectionWithExistingBlobs(blob) < 90.0:
                       self.current_blobs.append(blob)

                if self.blob_to_correct is not None:
                    mask_to_correct = self.blob_to_correct.getMask()
                    box_to_correct = self.blob_to_correct.bbox
                    biggest_blob = None
                    biggest_intersection = -1.0

                    for blob in self.current_blobs:
                        intersection = intersectMask(mask_to_correct, box_to_correct , blob.getMask(), blob.bbox)
                        if intersection is None:
                           intersecting_pixels = 0
                        else:
                           intersecting_pixels = np.count_nonzero(intersection[0])

                        if intersecting_pixels > biggest_intersection:
                           biggest_intersection = intersecting_pixels
                           biggest_blob = blob
                    if biggest_blob is not None:
                        self.current_blobs = [biggest_blob]

                for blob in self.current_blobs:
                    if blob is not None:
                      self.drawBlob(blob)

                self.infoMessage.emit("Segmentation done.")

        else:
            self.reset()

        self.log.emit("[TOOL][RITM] Segmentation ends.")

    def loadNetwork(self):

        if self.ritm_net is None:

            self.infoMessage.emit("Loading RITM network..")

            model_name = 'ritm_corals.pth'
            model_path = os.path.join(os.path.join(self.viewerplus.taglab_dir, "models"), model_name)

            if not torch.cuda.is_available():
                print("CUDA NOT AVAILABLE!")
                device = torch.device("cpu")
            else:
                device = torch.device("cuda:0")

            self.device = device

            try:
                self.ritm_net = ritmutils.load_is_model(model_path, device, cpu_dist_maps=False)
                self.ritm_net.to(device)
                # initialize predictor
                self.predictor = get_predictor(self.ritm_net, device=device, **self.predictor_params)

            except Exception as e:
                box = QMessageBox()
                box.setText("Could not load the Ritm network. You might need to run update.py.")
                box.exec()
                return False

        return True

    def resetNetwork(self):
        try:
            torch.cuda.empty_cache()
            if self.ritm_net is not None:
                del self.ritm_net
                self.ritm_net = None
        except:
            pass

    def apply(self, keep_selected=False):
        """
        Confirm the result and allow to segment another object. A corrected region keeps its id, class, attributes
        and "to check" flag (Blob.inheritInfo); with keep_selected it stays selected.
        """

        # finalize created blobs
        message = "[TOOL][RITM][BLOB-CREATED]"

        corrected = None
        # WARNING!! In the editing mode only the biggest blob is considered (e.g. it is not possible to subdivide
        # an existing blob in two blobs)
        if len(self.current_blobs)>0:
            if self.blob_to_correct is not None:
                new_blob = self.current_blobs[0]
                new_blob.inheritInfo(self.blob_to_correct)
                utils.undrawBlob(new_blob, self.viewerplus.scene, redraw=False)
                self.viewerplus.updateBlob(self.blob_to_correct, new_blob, selected=True)
                corrected = new_blob
                message = "[TOOL][RITM][BLOB-EDITED]"
                self.blobInfo.emit(new_blob, message)
            else:
                for blob in self.current_blobs:

                    # undraw preview
                    utils.undrawBlob(blob, self.viewerplus.scene, redraw=False)

                    # add blob
                    self.viewerplus.addBlob(blob, selected=True)

                    self.blobInfo.emit(blob, message)

                self.viewerplus.project.updateCorrespondences("ADD", self.viewerplus.image, self.current_blobs, None, "")
        elif self.blob_to_correct is not None:
            # nothing segmented: the region keeps its outline (it was hidden at the first click)
            self.viewerplus.drawBlob(self.blob_to_correct)
            corrected = self.blob_to_correct

        self.viewerplus.saveUndo()
        self.viewerplus.resetSelection()
        if keep_selected and corrected is not None:
            self.viewerplus.addToSelectedList(corrected)

        self.init_mask = None
        self.blob_to_correct = None
        self.correction_click = None
        self.current_blobs = []
        self.clicker.reset_clicks()
        self.points.reset()
        self.states = []    # each one holds a copy of the prediction on the whole work area
        self.resetWorkArea()

    @staticmethod
    def mostInteriorPoint(mask):
        """
        The (row, column) of the pixel of mask farthest from its border (None if the mask is empty).
        """
        if not mask.any():
            return None
        dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        row, col = np.unravel_index(int(np.argmax(dist)), dist.shape)
        return int(row), int(col)

    def resetWorkArea(self):
        """
        The reset of the working area causes also the re-initialization of the RITM.
        """
        self.work_area_bbox = [0, 0, 0, 0]
        self.work_area_mask = None
        if self.work_area_item is not None:
            self.viewerplus.scene.removeItem(self.work_area_item)
            self.work_area_item = None

    def reset(self):
        """
        Reset all the information. Called when ESCAPE is pressed.
        """

        self.resetNetwork()

        # re-add the blob removed
        if self.blob_to_correct is not None:
            #self.viewerplus.addBlob(self.blob_to_correct)
            self.viewerplus.drawBlob(self.blob_to_correct)
            self.blob_to_correct = None

        self.init_mask = None
        self.correction_click = None
        utils.undrawAllBlobs(self.current_blobs, self.viewerplus.scene)
        self.current_blobs = []
        self.clicker.reset_clicks()
        self.points.reset()
        self.states = []
        self.resetWorkArea()

    def drawBlob(self, blob):

        scene = self.viewerplus.scene

        # create the suitable brush
        if self.blob_to_correct is None:
            brush = QBrush(Qt.SolidPattern)
            brush.setColor(Qt.white)
        else:
            brush = self.viewerplus.project.classBrushFromName(self.blob_to_correct)

        utils.drawBlob(blob, brush, scene, self.viewerplus.transparency_value, redraw=False)
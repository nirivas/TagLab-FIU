# TagLab                                               
# A semi-automatic segmentation tool                                    
#
# Copyright(C) 2019                                         
# Visual Computing Lab                                           
# ISTI - Italian National Research Council                              
# All rights reserved.                                                      
                                                                          
# This program is free software; you can redistribute it and/or modify      
# it under the terms of the GNU General Public License as published by      
# the Free Software Foundation; either version 2 of the License, or         
# (at your option) any later version.                                       
                                                                           
# This program is distributed in the hope that it will be useful,           
# but WITHOUT ANY WARRANTY; without even the implied warranty of            
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             
#GNU General Public License (http://www.gnu.org/licenses/gpl.txt)          
# for more details.                                               

import os
import glob
import numpy as np
import pickle as pkl
import cv2

# PYTORCH
import torch

# DEEP EXTREME
import models.deeplab_resnet as resnet
from models.dataloaders import helpers as helpers

# DEEPLAB V3+
from models.deeplab import DeepLab

from PyQt5.QtCore import QCoreApplication, Qt, QObject, QThread, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QPainter, QImage, QColor, QPixmap, qRgb, qRed, qGreen, qBlue

from source import genutils
from source.Label import Label
from source.NewDataset import suggestColor


def classifierLabels(classifier_info, labels):
    """
    The labels a classifier is applied with: the given labels, plus a label for each class of the classifier that is
    not one of them, coloured as in its training ("Colors", stored with the classifiers trained since 10/2026). A
    colour that another label already uses is moved, since the label map is read back by colour. It returns
    (labels, added names, missing names without a stored colour).
    """
    colors = classifier_info.get("Colors", {})
    result = dict(labels)
    added = []
    unknown = []
    for name in classifier_info["Classes"]:
        if name == "Background" or name in labels:
            continue
        color = colors.get(name)
        if color is None:
            unknown.append(name)
            continue
        fill = [int(c) for c in color]
        used = [[int(c) for c in label.fill] for label in result.values()]
        if fill in used or fill == [0, 0, 0]:
            fill = suggestColor([fill], used)
        result[name] = Label(id=name, name=name, fill=fill)
        added.append(name)
    return result, added, unknown


class MapClassifier(QObject):
    """
    Given the name of the classifier, the MapClassifier loads and creates it. T
    The interface is common to all the classifier, a map is subdivide into overlapping tiles,
    the tiles are classified, the scores aggregated and put together to form the final
    classification map.
    """

    # custom signal
    updateProgress = pyqtSignal(float)

    def __init__(self, classifier_info, labels_dictionary, parent=None):
        super(QObject, self).__init__(parent)

        self.classifier_name = classifier_info['Classifier Name']
        self.nclasses = classifier_info['Num. Classes']
        self.labels_code_dict = classifier_info['Classes']

        self.background_index = -1

        self.label_colors = [[0,0,0]] * len(self.labels_code_dict)
        for key in self.labels_code_dict.keys():
            if key == "Background":
                self.background_index = self.labels_code_dict[key]
                self.label_colors[self.background_index] = [0,0,0]
            else:
                color = labels_dictionary[key].fill
                index = self.labels_code_dict[key]
                self.label_colors[index] = color

        self.average_norm = classifier_info['Average Norm.']
        self.net = self._load_classifier(classifier_info['Weights'])

        self.flagStopProcessing = False
        self.processing_step = 0
        self.total_processing_steps = 0
        self.scores = None

        self.scale_factor = 1.0
        self.input_image = None
        self.padding = 0
        self.wa_top = 0
        self.wa_left = 0
        self.wa_width = 0
        self.wa_height = 0

        self.temp_dir = "temp"


    def _load_classifier(self, modelName):

        models_dir = "models/"

        network_name = os.path.join(models_dir, modelName)

        classifier = DeepLab(backbone='resnet', output_stride=16, num_classes=self.nclasses)

        if torch.cuda.is_available():
            classifier.load_state_dict(torch.load(network_name, weights_only=False, map_location=torch.device("cuda")))
        else:
            classifier.load_state_dict(torch.load(network_name, weights_only=False, map_location=torch.device("cpu")))

        classifier.eval()

        return classifier

    def setup(self, img_map, pixel_size, target_pixel_size, working_area=[], padding=0):
        """
        Initialize the image to classify.
        """

        self.scale_factor = target_pixel_size / pixel_size
        if not working_area:
            working_area = [0, 0, img_map.width(), img_map.height()]

        # padding the working area (taking into account the scaling factor)
        self.padding = round(padding * self.scale_factor)
        top = int(working_area[0] - self.padding)
        left = int(working_area[1] - self.padding)
        width = int(max(513, working_area[2]) + 2*self.padding)
        height = int(max(513, working_area[3]) + 2*self.padding)

        # crop the input image
        crop_image = img_map.copy(left, top, width, height)
        self.input_image = genutils.qimageToNumpyArray(crop_image)

        # IMPORTNAT NOTE 1:
        # The following information, together with the scale_factor, are used to put the results on the orthoimage
        self.offset = [working_area[0], working_area[1]]

        # scale the input image
        w_target = round(crop_image.width() / self.scale_factor)
        h_target = round(crop_image.height() / self.scale_factor)

        self.input_image = cv2.resize(self.input_image, dsize=(w_target, h_target), interpolation=cv2.INTER_CUBIC)

        # IMPORTANT NOTE 2:
        # since the input image is cropped and resized the working area used for the tiles is
        # always the same, i.e. [padding, padding, w_target -  2 padding , h_target - 2 padding]
        self.padding = round(self.padding / self.scale_factor)
        self.wa_top = self.padding
        self.wa_left = self.padding
        self.wa_width = round(w_target - 2*self.padding)
        self.wa_height = round(h_target - 2*self.padding)


    def run(self, TILE_SIZE, AGGREGATION_WINDOW_SIZE, AGGREGATION_STEP, prediction_threshold=0.5,
            save_scores = False, autocolor = False,  autolevel = False, mode = "fast", batch_size = None):
        """
        Classify the working area and write temp/labelmap.png (and temp/assembled_scores.dat if save_scores is True).
        :param TILE_SIZE: Base tile. This corresponds to the INPUT SIZE of the network.
        :param AGGREGATION_WINDOW_SIZE: Size of the center window considered for the aggregation ("quality" mode).
        :param AGGREGATION_STEP: Step, in pixels, to calculate the different scores ("quality" mode).
        :param mode: "fast" (tiles with 50% overlap blended with a Hann window) or "quality" (9 shifted tiles for each
                     output window, about 9 times slower).
        :param batch_size: tiles classified together ("fast" mode); default 4 on a CUDA GPU, 1 on the CPU.
        """

        if mode == "quality":
            return self._runQuality(TILE_SIZE, AGGREGATION_WINDOW_SIZE, AGGREGATION_STEP, prediction_threshold,
                                    save_scores, autocolor, autolevel)
        return self._runFast(TILE_SIZE, prediction_threshold, save_scores, autocolor, autolevel, batch_size)

    def _prepareTile(self, top, left, size, autocolor, autolevel):
        """
        Crop a tile of the input image (zero padding outside of it) and normalize it as the network expects.
        """

        H, W = self.input_image.shape[0], self.input_image.shape[1]
        tile = np.zeros((size, size, 3), dtype=np.uint8)
        y0, x0 = max(top, 0), max(left, 0)
        y1, x1 = min(top + size, H), min(left + size, W)
        if y1 > y0 and x1 > x0:
            tile[y0 - top:y1 - top, x0 - left:x1 - left] = self.input_image[y0:y1, x0:x1, :3]

        if autocolor is True and autolevel is False:
            tile = genutils.whiteblance(tile)
        if autolevel is True and autocolor is False:
            tile = genutils.autolevel(tile, 1.0)
        if autolevel is True and autocolor is True:
            tile = genutils.autolevel(genutils.whiteblance(tile).astype(np.uint8), 1.0)

        tile = tile.astype(np.float32) / 255.0
        tile = tile.transpose(2, 0, 1)
        for c in range(3):
            tile[c] -= self.average_norm[c]
        return tile

    def _runFast(self, TILE_SIZE, prediction_threshold, save_scores, autocolor, autolevel, batch_size):
        """
        Tiles of TILE_SIZE pixels with 50% overlap; the class probabilities are blended with a Hann window, which
        avoids seams between tiles (the same scheme used e.g. by nnU-Net). Each pixel is computed by about 4 tiles
        instead of 36 tile-areas of the "quality" mode. The working area is processed one row of tiles at a time, so
        the memory needed does not depend on the size of the map.
        """

        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        else:
            for f in glob.glob(os.path.join(self.temp_dir, "*")):
                os.remove(f)

        T = TILE_SIZE
        S = T // 2
        margin = (T - S) // 2
        C = self.nclasses
        H_out, W_out = self.wa_height, self.wa_width

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda") if use_cuda else torch.device("cpu")
        if batch_size is None:
            batch_size = 4 if use_cuda else 1
        if use_cuda:
            torch.backends.cudnn.benchmark = True
            self.net = self.net.to(device, memory_format=torch.channels_last)
        self.net.eval()

        # tiles positions (relative to the working area): the first and last tiles extend beyond it by 'margin'
        nx = max(1, int(np.ceil((W_out + 2 * margin - T) / S)) + 1)
        ny = max(1, int(np.ceil((H_out + 2 * margin - T) / S)) + 1)
        xs = [-margin + k * S for k in range(nx)]
        ys = [-margin + k * S for k in range(ny)]

        # Hann window (never exactly zero, so that the borders of the map are always covered)
        w1 = np.sin(np.pi * (np.arange(T) + 0.5) / T) ** 2
        window = np.maximum(np.outer(w1, w1), 1e-6).astype(np.float32)

        # rolling buffers for one row of tiles: they cover the output rows [y, y + T)
        acc = np.zeros((C, T, W_out), dtype=np.float32)
        wsum = np.zeros((T, W_out), dtype=np.float32)

        # class index per pixel (255 = no class: no background class and a low score)
        NO_CLASS = 255
        labels = np.zeros((H_out, W_out), dtype=np.uint8)
        scores = np.zeros((C, H_out, W_out), dtype=np.float32) if save_scores else None

        self.processing_step = 0
        self.total_processing_steps = nx * ny

        def finalize(y, rows):
            # output rows [y, y + rows) of the buffer are complete
            y0, y1 = max(y, 0), min(y + rows, H_out)
            if y1 <= y0:
                return
            prob = acc[:, y0 - y:y1 - y, :] / wsum[None, y0 - y:y1 - y, :]
            pred = np.argmax(prob, axis=0).astype(np.uint8)
            pred[np.max(prob, axis=0) < prediction_threshold] = \
                self.background_index if self.background_index >= 0 else NO_CLASS
            labels[y0:y1, :] = pred
            if scores is not None:
                scores[:, y0:y1, :] = prob

        for (r, y) in enumerate(ys):
            if self.flagStopProcessing is True:
                break

            for b0 in range(0, nx, batch_size):
                if self.flagStopProcessing is True:
                    break
                bx = xs[b0:b0 + batch_size]
                batch = np.stack([self._prepareTile(self.wa_top + y, self.wa_left + x, T, autocolor, autolevel)
                                  for x in bx])

                with torch.inference_mode():
                    inp = torch.from_numpy(batch).to(device)
                    if use_cuda:
                        inp = inp.contiguous(memory_format=torch.channels_last)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                        out = self.net(inp)
                    prob = torch.softmax(out.float(), dim=1).cpu().numpy()

                for (k, x) in enumerate(bx):
                    x0, x1 = max(x, 0), min(x + T, W_out)
                    if x1 <= x0:
                        continue
                    wpart = window[:, x0 - x:x1 - x]
                    acc[:, :, x0:x1] += prob[k][:, :, x0 - x:x1 - x] * wpart[None]
                    wsum[:, x0:x1] += wpart

                    self.processing_step += 1
                    self.updateProgress.emit((100.0 * self.processing_step) / self.total_processing_steps)
                    self._processEventsIfMainThread()

            if self.flagStopProcessing is True:
                break

            if r < ny - 1:
                # the first S rows are not covered by the next row of tiles: finalize them and shift the buffers
                finalize(y, S)
                acc[:, :T - S, :] = acc[:, S:, :]
                acc[:, T - S:, :] = 0.0
                wsum[:T - S, :] = wsum[S:, :]
                wsum[T - S:, :] = 0.0
            else:
                finalize(y, T)

        if self.flagStopProcessing is False:
            # colors in BGR order (OpenCV); NO_CLASS is drawn black
            palette = np.zeros((256, 3), dtype=np.uint8)
            for i in range(C):
                palette[i] = self.label_colors[i][::-1]
            cv2.imwrite(os.path.join(self.temp_dir, "labelmap.png"), palette[labels])
            if scores is not None:
                with open(os.path.join(self.temp_dir, "assembled_scores.dat"), "wb") as fileobject:
                    pkl.dump(scores, fileobject)

        if use_cuda:
            torch.cuda.empty_cache()
        del self.net
        self.net = None

    def _processEventsIfMainThread(self):
        app = QCoreApplication.instance()
        if app is not None and QThread.currentThread() == app.thread():
            QCoreApplication.processEvents()

    def _runQuality(self, TILE_SIZE, AGGREGATION_WINDOW_SIZE, AGGREGATION_STEP, prediction_threshold=0.5,
                    save_scores = False, autocolor = False,  autolevel = False):
        """
        The original TagLab scheme: for each output window of AGGREGATION_WINDOW_SIZE pixels, 9 tiles of TILE_SIZE
        pixels shifted by AGGREGATION_STEP are classified and their scores averaged.
        """

        # create a temporary folder to store the processing
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        else:
            # if the folder exists, remove all files
            files = glob.glob(os.path.join(self.temp_dir, "*"))
            for f in files:
                os.remove(f)

        # prepare for running..
        DELTA_CROP = int((TILE_SIZE - AGGREGATION_WINDOW_SIZE) / 2)
        tile_cols = int(self.wa_width / AGGREGATION_WINDOW_SIZE) + 1
        tile_rows = int(self.wa_height / AGGREGATION_WINDOW_SIZE) + 1

        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.net.to(device)
            torch.cuda.synchronize()

        self.net.eval()

        # classification (per-tiles)
        tiles_number = tile_rows * tile_cols

        self.processing_step = 0
        self.total_processing_steps = 19 * tiles_number

        for row in range(tile_rows):

            if self.flagStopProcessing is True:
                break

            for col in range(tile_cols):

                if self.flagStopProcessing is True:
                    break

                scores = np.zeros((9, self.nclasses, TILE_SIZE, TILE_SIZE))

                k = 0
                for i in range(-1,2):
                    for j in range(-1,2):

                        top = self.wa_top - DELTA_CROP + row * AGGREGATION_WINDOW_SIZE + i * AGGREGATION_STEP
                        left = self.wa_left - DELTA_CROP + col * AGGREGATION_WINDOW_SIZE + j * AGGREGATION_STEP
                        img_np = genutils.cropImage(self.input_image, [top, left, TILE_SIZE, TILE_SIZE])

                        if autocolor is True and autolevel is False:
                            img_np = genutils.whiteblance(img_np)

                        if autolevel is True and autocolor is False:
                            img_np = genutils.autolevel(img_np, 1.0)

                        if autolevel is True and autocolor is True:
                            white = genutils.whiteblance(img_np)
                            white = white.astype(np.uint8)
                            img_np = genutils.autolevel(white, 1.0)

                        # if i == 0 and j == 0:
                        # tilename = "RGB_r=" + str(row) + "_c=" + str(col) + "_i=" + str(i) + "_j=" + str(j) + ".png"
                        # filename = os.path.join(self.temp_dir, tilename)
                        # tile = genutils.rgbToQImage(img_np)
                        # tile.save(filename)

                        img_np = img_np.astype(np.float32)
                        img_np = img_np / 255.0

                        # H x W x C --> C x H x W
                        img_np = img_np.transpose(2, 0, 1)

                        # Normalization (average subtraction)
                        img_np[0] = img_np[0] - self.average_norm[0]
                        img_np[1] = img_np[1] - self.average_norm[1]
                        img_np[2] = img_np[2] - self.average_norm[2]

                        with torch.no_grad():

                            img_tensor = torch.from_numpy(img_np)
                            input = img_tensor.unsqueeze(0)

                            if torch.cuda.is_available():
                                input = input.to(device)

                            outputs = self.net(input)

                            scores[k] = outputs[0].cpu().numpy()
                            k = k + 1

                            self.processing_step += 1
                            self.updateProgress.emit( (100.0 * self.processing_step) / self.total_processing_steps )
                            self._processEventsIfMainThread()

                if self.flagStopProcessing is True:
                    break

                # import matplotlib.pyplot as plt
                # for k in range(9):
                #     plt.imshow(scores[k,0,513-256:513+256, 513-256:513+256])
                #     filename = "C:\\temp\\predPorite_r=" + str(row) + "_c=" + str(col) + "-" + str(k) + ".png"
                #     plt.savefig(filename)
                #     plt.close('all')
                #     plt.imshow(scores[k,1,513-256:513+256, 513-256:513+256])
                #     filename = "C:\\temp\\predBack_r=" + str(row) + "_c=" + str(col) + "-" + str(k) + ".png"
                #     plt.savefig(filename)
                #     plt.close('all')
                #
                # por = scores[:,0,513-256:513+256, 513-256:513+256]
                # por = np.average(por, axis=0)
                # plt.imshow(por)
                # filename = "C:\\temp\\por_r=" + str(row) + "_c=" + str(col) + ".png"
                # plt.savefig(filename)
                #
                # back = scores[:,1,513-256:513+256, 513-256:513+256]
                # back = np.average(back, axis=0)
                # plt.imshow(back)
                # filename = "C:\\temp\\backg_r=" + str(row) + "_c=" + str(col) + ".png"
                # plt.savefig(filename)

                preds_avg = self.aggregateScores(scores, tile_sz=TILE_SIZE,
                                                     center_window_size=AGGREGATION_WINDOW_SIZE, step=AGGREGATION_STEP)

                values_t, predictions_t = torch.max(torch.from_numpy(preds_avg), 0)
                preds = predictions_t.cpu().numpy()
                values_t = values_t.cpu().numpy()
                unc_matrix = values_t < prediction_threshold
                preds[unc_matrix] = self.background_index  # assign background

                resimg = np.zeros((preds.shape[0], preds.shape[1], 3), dtype='uint8')
                for label_index in range(self.nclasses):
                    resimg[preds == label_index, :] = self.label_colors[label_index]

                tilename = str(row) + "_" + str(col) + ".png"
                filename = os.path.join(self.temp_dir, tilename)
                genutils.rgbToQImage(resimg).save(filename)

                if save_scores is True:
                    tilename = str(row) + "_" + str(col) + ".dat"
                    filename = os.path.join(self.temp_dir, tilename)
                    fileobject = open(filename, 'wb')
                    pkl.dump(preds_avg, fileobject)
                    fileobject.close()

                self.processing_step += 1
                self.updateProgress.emit( (100.0 * self.processing_step) / self.total_processing_steps )
                self._processEventsIfMainThread()

        # when stopped some tiles are missing
        if self.flagStopProcessing is False:
            self.assembleTiles(tile_rows, tile_cols, AGGREGATION_WINDOW_SIZE, ass_scores= save_scores)
        torch.cuda.empty_cache()
        del self.net
        self.net = None

    def loadScores(self):

        filename = os.path.join(self.temp_dir, "assembled_scores.dat")
        fileobject = open(filename, 'rb')
        self.scores = pkl.load(fileobject)
        fileobject.close()

    def assembleTiles(self, tile_rows, tile_cols, AGGREGATION_WINDOW_SIZE, ass_scores = False):

        # put tiles together

        W = AGGREGATION_WINDOW_SIZE * tile_cols
        H = AGGREGATION_WINDOW_SIZE * tile_rows

        if ass_scores is True:
            AWS = AGGREGATION_WINDOW_SIZE
            assembled_scores = np.zeros((self.nclasses, H, W))
            for r in range(tile_rows):
                for c in range(tile_cols):
                    tilename = str(r) + "_" + str(c) + ".dat"
                    filename = os.path.join(self.temp_dir, tilename)

                    fileobject = open(filename, 'rb')
                    scores = pkl.load(fileobject)
                    fileobject.close()

                    xoffset = c * AGGREGATION_WINDOW_SIZE
                    yoffset = r * AGGREGATION_WINDOW_SIZE

                    assembled_scores[:, yoffset: yoffset + AWS, xoffset: xoffset + AWS] = scores[:, 0:AWS, 0:AWS]

            working_area_scores = np.zeros((self.nclasses, self.wa_height, self.wa_width))
            working_area_scores = assembled_scores[:, 0:self.wa_height, 0: self.wa_width]

            filename = os.path.join(self.temp_dir, "assembled_scores.dat")
            fileobject = open(filename, 'wb')
            pkl.dump(working_area_scores, fileobject)
            fileobject.close()

        qimglabel = QImage(W, H, QImage.Format_RGB32)
        painter = QPainter(qimglabel)

        for r in range(tile_rows):
            for c in range(tile_cols):
                tilename = str(r) + "_" + str(c) + ".png"
                filename = os.path.join(self.temp_dir, tilename)
                qimg = QImage(filename)

                xoffset = c * AGGREGATION_WINDOW_SIZE
                yoffset = r * AGGREGATION_WINDOW_SIZE

                painter.drawImage(xoffset, yoffset, qimg)

        # detach the qimglabel otherwise the Qt EXPLODES when memory is free
        painter.end()

        # the classified area can exceed the working area
        qimgworkingarea = qimglabel.copy(0, 0, self.wa_width, self.wa_height)

        labelfile = os.path.join(self.temp_dir, "labelmap.png")
        qimgworkingarea.save(labelfile)

    def classify(self, tresh):
        """
        Given the output scores (C x H x W) it returns the label map.
        """

        scores_copy= self.scores.copy()
        predictions = np.argmax(self.scores, 0)
        max_scores = np.max(self.scores, 0)

        unc_matrix = max_scores < tresh
        predictions[unc_matrix] = self.background_index  # assign background

        resimg = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype='uint8')
        for label_index in range(self.nclasses):
            resimg[predictions == label_index, :] = self.label_colors[label_index]

        qimg = genutils.rgbToQImage(resimg)
        w = qimg.width() * self.scale_factor
        h = qimg.height() * self.scale_factor
        outimg = qimg.scaled(int(w), int(h), Qt.IgnoreAspectRatio, Qt.FastTransformation)
        return outimg

    def stopProcessing(self):

        self.flagStopProcessing = True

    def aggregateScores(self, scores, tile_sz, center_window_size, step):
        """
        Calcute the classification scores using a Bayesian fusion aggregation.
        """""

        nscores = scores.shape[0]
        nclasses = scores.shape[1]

        classification_scores = np.zeros((nscores, nclasses, center_window_size, center_window_size))
        scores_counter = np.zeros((center_window_size, center_window_size), dtype=np.int8)

        # aggregation limits
        top = int((tile_sz - center_window_size) / 2)
        left = int((tile_sz - center_window_size) / 2)

        k = 0
        for i in range(-1,2):
            for j in range(-1,2):

                x1dest = j * step - left
                y1dest = i * step - top

                x2dest = x1dest + tile_sz-1
                y2dest = y1dest + tile_sz-1

                x1src = 0
                if x1dest < 0:
                    x1src = -x1dest
                    x1dest = 0

                y1src = 0
                if y1dest < 0:
                    y1src = -y1dest
                    y1dest = 0

                if x2dest > center_window_size:
                    x2dest = center_window_size

                if y2dest > center_window_size:
                    y2dest = center_window_size

                x2src = x1src + x2dest - x1dest
                y2src = y1src + y2dest - y1dest

                classification_scores[k, :, y1dest:y2dest, x1dest:x2dest] = scores[k, :, y1src:y2src, x1src:x2src]
                scores_counter[y1dest:y2dest, x1dest:x2dest] += 1

                k = k + 1

                self.processing_step += 1
                self.updateProgress.emit( (100.0 * self.processing_step) / self.total_processing_steps )
                self._processEventsIfMainThread()

        #####   AGGREGATE SCORES BY AVERAGING THEM   ##################################################

        # NOTE: SOME APPROACHES AVERAGE THE SCORES DIRECTLY, OTHER ONES AVERAGE THE OUTPUT OF THE SOFTMAX
        #       HERE, WE AVERAGE THE OUTPUT OF THE SOFTMAX

        softmax = torch.nn.Softmax(dim=0)

        classification_scores_avg = np.zeros((nclasses, center_window_size, center_window_size))
        for i in range(nscores):
            prob = softmax(torch.from_numpy(classification_scores[i]))
            classification_scores_avg = classification_scores_avg + prob.numpy()

        classification_scores_avg = classification_scores_avg / nscores

        #classification_scores = np.average(classification_scores, axis=0)
        #classification_scores_avg = softmax(torch.from_numpy(classification_scores)).cpu().numpy()


        #####   AGGREGATE SCORES USING BAYESIAN FUSION   #############################################

        # NOTE THAT:
        #                                              _____
        #                                               | |
        #               p(y|s_N , s_N-1 , s_0) =  p(y)  | |  p(s_i | y)
        #                                             i=0..N
        # CORRESPONDS TO:
        #                                                          __
        #                                                      (   \                )
        #               p(y|s_N , s_N-1 , s_0) =  p(y) SOFTMAX (   /   p(s_i | y))  )
        #                                                      (   ==               )
        #                                                        i=0..N
        #
        # THIS AVOID NUMERICAL PROBLEMS FOR PRODUCTS WITH MANY TERMS.

        # # bayesian aggregation
        # classification_scores_bayes = np.zeros((nclasses, center_window_size, center_window_size))
        #
        # for i in range(nscores):
        #     classification_scores_bayes = classification_scores_bayes + classification_scores[i]
        #
        # classification_scores_bayesian = np.zeros((nclasses, center_window_size, center_window_size))
        #
        # res = softmax(torch.from_numpy(classification_scores_bayes))
        #
        # # PRIOR probabilities
        # prior = [0.7, 0.1, 0.1, 0.1]
        #
        # for i in range(nclasses):
        #     classification_scores_bayesian[i] = prior[i] * res[i].numpy()

        return classification_scores_avg


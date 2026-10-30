# TagLab
# A semi-automatic segmentation tool
#
# Carry the regions of a map over to another map of the same plot (e.g. the next survey). Each region is placed on
# the new map, looked for nearby if it moved or turned (e.g. a nursery brick that was re-laid), a small residual
# shift is corrected, its outline is fitted to the new image (RITM and GrabCut), and a few measures tell whether the
# colony is still there. Doubtful colonies are flagged ("check", "possibly gone") so that the user can review them;
# nothing is decided silently.
#
# The functions work on numpy arrays and QImages only (no widgets), so they can run in a worker thread and be tested
# without the TagLab window.

import math
import os

import cv2
import numpy as np
import rasterio as rio
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from rasterio.transform import Affine
from skimage import measure

from source.Alignment import applyMatrix, composeMatrix, phaseShift
from source.Blob import Blob
from source.DemMetrics import DemReader, DEM_METRIC_NAMES
from source.genutils import cropQImage, qimageToNumpyArray
from source.Mask import paintMask, checkIntersection

# region attributes (names <= 10 characters, the maximum length of a shapefile field name)
ATTR_STATUS = "carryover"   # keyword: ok / check / possibly gone / kept
ATTR_SCORE = "carry_conf"   # 0-1, how sure TagLab is that the colony is still there
ATTR_FROM = "carry_from"    # id of the region it was carried over from
CARRYOVER_ATTRIBUTES = [ATTR_STATUS, ATTR_SCORE, ATTR_FROM]

STATUS_OK = "ok"
STATUS_CHECK = "check"
STATUS_GONE = "possibly gone"
STATUS_KEPT = "kept"
STATUS_KEYWORDS = [STATUS_OK, STATUS_CHECK, STATUS_GONE, STATUS_KEPT]

# --- placement of the whole map ---
GLOBAL_MAX_SIDE = 2000          # longest side of the reduced maps used to look for a global offset
GLOBAL_MIN_RESPONSE = 0.05      # weaker phase-correlation peaks are ignored
GLOBAL_OFFSET_TOLERANCE = 2.0   # offsets below this (in reduced pixels) mean "same grid"

# --- per colony ---
MAX_LOCAL_SHIFT_MM = 50.0       # largest residual shift corrected around a colony
LOCAL_MIN_RESPONSE = 0.08       # weaker phase-correlation peaks are ignored (as in the Alignment Tool)
LOCAL_MAX_SIDE = 512            # patches are reduced to this size for the local phase correlation
MIN_COVERED = 0.5               # minimum fraction of the outline covered by valid pixels of the new map
EXISTING_OVERLAP = 0.5          # overlap (fraction of the smaller region) of a region already on the new map

# --- moved or turned colonies (e.g. nursery bricks re-laid between the surveys) ---
# The old map around the colony (a disc holding the colony and what it sits on) is looked for on the new map at every
# rotation and at shifts up to the search distance, matching the colour and, with a DEM on both maps, the height
# above the ground. Tuned on a sponge nursery (Nursery1, May -> Aug 2026: almost every brick turned, some moved
# ~10 cm) and on the official 2013-2014 reef sample (colonies do not move).
DEFAULT_SEARCH_MM = 150.0       # default search distance
RELOCATE_PIXEL_MM = 2.8         # the search works at about this pixel size (never finer than the map)
RELOCATE_BLUR = 1.0             # Gaussian blur (search pixels) applied before matching
RELOCATE_MIN_RADIUS_MM = 80.0   # radius of the disc matched: at least this..
RELOCATE_SIZE_FACTOR = 0.9      # ..and 0.9 x the longest side of the colony
RELOCATE_COARSE_STEP = 5        # degrees between the rotations tried, then..
RELOCATE_FINE_STEP = 1          # ..every degree around the best one
RELOCATE_RELIEF_MM = (-30.0, 150.0)   # height above the ground (mm) clipped to this range before matching
RELOCATE_MARGIN = 0.1           # a move is used only if it matches better than the colony left in place by this..
RELOCATE_MIN_SCORE = 0.3        # ..and matches at least this well (normalised cross-correlation, -1..1)
RELOCATE_FAR_MM = 50.0          # moves farther than this must match better..
RELOCATE_FAR_MARGIN = 0.02      # ..by this per cm beyond it (on a reef, the far matches were similar neighbours)

RITM_INPUT_SIDE = 400           # longest side of the RITM input (the size RITM works at after the first click)
RITM_EXPANSION = 1.4            # RITM crop = the colony box enlarged 1.4 times..
RITM_MIN_CROP = 200             # ..and at least 200 pixels
GRABCUT_EXPANSION = 2.0         # GrabCut crop = the colony box enlarged 2 times (room for the background colours)
GRABCUT_RING = 0.4              # GrabCut: the colony may grow by this fraction of its radius
GRABCUT_ITERATIONS = 5
GRABCUT_MAX_SIDE = 600          # larger GrabCut crops are reduced to this size
REFIT_AGREEMENT = 0.6           # the old outline kept by the re-fit counts as fitted if it agrees this well (IoU)
REFIT_MAX_GROWTH = 2.0          # re-fit candidates more than 2 times larger than the old outline are left out
MIN_REFIT_IOU = 0.3             # a re-fitted outline overlapping the old one less than this is not used
REFIT_AREA_RANGE = (0.3, 3.0)   # nor one whose area changed more than this

STRUCT_BAND_MM = (2.5, 7.5)     # mm; scale of the structure (branches, ridges) compared between the surveys

# --- "still there?" ---
# Each measure gives a 0-1 score (1 = looks like the same living colony; low below the first value of the range, 1
# above the second). Tuned on the official 2013-2014 sample plus synthetic removed / bleached / overgrown colonies:
# 94-100% of the changed colonies flagged, 75% of the real survivors "ok" (that pair differs a lot in colour and
# sharpness). Retune here for other plots.
STRUCT_RANGE = (0.0, 0.4)       # correlation of the structure inside the outline, year 1 vs year 2
COLOUR_RANGE = (12.0, 25.0)     # change (Lab units) of the colour of the colony relative to its surroundings
COLOUR_WEIGHT = 0.35            # "something else is there" = 0.65 x structure + 0.35 x colour
BRIGHT_RANGE = (15.0, 30.0)     # increase of the brightness relative to the surroundings (bleaching)
TEXTURE_RANGE = (0.4, 0.8)      # texture kept, relative to the surroundings (overgrown by algae, sediment)
DEM_MIN_HEIGHT = 0.02           # m; the height change is used only for colonies at least this high in year 1
DEM_RANGE = (0.3, 0.8)          # height year 2 / year 1 (broken, removed)
DEM_GONE_RATIO = 0.3            # a colony that lost more than 70% of its height is "possibly gone"

SCORE_GONE = 0.3                # score below: "possibly gone"
SCORE_CHECK = 0.6               # score below: "check"
MIN_GONE_AREA_CM2 = 5.0         # smaller colonies are at most "check" (too few pixels to judge)
AREA_CHECK_RANGE = (0.6, 1.6)   # a re-fitted area changing more than this is flagged "check"


#### MATRICES (2x3, pixel-centre coordinates as in the Alignment Tool: the pixel (i, j) covers [i-0.5, i+0.5])

def invertMatrix(M):
    M3 = np.vstack([M, [0.0, 0.0, 1.0]])
    return np.linalg.inv(M3)[:2]


def scaleMatrix(px_src, px_dst):
    """
    Two maps with the same origin (top-left corner) and different pixel sizes (mm).
    """
    s = px_src / px_dst
    t = 0.5 * (s - 1.0)
    return np.array([[s, 0.0, t], [0.0, s, t]])


def georefMatrix(src_filename, dst_filename):
    """
    Source pixel -> target pixel through the georeferencing of the two maps, or None if they are not both
    georeferenced in the same reference system.
    """
    try:
        with rio.open(src_filename) as src, rio.open(dst_filename) as dst:
            ts, tt = src.transform, dst.transform
            cs, ct = src.crs, dst.crs
    except Exception:
        return None

    if ts.is_identity or tt.is_identity or cs is None or ct is None or cs != ct:
        return None

    # rasterio transforms use pixel-corner coordinates
    A = Affine.translation(-0.5, -0.5) * ~tt * ts * Affine.translation(0.5, 0.5)
    return np.array([[A.a, A.b, A.c], [A.d, A.e, A.f]])


#### WHOLE-MAP PLACEMENT

def reducedGray(qimage, max_side):
    """
    The map reduced so that its longest side is at most max_side, in grey levels (float32), with the scale factors.
    """
    w, h = qimage.width(), qimage.height()
    f = min(1.0, max_side / max(w, h))
    rw = max(1, int(round(w * f)))
    rh = max(1, int(round(h * f)))
    small = qimage.scaled(rw, rh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation).convertToFormat(QImage.Format_RGB32)
    gray = cv2.cvtColor(qimageToNumpyArray(small), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return gray, rw / w, rh / h


# phaseShift (phase correlation restricted to the valid pixels) is in source/Alignment.py, imported above


def estimateGlobalOffset(src_qimage, dst_qimage, M, max_side=GLOBAL_MAX_SIDE):
    """
    Residual translation (target pixels) between the source map placed with M and the target map, from the
    reduced maps. It returns (dx, dy, response, reduced pixel in target pixels).
    """
    gs, fsx, fsy = reducedGray(src_qimage, max_side)
    gt, ftx, fty = reducedGray(dst_qimage, max_side)

    Fs = np.array([[fsx, 0.0, 0.0], [0.0, fsy, 0.0]])
    Ft = np.array([[ftx, 0.0, 0.0], [0.0, fty, 0.0]])
    A = composeMatrix(Ft, composeMatrix(M, invertMatrix(Fs)))

    warped = cv2.warpAffine(gs, A, (gt.shape[1], gt.shape[0]), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (warped > 0) & (gt > 0)
    dx, dy, response = phaseShift(warped, gt, valid)

    return dx / ftx, dy / fty, response, 1.0 / ftx


class Placement(object):
    """
    How the regions of the source map are placed on the target map: x_target = M @ [x_source, y_source, 1].
    """

    def __init__(self, M, method, message, warning=None):
        self.M = M
        self.method = method      # "same grid" | "georeference" | "offset"
        self.message = message
        self.warning = warning


def computePlacement(src_image, dst_image, src_qimage, dst_qimage):

    px_src = src_image.pixelSize()
    px_dst = dst_image.pixelSize()

    warning = None
    if abs(px_src - px_dst) / px_dst > 0.01:
        warning = "The two maps have different pixel sizes ({:.3f} and {:.3f} mm). Check them in the map " \
                  "settings: the regions are scaled accordingly.".format(px_src, px_dst)

    M = None
    if src_image.georef_filename and dst_image.georef_filename:
        M = georefMatrix(src_image.georef_filename, dst_image.georef_filename)

    if M is not None:
        dx, dy, response, red = estimateGlobalOffset(src_qimage, dst_qimage, M)
        message = "The maps are placed using their georeferencing."
        if response >= GLOBAL_MIN_RESPONSE and math.hypot(dx, dy) > GLOBAL_OFFSET_TOLERANCE * red:
            extra = "The images seem to be offset by about {:.0f} mm from what the georeferencing says: " \
                    "check the alignment.".format(math.hypot(dx, dy) * px_dst)
            warning = extra if warning is None else warning + "\n" + extra
        return Placement(M, "georeference", message, warning)

    M = scaleMatrix(px_src, px_dst)
    dx, dy, response, red = estimateGlobalOffset(src_qimage, dst_qimage, M)
    if response >= GLOBAL_MIN_RESPONSE and math.hypot(dx, dy) > GLOBAL_OFFSET_TOLERANCE * red:
        M = M.copy()
        M[0, 2] += dx
        M[1, 2] += dy
        message = "The maps are offset by {:.0f}, {:.0f} pixels (found from the images); the regions are moved " \
                  "accordingly.".format(dx, dy)
        return Placement(M, "offset", message, warning)

    return Placement(M, "same grid", "The maps line up (same pixel grid).", warning)


#### PER-COLONY RESULT

class CarriedColony(object):

    def __init__(self, source_blob):
        self.source_id = source_blob.id
        self.class_name = source_blob.class_name
        self.blob = None          # the new region (None if it is not created)
        self.status = None        # STATUS_OK | STATUS_CHECK | STATUS_GONE
        self.score = None         # 0-1
        self.covered = True       # False: outside the new map or over its empty (nodata) area
        self.existing_id = None   # id of a region already on the new map that is the same colony
        self.refit = False        # True if the outline was fitted to the new map (RITM / GrabCut)
        self.relocation = None    # Relocation: the colony was found moved or turned
        self.shift = (0.0, 0.0)   # residual shift corrected (target pixels)
        self.signals = {}
        self.reasons = []


#### IMAGE HELPERS

def rasterize(contour, inner_contours, left, top, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    origin = np.array([left, top], dtype=np.float64)
    cv2.fillPoly(mask, [np.round(contour - origin).astype(np.int32)], 1)
    for inner in inner_contours:
        cv2.fillPoly(mask, [np.round(inner - origin).astype(np.int32)], 0)
    return mask


def translate(img, dx, dy, flags=cv2.INTER_LINEAR):
    T = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(img, T, (img.shape[1], img.shape[0]), flags=flags, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def toGray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def toLab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def disk(radius):
    r = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if den <= 1e-9:
        return 0.0
    return float((a * b).sum()) / den


def largestComponent(mask, reference=None):
    """
    The connected component of mask with the largest overlap with reference (or the largest one).
    """
    labels = measure.label(mask > 0, connectivity=2)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    if reference is not None:
        counts = np.bincount(labels[reference > 0], minlength=labels.max() + 1)
        counts[0] = 0
        if counts.max() == 0:
            return np.zeros_like(mask, dtype=np.uint8)
    else:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
    return (labels == int(np.argmax(counts))).astype(np.uint8)


def blobFromMask(mask, left, top):
    """
    A region from a mask placed at (left, top) of the map (largest component only), or None if the mask is empty.
    """
    labels = measure.label(mask > 0, connectivity=2)
    regions = measure.regionprops(labels)
    if len(regions) == 0:
        return None
    region = max(regions, key=lambda r: r.area)
    return Blob(region, left, top, 0)


def heightAbovePlane(z, mask, ring_width):
    """
    90th percentile of the height (m) of the masked DEM values above a plane fitted to a ring around the mask
    (as the DEM metrics do), or None if the DEM does not cover enough of it.
    """
    m = mask > 0
    valid = m & np.isfinite(z)
    if valid.sum() < 0.5 * max(1, m.sum()):
        return None

    ring = cv2.dilate(m.astype(np.uint8), disk(ring_width)).astype(bool) & ~m & np.isfinite(z)
    rows, cols = np.nonzero(ring)
    zs = z[valid]
    if len(rows) >= 3:
        A = np.column_stack((cols, rows, np.ones(len(rows))))
        coeffs = np.linalg.lstsq(A, z[ring], rcond=None)[0]
        vr, vc = np.nonzero(valid)
        base = coeffs[0] * vc + coeffs[1] * vr + coeffs[2]
    else:
        base = np.full(len(zs), zs.min())

    return max(0.0, float(np.percentile(zs - base, 90)))


#### "STILL THERE?" MEASURES

def bandPass(gray, px_size):
    """
    Structure at the scale of branches, ridges and corallites (STRUCT_BAND_MM, converted to pixels): finer detail
    is mostly noise and compression artefacts, and differs between surveys even where nothing changed.
    """
    s1 = max(0.7, STRUCT_BAND_MM[0] / px_size)
    s2 = max(2.0 * s1, STRUCT_BAND_MM[1] / px_size)
    return cv2.GaussianBlur(gray, (0, 0), s1) - cv2.GaussianBlur(gray, (0, 0), s2)


def presenceSignals(src, tgt, mask, px_size):
    """
    Measures comparing the colony (mask) in the two maps, both on the same grid. The colony is compared with a ring
    around it, so that differences between the two surveys (lighting, colour balance, sharpness) cancel out:
     - struct_corr: correlation of the structure inside the outline, year 1 vs year 2 (low: something else is there)
     - texture_rel: how much of its texture the colony kept, relative to the surroundings (low: smoothed over, e.g.
       algae or sediment)
     - brightness_change, colour_change: change of the colour of the colony relative to the surroundings (Lab)
    """
    valid = src.any(axis=2) & tgt.any(axis=2)
    inside = (mask > 0) & valid
    n = int(inside.sum())
    signals = {"pixels": n}
    if n < 9:
        return signals

    size = math.sqrt(n)
    gap = max(1, int(round(0.05 * size)))
    width = max(3, int(round(0.15 * size)))
    m8 = (mask > 0).astype(np.uint8)
    ring = cv2.dilate(m8, disk(gap + width)).astype(bool) & ~cv2.dilate(m8, disk(gap)).astype(bool) & valid
    core = cv2.erode(m8, disk(1)).astype(bool) & valid
    if core.sum() < 9:
        core = inside

    bs = bandPass(toGray(src), px_size)
    bt = bandPass(toGray(tgt), px_size)
    signals["struct_corr"] = correlation(bs[core], bt[core])

    if ring.sum() < 9:
        return signals

    def ratio(a, b):
        return a / b if b > 1e-6 else None

    tex_in = ratio(float(bt[core].std()), float(bs[core].std()))
    tex_ring = ratio(float(bt[ring].std()), float(bs[ring].std()))
    if tex_in is not None and tex_ring is not None and tex_ring > 1e-6:
        signals["texture_rel"] = tex_in / tex_ring

    ls = toLab(src)
    lt = toLab(tgt)
    rel1 = ls[inside].mean(axis=0) - ls[ring].mean(axis=0)
    rel2 = lt[inside].mean(axis=0) - lt[ring].mean(axis=0)
    signals["colour_change"] = float(np.linalg.norm(rel2 - rel1))
    signals["brightness_change"] = float(rel2[0] - rel1[0])

    return signals


def rangeScore(value, low, high):
    """
    0 at low, 1 at high (low may be larger than high), clipped.
    """
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def decideStatus(signals, px_size, refit_tried, refit_ok):
    """
    Combine the measures into a score (0-1, 1 = looks like the same living colony) and a status. The score is the
    lowest of: "something else is there" (structure and colour), bleaching, loss of texture, loss of height. It
    returns (score, status, reasons).
    """
    if signals.get("struct_corr") is None:
        return None, STATUS_CHECK, ["too small to check"]

    scores = {"struct": rangeScore(signals["struct_corr"], *STRUCT_RANGE)}
    other = scores["struct"]
    if signals.get("colour_change") is not None:
        scores["colour"] = rangeScore(signals["colour_change"], COLOUR_RANGE[1], COLOUR_RANGE[0])
        other = (1.0 - COLOUR_WEIGHT) * scores["struct"] + COLOUR_WEIGHT * scores["colour"]
        scores["bright"] = rangeScore(signals["brightness_change"], BRIGHT_RANGE[1], BRIGHT_RANGE[0])
    if signals.get("texture_rel") is not None:
        scores["texture"] = rangeScore(signals["texture_rel"], *TEXTURE_RANGE)
    dem_ratio = signals.get("dem_ratio")
    if dem_ratio is not None:
        scores["dem"] = rangeScore(dem_ratio, *DEM_RANGE)

    score = min([other] + [scores[k] for k in ("bright", "texture", "dem") if k in scores])

    reasons = []
    labels = [("struct", "different structure"), ("colour", "colour changed"), ("bright", "brighter (bleached?)"),
              ("texture", "texture lost (overgrown?)"), ("dem", "lower (broken?)")]
    for key, text in labels:
        if scores.get(key, 1.0) < 0.5:
            reasons.append(text)

    status = STATUS_OK
    if score < SCORE_GONE or (dem_ratio is not None and dem_ratio < DEM_GONE_RATIO):
        status = STATUS_GONE
    elif score < SCORE_CHECK:
        status = STATUS_CHECK

    area_cm2 = signals.get("pixels", 0) * px_size * px_size / 100.0
    if status == STATUS_GONE and area_cm2 < MIN_GONE_AREA_CM2:
        status = STATUS_CHECK
        reasons.append("small")

    if refit_tried and not refit_ok:
        reasons.append("outline not re-fitted")
        if status == STATUS_OK:
            status = STATUS_CHECK

    area_ratio = signals.get("area_ratio")
    if refit_ok and area_ratio is not None and not (AREA_CHECK_RANGE[0] <= area_ratio <= AREA_CHECK_RANGE[1]):
        reasons.append("{} ({:+.0f}% area)".format("smaller" if area_ratio < 1.0 else "larger",
                                                   100.0 * (area_ratio - 1.0)))
        if status == STATUS_OK:
            status = STATUS_CHECK

    return score, status, reasons


#### CONTEXT AND MAIN LOOP

class ExistingRegion(object):
    """
    A region already present on the target map (copied, so the worker thread does not touch the annotations).
    """

    def __init__(self, blob):
        self.id = blob.id
        self.class_name = blob.class_name
        self.bbox = np.array(blob.bbox).astype(int)
        self.mask = blob.getMask()
        self.area = int(np.count_nonzero(self.mask))


class RitmRefit(object):
    """
    Re-fit outlines to the image with RITM (the network of the positive/negative clicks tool). The network is loaded
    on first use and freed by close().
    """

    def __init__(self, taglab_dir):
        self.taglab_dir = taglab_dir
        self.predictor = None
        self.device = None
        self.network = None
        self.choice = None          # outline kept by the last refit (see refit)
        self.agreement = None

    def load(self):

        if self.predictor is not None:
            return

        from models.isegm.inference import utils as ritmutils
        from models.isegm.inference.predictors import get_predictor

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model_path = os.path.join(self.taglab_dir, "models", "ritm_corals.pth")
        self.network = ritmutils.load_is_model(model_path, self.device, cpu_dist_maps=False)
        self.network.to(self.device)
        # no zoom-in and no flip: each colony is cropped and resized here, one prediction per colony
        self.predictor = get_predictor(self.network, 'NoBRS', self.device, with_flip=False, zoom_in_params=None)

    def close(self):
        self.predictor = None
        self.network = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, small, clicks):
        """
        RITM on the prepared image small (see refit) with the clicks [(row, column, positive)], from an empty starting
        mask. It returns the probability map (same size as small).
        """
        from models.isegm.inference.clicker import Clicker, Click

        clicker = Clicker()
        for row, col, positive in clicks:
            clicker.add_click(Click(is_positive=positive, coords=(int(row), int(col))))
        prev_mask = torch.zeros((1, 1) + small.shape[:2], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.predictor.get_prediction(clicker, prev_mask=prev_mask)

    def refit(self, img, mask):
        """
        The outline (mask) fitted to the image. Three candidate outlines are made around it:
         - RITM with one click at its most interior point, and RITM with two clicks (the two most interior points far
           apart), both without a starting outline: started from the old outline, RITM gives it back unchanged, so
           it could neither shrink nor grow;
         - GrabCut started from the outline (colour: inside = probably colony, a ring around it = probably not).
        Candidates more than REFIT_MAX_GROWTH times larger than the old outline are left out (on a sponge nursery they
        were mostly the whole brick the sponge sits on), then the one agreeing best with the others and with the old
        outline (sum of the IoUs) is returned; it may be the old outline itself. On the hand-made outlines of a sponge
        nursery made 30% too big, or turned by 20 degrees and shifted, this choice came back to them with a median IoU
        of 0.82 and 0.86 (0.57 and 0.64 before), closer than any of the candidates alone; an outline can at most
        double its area.
        It returns the new mask (same size), or None if the colony is too small to try. self.choice tells which
        outline was kept ("ritm 1 click", "ritm 2 clicks", "grabcut" or "old outline") and self.agreement how well it
        agrees with the others (mean IoU).
        """
        self.choice = None
        self.agreement = None

        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            return None
        qy0, qx0, qy1, qx1 = cropAround(mask.shape, cols.min(), rows.min(), cols.max() + 1, rows.max() + 1,
                                        RITM_EXPANSION)
        qw, qh = qx1 - qx0, qy1 - qy0

        crop = img[qy0:qy1, qx0:qx1]
        mcrop = mask[qy0:qy1, qx0:qx1]

        f = RITM_INPUT_SIDE / max(qw, qh)
        size = (max(1, int(round(qw * f))), max(1, int(round(qh * f))))
        small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA if f < 1.0 else cv2.INTER_LINEAR)
        msmall = cv2.resize(mcrop.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        binary = (msmall > 0.5).astype(np.uint8)
        if binary.sum() == 0:
            return None

        clicks = [(r, c, True) for r, c in interiorPoints(binary, 2)]

        self.load()
        with torch.no_grad():
            self.predictor.set_input_image(np.ascontiguousarray(small))

        placed = (mask > 0).astype(np.uint8)
        candidates = []
        for name, n in (("ritm 1 click", 1), ("ritm 2 clicks", 2)):
            if len(clicks) < n:
                continue
            prob = cv2.resize(self.predict(small, clicks[:n]).astype(np.float32), (qw, qh),
                              interpolation=cv2.INTER_LINEAR)
            out = np.zeros_like(mask, dtype=np.uint8)
            out[qy0:qy1, qx0:qx1] = largestComponent((prob > 0.5).astype(np.uint8), reference=mcrop)
            candidates.append((name, out))
        candidates.append(("grabcut", grabCutRefit(img, mask)))

        limit = REFIT_MAX_GROWTH * np.count_nonzero(placed)
        candidates = [("old outline", placed)] + [(name, m) for name, m in candidates
                                                  if 0 < np.count_nonzero(m) <= limit]
        self.choice, chosen, self.agreement = consensus(candidates)
        return chosen


def cropAround(shape, x0, y0, x1, y1, expansion):
    """
    The crop (top, left, bottom, right) of an image of the given shape around the box [x0, x1) x [y0, y1), enlarged
    by expansion and at least RITM_MIN_CROP pixels wide and high, clipped to the image.
    """
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    ew = max(RITM_MIN_CROP, expansion * (x1 - x0))
    eh = max(RITM_MIN_CROP, expansion * (y1 - y0))
    H, W = shape[:2]
    return (max(0, int(math.floor(cy - eh / 2))), max(0, int(math.floor(cx - ew / 2))),
            min(H, int(math.ceil(cy + eh / 2))), min(W, int(math.ceil(cx + ew / 2))))


def interiorPoints(mask, n):
    """
    Up to n points (row, column) well inside the mask: the most interior one, then the most interior one outside
    discs around the previous ones (1.2 x their distance to the border).
    """
    dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    left = dist.copy()
    rows, cols = np.ogrid[:dist.shape[0], :dist.shape[1]]
    points = []
    for i in range(n):
        r, c = np.unravel_index(int(np.argmax(left)), left.shape)
        if dist[r, c] < 2.0:
            break
        points.append((int(r), int(c)))
        left[(rows - r) ** 2 + (cols - c) ** 2 <= (1.2 * dist[r, c]) ** 2] = 0
    return points


def grabCutRefit(img, mask):
    """
    The outline (mask) fitted to the colours of the image with GrabCut: inside the outline = probably colony, a ring
    around it (GRABCUT_RING of its radius) = probably not, farther = not. It returns the new mask (same size; empty if
    GrabCut fails).
    """
    out = np.zeros_like(mask, dtype=np.uint8)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return out
    qy0, qx0, qy1, qx1 = cropAround(mask.shape, cols.min(), rows.min(), cols.max() + 1, rows.max() + 1,
                                    GRABCUT_EXPANSION)
    crop = img[qy0:qy1, qx0:qx1]
    start = (mask[qy0:qy1, qx0:qx1] > 0).astype(np.uint8)

    f = min(1.0, GRABCUT_MAX_SIDE / max(crop.shape[:2]))
    if f < 1.0:
        size = (max(1, int(round(crop.shape[1] * f))), max(1, int(round(crop.shape[0] * f))))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        start = cv2.resize(start, size, interpolation=cv2.INTER_NEAREST)
    if start.sum() == 0:
        return out

    radius = math.sqrt(start.sum() / math.pi)
    outer = cv2.dilate(start, disk(max(3.0, GRABCUT_RING * radius)))
    labels = np.full(start.shape, cv2.GC_BGD, dtype=np.uint8)
    labels[outer > 0] = cv2.GC_PR_BGD
    labels[start > 0] = cv2.GC_PR_FGD
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2BGR), labels, None, bgd, fgd,
                    GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return out
    seg = largestComponent(((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8), reference=start)

    if f < 1.0:
        seg = cv2.resize(seg, (qx1 - qx0, qy1 - qy0), interpolation=cv2.INTER_NEAREST)
    out[qy0:qy1, qx0:qx1] = seg
    return out


def maskIoU(a, b):
    union = int(np.count_nonzero(a | b))
    return int(np.count_nonzero(a & b)) / union if union > 0 else 0.0


def consensus(candidates):
    """
    Of the candidate outlines [(name, mask)], the one agreeing best with the others (largest sum of IoUs); empty
    masks are left out. It returns (name, mask, mean IoU with the others).
    """
    candidates = [(name, m) for name, m in candidates if m is not None and m.any()]
    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1], None
    best = None
    for i, (name, m) in enumerate(candidates):
        total = sum(maskIoU(m, other) for j, (_, other) in enumerate(candidates) if j != i)
        if best is None or total > best[2]:
            best = (name, m, total)
    return best[0], best[1], best[2] / (len(candidates) - 1)


def acceptRefit(mask, refitted):
    """
    Whether the re-fitted outline replaces the old one (mask): it must overlap it enough and its area must not change
    too much. Returns (accepted, IoU, area ratio).
    """
    if refitted is None:
        return False, None, None
    inter = int(np.count_nonzero(refitted & mask))
    union = int(np.count_nonzero(refitted | mask))
    iou = inter / union if union > 0 else 0.0
    ratio = float(refitted.sum()) / float(max(1, mask.sum()))
    return iou >= MIN_REFIT_IOU and REFIT_AREA_RANGE[0] <= ratio <= REFIT_AREA_RANGE[1], iou, ratio


def refitBox(blob, width, height):
    """
    The box [top, left, w, h] of the map around a region with room for the re-fit crops (see RitmRefit.refit),
    clipped to the map.
    """
    top, left, w, h = [float(v) for v in blob.bbox]
    cx, cy = left + w / 2.0, top + h / 2.0
    expansion = max(RITM_EXPANSION, GRABCUT_EXPANSION)
    ew = max(RITM_MIN_CROP, expansion * w) / 2.0 + 4
    eh = max(RITM_MIN_CROP, expansion * h) / 2.0 + 4
    x0 = max(0, int(math.floor(cx - ew)))
    y0 = max(0, int(math.floor(cy - eh)))
    x1 = min(int(width), int(math.ceil(cx + ew)))
    y1 = min(int(height), int(math.ceil(cy + eh)))
    return [y0, x0, x1 - x0, y1 - y0]


class CarryOverContext(object):

    def __init__(self, src_image, dst_image, src_qimage, dst_qimage, placement, taglab_dir, refit=True,
                 use_dem=False, existing_blobs=None, search_mm=0.0):

        self.src_image = src_image
        self.dst_image = dst_image
        self.src_qimage = src_qimage
        self.dst_qimage = dst_qimage
        self.placement = placement
        self.taglab_dir = taglab_dir
        self.refit = refit
        self.use_dem = use_dem            # the height change counts in the "still there?" score
        self.search_mm = search_mm        # > 0: look for colonies that moved or turned, up to this distance

        self.px_src = src_image.pixelSize()
        self.px_dst = dst_image.pixelSize()
        self.max_shift_px = MAX_LOCAL_SHIFT_MM / self.px_dst

        self.existing = [ExistingRegion(b) for b in (existing_blobs or []) if b.class_name != "Empty"]

        self.ritm = RitmRefit(taglab_dir)

        # the DEMs are read for the height change and to find moved colonies
        self.src_dem = None
        self.dst_dem = None
        if (use_dem or search_mm > 0) and src_image.hasDEM() and dst_image.hasDEM():
            self.src_dem = DemReader(os.path.join(taglab_dir, src_image.getDEMChannel().filename), self.px_src)
            self.dst_dem = DemReader(os.path.join(taglab_dir, dst_image.getDEMChannel().filename), self.px_dst)

    def close(self):
        for reader in (self.src_dem, self.dst_dem):
            if reader is not None:
                reader.close()
        self.src_dem = self.dst_dem = None
        self.ritm.close()

    # the RITM network (shared by the tests between contexts)
    @property
    def predictor(self):
        return self.ritm.predictor

    @predictor.setter
    def predictor(self, value):
        self.ritm.predictor = value

    @property
    def device(self):
        return self.ritm.device

    @device.setter
    def device(self, value):
        self.ritm.device = value

    def loadRitm(self):
        self.ritm.load()

    def refitWithRitm(self, img, mask):
        return self.ritm.refit(img, mask)

    def cropTarget(self, box):
        return qimageToNumpyArray(cropQImage(self.dst_qimage, box))

    def cropSourceOnTarget(self, box, M=None):
        """
        The source map resampled on the target grid, in the box [top, left, w, h] of the target map. M (source pixel
        -> target pixel) is the placement of the maps unless given (e.g. with the move of a colony).
        """
        M = self.placement.M if M is None else M
        top, left, w, h = box
        sl, st, sw, sh = self.sourceWindow(box, M)
        src = qimageToNumpyArray(cropQImage(self.src_qimage, [st, sl, sw, sh]))
        return cv2.warpAffine(src, self.cropMatrix(M, sl, st, left, top), (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def sourceReliefOnTarget(self, box, M=None):
        """
        Height (mm) above the ground of the source DEM resampled on the target grid, in the box [top, left, w, h] of
        the target map (see relief); 0 where there is no DEM.
        """
        M = self.placement.M if M is None else M
        top, left, w, h = box
        sl, st, sw, sh = self.sourceWindow(box, M)
        z = relief(self.src_dem.readWindow(st, sl, sh, sw))
        return cv2.warpAffine(z, self.cropMatrix(M, sl, st, left, top), (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def sourceWindow(self, box, M):
        """
        The window (left, top, w, h) of the source map covering the box [top, left, w, h] of the target map.
        """
        top, left, w, h = box
        corners = np.array([[left, top], [left + w, top], [left, top + h], [left + w, top + h]], dtype=np.float64)
        src_corners = applyMatrix(invertMatrix(M), corners)
        sl = int(math.floor(src_corners[:, 0].min())) - 2
        st = int(math.floor(src_corners[:, 1].min())) - 2
        sw = int(math.ceil(src_corners[:, 0].max())) + 2 - sl
        sh = int(math.ceil(src_corners[:, 1].max())) + 2 - st
        return sl, st, sw, sh

    @staticmethod
    def cropMatrix(M, sl, st, left, top):
        """
        M between the source window starting at (sl, st) and the target box starting at (left, top).
        """
        A = np.array(M, dtype=np.float64).copy()
        A[:, 2] = A[:, :2] @ np.array([sl, st], dtype=np.float64) + A[:, 2] - np.array([left, top], dtype=np.float64)
        return A

    def findExisting(self, class_name, mask, box):
        """
        The id of the region of the same class already on the target map that overlaps the colony most, if the
        overlap is at least EXISTING_OVERLAP of the smaller of the two; otherwise None.
        """
        area = int(mask.sum())
        best_id, best_inter = None, 0
        for region in self.existing:
            if region.class_name != class_name or not checkIntersection(box, region.bbox):
                continue
            painted = np.zeros_like(mask)
            paintMask(painted, box, region.mask, region.bbox, 1)
            inter = int(np.count_nonzero(painted & mask))
            if inter >= EXISTING_OVERLAP * min(area, region.area) and inter > best_inter:
                best_id, best_inter = region.id, inter
        return best_id

    def demSignals(self, blob, mask, box):
        """
        Height (above the surrounding plane) of the colony in year 1 (its outline on the source DEM) and in year 2
        (the same outline on the target DEM).
        """
        if not self.use_dem or self.src_dem is None or self.dst_dem is None:
            return {}

        pad = 3 + max(2, int(0.15 * math.sqrt(max(1.0, blob.area))))
        top, left = int(blob.bbox[0]) - pad, int(blob.bbox[1]) - pad
        bh, bw = int(blob.bbox[3]), int(blob.bbox[2])
        z1 = self.src_dem.readWindow(top, left, bh + 2 * pad, bw + 2 * pad)
        m1 = np.zeros(z1.shape, dtype=np.uint8)
        m1[pad:pad + bh, pad:pad + bw] = blob.getMask()
        ring1 = max(2, int(0.15 * math.sqrt(max(1, m1.sum()))))
        h1 = heightAbovePlane(z1, m1, ring1)

        z2 = self.dst_dem.readWindow(int(box[0]), int(box[1]), int(box[3]), int(box[2]))
        ring2 = max(2, int(0.15 * math.sqrt(max(1, mask.sum()))))
        h2 = heightAbovePlane(z2, mask, ring2)

        signals = {"height1_cm": None if h1 is None else round(h1 * 100.0, 1),
                   "height2_cm": None if h2 is None else round(h2 * 100.0, 1)}
        if h1 is not None and h2 is not None and h1 >= DEM_MIN_HEIGHT:
            signals["dem_ratio"] = h2 / h1
        return signals


#### MOVED OR TURNED COLONIES

class Relocation(object):
    """
    A colony found moved or turned on the new map: L (2x3, target pixel -> target pixel) moves its placed outline.
    """

    def __init__(self, L, angle, shift, score, in_place):
        self.L = L
        self.angle = angle          # degrees, clockwise on the screen
        self.shift = shift          # (dx, dy) of the colony centre, target pixels
        self.score = score          # match at the new place (normalised cross-correlation, less the distance cost)
        self.in_place = in_place    # match with the colony left in place

    def signedAngle(self):
        return ((self.angle + 180.0) % 360.0) - 180.0

    def description(self, px_size):
        """
        E.g. "moved 6 cm, turned 90° clockwise" (px_size: mm per target pixel).
        """
        parts = []
        dist_cm = math.hypot(*self.shift) * px_size / 10.0
        if dist_cm >= 0.5:
            parts.append("moved {:.0f} cm".format(dist_cm))
        a = self.signedAngle()
        if abs(a) >= 1.0:
            parts.append("turned {:.0f}° {}".format(abs(a), "clockwise" if a > 0 else "anticlockwise"))
        return ", ".join(parts) if parts else "moved"


def relief(z):
    """
    Height (mm) of the DEM window z (m) above the plane fitted to its valid values, clipped to RELOCATE_RELIEF_MM;
    0 where z is not valid (float32).
    """
    out = np.zeros(z.shape, dtype=np.float32)
    valid = np.isfinite(z)
    sub = valid[::4, ::4]
    if sub.sum() < 3:
        return out
    rows, cols = np.nonzero(sub)
    A = np.column_stack((4 * cols, 4 * rows, np.ones(len(rows))))
    coeffs = np.linalg.lstsq(A, z[::4, ::4][sub], rcond=None)[0]
    rows, cols = np.nonzero(valid)
    base = coeffs[0] * cols + coeffs[1] * rows + coeffs[2]
    out[valid] = np.clip((z[valid] - base) * 1000.0, *RELOCATE_RELIEF_MM)
    return out


def subpixelPeak(res, row, col):
    """
    Position of the maximum of res near (row, col), refined with a parabola along each axis.
    """
    def offset(a, b, c):
        den = a - 2.0 * b + c
        return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))

    r, c = float(row), float(col)
    if 0 < row < res.shape[0] - 1:
        r += offset(res[row - 1, col], res[row, col], res[row + 1, col])
    if 0 < col < res.shape[1] - 1:
        c += offset(res[row, col - 1], res[row, col], res[row, col + 1])
    return r, c


def relocateColony(ctx, blob):
    """
    Look for the colony around its place on the new map, turned by any angle and moved by up to ctx.search_mm. The
    old map around it (a disc holding the colony and what it sits on, resampled on the new map) is matched with the
    new map at about RELOCATE_PIXEL_MM per pixel: normalised cross-correlation of the grey levels and, with a DEM on
    both maps, of the height above the ground (mean of the two), every RELOCATE_COARSE_STEP degrees and then every
    RELOCATE_FINE_STEP degrees around the best angle. It returns a Relocation if the colony matches clearly better
    elsewhere than in place (RELOCATE_MARGIN, RELOCATE_MIN_SCORE), otherwise None. Matches farther than
    RELOCATE_FAR_MM count less (RELOCATE_FAR_MARGIN), and nothing beyond ctx.search_mm is looked at.
    """
    M = ctx.placement.M
    top, left, w, h = [float(v) for v in blob.bbox]
    centre = applyMatrix(M, np.array([[left + 0.5 * (w - 1.0), top + 0.5 * (h - 1.0)]]))[0]
    scale = math.sqrt(abs(float(np.linalg.det(np.asarray(M, dtype=np.float64)[:, :2]))))
    R = int(math.ceil(max(RELOCATE_MIN_RADIUS_MM / ctx.px_dst, RELOCATE_SIZE_FACTOR * max(w, h) * scale)))
    S = int(math.ceil(ctx.search_mm / ctx.px_dst))
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    tbox = [cy - R, cx - R, 2 * R + 1, 2 * R + 1]
    sbox = [cy - R - S, cx - R - S, 2 * (R + S) + 1, 2 * (R + S) + 1]

    src_rgb = ctx.cropSourceOnTarget(tbox)
    tgt_rgb = ctx.cropTarget(sbox)
    tvalid = src_rgb.any(axis=2)
    svalid = tgt_rgb.any(axis=2)
    if tvalid.mean() < 0.5 or svalid.mean() < 0.25:
        return None

    templates = [toGray(src_rgb)]
    searches = [toGray(tgt_rgb)]
    if ctx.src_dem is not None and ctx.dst_dem is not None:
        templates.append(ctx.sourceReliefOnTarget(tbox))
        searches.append(relief(ctx.dst_dem.readWindow(sbox[0], sbox[1], sbox[3], sbox[2])))

    g = min(1.0, ctx.px_dst / RELOCATE_PIXEL_MM)

    def prepare(img, valid):
        img = img.astype(np.float32)
        img[~valid] = float(img[valid].mean())
        if g < 1.0:
            img = cv2.resize(img, None, fx=g, fy=g, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(img, (0, 0), RELOCATE_BLUR)

    templates = [prepare(t, tvalid) for t in templates]
    searches = [prepare(s, svalid) for s in searches]
    th, tw = templates[0].shape
    rows, cols = np.mgrid[0:th, 0:tw]
    tc = ((tw - 1) / 2.0, (th - 1) / 2.0)
    disc = (cols - tc[0]) ** 2 + (rows - tc[1]) ** 2 <= (0.5 * min(tw, th) - 1.0) ** 2
    tval = cv2.resize(tvalid.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0
    base_mask = (disc & tval).astype(np.uint8)

    def scoreMap(angle):
        rot = cv2.getRotationMatrix2D(tc, -angle, 1.0)
        mask = cv2.warpAffine(base_mask, rot, (tw, th), flags=cv2.INTER_NEAREST)
        if mask.sum() < 0.25 * disc.sum():
            return None
        total = None
        for t, s in zip(templates, searches):
            turned = cv2.warpAffine(t, rot, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            res = cv2.matchTemplate(s, turned, cv2.TM_CCOEFF_NORMED, mask=mask)
            res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
            total = res if total is None else total + res
        total /= len(templates)
        total -= penalty
        total[far] = -1.0
        return total

    rh, rw = searches[0].shape[0] - th + 1, searches[0].shape[1] - tw + 1
    rr, cc = np.ogrid[:rh, :rw]
    dist_mm = np.sqrt((rr - (rh - 1) / 2.0) ** 2 + (cc - (rw - 1) / 2.0) ** 2) * ctx.px_dst / g
    far = dist_mm > ctx.search_mm
    penalty = (RELOCATE_FAR_MARGIN / 10.0) * np.maximum(0.0, dist_mm - RELOCATE_FAR_MM).astype(np.float32)
    best = None
    in_place = None

    def tryAngle(angle):
        nonlocal best, in_place
        res = scoreMap(angle)
        if res is None:
            return
        if angle == 0:
            in_place = float(res[dist_mm <= ctx.px_dst / g].max())
        row, col = np.unravel_index(int(np.argmax(res)), res.shape)
        if best is None or res[row, col] > best[0]:
            best = (float(res[row, col]), angle, res, int(row), int(col))

    for angle in range(0, 360, RELOCATE_COARSE_STEP):
        tryAngle(angle)
    if best is None or in_place is None:
        return None
    coarse = best[1]
    for k in range(-RELOCATE_COARSE_STEP + 1, RELOCATE_COARSE_STEP):
        if k != 0 and k % RELOCATE_FINE_STEP == 0:
            tryAngle((coarse + k) % 360)

    score, angle, res, row, col = best
    if score < RELOCATE_MIN_SCORE or score < in_place + RELOCATE_MARGIN:
        return None

    r, c = subpixelPeak(res, row, col)
    dx = (c - (res.shape[1] - 1) / 2.0) / g
    dy = (r - (res.shape[0] - 1) / 2.0) / g
    L = cv2.getRotationMatrix2D((float(cx), float(cy)), -float(angle), 1.0)
    L[:, 2] += (dx, dy)
    return Relocation(L, float(angle), (dx, dy), score, in_place)


class ColonyWindow(object):
    """
    The neighbourhood of a colony on the target map: box [top, left, w, h] of the target map, the target crop, the
    source map resampled on the same pixels, and the colony mask (after the local shift).
    """

    def __init__(self, box, src, tgt, mask):
        self.box = box
        self.left = box[1]
        self.top = box[0]
        self.src = src
        self.tgt = tgt
        self.mask = mask


def colonyWindow(ctx, blob, result, M=None):
    """
    Place the colony on the target map (with M, source pixel -> target pixel; by default the placement of the maps)
    and correct the residual shift around it. It returns a ColonyWindow, or None if the colony is not carried over
    (result.covered / result.existing_id say why).
    """
    M = ctx.placement.M if M is None else M
    contour = applyMatrix(M, blob.contour)
    inner = [applyMatrix(M, c) for c in blob.inner_contours]

    x0, y0 = contour.min(axis=0)
    x1, y1 = contour.max(axis=0)
    w, h = x1 - x0 + 1.0, y1 - y0 + 1.0
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    # analysis box: room for the RITM crop after any accepted shift, plus the ring around the colony
    margin = ctx.max_shift_px + 0.25 * max(w, h) + 8
    ew = max(RITM_MIN_CROP, RITM_EXPANSION * w) / 2 + margin
    eh = max(RITM_MIN_CROP, RITM_EXPANSION * h) / 2 + margin
    left = int(math.floor(cx - ew))
    top = int(math.floor(cy - eh))
    box = [top, left, int(math.ceil(cx + ew)) - left, int(math.ceil(cy + eh)) - top]

    mask = rasterize(contour, inner, left, top, (box[3], box[2]))
    if mask.sum() == 0:
        result.covered = False
        return None

    tgt = ctx.cropTarget(box)
    valid_tgt = tgt.any(axis=2)
    if float(valid_tgt[mask > 0].mean()) < MIN_COVERED:
        result.covered = False
        return None

    existing_id = ctx.findExisting(blob.class_name, mask, box)
    if existing_id is not None:
        result.existing_id = existing_id
        return None

    src = ctx.cropSourceOnTarget(box, M)

    # residual shift around the colony
    g = min(1.0, LOCAL_MAX_SIDE / max(box[2], box[3]))
    gs = cv2.resize(toGray(src), None, fx=g, fy=g, interpolation=cv2.INTER_AREA) if g < 1.0 else toGray(src)
    gt = cv2.resize(toGray(tgt), None, fx=g, fy=g, interpolation=cv2.INTER_AREA) if g < 1.0 else toGray(tgt)
    dx, dy, response = phaseShift(gs, gt, (gs > 0) & (gt > 0))
    dx, dy = dx / g, dy / g
    result.signals["shift_response"] = response
    if response >= LOCAL_MIN_RESPONSE and math.hypot(dx, dy) <= ctx.max_shift_px:
        result.shift = (dx, dy)
        src = translate(src, dx, dy)
        mask = (translate(mask.astype(np.float32), dx, dy) > 0.5).astype(np.uint8)
        if mask.sum() == 0:
            result.covered = False
            return None

    return ColonyWindow(box, src, tgt, mask)


def processColony(ctx, blob):

    result = CarriedColony(blob)

    M = None
    if ctx.search_mm > 0:
        result.relocation = relocateColony(ctx, blob)
        if result.relocation is not None:
            M = composeMatrix(result.relocation.L, ctx.placement.M)
            result.signals["move_mm"] = math.hypot(*result.relocation.shift) * ctx.px_dst
            result.signals["turn_deg"] = result.relocation.signedAngle()

    window = colonyWindow(ctx, blob, result, M)
    if window is None:
        return result

    src, tgt, mask, box = window.src, window.tgt, window.mask, window.box
    left, top = window.left, window.top

    new_mask = mask
    refit_ok = False
    if ctx.refit:
        refitted = ctx.refitWithRitm(tgt, mask)
        if refitted is not None:
            accepted, iou, ratio = acceptRefit(mask, refitted)
            choice = getattr(ctx.ritm, "choice", None)
            agreement = getattr(ctx.ritm, "agreement", None)
            result.signals["refit_iou"] = iou
            result.signals["area_ratio"] = ratio
            result.signals["refit_choice"] = choice
            if accepted:
                new_mask = refitted
                # the old outline kept because the re-fits disagree with each other is not a fitted outline
                refit_ok = choice != "old outline" or (agreement is not None and agreement >= REFIT_AGREEMENT)
    result.refit = refit_ok

    result.signals.update(presenceSignals(src, tgt, mask, ctx.px_dst))
    result.signals.update(ctx.demSignals(blob, mask, box))
    result.score, result.status, result.reasons = decideStatus(result.signals, ctx.px_dst, ctx.refit, refit_ok)
    if result.relocation is not None:
        result.reasons.insert(0, result.relocation.description(ctx.px_dst))

    new_blob = blobFromMask(new_mask, left, top)
    if new_blob is None:
        new_blob = blobFromMask(mask, left, top)
    if new_blob is None:
        result.covered = False
        return result

    new_blob.class_name = blob.class_name
    new_blob.instance_name = blob.instance_name
    new_blob.data = {k: v for k, v in blob.data.items() if k not in DEM_METRIC_NAMES and k not in CARRYOVER_ATTRIBUTES}
    new_blob.data[ATTR_STATUS] = result.status
    new_blob.data[ATTR_SCORE] = None if result.score is None else round(result.score, 3)
    new_blob.data[ATTR_FROM] = int(blob.id)
    result.blob = new_blob

    return result


def carryOver(blobs, ctx, progress=None, stop=None):
    """
    Process the given regions of the source map. progress, if given, is called with the percentage done; stop, if
    given, is called before each region and stops the processing when it returns True. It returns the list of
    CarriedColony, one per processed region.
    """
    results = []
    n = len(blobs)
    for i, blob in enumerate(blobs):
        if stop is not None and stop():
            break
        results.append(processColony(ctx, blob))
        if progress is not None:
            progress(100.0 * (i + 1) / n)
    return results


def registerAttributes(project):
    """
    Add the carry-over attributes to the region attributes of the project. Returns the attributes added.
    """
    added = []
    specs = [(ATTR_STATUS, 'keyword', STATUS_KEYWORDS), (ATTR_SCORE, 'decimal number', []),
             (ATTR_FROM, 'integer number', [])]
    for name, type, keywords in specs:
        if not project.region_attributes.has(name):
            project.region_attributes.data.append({'name': name, 'type': type, 'min': None, 'max': None,
                                                   'keywords': list(keywords)})
            added.append(name)
    return added

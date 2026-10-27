# TagLab
# A semi-automatic segmentation tool
#
# Alignment (co-registration) of two maps: transform estimation from control points, output canvas,
# raster warping, transformation of the annotations and automatic search of control points.
#
# Conventions:
#  - points are (x, y), with y pointing down (image convention);
#  - the transformation maps the image to align ("moving") onto the reference image;
#  - transformations are 2x3 matrices [[a, b, tx], [c, d, ty]] acting on column vectors (x, y, 1).

import math
import os
import numpy as np
import cv2
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import Affine

from source.Point import Point
from source.DemMetrics import DEM_METRIC_NAMES, PLOT_METRICS_KEY

DEM_DEFAULT_NODATA = -32767.0


# ---------------------------------------------------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------------------------------------------------

def paramsToMatrix(angle_deg, scale, tx, ty):
    """
    Similarity transformation (rotation, uniform scale, translation) as a 2x3 matrix.
    """

    a = math.radians(angle_deg)
    c = scale * math.cos(a)
    s = scale * math.sin(a)
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float64)


def matrixToParams(A):
    """
    Return (angle in degrees, scale, tx, ty) of a similarity transformation.
    """

    scale = math.hypot(A[0, 0], A[1, 0])
    angle = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    return angle, scale, A[0, 2], A[1, 2]


def applyMatrix(A, points):
    """
    Apply the 2x3 matrix A to the points (N x 2).
    """

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return points @ A[:, :2].T + A[:, 2]


def composeMatrix(A, B):
    """
    Return the matrix of the transformation A(B(p)).
    """

    A3 = np.vstack([A, [0.0, 0.0, 1.0]])
    B3 = np.vstack([B, [0.0, 0.0, 1.0]])
    return (A3 @ B3)[:2, :]


def adjustMatrix(A, d_angle_deg, d_tx, d_ty, pivot):
    """
    Apply a manual correction to the transformation A: a rotation of d_angle_deg around the pivot (in the
    reference space) followed by a translation (d_tx, d_ty).
    """

    a = math.radians(d_angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pivot = np.asarray(pivot, dtype=np.float64)
    C = np.hstack([R, (pivot - R @ pivot + np.array([d_tx, d_ty]))[:, None]])
    return composeMatrix(C, A)


def fitTransform(ref_points, mov_points, weights=None, with_scale=False):
    """
    Weighted least squares (Umeyama) estimation of the rigid (or similarity, if with_scale is True) transformation
    that maps mov_points onto ref_points. The points are given in millimeters (isotropic coordinates).

    It returns a dictionary with:
      'matrix': 2x3 matrix (moving mm -> reference mm), 'angle', 'scale', 'tx', 'ty',
      'residuals': N x 2 (predicted - reference, mm), 'errors': N (mm), 'rmse' (mm).
    """

    ref = np.asarray(ref_points, dtype=np.float64).reshape(-1, 2)
    mov = np.asarray(mov_points, dtype=np.float64).reshape(-1, 2)
    n = len(ref)
    if n < 2 or len(mov) != n:
        raise ValueError("At least two pairs of points are required.")

    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    W = w.sum()

    mu_ref = (w[:, None] * ref).sum(axis=0) / W
    mu_mov = (w[:, None] * mov).sum(axis=0) / W
    r0 = ref - mu_ref
    m0 = mov - mu_mov

    # cross-covariance and its SVD
    C = (w[:, None] * r0).T @ m0 / W
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_mov = (w * (m0 ** 2).sum(axis=1)).sum() / W
        scale = float(np.trace(np.diag(D) @ S) / var_mov) if var_mov > 0.0 else 1.0
    else:
        scale = 1.0

    t = mu_ref - scale * R @ mu_mov
    A = np.hstack([scale * R, t[:, None]])

    residuals = applyMatrix(A, mov) - ref
    errors = np.linalg.norm(residuals, axis=1)
    angle, scale, tx, ty = matrixToParams(A)

    return {'matrix': A, 'angle': angle, 'scale': scale, 'tx': tx, 'ty': ty,
            'residuals': residuals, 'errors': errors, 'rmse': float(math.sqrt(np.mean(errors ** 2)))}


def pixelMatrix(A_mm, px_ref, px_mov, offset=(0.0, 0.0)):
    """
    Convert a transformation in millimeters (moving mm -> reference mm) into a transformation in pixels
    (moving pixels -> pixels of the output canvas, that has the pixel size of the reference and the given
    (left, top) offset with respect to the reference image).
    """

    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = A_mm[:, :2] * (px_mov / px_ref)
    M[:, 2] = A_mm[:, 2] / px_ref + np.asarray(offset, dtype=np.float64)
    return M


def unionCanvas(M, mov_w, mov_h, ref_w, ref_h):
    """
    Compute the canvas containing both the reference image (placed at the origin) and the image to align
    transformed by M (moving pixels -> reference pixels).
    It returns the borders to add to the reference image (left, right, top, bottom) and the canvas size.
    """

    corners = applyMatrix(M, [[0, 0], [mov_w, 0], [mov_w, mov_h], [0, mov_h]])
    xmin, ymin = corners.min(axis=0)
    xmax, ymax = corners.max(axis=0)

    left = int(math.ceil(max(0.0, -xmin) - 1e-6))
    top = int(math.ceil(max(0.0, -ymin) - 1e-6))
    right = int(math.ceil(max(0.0, xmax - ref_w) - 1e-6))
    bottom = int(math.ceil(max(0.0, ymax - ref_h) - 1e-6))

    return {'left': left, 'right': right, 'top': top, 'bottom': bottom,
            'width': ref_w + left + right, 'height': ref_h + top + bottom}


# ---------------------------------------------------------------------------------------------------------------------
# Raster warping
# ---------------------------------------------------------------------------------------------------------------------

def warpArray(src, M, out_w, out_h, border_value=0.0, flags=cv2.INTER_LINEAR):
    """
    Warp the array src (H x W or H x W x C, C <= 4) with the 2x3 matrix M (src pixels -> output pixels),
    bilinear interpolation by default. Only the part of src needed by the output is processed (this also
    avoids the OpenCV limit of 32767 pixels on the source size). NaN values (float arrays) are propagated.
    """

    window = _sourceWindow(M, out_w, out_h, src.shape)
    if window is None:
        return np.full((out_h, out_w) + src.shape[2:], border_value, dtype=src.dtype)

    x0, y0, x1, y1, Ms = window
    out = cv2.warpAffine(src[y0:y1, x0:x1], Ms, (out_w, out_h), flags=flags, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=border_value)
    if src.ndim == 3 and out.ndim == 2:
        out = out[:, :, None]
    return out


def _sourceWindow(M, out_w, out_h, src_shape):
    """
    Part of the source needed to compute an output of out_w x out_h pixels with M (src -> output), and the matrix
    to use with that part. Returns (x0, y0, x1, y1, Ms) or None if the output does not see the source.
    """

    Minv = cv2.invertAffineTransform(M)
    corners = applyMatrix(Minv, [[0, 0], [out_w, 0], [0, out_h], [out_w, out_h]])
    x0 = max(int(math.floor(corners[:, 0].min())) - 2, 0)
    y0 = max(int(math.floor(corners[:, 1].min())) - 2, 0)
    x1 = min(int(math.ceil(corners[:, 0].max())) + 2, src_shape[1])
    y1 = min(int(math.ceil(corners[:, 1].max())) + 2, src_shape[0])
    if x1 <= x0 or y1 <= y0:
        return None

    Ms = M.copy()
    Ms[:, 2] = M[:, 2] + M[:, :2] @ np.array([x0, y0], dtype=np.float64)
    return x0, y0, x1, y1, Ms


def warpDem(values, valid, M, out_w, out_h, nodata):
    """
    Warp a DEM (bilinear) without ever mixing valid and invalid values: values has 0 where the DEM is not valid,
    valid is a uint8 mask (1 = valid). An output pixel is valid only if all the source pixels contributing to it
    (with a non-zero weight) are valid; exact positions (e.g. rotations by 90 degrees) are copied exactly.
    """

    window = _sourceWindow(M, out_w, out_h, values.shape)
    if window is None:
        return np.full((out_h, out_w), nodata, dtype=np.float32)

    x0, y0, x1, y1, Ms = window
    v = cv2.warpAffine(values[y0:y1, x0:x1], Ms, (out_w, out_h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    m = cv2.warpAffine(valid[y0:y1, x0:x1].astype(np.float32), Ms, (out_w, out_h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    v[m < 0.999] = nodata
    return v


def warpRaster(src_filename, dst_filename, M, out_w, out_h, is_dem, crs=None, transform=None, tile=2048, progress=None):
    """
    Write dst_filename as the src_filename image warped by M (src pixels -> output pixels) on a canvas of
    out_w x out_h pixels. The output is a tiled and compressed GeoTIFF, processed tile by tile.
    DEM: the nodata values are never interpolated, the pixels outside the source are nodata.
    """

    # a small GDAL block cache: the source is read once and the output is written tile by tile
    with rio.Env(GDAL_CACHEMAX=256):
        return _warpRaster(src_filename, dst_filename, M, out_w, out_h, is_dem, crs, transform, tile, progress)


def _warpRaster(src_filename, dst_filename, M, out_w, out_h, is_dem, crs, transform, tile, progress):

    with rio.open(src_filename) as src:
        data = src.read()
        src_nodata = src.nodata

    count = data.shape[0]

    profile = {'driver': 'GTiff', 'width': out_w, 'height': out_h, 'count': count, 'tiled': True,
               'blockxsize': 512, 'blockysize': 512, 'compress': 'deflate', 'BIGTIFF': 'IF_SAFER'}

    if is_dem:
        out_nodata = float(src_nodata) if src_nodata is not None and np.isfinite(src_nodata) else DEM_DEFAULT_NODATA
        profile['dtype'] = 'float32'
        profile['nodata'] = out_nodata
        profile['predictor'] = 3
        data = data.astype(np.float32, copy=False)
        valid = np.isfinite(data)
        if src_nodata is not None:
            valid &= data != src_nodata
        data[~valid] = 0.0
        valid = valid.astype(np.uint8)
        border = 0.0
    else:
        profile['dtype'] = data.dtype.name
        profile['predictor'] = 2
        border = 0

    if crs is not None and transform is not None:
        profile['crs'] = crs
        profile['transform'] = transform

    tiles = [(r0, c0) for r0 in range(0, out_h, tile) for c0 in range(0, out_w, tile)]

    with rio.open(dst_filename, 'w', **profile) as dst:
        for k, (r0, c0) in enumerate(tiles):
            th = min(tile, out_h - r0)
            tw = min(tile, out_w - c0)
            Mt = M.copy()
            Mt[:, 2] -= np.array([c0, r0], dtype=np.float64)
            out = np.empty((count, th, tw), dtype=profile['dtype'])
            for b in range(count):
                if is_dem:
                    out[b] = warpDem(data[b], valid[b], Mt, tw, th, out_nodata)
                else:
                    out[b] = warpArray(data[b], Mt, tw, th, border_value=border)
            dst.write(out, window=rio.windows.Window(c0, r0, tw, th))
            if progress is not None:
                progress(100.0 * (k + 1) / len(tiles))

    return out_w, out_h


def georeferenceOfCanvas(reference_filename, left, top):
    """
    CRS and transform of the output canvas, given the georeferenced reference image and the borders added
    on its left and top side. Returns (None, None) if the reference is not georeferenced.
    """

    if reference_filename is None or reference_filename == "":
        return None, None

    with rio.open(reference_filename) as ref:
        if ref.crs is None:
            return None, None
        return ref.crs, ref.transform * Affine.translation(-left, -top)


def transformedGeoreference(georef_filename, M):
    """
    CRS and transform of an image obtained transforming the georeferenced image georef_filename with M
    (pixels -> pixels, pixel-center convention), so that each pixel keeps its geographic position.
    Returns (None, None) if the image is not georeferenced.
    """

    if georef_filename is None or georef_filename == "":
        return None, None

    with rio.open(georef_filename) as src:
        if src.crs is None:
            return None, None
        crs = src.crs
        transform = src.transform

    # rasterio transforms use the pixel-corner convention
    C = composeMatrix(composeMatrix(np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]), M),
                      np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5]]))
    A = Affine(C[0, 0], C[0, 1], C[0, 2], C[1, 0], C[1, 1], C[1, 2])
    return crs, transform * ~A


def rotationMatrix(width, height, angle_deg):
    """
    Matrix (pixels -> pixels) and size of the output to rotate an image of width x height pixels by angle_deg
    (positive = clockwise on screen). Multiples of 90 degrees are exact index mappings (no resampling); other
    angles rotate around the center of the image and the canvas is enlarged to contain the whole image.
    Returns (M, out_w, out_h, exact).
    """

    a = angle_deg % 360.0
    W, H = width, height
    for (k, M, ow, oh) in [(0, [[1, 0, 0], [0, 1, 0]], W, H),
                           (90, [[0, -1, H - 1], [1, 0, 0]], H, W),
                           (180, [[-1, 0, W - 1], [0, -1, H - 1]], W, H),
                           (270, [[0, 1, 0], [-1, 0, W - 1]], H, W),
                           (360, [[1, 0, 0], [0, 1, 0]], W, H)]:
        if abs(a - k) < 1e-9:
            return np.array(M, dtype=np.float64), ow, oh, True

    R = paramsToMatrix(angle_deg, 1.0, 0.0, 0.0)
    # extent of the image (pixel centers from 0 to W-1, so the image covers -0.5 .. W-0.5)
    corners = applyMatrix(R, [[-0.5, -0.5], [W - 0.5, -0.5], [W - 0.5, H - 0.5], [-0.5, H - 0.5]])
    cmin = corners.min(axis=0)
    cmax = corners.max(axis=0)
    out_w = int(math.ceil(cmax[0] - cmin[0] - 1e-6))
    out_h = int(math.ceil(cmax[1] - cmin[1] - 1e-6))
    R[:, 2] = -0.5 - cmin
    return R, out_w, out_h, False


def uniqueSuffix(project, image, base, taglab_dir):
    """
    A suffix (base, base2, base3, ..) such that neither the new image id nor the new files already exist.
    """

    ids = set(im.id for im in project.images)
    k = 1
    while True:
        suffix = base if k == 1 else base + str(k)
        exists = (image.id + suffix) in ids
        for ch in image.channels:
            if os.path.exists(os.path.join(taglab_dir, os.path.splitext(ch.filename)[0] + suffix + ".tif")):
                exists = True
        if not exists:
            return suffix
        k += 1


def createTransformedImage(image, suffix, M, width, height, crs, transform, px_size, taglab_dir, applied_scale=1.0,
                           progress=None):
    """
    Create a new Image transforming all the channels and the annotations of image with M (pixels -> pixels) on a
    canvas of width x height pixels. The new files (GeoTIFF) are written next to the original ones, with the suffix
    added to their names. Annotation grids and shape layers are not copied.
    progress, if given, is called as progress(percentage, message).
    """

    from source.Image import Image

    # the plot values (DEM metrics) refer to an area of the original map
    metadata = {key: value for key, value in (image.metadata or {}).items() if key != PLOT_METRICS_KEY}

    new_image = Image(rect=[0.0, 0.0, 0.0, 0.0], map_px_to_mm_factor=str(px_size), width=None, height=None,
                      channels=[], id=image.id + suffix, name=image.name + suffix,
                      acquisition_date=image.acquisition_date, georef_filename="", workspace=[],
                      metadata=metadata, annotations={}, layers=[], grid={},
                      export_dataset_area=[], sampling_areas=[])

    n = len(image.channels)
    for (i, ch) in enumerate(image.channels):
        src = os.path.join(taglab_dir, ch.filename)
        dst_rel = os.path.splitext(ch.filename)[0] + suffix + ".tif"
        dst = os.path.join(taglab_dir, dst_rel)
        msg = "Writing " + os.path.basename(dst_rel) + ".."
        cb = None
        if progress is not None:
            cb = lambda p, i=i, msg=msg: progress(100.0 * (i + p / 100.0) / n, msg)
        warpRaster(src, dst, M, width, height, is_dem=(ch.type == "DEM"), crs=crs, transform=transform, progress=cb)
        new_image.addChannel(dst_rel, ch.type)

    transformAnnotations(image.annotations, new_image.annotations, M, applied_scale)
    return new_image


# ---------------------------------------------------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------------------------------------------------

def transformAnnotations(src_annotations, dst_annotations, M, applied_scale=1.0):
    """
    Copy the regions and the points of src_annotations into dst_annotations transformed by M (pixels -> pixels).
    Areas and perimeters (in pixels) are updated according to the scale factor of M. The DEM metrics are kept
    only if the physical size of the map is unchanged (applied_scale == 1).
    """

    k = math.hypot(M[0, 0], M[1, 0])
    R = M[:, :2]
    T = M[:, 2]

    blobs = src_annotations.computeBlobsAffineTransform(R, T)
    for blob in blobs:
        blob.area = blob.area * k * k
        blob.perimeter = blob.perimeter * k
        if abs(applied_scale - 1.0) < 1e-9:
            blob.surface_area = blob.surface_area * k * k
        else:
            blob.surface_area = 0.0
            for name in DEM_METRIC_NAMES:
                blob.data.pop(name, None)
        dst_annotations.addBlob(blob)

    for point in src_annotations.annpoints:
        x, y = applyMatrix(M, [[point.coordx, point.coordy]])[0]
        new_point = Point(float(x), float(y), point.class_name, point.id)
        new_point.note = point.note
        new_point.data = point.data.copy()
        dst_annotations.addPoint(new_point)

    return len(blobs), len(src_annotations.annpoints)


# ---------------------------------------------------------------------------------------------------------------------
# Automatic control points
# ---------------------------------------------------------------------------------------------------------------------

def _readGray(filename, out_shape=None):
    """
    Read an RGB(A) image as an 8-bit grayscale array, optionally resampled to out_shape (rows, cols).
    """

    with rio.open(filename) as src:
        nb = min(src.count, 3)
        if out_shape is None:
            data = src.read(list(range(1, nb + 1)))
        else:
            data = src.read(list(range(1, nb + 1)), out_shape=(nb, out_shape[0], out_shape[1]),
                            resampling=Resampling.average)

    if nb == 1:
        gray = data[0].astype(np.float32)
    else:
        gray = 0.299 * data[0].astype(np.float32) + 0.587 * data[1] + 0.114 * data[2]

    return np.clip(gray, 0, 255).astype(np.uint8)


def autoMarkers(ref_filename, mov_filename, px_ref, px_mov, max_markers=30, max_side=4000, patch=256,
                ref_gray=None, mov_gray=None, progress=None):
    """
    Find automatically pairs of corresponding points between the reference image and the image to align.

    1. SIFT features are extracted on reduced versions of the two maps (same mm/pixel, longest side <= max_side),
       matched (ratio test) and filtered with RANSAC (similarity transformation).
    2. A set of well distributed matches is refined at full resolution: a patch of the image to align is warped
       with the coarse transformation and the residual shift is estimated with phase correlation.

    It returns (pairs, info): pairs is a list of ((xr, yr), (xm, ym)) in millimeters, info is a dictionary with the
    coarse estimation (number of matches/inliers, angle, scale).
    """

    def report(perc, msg):
        if progress is not None:
            progress(perc, msg)

    with rio.open(ref_filename) as f:
        ref_w, ref_h = f.width, f.height
    with rio.open(mov_filename) as f:
        mov_w, mov_h = f.width, f.height

    # common resolution (mm/pixel) of the reduced maps
    mm = max(max(ref_w, ref_h) * px_ref, max(mov_w, mov_h) * px_mov) / float(max_side)
    mm = max(mm, px_ref, px_mov)
    fr = px_ref / mm      # full-res reference pixels -> reduced pixels
    fm = px_mov / mm
    shape_r = (max(1, int(round(ref_h * fr))), max(1, int(round(ref_w * fr))))
    shape_m = (max(1, int(round(mov_h * fm))), max(1, int(round(mov_w * fm))))

    report(5, "Reading the maps..")
    small_r = _readGray(ref_filename, shape_r)
    small_m = _readGray(mov_filename, shape_m)
    # actual factors (rounding of the reduced size)
    fr_x, fr_y = shape_r[1] / ref_w, shape_r[0] / ref_h
    fm_x, fm_y = shape_m[1] / mov_w, shape_m[0] / mov_h

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    small_r = clahe.apply(small_r)
    small_m = clahe.apply(small_m)

    report(15, "Extracting features..")
    sift = cv2.SIFT_create(nfeatures=20000)
    kp_r, des_r = sift.detectAndCompute(small_r, None)
    kp_m, des_m = sift.detectAndCompute(small_m, None)
    if des_r is None or des_m is None or len(kp_r) < 10 or len(kp_m) < 10:
        raise Exception("Not enough image features were found.")

    report(40, "Matching features..")
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    knn = matcher.knnMatch(des_m, des_r, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 8:
        raise Exception("Not enough matches between the two maps ({:d}).".format(len(good)))

    pts_m = np.float32([kp_m[m.queryIdx].pt for m in good])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good])

    report(50, "Estimating the transformation..")
    A_small, inliers = cv2.estimateAffinePartial2D(pts_m, pts_r, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                                   maxIters=10000, confidence=0.999)
    if A_small is None or inliers is None or int(inliers.sum()) < 8:
        raise Exception("The two maps could not be matched reliably.")

    inl = inliers.ravel().astype(bool)
    pts_m = pts_m[inl]
    pts_r = pts_r[inl]
    dist = np.array([m.distance for m, ok in zip(good, inl) if ok])

    # coarse transformation at full resolution (moving px -> reference px)
    S_m = np.array([[fm_x, 0, 0], [0, fm_y, 0]], dtype=np.float64)              # moving full -> moving small
    S_r_inv = np.array([[1.0 / fr_x, 0, 0], [0, 1.0 / fr_y, 0]], dtype=np.float64)  # ref small -> ref full
    A_full = composeMatrix(S_r_inv, composeMatrix(A_small, S_m))
    B_full = cv2.invertAffineTransform(A_full)                                  # ref px -> moving px

    # well distributed subset: best match of each cell of a grid on the reference
    ncells = max(2, int(math.ceil(math.sqrt(max_markers * 1.5))))
    cell_w = shape_r[1] / ncells
    cell_h = shape_r[0] / ncells
    best = {}
    for i in range(len(pts_r)):
        key = (int(pts_r[i, 0] // cell_w), int(pts_r[i, 1] // cell_h))
        if key not in best or dist[i] < dist[best[key]]:
            best[key] = i
    candidates = sorted(best.values(), key=lambda i: dist[i])

    report(60, "Refining the control points..")
    if ref_gray is None:
        ref_gray = _readGray(ref_filename)
    if mov_gray is None:
        mov_gray = _readGray(mov_filename)

    half = patch // 2
    window = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
    pairs = []
    for count, i in enumerate(candidates):
        if len(pairs) >= max_markers:
            break
        xr = pts_r[i, 0] / fr_x
        yr = pts_r[i, 1] / fr_y
        x0 = int(round(xr)) - half
        y0 = int(round(yr)) - half
        if x0 < 0 or y0 < 0 or x0 + patch > ref_gray.shape[1] or y0 + patch > ref_gray.shape[0]:
            continue

        ref_patch = ref_gray[y0:y0 + patch, x0:x0 + patch].astype(np.float32)
        if ref_patch.std() < 5.0:
            continue

        # moving image resampled on the reference patch grid (patch pixel -> moving pixel)
        T = np.array([[1, 0, x0], [0, 1, y0]], dtype=np.float64)
        P = composeMatrix(B_full, T)
        mov_patch = cv2.warpAffine(mov_gray, P, (patch, patch), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        if (mov_patch == 0).mean() > 0.05:
            continue

        (dx, dy), response = cv2.phaseCorrelate(ref_patch, mov_patch, window)
        if response < 0.08 or abs(dx) > half / 4 or abs(dy) > half / 4:
            continue

        # the content at (u, v) of the reference patch is at (u + dx, v + dy) in the warped moving patch
        u = xr - x0
        v = yr - y0
        xm, ym = applyMatrix(P, [[u + dx, v + dy]])[0]
        pairs.append(((xr * px_ref, yr * px_ref), (xm * px_mov, ym * px_mov)))

        report(60 + 40.0 * (count + 1) / len(candidates), "Refining the control points..")

    angle, scale, _, _ = matrixToParams(A_full)
    # physical scale (mm -> mm) of the coarse transformation
    info = {'matches': len(good), 'inliers': int(inl.sum()), 'angle': angle, 'scale': scale * px_ref / px_mov,
            'markers': len(pairs)}

    return pairs, info

# TagLab
# A semi-automatic segmentation tool
#
# Alignment (co-registration) of two maps: transform estimation from control points, output canvas,
# raster warping, transformation of the annotations and automatic search of control points.
#
# Conventions:
#  - points are (x, y), with y pointing down (image convention);
#  - the transformation maps the image to align ("moving") onto the reference image;
#  - transformations are 2x3 matrices [[a, b, tx], [c, d, ty]] acting on column vectors (x, y, 1);
#  - positions in millimeters (the markers) are measured from the top-left corner of the map: the center of the pixel
#    (i, j) is at ((i + 0.5) * px, (j + 0.5) * px) mm; matrices in pixels use the pixel centers (as OpenCV does), the
#    center of the pixel (i, j) is at (i, j).

import math
import os
import numpy as np
import cv2
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import Affine

from source.Point import Point
from source.DemMetrics import DEM_METRIC_NAMES, PLOT_METRICS_KEY, DemReader, hornGradient

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
    (left, top) offset with respect to the reference image). The millimeters are measured from the corner of the
    maps, the pixels from the pixel centers (see the conventions at the top).
    """

    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = A_mm[:, :2] * (px_mov / px_ref)
    M[:, 2] = A_mm[:, 2] / px_ref + M[:, :2] @ np.array([0.5, 0.5]) - 0.5 + np.asarray(offset, dtype=np.float64)
    return M


def mmToPixels(points_mm, px):
    """
    Positions in millimeters (from the corner of the map) -> pixel coordinates (pixel centers at integers).
    """
    return np.asarray(points_mm, dtype=np.float64).reshape(-1, 2) / px - 0.5


def pixelsToMm(points, px):
    """
    Pixel coordinates (pixel centers at integers) -> positions in millimeters (from the corner of the map).
    """
    return (np.asarray(points, dtype=np.float64).reshape(-1, 2) + 0.5) * px


def transformMarkerPositions(points_mm, M, px_old, px_new):
    """
    Positions in millimeters on a map -> the same positions on its transformed copy (M: pixels of the map ->
    pixels of the copy, px_old / px_new: pixel sizes of the map and of the copy).
    """
    return pixelsToMm(applyMatrix(M, mmToPixels(points_mm, px_old)), px_new)


def unionCanvas(M, mov_w, mov_h, ref_w, ref_h):
    """
    Compute the canvas containing both the reference image (placed at the origin) and the image to align
    transformed by M (moving pixels -> reference pixels).
    It returns the borders to add to the reference image (left, right, top, bottom) and the canvas size.
    """

    # outer corners of the moving image (pixel centers at integers), back to edge coordinates of the reference
    corners = applyMatrix(M, [[-0.5, -0.5], [mov_w - 0.5, -0.5], [mov_w - 0.5, mov_h - 0.5], [-0.5, mov_h - 0.5]])
    corners = corners + 0.5
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


def warpRaster(src_filename, dst_filename, M, out_w, out_h, is_dem, crs=None, transform=None, tile=2048, progress=None,
               height_plane=None):
    """
    Write dst_filename as the src_filename image warped by M (src pixels -> output pixels) on a canvas of
    out_w x out_h pixels. The output is a tiled and compressed GeoTIFF, processed tile by tile.
    DEM: the nodata values are never interpolated, the pixels outside the source are nodata. height_plane, if given,
    is (a, b, c): a x + b y + c is subtracted from the heights (x, y: output pixels).
    """

    # a small GDAL block cache: the source is read once and the output is written tile by tile
    with rio.Env(GDAL_CACHEMAX=256):
        return _warpRaster(src_filename, dst_filename, M, out_w, out_h, is_dem, crs, transform, tile, progress,
                           height_plane)


def _warpRaster(src_filename, dst_filename, M, out_w, out_h, is_dem, crs, transform, tile, progress, height_plane):

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
                    if height_plane is not None:
                        pa, pb, pc = height_plane
                        plane = (pa * np.arange(c0, c0 + tw, dtype=np.float64)[None, :] +
                                 pb * np.arange(r0, r0 + th, dtype=np.float64)[:, None] + pc)
                        ok = out[b] != out_nodata
                        out[b][ok] = (out[b][ok] - plane[ok]).astype(np.float32)
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


def notCarriedOver(image):
    """
    What the map has that its transformed copy (createTransformedImage) does not get, as a list of descriptions.
    """
    items = []
    if image.grid is not None:
        items.append("the annotation grid")
    if len(image.layers) > 0:
        items.append("the shape layers")
    if len(image.sampling_areas) > 0:
        items.append("the sampling areas")
    if image.export_dataset_area:
        items.append("the dataset export area")
    return items


def createTransformedImage(image, suffix, M, width, height, crs, transform, px_size, taglab_dir, applied_scale=1.0,
                           progress=None, height_plane=None):
    """
    Create a new Image transforming all the channels and the annotations of image with M (pixels -> pixels) on a
    canvas of width x height pixels. The new files (GeoTIFF) are written next to the original ones, with the suffix
    added to their names. Annotation grids and shape layers are not copied.
    height_plane, if given, is (a, b, c) in canvas pixels: it is subtracted from the heights of the DEM (the regions
    then lose their DEM metrics and surface area, which must be computed again).
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
        is_dem = ch.type == "DEM"
        warpRaster(src, dst, M, width, height, is_dem=is_dem, crs=crs, transform=transform, progress=cb,
                   height_plane=height_plane if is_dem else None)
        new_image.addChannel(dst_rel, ch.type)

    transformAnnotations(image.annotations, new_image.annotations, M, applied_scale,
                         heights_changed=height_plane is not None and image.hasDEM())
    return new_image


# ---------------------------------------------------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------------------------------------------------

def transformAnnotations(src_annotations, dst_annotations, M, applied_scale=1.0, heights_changed=False):
    """
    Copy the regions and the points of src_annotations into dst_annotations transformed by M (pixels -> pixels).
    Areas and perimeters (in pixels) are updated according to the scale factor of M. The DEM metrics and the
    surface areas are kept only if the physical size of the map is unchanged (applied_scale == 1) and the heights of
    its DEM have not been changed.
    """

    k = math.hypot(M[0, 0], M[1, 0])
    R = M[:, :2]
    T = M[:, 2]

    blobs = src_annotations.computeBlobsAffineTransform(R, T)
    for blob in blobs:
        blob.area = blob.area * k * k
        blob.perimeter = blob.perimeter * k
        if abs(applied_scale - 1.0) < 1e-9 and not heights_changed:
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

def _toGray(data):
    """
    Bands (count, h, w) of an RGB(A) image -> 8-bit grey levels.
    """
    if data.shape[0] == 1:
        gray = data[0].astype(np.float32)
    else:
        gray = 0.299 * data[0].astype(np.float32) + 0.587 * data[1] + 0.114 * data[2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def _readGray(filename, out_shape=None, window=None):
    """
    Read an RGB(A) image as an 8-bit grayscale array, optionally resampled to out_shape (rows, cols), or only a
    window (rasterio Window) of it.
    """

    with rio.open(filename) as src:
        nb = min(src.count, 3)
        if out_shape is None:
            data = src.read(list(range(1, nb + 1)), window=window)
        else:
            data = src.read(list(range(1, nb + 1)), out_shape=(nb, out_shape[0], out_shape[1]),
                            resampling=Resampling.average)

    return _toGray(data)


def _pasteWindow(top, left, height, width, map_w, map_h, read, fill, dtype):
    """
    A height x width window of a map at (top, left): read(r0, c0, r1, c1) gives the part inside the map, the rest is
    fill.
    """
    out = np.full((height, width), fill, dtype=dtype)
    r0, c0 = max(top, 0), max(left, 0)
    r1, c1 = min(top + height, map_h), min(left + width, map_w)
    if r1 > r0 and c1 > c0:
        out[r0 - top:r1 - top, c0 - left:c1 - left] = read(r0, c0, r1, c1)
    return out


class ColourSource(object):
    """
    The grey levels of a map (0 = no data) for autoMarkers(): a reduced version of the whole map, or a window at full
    resolution. They come from the image already loaded (array: BGRA view of the QImage) or, without it, from the
    file (only the part needed is read).
    """

    label = "colour"

    def __init__(self, filename, array=None):
        self.filename = filename
        self.array = array
        if array is not None:
            self.height, self.width = array.shape[:2]
        else:
            with rio.open(filename) as f:
                self.width, self.height = f.width, f.height

    def reduced(self, shape):
        if self.array is None:
            return _readGray(self.filename, shape)
        small = cv2.resize(self.array, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGRA2GRAY if small.shape[2] == 4 else cv2.COLOR_BGR2GRAY)

    def window(self, top, left, height, width):
        if self.array is None:
            read = lambda r0, c0, r1, c1: _readGray(self.filename, window=rio.windows.Window(c0, r0, c1 - c0, r1 - r0))
        else:
            code = cv2.COLOR_BGRA2GRAY if self.array.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            read = lambda r0, c0, r1, c1: cv2.cvtColor(np.ascontiguousarray(self.array[r0:r1, c0:c1]), code)
        return _pasteWindow(top, left, height, width, self.width, self.height, read, 0, np.uint8)

    def close(self):
        self.array = None


class DemReliefSource(object):
    """
    The local relief of a DEM (height above the mean surface within about RELIEF_MM) as grey levels (0 = no data),
    for autoMarkers(): it shows the shapes of the colonies whatever their colours and whatever the direction of the
    map (a shading would change with the direction of the light). Windows are read from the file.
    """

    label = "DEM"

    def __init__(self, filename, px_mm, relief_mm=None):
        self.reader = DemReader(filename, px_mm)
        self.width, self.height = self.reader.dataset.width, self.reader.dataset.height
        self.px_mm = px_mm
        self.relief_mm = RELIEF_MM if relief_mm is None else relief_mm

    def reduced(self, shape):
        z = readDem(self.reader.dataset, shape)
        return localRelief(z, self.relief_mm / (self.px_mm * self.width / float(shape[1])))

    def window(self, top, left, height, width):
        sigma = self.relief_mm / self.px_mm
        pad = int(math.ceil(3.0 * sigma)) + 1
        z = self.reader.readWindow(top - pad, left - pad, height + 2 * pad, width + 2 * pad)
        return localRelief(z, sigma)[pad:pad + height, pad:pad + width]

    def close(self):
        self.reader.close()


def _sourceList(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        return [ColourSource(x)]
    return [x]


def autoMarkers(ref, mov, px_ref, px_mov, max_markers=30, max_side=4000, patch=256, progress=None):
    """
    Find automatically pairs of corresponding points between the reference image and the image to align.

    1. SIFT features are extracted on reduced versions of the two maps (same mm/pixel, longest side <= max_side),
       matched (ratio test) and filtered with RANSAC (similarity transformation).
    2. A set of well distributed matches is refined at full resolution: a patch of the image to align is warped
       with the coarse transformation and the residual shift is estimated with phase correlation. Only the windows
       needed are read.

    ref and mov are the two maps: a file name (the colours), a source (ColourSource, DemReliefSource) or a list of
    sources of the same kinds in the same order (e.g. colour and DEM: their matches are pooled before RANSAC and each
    control point is refined on the source where it was found).

    It returns (pairs, info): pairs is a list of ((xr, yr), (xm, ym)) in millimeters, info is a dictionary with the
    coarse estimation (number of matches/inliers, angle, scale) and the number of markers per source.
    """

    def report(perc, msg):
        if progress is not None:
            progress(perc, msg)

    refs = _sourceList(ref)
    movs = _sourceList(mov)
    ref_w, ref_h = refs[0].width, refs[0].height
    mov_w, mov_h = movs[0].width, movs[0].height

    # common resolution (mm/pixel) of the reduced maps
    mm = max(max(ref_w, ref_h) * px_ref, max(mov_w, mov_h) * px_mov) / float(max_side)
    mm = max(mm, px_ref, px_mov)
    fr = px_ref / mm      # full-res reference pixels -> reduced pixels
    fm = px_mov / mm
    shape_r = (max(1, int(round(ref_h * fr))), max(1, int(round(ref_w * fr))))
    shape_m = (max(1, int(round(mov_h * fm))), max(1, int(round(mov_w * fm))))
    # actual factors (rounding of the reduced size)
    fr_x, fr_y = shape_r[1] / ref_w, shape_r[0] / ref_h
    fm_x, fm_y = shape_m[1] / mov_w, shape_m[0] / mov_h

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sift = cv2.SIFT_create(nfeatures=20000)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))

    nsrc = len(refs)
    pts_m, pts_r, dist, kind = [], [], [], []
    n_good = 0
    featureless = []
    for k in range(nsrc):
        name = refs[k].label if nsrc > 1 else ""
        step = 50.0 / nsrc
        report(5 + k * step, "Reading the maps{}..".format(" (" + name + ")" if name else ""))
        small_r = clahe.apply(refs[k].reduced(shape_r))
        small_m = clahe.apply(movs[k].reduced(shape_m))

        report(5 + (k + 0.3) * step, "Extracting features{}..".format(" (" + name + ")" if name else ""))
        kp_r, des_r = sift.detectAndCompute(small_r, None)
        kp_m, des_m = sift.detectAndCompute(small_m, None)
        if des_r is None or des_m is None or len(kp_r) < 10 or len(kp_m) < 10:
            featureless.append(refs[k].label)
            continue

        report(5 + (k + 0.7) * step, "Matching features{}..".format(" (" + name + ")" if name else ""))
        knn = matcher.knnMatch(des_m, des_r, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        n_good += len(good)
        pts_m += [kp_m[m.queryIdx].pt for m in good]
        pts_r += [kp_r[m.trainIdx].pt for m in good]
        dist += [m.distance for m in good]
        kind += [k] * len(good)

    if len(featureless) == nsrc:
        raise Exception("Not enough image features were found.")
    if n_good < 8:
        raise Exception("Not enough matches between the two maps ({:d}).".format(n_good))

    pts_m = np.float32(pts_m)
    pts_r = np.float32(pts_r)

    report(55, "Estimating the transformation..")
    A_small, inliers = cv2.estimateAffinePartial2D(pts_m, pts_r, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                                   maxIters=10000, confidence=0.999)
    if A_small is None or inliers is None or int(inliers.sum()) < 8:
        raise Exception("The two maps could not be matched reliably.")

    inl = inliers.ravel().astype(bool)
    pts_m = pts_m[inl]
    pts_r = pts_r[inl]
    dist = np.array(dist)[inl]
    kind = np.array(kind)[inl]

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
    half = patch // 2
    window = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
    corners = np.array([[0.0, 0.0], [patch, 0.0], [0.0, patch], [patch, patch]])
    pairs = []
    per_source = [0] * nsrc
    for count, i in enumerate(candidates):
        if len(pairs) >= max_markers:
            break
        k = int(kind[i])
        xr = pts_r[i, 0] / fr_x
        yr = pts_r[i, 1] / fr_y
        x0 = int(round(xr)) - half
        y0 = int(round(yr)) - half
        if x0 < 0 or y0 < 0 or x0 + patch > ref_w or y0 + patch > ref_h:
            continue

        ref_patch = refs[k].window(y0, x0, patch, patch).astype(np.float32)
        if ref_patch.std() < 5.0:
            continue

        # moving image resampled on the reference patch grid (patch pixel -> moving pixel), from the window of the
        # moving map that the patch covers
        T = np.array([[1, 0, x0], [0, 1, y0]], dtype=np.float64)
        P = composeMatrix(B_full, T)
        c = applyMatrix(P, corners)
        mx0 = int(math.floor(c[:, 0].min())) - 2
        my0 = int(math.floor(c[:, 1].min())) - 2
        mx1 = int(math.ceil(c[:, 0].max())) + 3
        my1 = int(math.ceil(c[:, 1].max())) + 3
        mov_win = movs[k].window(my0, mx0, my1 - my0, mx1 - mx0)
        Pw = P.copy()
        Pw[:, 2] -= np.array([mx0, my0], dtype=np.float64)
        mov_patch = cv2.warpAffine(mov_win, Pw, (patch, patch), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
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
        pairs.append(((float(xr + 0.5) * px_ref, float(yr + 0.5) * px_ref),
                      (float(xm + 0.5) * px_mov, float(ym + 0.5) * px_mov)))
        per_source[k] += 1

        report(60 + 40.0 * (count + 1) / len(candidates), "Refining the control points..")

    angle, scale, _, _ = matrixToParams(A_full)
    # physical scale (mm -> mm) of the coarse transformation
    info = {'matches': n_good, 'inliers': int(inl.sum()), 'angle': angle, 'scale': scale * px_ref / px_mov,
            'markers': len(pairs), 'sources': {refs[k].label: per_source[k] for k in range(nsrc)}}

    return pairs, info


# ---------------------------------------------------------------------------------------------------------------------
# DEM: shading, relief and heights
# ---------------------------------------------------------------------------------------------------------------------

SHADE_AZIMUTH = 315.0        # light of the DEM shading: from the top left of the map (degrees clockwise from the top)
SHADE_ALTITUDE = 45.0        # elevation of the light (degrees)
SHADE_MAX_SIDE = 8000        # the DEM shading of the preview is computed on a DEM reduced to at most this size
RELIEF_MM = 20.0             # matching on the DEM: height above the mean surface within about this distance
HEIGHT_MAX_SIDE = 1500       # the heights of the two DEMs are compared on a grid of at most this size
HEIGHT_FIT_POINTS = 250000   # the height plane is fitted on at most this many cells of the grid
HEIGHT_FIT_ITERATIONS = 10
HEIGHT_OUTLIER_SIGMAS = 3.0  # cells farther than this many robust standard deviations from the plane are left out
HEIGHT_MIN_TOLERANCE = 0.003  # (m) ... but never the cells within 3 mm of it
HEIGHT_MIN_CELLS = 100


def readDem(dataset, shape=None):
    """
    Read a DEM (an open rasterio dataset) as float32 with NaN where there is no data, optionally reduced to shape
    (rows, cols) by averaging the valid values.
    """
    if shape is None or (shape[0] == dataset.height and shape[1] == dataset.width):
        z = dataset.read(1, masked=True)
    else:
        z = dataset.read(1, out_shape=(shape[0], shape[1]), resampling=Resampling.average, masked=True)
    z = np.ma.filled(z.astype(np.float32), np.nan)
    z[~np.isfinite(z)] = np.nan
    return z


def reducedShape(width, height, max_side):
    """
    (rows, cols) of a map of width x height pixels reduced so that its longest side is at most max_side.
    """
    f = min(1.0, max_side / float(max(width, height)))
    return max(1, int(round(height * f))), max(1, int(round(width * f)))


def hillshade(z, dx, dy, azimuth=SHADE_AZIMUTH, altitude=SHADE_ALTITUDE, band=512):
    """
    Shading of the DEM z (NaN = no data) lit from azimuth (degrees clockwise from the top of the map) and altitude
    (degrees above the horizon), as grey levels 1..255 (0 = no data). dx, dy: spacing of the pixels, in the units of
    the heights. Processed in bands of rows (large DEMs).
    """
    az = math.radians(azimuth)
    alt = math.radians(altitude)
    # direction of the light (x right, y down, z up)
    lx = math.cos(alt) * math.sin(az)
    ly = -math.cos(alt) * math.cos(az)
    lz = math.sin(alt)

    h, w = z.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for r0 in range(0, h, band):
        r1 = min(h, r0 + band)
        zb = z[max(r0 - 1, 0):min(r1 + 1, h)].astype(np.float32)
        # the border of the map: the edge values are repeated (one-sided gradient)
        zb = np.pad(zb, ((1 if r0 == 0 else 0, 1 if r1 == h else 0), (1, 1)), mode='edge')
        gx, gy = hornGradient(zb, dx, dy)
        # the surface normal is (-gx, -gy, 1) / norm
        shade = (lz - gx * lx - gy * ly) / np.sqrt(1.0 + gx * gx + gy * gy)
        valid = np.isfinite(shade)
        band_out = np.clip(1.0 + 254.0 * np.clip(np.where(valid, shade, 0.0), 0.0, 1.0), 1.0, 255.0).astype(np.uint8)
        band_out[~valid] = 0
        out[r0:r1] = band_out
    return out


def localRelief(z, sigma_px):
    """
    Height above the local mean surface (Gaussian of sigma_px pixels, computed on the valid values only), as grey
    levels around 128 (scaled by its robust spread; 0 = no data).
    """
    valid = np.isfinite(z)
    out = np.zeros(z.shape, dtype=np.uint8)
    if valid.sum() < 16:
        return out
    zf = np.where(valid, z, 0.0).astype(np.float32)
    num = cv2.GaussianBlur(zf, (0, 0), sigma_px)
    den = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigma_px)
    r = zf - num / np.maximum(den, 1e-6)
    rv = r[valid]
    if rv.size > 1000000:
        rv = rv[::int(rv.size // 1000000) + 1]
    med = float(np.median(rv))
    spread = max(1.4826 * float(np.median(np.abs(rv - med))), 1e-9)
    out[:] = np.clip(128.0 + 40.0 * (r - med) / spread, 1.0, 255.0).astype(np.uint8)
    out[~valid] = 0
    return out


def heightDifference(ref_z, mov_z, M, ref_size, mov_size):
    """
    Heights of the map to align minus the heights of the reference, on the grid of ref_z (the reference DEM, possibly
    reduced; NaN = no data). mov_z is the DEM of the map to align (possibly reduced), M maps pixels of the map to
    align to pixels of the reference map; ref_size and mov_size are the sizes (width, height) of the maps.
    """
    sr = (ref_z.shape[1] / float(ref_size[0]), ref_z.shape[0] / float(ref_size[1]))
    sm = (mov_z.shape[1] / float(mov_size[0]), mov_z.shape[0] / float(mov_size[1]))
    toRef = composeMatrix(resizeMatrix(sr[0], sr[1]), composeMatrix(M, invertMatrix(resizeMatrix(sm[0], sm[1]))))
    mov_on_ref = warpArray(mov_z.astype(np.float32), toRef, ref_z.shape[1], ref_z.shape[0], border_value=np.nan)
    return mov_on_ref - ref_z


def fitHeightPlane(diff, ref_size, px_ref):
    """
    Robust plane fitted to diff (heights of the aligned map minus heights of the reference on a grid of the reference,
    see heightDifference(); NaN = no data): diff ~ a x + b y + c, with x, y pixels of the reference map
    (pixel-centre convention). The cells that changed (colonies that grew or died, DEM artefacts) are left out
    iteratively: those farther from the plane than HEIGHT_OUTLIER_SIGMAS robust standard deviations.

    px_ref: pixel size of the reference map (mm); the heights are assumed in metres. Returns None if fewer than
    HEIGHT_MIN_CELLS cells are valid, otherwise a dictionary: a, b, c, offset (m, at the centre of the cells
    compared), tilt (degrees), spread (m, robust standard deviation of the heights left), n (cells compared),
    n_used (cells used for the plane).
    """
    h, w = diff.shape
    step = max(1, int(math.ceil(math.sqrt(h * w / float(HEIGHT_FIT_POINTS)))))
    sub = diff[::step, ::step]
    jj, ii = np.nonzero(np.isfinite(sub))
    if len(ii) < HEIGHT_MIN_CELLS:
        return None

    sx, sy = w / float(ref_size[0]), h / float(ref_size[1])
    x = (ii * step + 0.5) / sx - 0.5
    y = (jj * step + 0.5) / sy - 0.5
    d = sub[jj, ii].astype(np.float64)
    cx, cy = float(x.mean()), float(y.mean())
    X = np.column_stack([x - cx, y - cy, np.ones_like(x)])

    use = np.ones(len(d), dtype=bool)
    coef = np.zeros(3)
    for _ in range(HEIGHT_FIT_ITERATIONS):
        coef = np.linalg.lstsq(X[use], d[use], rcond=None)[0]
        r = d - X @ coef
        spread = 1.4826 * float(np.median(np.abs(r[use])))
        new = np.abs(r) <= max(HEIGHT_OUTLIER_SIGMAS * spread, HEIGHT_MIN_TOLERANCE)
        if new.sum() < HEIGHT_MIN_CELLS or np.array_equal(new, use):
            break
        use = new

    r = d - X @ coef
    a, b, c0 = (float(v) for v in coef)
    gradient = math.hypot(a, b) / (px_ref / 1000.0)     # m per m
    return {'a': a, 'b': b, 'c': c0 - a * cx - b * cy, 'offset': c0, 'tilt': math.degrees(math.atan(gradient)),
            'spread': 1.4826 * float(np.median(np.abs(r[use]))), 'n': int(len(d)), 'n_used': int(use.sum())}


def planeOnGrid(plane, shape, map_size):
    """
    The plane of fitHeightPlane() evaluated on a grid of shape (rows, cols) covering a map of map_size (width,
    height) pixels.
    """
    sx, sy = shape[1] / float(map_size[0]), shape[0] / float(map_size[1])
    x = (np.arange(shape[1]) + 0.5) / sx - 0.5
    y = (np.arange(shape[0]) + 0.5) / sy - 0.5
    return (plane['a'] * x[None, :] + plane['b'] * y[:, None] + plane['c']).astype(np.float32)


def shiftedPlane(plane, left, top):
    """
    The plane (a, b, c) on a canvas whose pixel (left, top) is the pixel (0, 0) of the reference.
    """
    return plane['a'], plane['b'], plane['c'] - plane['a'] * left - plane['b'] * top


def heightPlaneText(plane):
    if plane is None:
        return "Heights: the two DEMs do not overlap enough to be compared."
    return "Heights (aligned - reference): offset {:+.1f} cm, tilt {:.2f}°, spread left {:.1f} mm ({:d} of {:d} " \
           "cells used; the others changed or are noisy).".format(plane['offset'] * 100.0, plane['tilt'],
                                                                  plane['spread'] * 1000.0, plane['n_used'], plane['n'])


# ---------------------------------------------------------------------------------------------------------------------
# Alignment check (quality map)
# ---------------------------------------------------------------------------------------------------------------------

QUALITY_GRID = 12            # about this many cells per side over the overlap of the two maps
QUALITY_PATCH = 256          # side (pixels of the measured images) of the patch measured at the centre of each cell
QUALITY_MIN_VALID = 0.6      # minimum fraction of the patch with data in both maps
QUALITY_MIN_TEXTURE = 3.0    # minimum standard deviation of the grey levels of the reference patch
QUALITY_MIN_RESPONSE = 0.08  # weaker phase-correlation peaks: not measurable (no texture, or a real change)
QUALITY_GOOD_MM = 1.0        # residual up to this: good (green)
QUALITY_FAIR_MM = 3.0        # up to this: fair (yellow); above: poor (red)

QUALITY_GOOD = "good"
QUALITY_FAIR = "fair"
QUALITY_POOR = "poor"
QUALITY_NONE = "not measurable"


def phaseShift(a, b, valid):
    """
    Phase correlation of two images of the same size, using only the valid pixels. It returns (dx, dy, response):
    the content at (x, y) of a is at (x + dx, y + dy) of b. The response is the sum of the correlation around the
    peak, as cv2.phaseCorrelate gives it (1 = identical images).

    cv2.phaseCorrelate is not used: it pads the images to an "optimal" DFT size, and when that size is odd the
    shift it returns is off by half a pixel.
    """
    h, w = a.shape
    if h < 8 or w < 8 or valid.sum() < 16:
        return 0.0, 0.0, 0.0

    # the weights fade to zero smoothly towards the invalid pixels (e.g. the edge of the map): a sharp edge at the
    # same place in both images would give a false peak at no shift
    fade = max(4.0, 0.04 * min(h, w))
    dist = cv2.distanceTransform(np.pad(valid.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    taper = np.clip(dist / fade, 0.0, 1.0)
    weight = np.outer(np.hanning(h), np.hanning(w)) * (taper * taper * (3.0 - 2.0 * taper))

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    A = np.fft.rfft2((a - a[valid].mean()) * weight)
    B = np.fft.rfft2((b - b[valid].mean()) * weight)
    R = np.conj(A) * B
    R /= np.maximum(np.abs(R), 1e-12)
    corr = np.fft.irfft2(R, s=(h, w))

    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    rows = (py + np.arange(-2, 3)) % h
    cols = (px + np.arange(-2, 3)) % w
    patch = corr[np.ix_(rows, cols)]
    response = float(patch.sum())

    # sub-pixel peak: centroid of the positive values of the 3 x 3 neighbourhood
    weights = np.clip(patch[1:4, 1:4], 0.0, None)
    total = weights.sum()
    oy, ox = 0.0, 0.0
    if total > 0.0:
        oy = float((weights.sum(axis=1) * np.arange(-1, 2)).sum() / total)
        ox = float((weights.sum(axis=0) * np.arange(-1, 2)).sum() / total)

    dy = py + oy
    dx = px + ox
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return dx, dy, response


def resizeMatrix(sx, sy):
    """
    Map pixels -> pixels of a version of the map resized by (sx, sy) (pixel-centre coordinates).
    """
    return np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5]])


def invertMatrix(M):
    M3 = np.vstack([np.asarray(M, dtype=np.float64), [0.0, 0.0, 1.0]])
    return np.linalg.inv(M3)[:2]


def _grayAndValid(bgra):
    """
    Grey levels (float32) and data mask of a BGRA patch (alpha 0 or black = no data).
    """
    if bgra.ndim == 2:
        gray = bgra.astype(np.float32)
        valid = gray > 0
    else:
        code = cv2.COLOR_BGRA2GRAY if bgra.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(np.ascontiguousarray(bgra), code).astype(np.float32)
        valid = bgra[:, :, :3].max(axis=2) > 0
        if bgra.shape[2] == 4:
            valid &= bgra[:, :, 3] > 0
    return gray, valid


def residualGrid(ref, mov, M, px_ref, ref_size, mov_size, grid=QUALITY_GRID, patch=QUALITY_PATCH, stop=None):
    """
    Check an alignment: the shift left between the reference map and the aligned map, measured by phase
    correlation on a patch at the centre of each cell of a grid (about grid x grid cells) over the part of the
    reference covered by the aligned map.

    ref and mov are the images measured: BGRA arrays (alpha 0 or black = no data), possibly reduced versions of the
    maps (e.g. the DEM shading); ref_size and mov_size are the sizes (width, height) of the maps in pixels. M maps
    pixels of the map to align to pixels of the reference map. px_ref is the pixel size of the reference map (mm).

    It returns (cells, stats). Each cell is a dictionary: rect and patch (x0, y0, x1, y1 in reference pixels),
    status (good / fair / poor / not measurable), reason (why not measurable), dx_mm, dy_mm, mm (residual: where
    the reference content is in the aligned map), response. stats: see qualityStats().
    """
    M = np.asarray(M, dtype=np.float64)
    ref_w, ref_h = ref_size
    mov_w, mov_h = mov_size
    sr = (ref.shape[1] / ref_w, ref.shape[0] / ref_h)
    sm = (mov.shape[1] / mov_w, mov.shape[0] / mov_h)

    # part of the reference covered by the aligned map
    corners = applyMatrix(M, [[-0.5, -0.5], [mov_w - 0.5, -0.5], [mov_w - 0.5, mov_h - 0.5], [-0.5, mov_h - 0.5]])
    x0 = max(-0.5, corners[:, 0].min())
    y0 = max(-0.5, corners[:, 1].min())
    x1 = min(ref_w - 0.5, corners[:, 0].max())
    y1 = min(ref_h - 0.5, corners[:, 1].max())
    if x1 - x0 < 8 or y1 - y0 < 8:
        return [], qualityStats([])

    side = math.sqrt((x1 - x0) * (y1 - y0)) / grid
    nx = max(1, int(round((x1 - x0) / side)))
    ny = max(1, int(round((y1 - y0) / side)))
    cw = (x1 - x0) / nx
    ch = (y1 - y0) / ny

    # patch side in pixels of the measured reference image (not larger than 90% of a cell)
    p = int(min(patch, 0.9 * min(cw * sr[0], ch * sr[1])))
    p -= p % 2
    Sr = resizeMatrix(sr[0], sr[1])
    toRef = composeMatrix(Sr, composeMatrix(M, invertMatrix(resizeMatrix(sm[0], sm[1]))))  # mov image -> ref image

    cells = []
    for j in range(ny):
        for i in range(nx):
            if stop is not None and stop():
                return cells, qualityStats(cells)
            rect = (x0 + i * cw, y0 + j * ch, x0 + (i + 1) * cw, y0 + (j + 1) * ch)
            cx, cy = 0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3])
            cell = {'rect': rect, 'center': (cx, cy), 'status': QUALITY_NONE, 'reason': "", 'dx_mm': None,
                    'dy_mm': None, 'mm': None, 'response': 0.0, 'patch': None}
            cells.append(cell)
            if p < 32:
                cell['reason'] = "cell too small"
                continue

            # patch of the reference image around the centre, and the aligned image on the same pixels
            icx, icy = applyMatrix(Sr, [[cx, cy]])[0]
            px0 = int(round(icx - p / 2.0))
            py0 = int(round(icy - p / 2.0))
            T = np.array([[1.0, 0.0, -px0], [0.0, 1.0, -py0]])
            ref_patch = warpArray(ref, T, p, p, border_value=0, flags=cv2.INTER_NEAREST)
            mov_patch = warpArray(mov, composeMatrix(T, toRef), p, p, border_value=0, flags=cv2.INTER_LINEAR)
            corners_p = applyMatrix(invertMatrix(composeMatrix(T, Sr)), [[-0.5, -0.5], [p - 0.5, p - 0.5]])
            cell['patch'] = (corners_p[0, 0], corners_p[0, 1], corners_p[1, 0], corners_p[1, 1])

            gr, vr = _grayAndValid(ref_patch)
            gm, vm = _grayAndValid(mov_patch)
            # interpolated pixels next to the edge of the data mix in the no-data colour
            valid = cv2.erode((vr & vm).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if valid.mean() < QUALITY_MIN_VALID:
                cell['reason'] = "no data"
                continue
            if gr[valid].std() < QUALITY_MIN_TEXTURE:
                cell['reason'] = "no texture"
                continue

            dx, dy, response = phaseShift(gr, gm, valid)
            cell['response'] = response
            if response < QUALITY_MIN_RESPONSE or abs(dx) > p / 4.0 or abs(dy) > p / 4.0:
                cell['reason'] = "no match (little texture, or the place changed)"
                continue

            # measured-image pixels -> reference map pixels -> mm
            cell['dx_mm'] = dx / sr[0] * px_ref
            cell['dy_mm'] = dy / sr[1] * px_ref
            cell['mm'] = math.hypot(cell['dx_mm'], cell['dy_mm'])
            if cell['mm'] <= QUALITY_GOOD_MM:
                cell['status'] = QUALITY_GOOD
            elif cell['mm'] <= QUALITY_FAIR_MM:
                cell['status'] = QUALITY_FAIR
            else:
                cell['status'] = QUALITY_POOR

    return cells, qualityStats(cells)


def qualityStats(cells):
    """
    Summary of the cells of residualGrid(): number of cells (n_cells) and of measured cells (n), median, 90th
    percentile and worst residual (mm), mean residual vector (mean_dx_mm, mean_dy_mm) and whether it looks like a
    shift left over (most cells pointing the same way).
    """
    measured = [c for c in cells if c['mm'] is not None]
    stats = {'n_cells': len(cells), 'n': len(measured), 'median': None, 'p90': None, 'worst': None,
             'mean_dx_mm': None, 'mean_dy_mm': None, 'shift_left': False}
    if len(measured) == 0:
        return stats
    mm = np.array([c['mm'] for c in measured])
    dx = np.array([c['dx_mm'] for c in measured])
    dy = np.array([c['dy_mm'] for c in measured])
    stats['median'] = float(np.median(mm))
    stats['p90'] = float(np.percentile(mm, 90))
    stats['worst'] = float(mm.max())
    stats['mean_dx_mm'] = float(dx.mean())
    stats['mean_dy_mm'] = float(dy.mean())
    mean = math.hypot(stats['mean_dx_mm'], stats['mean_dy_mm'])
    stats['shift_left'] = len(measured) >= 5 and mean >= 0.5 and mean >= 0.5 * stats['median']
    return stats


def qualityText(stats, source=None):
    """
    One or two sentences about the alignment check (source: what was compared, if not the colours).
    """
    head = "Alignment check" + ("" if source is None else " (" + source + ")")
    if stats['n'] == 0:
        return head + ": no cell could be measured ({:d} cells: no texture, no data or changed).".format(
            stats['n_cells'])
    txt = head + ": {:d} of {:d} cells measured; leftover shift median {:.2f} mm, 90% {:.2f} mm, " \
          "worst {:.2f} mm.".format(stats['n'], stats['n_cells'], stats['median'], stats['p90'], stats['worst'])
    if stats['shift_left']:
        txt += " Most cells point the same way: the aligned map sits about ({:+.2f}, {:+.2f}) mm (x right, y down) " \
               "from the reference; check the markers, or move it back with Adjust Tx / Ty.".format(
                   stats['mean_dx_mm'], stats['mean_dy_mm'])
    return txt

# TagLab
# A semi-automatic segmentation tool
#
# Per-region metrics computed from the DEM channel (slope, elevation, relief, height, volume, rugosity).
# Each region is processed on a small window of the DEM around its bounding box, so the whole DEM
# is never loaded or processed at once.

import numpy as np
import rasterio as rio
from rasterio.windows import Window
from cv2 import dilate

# names of the region attributes (<= 10 characters, the maximum length of a shapefile field name)
DEM_METRIC_NAMES = ["slope_deg", "elev_mean", "elev_min", "elev_max", "relief_cm", "height_cm", "volume_cm3", "rugosity"]

MAX_SLOPE = 87.0            # steeper slopes are considered DEM artifacts (jumps)
MIN_VALID_FRACTION = 0.5    # minimum fraction of the region covered by valid DEM values
PAD = 3                     # pixels read around the bounding box (slope kernel + base plane ring)
RING_SIZE = 2               # width (in pixels) of the ring used to fit the base plane


def hornSlope(z, dx, dy):
    """
    Slope (in degrees) of the interior pixels of z computed with Horn's method (the same used by gdaldem).
    The result has shape (h-2, w-2) and it is NaN where one of the 3x3 neighbors is missing.
    """

    a = z[:-2, :-2]
    b = z[:-2, 1:-1]
    c = z[:-2, 2:]
    d = z[1:-1, :-2]
    f = z[1:-1, 2:]
    g = z[2:, :-2]
    h = z[2:, 1:-1]
    i = z[2:, 2:]

    dzdx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * dx)
    dzdy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * dy)

    return np.degrees(np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy)))


class DemReader(object):
    """
    Read windows of a DEM (nodata -> NaN). The horizontal spacing is taken from the geotransform; if the DEM
    is not georeferenced (or it uses a geographic CRS) the pixel size of the map is used, assuming metres.
    """

    def __init__(self, filename, pixel_size_mm=None):

        self.dataset = rio.open(filename)
        self.nodata = self.dataset.nodata

        transform = self.dataset.transform
        crs = self.dataset.crs
        self.georeferenced = not transform.is_identity and not (crs is not None and crs.is_geographic)

        if self.georeferenced:
            self.dx = abs(transform.a)
            self.dy = abs(transform.e)
        elif pixel_size_mm is not None:
            self.dx = self.dy = pixel_size_mm / 1000.0
        else:
            # same assumption of gdaldem on a DEM without georeferencing
            self.dx = self.dy = 1.0

    def close(self):
        self.dataset.close()

    def readWindow(self, top, left, height, width):
        """
        Read the window [top, left, width, height]; the pixels outside the DEM are NaN.
        """

        H = self.dataset.height
        W = self.dataset.width

        out = np.full((height, width), np.nan, dtype=np.float64)

        r0 = max(top, 0)
        c0 = max(left, 0)
        r1 = min(top + height, H)
        c1 = min(left + width, W)

        if r1 > r0 and c1 > c0:
            data = self.dataset.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64)
            if self.nodata is not None:
                data[data == self.nodata] = np.nan
            out[r0 - top:r1 - top, c0 - left:c1 - left] = data

        return out


def computeBlobMetrics(reader, blob, pixel_size_mm):
    """
    Compute the DEM metrics of a blob. It returns a dictionary {metric name: value} (values are None if
    the DEM does not cover enough of the region) and the surface area in pixels.
    """

    metrics = {name: None for name in DEM_METRIC_NAMES}

    bh = int(blob.bbox[3])
    bw = int(blob.bbox[2])
    top = int(blob.bbox[0]) - PAD
    left = int(blob.bbox[1]) - PAD
    h = bh + 2 * PAD
    w = bw + 2 * PAD

    z = reader.readWindow(top, left, h, w)

    mask = np.zeros((h, w), dtype=bool)
    mask[PAD:PAD + bh, PAD:PAD + bw] = blob.getMask() > 0
    n_pixels = int(mask.sum())
    if n_pixels == 0:
        return metrics, 0.0

    slope = np.full((h, w), np.nan)
    slope[1:-1, 1:-1] = hornSlope(z, reader.dx, reader.dy)
    slope_ok = np.isfinite(slope) & (slope <= MAX_SLOPE)

    # surface area: each pixel is weighted by 1/cos(slope), pixels without a valid slope are considered flat
    factor = np.ones((h, w))
    factor[slope_ok] = 1.0 / np.cos(np.radians(slope[slope_ok]))
    surface_area = float(factor[mask].sum())

    valid = mask & np.isfinite(z)
    if valid.sum() < MIN_VALID_FRACTION * n_pixels:
        return metrics, surface_area

    zs = z[valid]
    elev_min = float(zs.min())
    elev_max = float(zs.max())

    # base plane fitted (least squares) on the valid DEM values of a thin ring just outside the region
    kernel = np.ones((3, 3), dtype=np.uint8)
    ring = dilate(mask.astype(np.uint8), kernel, iterations=RING_SIZE).astype(bool) & ~mask & np.isfinite(z)
    rows, cols = np.nonzero(ring)
    if len(rows) >= 3:
        A = np.column_stack((cols, rows, np.ones(len(rows))))
        coeffs = np.linalg.lstsq(A, z[ring], rcond=None)[0]
        vr, vc = np.nonzero(valid)
        base = coeffs[0] * vc + coeffs[1] * vr + coeffs[2]
    else:
        base = np.full(len(zs), elev_min)

    height = zs - base                              # metres
    pixel_area_cm2 = (pixel_size_mm / 10.0) ** 2

    slope_in = slope[mask & slope_ok]

    metrics["slope_deg"] = round(float(slope_in.mean()), 2) if len(slope_in) > 0 else None
    metrics["elev_mean"] = round(float(zs.mean()), 4)
    metrics["elev_min"] = round(elev_min, 4)
    metrics["elev_max"] = round(elev_max, 4)
    metrics["relief_cm"] = round((elev_max - elev_min) * 100.0, 2)
    metrics["height_cm"] = round(max(float(height.max()), 0.0) * 100.0, 2)
    metrics["volume_cm3"] = round(float(np.clip(height, 0.0, None).sum()) * 100.0 * pixel_area_cm2, 2)
    metrics["rugosity"] = round(surface_area / n_pixels, 4)

    return metrics, surface_area


def pixelSizeWarning(reader, pixel_size_mm):
    """
    Return a warning message if the pixel size of the DEM and the one of the map are different (more than 5%).
    """

    if not reader.georeferenced or pixel_size_mm is None or pixel_size_mm <= 0.0:
        return None

    dem_mm = reader.dx * 1000.0
    if abs(dem_mm - pixel_size_mm) / pixel_size_mm > 0.05:
        return "The DEM pixel size ({:.3f} mm) is different from the pixel size of the map ({:.3f} mm). " \
               "Check the pixel size in the map settings: areas and volumes use the map pixel size.".format(dem_mm, pixel_size_mm)

    return None


def computeDemMetrics(dem_filename, blobs, pixel_size_mm, progress=None):
    """
    Compute the DEM metrics of the given blobs, storing them in blob.data (and the surface area in blob.surface_area).
    progress, if given, is called with the percentage of the processed blobs.
    It returns a summary {'processed': n, 'skipped': n, 'warning': message or None}.
    """

    reader = DemReader(dem_filename, pixel_size_mm)
    skipped = 0

    try:
        n = len(blobs)
        for i, blob in enumerate(blobs):
            metrics, surface_area = computeBlobMetrics(reader, blob, pixel_size_mm)
            blob.surface_area = surface_area
            for name in DEM_METRIC_NAMES:
                blob.data[name] = metrics[name]
            if metrics["elev_mean"] is None:
                skipped += 1
            if progress is not None:
                progress(100.0 * (i + 1) / n)

        warning = pixelSizeWarning(reader, pixel_size_mm)
    finally:
        reader.close()

    return {"processed": len(blobs) - skipped, "skipped": skipped, "warning": warning}


def computeSurfaceArea(dem_filename, blobs, pixel_size_mm=None):
    """
    Compute only the surface area (in pixels) of the given blobs.
    """

    reader = DemReader(dem_filename, pixel_size_mm)
    try:
        for blob in blobs:
            metrics, surface_area = computeBlobMetrics(reader, blob, 1.0 if pixel_size_mm is None else pixel_size_mm)
            blob.surface_area = surface_area
    finally:
        reader.close()


def registerAttributes(project):
    """
    Add the DEM metrics to the region attributes of the project (decimal numbers). Returns the attributes added.
    """

    added = []
    for name in DEM_METRIC_NAMES:
        if not project.region_attributes.has(name):
            project.region_attributes.data.append({'name': name, 'type': 'decimal number', 'min': None, 'max': None,
                                                   'keywords': []})
            added.append(name)

    return added

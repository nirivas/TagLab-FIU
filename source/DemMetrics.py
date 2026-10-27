# TagLab
# A semi-automatic segmentation tool
#
# Per-region metrics computed from the DEM channel (slope, elevation, relief, height, volume, rugosity).
# Each region is processed on a small window of the DEM around its bounding box, so the whole DEM
# is never loaded or processed at once. The values of the whole plot (rugosity, slope, relief) are
# computed tile by tile (computePlotMetrics).

import csv
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

# values of the whole plot (computePlotMetrics), stored in image.metadata[PLOT_METRICS_KEY]
PLOT_METRICS_KEY = "plot_dem_metrics"
PLOT_CELL_MM = 10.0        # cell of the averaged DEM giving values comparable between surveys (whole DEM pixels)
PLOT_TILE = 2048            # the plot is processed in tiles of about this size (pixels)
PLOT_MAX_CELLS = 16000000   # larger cells are used if the averaged DEM would have more cells than this
PLOT_RELIEF_PERCENTILES = (1.0, 99.0)   # the relief is the range of heights between these percentiles


def hornGradient(z, dx, dy):
    """
    Gradient (dz/dx, dz/dy) of the interior pixels of z computed with Horn's method (the same used by gdaldem).
    The results have shape (h-2, w-2) and they are NaN where one of the 3x3 neighbors is missing.
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

    return dzdx, dzdy


def hornSlope(z, dx, dy):
    """
    Slope (in degrees) of the interior pixels of z computed with Horn's method (the same used by gdaldem).
    The result has shape (h-2, w-2) and it is NaN where one of the 3x3 neighbors is missing.
    """

    dzdx, dzdy = hornGradient(z, dx, dy)

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


def surfaceSums(z, dx, dy):
    """
    Sums over the interior pixels of z (z has a border of one pixel around them) used for the plot values:
    [pixels with a DEM value, valid slopes, sum of their surface factors 1/cos(slope), sum of the slopes (degrees)].
    Unlike for the regions, steep slopes are kept (walls of colonies and boulders are real surface: excluding
    the slopes above MAX_SLOPE made the rugosity of a steep sample plot 1.68 instead of 2.54), and the pixels
    without a slope (next to a DEM hole) are left out instead of counting as flat: the rugosity of the plot is
    the mean factor of the pixels with a slope.
    """

    dzdx, dzdy = hornGradient(z, dx, dy)
    g2 = dzdx * dzdx + dzdy * dzdy

    valid = np.isfinite(z[1:-1, 1:-1])
    slope_ok = valid & np.isfinite(g2)
    g2 = g2[slope_ok]

    n_valid = float(valid.sum())
    n_slope = float(len(g2))
    factor_sum = float(np.sqrt(1.0 + g2).sum())
    slope_sum = float(np.degrees(np.arctan(np.sqrt(g2))).sum())

    return np.array([n_valid, n_slope, factor_sum, slope_sum])


def computePlotMetrics(dem_filename, area, pixel_size_mm, progress=None, tile=PLOT_TILE):
    """
    Values of the whole plot area = [top, left, width, height] (pixels; it is clipped to the DEM): the rugosity
    (3D surface area / planar area) and the mean slope, at the resolution of the DEM and on the DEM averaged to
    cells of about PLOT_CELL_MM (comparable between surveys with different resolutions), and the relief: the
    range of heights above the plane fitted to the averaged DEM (PLOT_RELIEF_PERCENTILES, so that single spikes
    do not count). Only the part of the plot covered by the DEM is used.
    The DEM is read once, in tiles of whole cells: each tile gives the sums at the DEM resolution (surfaceSums)
    of its pixels inside the plot, and its cells. The cells form a grid with a border of one cell around the plot,
    so that the slopes at its edges use the DEM around it, as at the DEM resolution.
    progress, if given, is called with the percentage of the DEM processed.
    """

    reader = DemReader(dem_filename, pixel_size_mm)

    try:
        top = max(int(area[0]), 0)
        left = max(int(area[1]), 0)
        bottom = min(int(area[0]) + int(area[3]), reader.dataset.height)
        right = min(int(area[1]) + int(area[2]), reader.dataset.width)
        height = max(bottom - top, 0)
        width = max(right - left, 0)

        spacing_mm = 500.0 * (reader.dx + reader.dy)
        f = max(1, int(round(PLOT_CELL_MM / spacing_mm)))
        f = max(f, int(np.ceil(np.sqrt(float(width) * height / PLOT_MAX_CELLS))))

        result = {"area": [top, left, width, height], "planar_m2": None, "dem_coverage": 0.0,
                  "dem_mm": round(spacing_mm, 4), "rugosity": None, "slope_deg": None,
                  "cell_mm": round(f * spacing_mm, 4), "rugosity_1cm": None, "slope_1cm": None, "relief_cm": None}
        if width == 0 or height == 0:
            return result

        # grid of cells: whole cells inside the plot plus a border of one cell
        nbr = height // f
        nbc = width // f
        grid_top = top - f
        grid_left = left - f
        cells = np.full((nbr + 2, nbc + 2), np.nan)

        native = np.zeros(4)
        kb = max(1, tile // f)  # cells per tile side
        n_tiles = ((nbr + 1) // kb + 1) * ((nbc + 1) // kb + 1)
        done = 0
        for bi0 in range(0, nbr + 2, kb):
            bi1 = min(bi0 + kb, nbr + 2)
            for bj0 in range(0, nbc + 2, kb):
                bj1 = min(bj0 + kb, nbc + 2)

                r0 = grid_top + bi0 * f
                c0 = grid_left + bj0 * f
                th = (bi1 - bi0) * f
                tw = (bj1 - bj0) * f
                z = reader.readWindow(r0 - 1, c0 - 1, th + 2, tw + 2)   # one more pixel for the slopes

                # pixels of the tile inside the plot, with their border
                rr0 = max(r0, top)
                rr1 = min(r0 + th, bottom)
                cc0 = max(c0, left)
                cc1 = min(c0 + tw, right)
                if rr1 > rr0 and cc1 > cc0:
                    native += surfaceSums(z[rr0 - r0:rr1 - r0 + 2, cc0 - c0:cc1 - c0 + 2], reader.dx, reader.dy)

                # cells: mean of the valid DEM values, if they cover at least half of the cell
                block = z[1:-1, 1:-1].reshape(bi1 - bi0, f, bj1 - bj0, f)
                valid = np.isfinite(block)
                count = valid.sum(axis=(1, 3))
                total = np.where(valid, block, 0.0).sum(axis=(1, 3))
                with np.errstate(invalid='ignore', divide='ignore'):
                    cells[bi0:bi1, bj0:bj1] = np.where(count >= 0.5 * f * f, total / count, np.nan)

                done += 1
                if progress is not None:
                    progress(min(100.0, 100.0 * done / n_tiles))

        n_valid, n_slope, factor_sum, slope_sum = native.tolist()
        if n_valid == 0:
            return result

        result["planar_m2"] = round(n_valid * pixel_size_mm * pixel_size_mm / 1.0e6, 4)
        result["dem_coverage"] = round(n_valid / (float(width) * height), 4)
        if n_slope > 0:
            result["rugosity"] = round(factor_sum / n_slope, 4)
            result["slope_deg"] = round(slope_sum / n_slope, 2)

        if nbr > 0 and nbc > 0:
            coarse = np.zeros(4)
            for r in range(0, nbr, tile):
                for c in range(0, nbc, tile):
                    coarse += surfaceSums(cells[r:min(r + tile, nbr) + 2, c:min(c + tile, nbc) + 2],
                                          f * reader.dx, f * reader.dy)
            coarse = coarse.tolist()
            if coarse[1] > 0:
                result["rugosity_1cm"] = round(coarse[2] / coarse[1], 4)
                result["slope_1cm"] = round(coarse[3] / coarse[1], 2)

            # relief: heights above the least-squares plane of the cells inside the plot
            inner = cells[1:-1, 1:-1]
            rows, cols = np.nonzero(np.isfinite(inner))
            if len(rows) >= 3:
                x = cols * (f * reader.dx)
                y = rows * (f * reader.dy)
                zs = inner[rows, cols]
                A = np.column_stack((x, y, np.ones(len(rows))))
                coeffs = np.linalg.lstsq(A, zs, rcond=None)[0]
                heights = zs - A @ coeffs
                low, high = np.percentile(heights, PLOT_RELIEF_PERCENTILES)
                result["relief_cm"] = round(float(high - low) * 100.0, 2)

        return result

    finally:
        reader.close()


def plotArea(working_area, width, height):
    """
    The plot of a map of the given size: the working area [top, left, width, height] clipped to the map or, if
    there is none, the whole map. It returns (area, "working area" or "whole map").
    """

    if working_area is None or len(working_area) != 4:
        return [0, 0, int(width), int(height)], "whole map"

    top = max(int(working_area[0]), 0)
    left = max(int(working_area[1]), 0)
    bottom = min(int(working_area[0]) + int(working_area[3]), int(height))
    right = min(int(working_area[1]) + int(working_area[2]), int(width))

    return [top, left, max(right - left, 0), max(bottom - top, 0)], "working area"


PLOT_CSV_COLUMNS = ["map", "date", "area", "top", "left", "width", "height", "planar_m2", "dem_coverage", "dem_mm",
                    "rugosity", "slope_deg", "cell_mm", "rugosity_1cm", "slope_1cm", "relief_cm"]


def exportPlotMetrics(filename, images):
    """
    Write the plot values of the given maps (image.metadata[PLOT_METRICS_KEY]) as CSV: one row per map.
    """

    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(PLOT_CSV_COLUMNS)
        for image in images:
            plot = image.metadata[PLOT_METRICS_KEY]
            top, left, width, height = plot["area"]
            row = [image.name, image.acquisition_date, plot.get("area_source"), top, left, width, height]
            row += [plot.get(name) for name in PLOT_CSV_COLUMNS[7:]]
            writer.writerow(["" if value is None else value for value in row])


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

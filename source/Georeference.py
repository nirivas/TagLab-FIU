# TagLab
# A semi-automatic segmentation tool
#
# Georeference of the maps: read it (TagLab's own, stored in the project, or the image file's), set it from a file
# georeferenced in ArcGIS Pro (its side files, or by matching the map to a new georeferenced raster), or from
# coordinates typed for points of the map; share it with the maps that are on the same pixel grid.
#
# Conventions:
#  - transforms are rasterio / GDAL affine transforms in the corner convention: the corner of the pixel (0, 0) is at
#    transform * (0, 0), the centre of the pixel (i, j) at transform * (i + 0.5, j + 0.5);
#  - TagLab's pixel coordinates (contours, centroids, points) put the centre of the pixel (i, j) at (i, j);
#  - image.metadata["georeference"] (TagLab's georeference) wins over the georeference of the image file, which is
#    never changed;
#  - image.metadata["pixel_grid"] = {"group", "origin"}: maps of the same group are on the same pixel grid; the pixel
#    (i, j) of a map is at (i + origin[0], j + origin[1]) of the group's common frame.

import datetime
import math
import os
import re
import uuid
from collections import namedtuple

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window

from source import Alignment

KEY = "georeference"
GRID_KEY = "pixel_grid"

# side files written by ArcGIS Pro (Georeference > Save) or other GIS next to an image
SIDE_FILE_SUFFIXES = (".aux.xml", ".tfw", ".tfwx", ".tifw", ".tiffw", ".wld", ".jgw", ".pgw")

GRID_CHECK_MAX_SIDE = 2000      # coarse offset between two maps: on versions reduced to this size
GRID_CHECK_WINDOWS = 5          # then 5 x 5 windows ...
GRID_CHECK_WINDOW = 512         # ... of this size, at full resolution
GRID_CHECK_MIN_FRACTION = 0.4   # at least this fraction of the windows must be measurable
GRID_CHECK_MAX_MM = 1.0         # median distance of the windows from the common offset
GRID_CHECK_MAX_FRACTION = 0.25  # the common offset must be this close to whole pixels
PIXEL_SIZE_TOLERANCE = 0.001    # maps of the same grid have the same pixel size (relative)

PIXEL_SIZE_WARNING = 0.02       # the georeference's pixel size differs from TagLab's by more than this
STRETCH_WARNING = 0.005         # the georeference stretches or shears the map by more than this
MATCH_MIN_MARKERS = 6
MATCH_MAX_RMSE_MM = 2.0

FIT_RIGID = "rigid"
FIT_SIMILARITY = "similarity"
FIT_AFFINE = "affine"

WEB_MERCATOR_EPSG = (3857, 900913, 102100, 102113, 3785)

GeoRef = namedtuple("GeoRef", ["crs", "transform", "source", "method"])


class GeoreferenceError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# TagLab's georeference of a map
# ---------------------------------------------------------------------------------------------------------------------

def get(image):
    """
    The georeference of the map: GeoRef(crs, transform, source, method), or None. TagLab's own georeference
    (image.metadata) wins over the one of the image file.
    """
    entry = image.metadata.get(KEY) if image.metadata else None
    if entry:
        try:
            return GeoRef(CRS.from_wkt(entry["crs"]), Affine(*entry["transform"][:6]), entry.get("source", ""),
                          entry.get("method", ""))
        except Exception:
            pass

    filename = image.georef_filename
    if not filename:
        return None
    try:
        with rio.open(filename) as f:
            crs, transform = f.crs, f.transform
    except Exception:
        return None
    if crs is None or transform.is_identity:
        return None
    return GeoRef(crs, transform, filename, "file")


def isGeoreferenced(image):
    return get(image) is not None


def isSetInTagLab(image):
    return bool(image.metadata.get(KEY)) if image.metadata else False


def setGeoreference(image, crs, transform, method, source="", from_map=None, rmse_mm=None, extra=None):
    """
    Store TagLab's georeference of the map (a new dictionary, see Image.__init__: metadata is copied shallowly).
    """
    crs = CRS.from_user_input(crs)
    entry = {"crs": crs.to_wkt(), "transform": [float(v) for v in tuple(transform)[:6]], "method": method,
             "source": source or "", "from_map": from_map if from_map is not None else image.id,
             "date": datetime.date.today().isoformat()}
    if rmse_mm is not None:
        entry["rmse_mm"] = float(rmse_mm)
    if extra:
        entry.update(extra)
    image.metadata = dict(image.metadata)
    image.metadata[KEY] = entry
    return entry


def clear(image):
    if image.metadata and KEY in image.metadata:
        image.metadata = dict(image.metadata)
        del image.metadata[KEY]


def mapFile(image):
    """
    The image file used to recognise the map: the colour channel, else the DEM.
    """
    ch = image.getRGBChannel() or image.getDEMChannel()
    return ch.filename if ch is not None else None


# ---------------------------------------------------------------------------------------------------------------------
# Describing a georeference
# ---------------------------------------------------------------------------------------------------------------------

def crsName(crs):
    if crs is None:
        return "no coordinate system"
    epsg = crs.to_epsg()
    try:
        from osgeo import osr
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs.to_wkt())
        name = srs.GetName()
    except Exception:
        name = None
    if not name:
        name = crs.to_string()
    return "{} (EPSG:{})".format(name, epsg) if epsg else name


def metresPerUnit(crs, transform=None, size=None):
    """
    (metres per unit along x, along y) of the coordinate system; for latitude / longitude, at the centre of the map.
    """
    if crs is not None and crs.is_geographic:
        lat = 0.0
        if transform is not None and size is not None:
            cx, cy = transform * (size[0] / 2.0, size[1] / 2.0)
            lat = cy
        return 111320.0 * math.cos(math.radians(lat)), 110574.0
    factor = 1.0
    if crs is not None:
        try:
            factor = float(crs.linear_units_factor[1])
        except Exception:
            factor = 1.0
    return factor, factor


def describe(crs, transform, size=None, taglab_px=None):
    """
    A dictionary describing the georeference: CRS name, pixel size (mm), direction of the top of the map
    (degrees clockwise from north), stretch, and warnings (as text).
    """
    a, b, c, d, e, f = tuple(transform)[:6]
    mx, my = metresPerUnit(crs, transform, size)
    sx = math.hypot(a * mx, d * my) * 1000.0
    sy = math.hypot(b * mx, e * my) * 1000.0
    px = math.sqrt(sx * sy) if sx > 0 and sy > 0 else 0.0
    # top of the map: the direction of the pixel axis -y in the world (x east, y north)
    bearing = math.degrees(math.atan2(-b * mx, -e * my)) % 360.0
    cos_axes = abs(a * mx * b * mx + d * my * e * my) / max(sx * sy / 1.0e6, 1e-30)
    stretch = max(abs(sx / sy - 1.0) if sy > 0 else 0.0, cos_axes)
    det = a * e - b * d

    warnings = []
    if crs is None:
        warnings.append("No coordinate system.")
    elif crs.is_geographic:
        warnings.append("Latitude / longitude: areas in ArcGIS will not be in m². Use a projected coordinate system "
                        "in metres (e.g. WGS 84 / UTM zone 20N).")
    elif crs.to_epsg() in WEB_MERCATOR_EPSG:
        warnings.append("Web Mercator stretches distances and areas (about 10% in the Caribbean). Use a projected "
                        "coordinate system such as WGS 84 / UTM zone 20N.")
    if det > 0 and (crs is None or not crs.is_geographic):
        warnings.append("The georeference mirrors the map (north is down): check the control points.")
    if taglab_px and px > 0:
        diff = px / taglab_px - 1.0
        if abs(diff) > PIXEL_SIZE_WARNING:
            warnings.append("The pixel size of the georeference ({:.3f} mm) differs from TagLab's ({:.3f} mm) by "
                            "{:+.1f}%. TagLab keeps its own pixel size for areas; ArcGIS's Shape_Area will differ by "
                            "about {:+.1f}%.".format(px, taglab_px, 100.0 * diff, 100.0 * ((1 + diff) ** 2 - 1)))
    if stretch > STRETCH_WARNING:
        warnings.append("The georeference stretches or shears the map by {:.1f}%.".format(100.0 * stretch))

    return {"crs": crsName(crs), "epsg": crs.to_epsg() if crs is not None else None,
            "geographic": bool(crs is not None and crs.is_geographic), "pixel_mm": px, "pixel_x_mm": sx,
            "pixel_y_mm": sy, "top_bearing": bearing, "stretch": stretch, "warnings": warnings}


def describeText(desc):
    lines = ["Coordinate system: " + desc["crs"],
             "Pixel size: {:.4f} mm".format(desc["pixel_mm"]),
             "Top of the map points to {:.1f}° (clockwise from north)".format(desc["top_bearing"])]
    if desc["stretch"] > 1e-4:
        lines.append("Stretch / shear: {:.2f}%".format(100.0 * desc["stretch"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------------------------------------------------

def fitAffine(src, dst):
    """
    Least-squares affine transform (2 x 3) mapping src points onto dst points (N >= 3). Returns (A, residuals),
    residuals = A(src) - dst.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) < 3 or len(dst) != len(src):
        raise GeoreferenceError("An affine fit needs at least three points.")
    # centre for numerical stability (world coordinates can be large)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    X = np.hstack([src - cs, np.ones((len(src), 1))])
    sol, _, rank, _ = np.linalg.lstsq(X, dst - cd, rcond=None)
    if rank < 3:
        raise GeoreferenceError("The points are on a line: place them farther apart.")
    L = sol[:2].T
    t = sol[2] + cd - L @ cs
    A = np.hstack([L, t[:, None]])
    residuals = Alignment.applyMatrix(A, src) - dst
    return A, residuals


def matrixToAffine(A):
    return Affine(A[0, 0], A[0, 1], A[0, 2], A[1, 0], A[1, 1], A[1, 2])


def centreToCorner(A):
    """
    A maps TagLab pixel coordinates (pixel centres at integers) to the world: the corner-convention transform.
    """
    return matrixToAffine(A) * Affine.translation(-0.5, -0.5)


def pixelToWorld(transform, points):
    """
    TagLab pixel coordinates (centres at integers) -> world coordinates.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2) + 0.5
    a, b, c, d, e, f = tuple(transform)[:6]
    return np.stack([a * pts[:, 0] + b * pts[:, 1] + c, d * pts[:, 0] + e * pts[:, 1] + f], axis=1)


# ---------------------------------------------------------------------------------------------------------------------
# From a file georeferenced in ArcGIS Pro
# ---------------------------------------------------------------------------------------------------------------------

def sideFiles(filename):
    base, _ = os.path.splitext(filename)
    found = []
    for suffix in SIDE_FILE_SUFFIXES:
        for candidate in (filename + suffix, base + suffix):
            if os.path.exists(candidate) and candidate not in found:
                found.append(candidate)
    return found


def sameFile(a, b):
    try:
        return os.path.samefile(a, b)
    except Exception:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def readFile(filename, epsg=None):
    """
    The georeference of a file as GIS software sees it, side files first (ArcGIS Pro's Save writes .aux.xml and world
    files next to the image, the GeoTIFF tags may still hold the old georeference). Without a transform but with
    control points (ESRI GeodataXform), the transform is fitted to them.

    Returns a dictionary: crs (None when the file has none and no epsg is given), transform, width, height,
    method ("side files", "image file" or "control points"), rmse (map units, control points only), side_files.
    """
    if not os.path.exists(filename):
        raise GeoreferenceError("The file does not exist:\n" + filename)
    with rio.Env(GDAL_GEOREF_SOURCES="PAM,WORLDFILE,INTERNAL,TABFILE"):
        with rio.open(filename) as f:
            crs, transform, width, height = f.crs, f.transform, f.width, f.height
            try:
                gcps, gcp_crs = f.gcps
            except Exception:
                gcps, gcp_crs = [], None

    side = sideFiles(filename)
    method = "side files" if side else "image file"
    rmse = None
    if transform.is_identity and gcps:
        src = [(g.col, g.row) for g in gcps]
        dst = [(g.x, g.y) for g in gcps]
        A, res = fitAffine(src, dst)
        transform = matrixToAffine(A)
        rmse = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
        crs = crs or gcp_crs
        method = "control points"
    if transform.is_identity:
        raise GeoreferenceError("The file has no georeference. In ArcGIS Pro, save the georeference (Save as New) "
                                "and choose the new file here.")
    if crs is None and epsg:
        crs = CRS.from_epsg(int(epsg))
    return {"crs": crs, "transform": transform, "width": width, "height": height, "method": method, "rmse": rmse,
            "side_files": side}


def fromFile(image, filename, epsg=None, progress=None):
    """
    TagLab's georeference of the map from a file georeferenced in ArcGIS Pro:
      - the map's own image file (ArcGIS Pro's Save: side files): its georeference is the map's;
      - any other georeferenced raster showing the map (Save as New writes a new, resampled raster): the map is
        matched to it (automatic markers) and an affine transform is fitted.

    Returns a dictionary: crs, transform, method, source, rmse_mm, markers, warnings.
    """
    own = [ch.filename for ch in image.channels if ch.filename]
    if any(sameFile(filename, f) for f in own):
        info = readFile(filename, epsg)
        if (info["width"], info["height"]) != (image.width, image.height):
            raise GeoreferenceError("The file does not have the size of the map.")
        if info["crs"] is None:
            raise GeoreferenceError("The file has a georeference but no coordinate system: type its EPSG code.")
        rmse_mm = None
        if info["rmse"] is not None:
            mx, my = metresPerUnit(info["crs"], info["transform"], (image.width, image.height))
            rmse_mm = info["rmse"] * math.sqrt(mx * my) * 1000.0
        method = "control points" if info["method"] == "control points" else "sidecar"
        return {"crs": info["crs"], "transform": info["transform"], "method": method, "source": filename,
                "rmse_mm": rmse_mm, "markers": None, "warnings": []}
    return fromMatch(image, filename, epsg, progress)


def fromMatch(image, filename, epsg=None, progress=None):
    """
    Match the map to a georeferenced raster (e.g. ArcGIS Pro's Save as New) and fit the affine transform map pixel ->
    world through the matched points.
    """
    info = readFile(filename, epsg)
    crs, tf = info["crs"], info["transform"]
    if crs is None:
        raise GeoreferenceError("The file has no coordinate system: type its EPSG code.")
    if crs.is_geographic:
        raise GeoreferenceError("The file is in latitude / longitude. In ArcGIS Pro, set the map to a projected "
                                "coordinate system (e.g. WGS 84 / UTM zone 20N) and save the raster again.")
    mx, my = metresPerUnit(crs)
    px_file = math.sqrt(abs(tf.a * tf.e - tf.b * tf.d)) * math.sqrt(mx * my) * 1000.0
    px_map = image.pixelSize()
    map_file = mapFile(image)
    if map_file is None:
        raise GeoreferenceError("The map has no image file.")

    try:
        pairs, minfo = Alignment.autoMarkers(map_file, filename, px_map, px_file, progress=progress)
    except Exception as e:
        raise GeoreferenceError("The map could not be recognised in the file ({}). Check that the file shows the "
                                "same area, and that its georeference was saved in ArcGIS Pro.".format(e))
    if len(pairs) < 3:
        raise GeoreferenceError("The map could not be recognised in the file ({} matching points). Check that the "
                                "file shows the same area.".format(len(pairs)))

    map_px = Alignment.mmToPixels([p[0] for p in pairs], px_map)
    file_px = Alignment.mmToPixels([p[1] for p in pairs], px_file)
    world = pixelToWorld(tf, file_px)

    # robust: drop the points far from the fit of the others (autoMarkers' RANSAC already removed most); never
    # those within half a pixel of the file
    floor = 0.5 * px_file / 1000.0 / math.sqrt(mx * my)
    keep = np.ones(len(pairs), dtype=bool)
    for _ in range(3):
        A, res = fitAffine(map_px[keep], world[keep])
        err = np.linalg.norm(Alignment.applyMatrix(A, map_px) - world, axis=1)
        med = np.median(err[keep])
        mad = np.median(np.abs(err[keep] - med))
        new_keep = err <= max(med + 3.0 * 1.4826 * mad, 2.0 * med, floor)
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    A, res = fitAffine(map_px[keep], world[keep])
    rmse_mm = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))) * math.sqrt(mx * my) * 1000.0

    warnings = []
    if keep.sum() < MATCH_MIN_MARKERS:
        warnings.append("Only {} matching points: check the result.".format(int(keep.sum())))
    if rmse_mm > max(MATCH_MAX_RMSE_MM, 1.5 * px_file):
        warnings.append("The matching points disagree by {:.1f} mm (RMSE).".format(rmse_mm))
    return {"crs": crs, "transform": centreToCorner(A), "method": "matched", "source": filename,
            "rmse_mm": rmse_mm, "markers": int(keep.sum()), "warnings": warnings}


# ---------------------------------------------------------------------------------------------------------------------
# From typed coordinates
# ---------------------------------------------------------------------------------------------------------------------

_SEPARATORS = re.compile(r"[\u00b0\u00ba'\u2032\u2019\"\u2033\u201d:\s]+")


def parseCoordinate(text):
    """
    A coordinate typed as a decimal number (18.3215, -65.30, 650123.45, 18,3215) or as degrees / minutes / seconds
    (18°19'12.3"N, 65 18 12.3 W, 65°18.205'W, N18 19 12.3). S and W make it negative.
    """
    if text is None:
        raise ValueError("empty")
    t = str(text).strip()
    if not t:
        raise ValueError("empty")
    hemi = None
    if t[0] in "NSEWnsew":
        hemi, t = t[0].upper(), t[1:].strip()
    if t and t[-1] in "NSEWnsew":
        if hemi is not None:
            raise ValueError("two hemispheres: " + str(text))
        hemi, t = t[-1].upper(), t[:-1].strip()
    if "." not in t:
        t = t.replace(",", ".")
    parts = [p for p in _SEPARATORS.split(t) if p]
    if not 1 <= len(parts) <= 3:
        raise ValueError("not a coordinate: " + str(text))
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError("not a coordinate: " + str(text))
    if any(p.startswith(("-", "+")) for p in parts[1:]) or any(v >= 60 for v in values[1:]):
        raise ValueError("bad minutes or seconds: " + str(text))
    if len(parts) > 1 and float(values[0]) != int(values[0]):
        raise ValueError("degrees with decimals and minutes: " + str(text))
    value = abs(values[0]) + sum(v / 60.0 ** (i + 1) for i, v in enumerate(values[1:]))
    negative = parts[0].startswith("-")
    if hemi is not None:
        if negative:
            raise ValueError("a sign and a hemisphere: " + str(text))
        negative = hemi in "SW"
    return -value if negative else value


def utmZone(lon, lat):
    """
    EPSG code of the WGS 84 / UTM zone of a position.
    """
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def transformPoints(points, src_crs, dst_crs):
    """
    (c1, c2) points from src_crs to dst_crs. For latitude / longitude systems the points are (latitude, longitude),
    the order GPS units show; otherwise (x / easting, y / northing).
    """
    src_crs = CRS.from_user_input(src_crs)
    dst_crs = CRS.from_user_input(dst_crs)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xy = pts[:, ::-1] if src_crs.is_geographic else pts
    if src_crs == dst_crs:
        out = xy.copy()
    else:
        from rasterio.warp import transform as rio_transform
        xs, ys = rio_transform(src_crs, dst_crs, xy[:, 0].tolist(), xy[:, 1].tolist())
        out = np.stack([xs, ys], axis=1)
    return out[:, ::-1] if dst_crs.is_geographic else out


def fromCoordinates(image, points, input_crs, output_crs, fit=FIT_RIGID, angle_deg=0.0):
    """
    Fit the georeference of the map to points of the map with known coordinates.

    points: list of ((x, y) TagLab pixel coordinates, (c1, c2) coordinates in input_crs: latitude / longitude for a
    geographic system, else x / y). output_crs must be projected (metres or feet).
    fit: "rigid" keeps TagLab's pixel size (1 point: angle_deg gives the direction of the top of the map, degrees
    clockwise from north), "similarity" fits the scale too (>= 2 points), "affine" (>= 3 points) also stretch / shear.

    Returns a dictionary: crs, transform, residuals_m (per point, x / y in metres), errors_m, rmse_m, fit, warnings.
    """
    output_crs = CRS.from_user_input(output_crs)
    if output_crs.is_geographic:
        raise GeoreferenceError("Choose a projected coordinate system (in metres) for the map.")
    n = len(points)
    need = {FIT_RIGID: 1, FIT_SIMILARITY: 2, FIT_AFFINE: 3}[fit]
    if n < need:
        raise GeoreferenceError("This fit needs at least {} point{}.".format(need, "s" if need > 1 else ""))

    P = np.array([p[0] for p in points], dtype=np.float64).reshape(-1, 2) + 0.5    # corner convention
    W = transformPoints([p[1] for p in points], input_crs, output_crs)
    k = metresPerUnit(output_crs)[0] * 1000.0            # mm per unit of the output system
    px = image.pixelSize()

    # isotropic millimetres, world y turned down like the image's (so that the fits are rotations, not mirrors)
    Pm = P * px
    Wm = np.stack([W[:, 0] * k, -W[:, 1] * k], axis=1)
    origin = Wm.mean(axis=0)
    Wm0 = Wm - origin

    if fit == FIT_AFFINE:
        A, _ = fitAffine(Pm, Wm0)
    elif n == 1:
        t = math.radians(angle_deg)
        R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        A = np.hstack([R, (Wm0[0] - R @ Pm[0])[:, None]])
    else:
        A = Alignment.fitTransform(Wm0, Pm, with_scale=(fit == FIT_SIMILARITY))["matrix"]

    # back to the output system: world = F (A (P px) + origin) / k, with F = flip of y
    F = np.diag([1.0, -1.0])
    L = F @ A[:, :2] * px / k
    t = F @ (A[:, 2] + origin) / k
    transform = Affine(L[0, 0], L[0, 1], t[0], L[1, 0], L[1, 1], t[1])

    pred = np.stack([transform.a * P[:, 0] + transform.b * P[:, 1] + transform.c,
                     transform.d * P[:, 0] + transform.e * P[:, 1] + transform.f], axis=1)
    res = (pred - W) * (k / 1000.0)
    errors = np.linalg.norm(res, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2))) if n > 0 else 0.0

    warnings = []
    if fit == FIT_RIGID and n == 1:
        warnings.append("One point: the direction of the map comes from the rotation you typed.")
    if fit == FIT_SIMILARITY and n == 2:
        warnings.append("Two points fit exactly: there is no check of their accuracy.")
    if fit == FIT_AFFINE and n == 3:
        warnings.append("Three points fit exactly: there is no check of their accuracy.")
    return {"crs": output_crs, "transform": transform, "residuals_m": res.tolist(), "errors_m": errors.tolist(),
            "rmse_m": rmse, "fit": fit, "warnings": warnings}


# ---------------------------------------------------------------------------------------------------------------------
# Pixel grids (maps aligned with the Alignment Tool)
# ---------------------------------------------------------------------------------------------------------------------

def gridOf(image):
    g = image.metadata.get(GRID_KEY) if image.metadata else None
    if not g or "group" not in g:
        return None
    o = g.get("origin", [0, 0])
    return g["group"], (float(o[0]), float(o[1]))


def setGrid(image, group, origin):
    image.metadata = dict(image.metadata)
    image.metadata[GRID_KEY] = {"group": group, "origin": [float(origin[0]), float(origin[1])]}


def newGroup():
    return uuid.uuid4().hex[:8]


def ensureGrid(image):
    """
    The pixel grid of the map, creating a new group (origin 0, 0) if it has none.
    """
    g = gridOf(image)
    if g is None:
        setGrid(image, newGroup(), (0.0, 0.0))
        g = gridOf(image)
    return g


def gridOffset(a, b):
    """
    origin(b) - origin(a) if the two maps are on the same pixel grid, else None. Then
    transform(b) = transform(a) * translation(offset).
    """
    ga, gb = gridOf(a), gridOf(b)
    if ga is None or gb is None or ga[0] != gb[0]:
        return None
    return gb[1][0] - ga[1][0], gb[1][1] - ga[1][1]


def shifted(transform, dx, dy):
    return transform * Affine.translation(dx, dy)


def joinGrid(project, image, group, origin):
    """
    Put the map on the given grid with the given origin. The other maps of its old group (if any) follow it, so their
    relation to it is kept.
    """
    old = gridOf(image)
    if old is not None and old[0] != group:
        dx, dy = origin[0] - old[1][0], origin[1] - old[1][1]
        for other in project.images:
            if other is image:
                continue
            go = gridOf(other)
            if go is not None and go[0] == old[0]:
                setGrid(other, group, (go[1][0] + dx, go[1][1] + dy))
    setGrid(image, group, origin)


def samePixelSize(a, b):
    pa, pb = a.pixelSize(), b.pixelSize()
    return pa > 0 and pb > 0 and abs(pa / pb - 1.0) <= PIXEL_SIZE_TOLERANCE


def coRegistered(project, image):
    """
    The other maps of the project and how they relate to this one: a list of dictionaries with
    image, status ("same grid": recorded by the Alignment Tool; "to check": same pixel size, no record;
    "different pixel size") and offset (same grid only).
    """
    out = []
    for other in project.images:
        if other is image:
            continue
        offset = gridOffset(image, other)
        if offset is not None:
            out.append({"image": other, "status": "same grid", "offset": offset})
        elif samePixelSize(image, other):
            out.append({"image": other, "status": "to check", "offset": None})
        else:
            out.append({"image": other, "status": "different pixel size", "offset": None})
    return out


def _grayValid(gray):
    return gray > 0


def gridCheck(image_a, image_b, progress=None, stop=None):
    """
    Are the two maps on the same pixel grid (e.g. aligned with the Alignment Tool before TagLab recorded it)?
    A coarse offset on reduced versions, then windows at full resolution.

    Returns a dictionary: ok, offset (origin(b) - origin(a), whole pixels, when ok), residual_mm (median),
    measured (windows), reason (when not ok).
    """
    def report(p, msg):
        if progress is not None:
            progress(p, msg)

    if not samePixelSize(image_a, image_b):
        return {"ok": False, "reason": "different pixel size"}
    fa, fb = mapFile(image_a), mapFile(image_b)
    if not fa or not fb:
        return {"ok": False, "reason": "no image file"}
    px = image_a.pixelSize()
    wa, ha, wb, hb = image_a.width, image_a.height, image_b.width, image_b.height

    # coarse: both reduced by the same factor, on a common canvas
    f = max(1.0, max(wa, ha, wb, hb) / float(GRID_CHECK_MAX_SIDE))
    sa = (max(8, int(round(ha / f))), max(8, int(round(wa / f))))
    sb = (max(8, int(round(hb / f))), max(8, int(round(wb / f))))
    report(5, "Reading " + image_a.name)
    ga = Alignment._readGray(fa, out_shape=sa)
    if stop is not None and stop():
        return {"ok": False, "reason": "stopped"}
    report(25, "Reading " + image_b.name)
    gb = Alignment._readGray(fb, out_shape=sb)
    H, W = max(sa[0], sb[0]), max(sa[1], sb[1])
    A = np.zeros((H, W), np.uint8)
    B = np.zeros((H, W), np.uint8)
    A[:sa[0], :sa[1]] = ga
    B[:sb[0], :sb[1]] = gb
    valid = _grayValid(A) | _grayValid(B)
    dx, dy, response = Alignment.phaseShift(A, B, valid)
    # the content at (x, y) of a is at (x + d) of b
    cdx = dx * (wa / float(sa[1]))
    cdy = dy * (ha / float(sa[0]))
    if response < 0.02:
        return {"ok": False, "reason": "the maps do not look alike"}

    # fine: windows of a, at the coarse offset in b
    win = GRID_CHECK_WINDOW
    n = GRID_CHECK_WINDOWS
    shifts = []
    tried = 0
    ix, iy = int(round(cdx)), int(round(cdy))
    for r in range(n):
        for c in range(n):
            if stop is not None and stop():
                return {"ok": False, "reason": "stopped"}
            x0 = int((c + 0.5) * wa / n - win / 2)
            y0 = int((r + 0.5) * ha / n - win / 2)
            x0 = min(max(x0, 0), max(wa - win, 0))
            y0 = min(max(y0, 0), max(ha - win, 0))
            tried += 1
            report(40 + 55 * tried / float(n * n), "Comparing the maps..")
            bx, by = x0 + ix, y0 + iy
            if bx < 0 or by < 0 or bx + win > wb or by + win > hb or x0 + win > wa or y0 + win > ha:
                continue
            wa_ = Alignment._readGray(fa, window=Window(x0, y0, win, win))
            wb_ = Alignment._readGray(fb, window=Window(bx, by, win, win))
            v = _grayValid(wa_) & _grayValid(wb_)
            if v.mean() < 0.6 or wa_[v].std() < 3.0:
                continue
            sx, sy, resp = Alignment.phaseShift(wa_, wb_, v)
            if resp < 0.08 or abs(sx) > win / 4 or abs(sy) > win / 4:
                continue
            shifts.append((ix + sx, iy + sy))

    if len(shifts) < max(3, GRID_CHECK_MIN_FRACTION * n * n):
        return {"ok": False, "reason": "too few parts of the maps could be compared ({} of {})".format(len(shifts),
                                                                                                    n * n),
                "measured": len(shifts)}
    s = np.array(shifts)
    med = np.median(s, axis=0)
    spread_mm = float(np.median(np.linalg.norm(s - med, axis=1))) * px
    frac = np.abs(med - np.round(med))
    if spread_mm > GRID_CHECK_MAX_MM:
        return {"ok": False, "reason": "the maps do not line up ({:.1f} mm apart)".format(spread_mm),
                "residual_mm": spread_mm, "measured": len(shifts)}
    if frac.max() > GRID_CHECK_MAX_FRACTION:
        return {"ok": False, "reason": "not on the same pixel grid (shifted by {:.2f}, {:.2f} px)".format(med[0],
                                                                                                        med[1]),
                "residual_mm": spread_mm, "measured": len(shifts)}
    d = np.round(med)
    # b pixel = a pixel + d  ->  origin(b) - origin(a) = -d
    return {"ok": True, "offset": (-float(d[0]) + 0.0, -float(d[1]) + 0.0), "residual_mm": spread_mm,
            "measured": len(shifts)}


def applyToGroup(project, source_image, crs, transform, method, source="", rmse_mm=None, extra=None, targets=()):
    """
    Set the georeference on the map and on the target maps. targets: list of (image, offset, from_check) with
    offset = origin(target) - origin(source); maps that passed gridCheck (from_check True) join the grid of the
    source map. Returns the list of maps changed.
    """
    setGeoreference(source_image, crs, transform, method, source, source_image.id, rmse_mm, extra)
    changed = [source_image]
    if any(t[2] for t in targets):
        group, origin = ensureGrid(source_image)
    # the control points are on the source map: the other maps don't keep them
    target_extra = {k: v for k, v in extra.items() if k != "control_points"} if extra else None
    for image, offset, from_check in targets:
        if image is source_image:
            continue
        if from_check:
            joinGrid(project, image, group, (origin[0] + offset[0], origin[1] + offset[1]))
        setGeoreference(image, crs, shifted(transform, offset[0], offset[1]), method, source, source_image.id,
                        rmse_mm, target_extra)
        changed.append(image)
    return changed

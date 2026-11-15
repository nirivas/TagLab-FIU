"""
GeoPackage export for ArcGIS Pro and QGIS (Round 16). Pure: GDAL / OGR and numpy, no Qt, so that the command line
(taglab_gis.py) and the ArcGIS Pro toolbox use it too.

One GeoPackage holds:
- regions_<map>: one polygon (with holes) per region, in the map's georeference (Georeference.get), with the values
  of the data table (Annotation.export_data_table) and every region attribute
- points_<map>: the point annotations
- matches_<map1>_<map2>: the comparison tables (the rows of Correspondences.tidyTable), without geometry
- colonies: one row per region per map (the rows of Genet.colonyRows), without geometry
Next to it, <base>_<map>_ortho.vrt / _dem.vrt: small GDAL VRT files that point to the map's images with TagLab's
georeference (nothing is copied, the images are not changed).
"""

import os
import re
import time
import math
import datetime
import struct

import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr

from source import Georeference as GR
from source.Genet import Genet

REPLACE_ATTEMPTS = 5
REPLACE_WAIT_S = 0.2
TEXT_MIN_WIDTH = 255
NAME_MAX = 60

DEFAULT_OPTIONS = {"points": True, "matches": True, "colonies": True, "views": True, "working_area": True}

# field kinds -> OGR types
INT, REAL, TEXT, DATE, BOOL = "int", "real", "text", "date", "bool"

REGION_FIELDS = [
    ("region_id", INT, "Region id"),
    ("map", TEXT, "Map"),
    ("date", DATE, "Acquisition date"),
    ("class", TEXT, "Class"),
    ("colony", TEXT, "Colony (map:id of its first region)"),
    ("genet", INT, "Genet"),
    ("area_cm2", REAL, "Area (cm2)"),
    ("surf_area_cm2", REAL, "Surface area (cm2)"),
    ("perimeter_cm", REAL, "Perimeter (cm)"),
    ("centroid_x", REAL, "Centroid x (map pixels)"),
    ("centroid_y", REAL, "Centroid y (map pixels)"),
    ("note", TEXT, "Note"),
    ("to_check", BOOL, "Match to check"),
]

POINT_FIELDS = [
    ("point_id", INT, "Point id"),
    ("map", TEXT, "Map"),
    ("date", DATE, "Acquisition date"),
    ("class", TEXT, "Class"),
    ("note", TEXT, "Note"),
]

MATCH_KINDS = {"colony": TEXT, "genet": INT, "map1": TEXT, "date1": DATE, "id1": INT, "class1": TEXT,
               "area1_cm2": REAL, "surf_area1_cm2": REAL, "map2": TEXT, "date2": DATE, "id2": INT, "class2": TEXT,
               "area2_cm2": REAL, "surf_area2_cm2": REAL, "fate": TEXT, "split_fuse": TEXT, "change_cm2": REAL,
               "change_pct": REAL, "years": REAL, "to_check": BOOL}

COLONY_KINDS = {"colony": TEXT, "genet": INT, "map": TEXT, "date": DATE, "id": INT, "class": TEXT,
                "area_cm2": REAL, "surf_area_cm2": REAL}

ATTRIBUTE_KINDS = {"string": TEXT, "keyword": TEXT, "integer number": INT, "boolean": BOOL, "decimal number": REAL}

RESERVED = {"fid", "geom", "shape", "objectid", "oid", "shape_length", "shape_area"}


class GeoExportError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------------------------------------------------

def safeName(text, default="x"):
    """
    Letters, digits and underscores only, starting with a letter (valid in a GeoPackage, a file geodatabase and SQL
    without quotes).
    """
    name = re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_")
    name = re.sub(r"_+", "_", name)
    if not name:
        name = default
    if not name[0].isalpha():
        name = default + "_" + name
    return name[:NAME_MAX]


def uniqueName(name, used):
    """
    name, or name_2, name_3 ... not in used (case-insensitive); the result is added to used.
    """
    candidate, k = name, 2
    while candidate.lower() in used:
        suffix = "_{}".format(k)
        candidate = name[:NAME_MAX - len(suffix)] + suffix
        k += 1
    used.add(candidate.lower())
    return candidate


def attributeFields(attributes, used):
    """
    [(field name, kind, alias = the attribute's own name, attribute name)] for the region attributes of the project;
    attributes of unknown type are left out (as in the data table).
    """
    out = []
    for attribute in attributes:
        kind = ATTRIBUTE_KINDS.get(attribute.get("type"))
        if kind is None:
            continue
        base = safeName(attribute["name"], "attr")
        if base.lower() in RESERVED:
            base = base + "_attr"
        out.append((uniqueName(base, used), kind, attribute["name"], attribute["name"]))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------------------------------------------------

def isMissing(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, np.floating) and np.isnan(value):
        return True
    return False


def parseDate(text):
    try:
        return datetime.datetime.strptime(str(text), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def convert(value, kind):
    """
    The value for a field of the kind, or None (NULL).
    """
    if isMissing(value):
        return None
    try:
        if kind == INT:
            return int(value)
        if kind == REAL:
            v = float(value)
            return None if math.isnan(v) else v
        if kind == BOOL:
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("true", "yes", "1"):
                    return True
                if v in ("false", "no", "0"):
                    return False
                return None
            return bool(value)
        if kind == DATE:
            return value if isinstance(value, datetime.date) else parseDate(value)
    except (TypeError, ValueError):
        return None
    return str(value)


def regionValues(blob, image, attribute_fields, colony=None):
    """
    The values of a region, in the order of REGION_FIELDS + attribute_fields. Areas, perimeter and centroid are
    computed and rounded as in the data table (Annotation.export_data_table); a surface area not computed and an
    attribute without a value are None (NULL) instead of 0.
    """
    px = image.pixelSize()
    surface = float(blob.surface_area or 0.0)
    values = [blob.id, image.name, parseDate(image.acquisition_date), blob.class_name, colony, blob.genet,
              round(blob.area * px * px / 100, 2),
              round(surface * px * px / 100, 2) if surface > 0.0 else None,
              round(blob.perimeter * px / 10, 1),
              round(float(blob.centroid[0]), 1), round(float(blob.centroid[1]), 1),
              blob.note or None, bool(blob.correspondence_to_check)]
    data = blob.data or {}
    for name, kind, alias, key in attribute_fields:
        values.append(convert(data.get(key), kind))
    return values


def inferKind(values):
    """
    The field kind for the values of a point attribute (bool, int, real, else text).
    """
    present = [v for v in values if not isMissing(v)]
    if not present:
        return TEXT
    if all(isinstance(v, (bool, np.bool_)) for v in present):
        return BOOL
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in present):
        return INT
    if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
           for v in present):
        return REAL
    return TEXT


# ---------------------------------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------------------------------

def signedArea(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygonWkb(contours, transform):
    """
    WKB of a polygon: the first contour is the outline, the others are holes (TagLab pixel coordinates, centres at
    integers). Rings are closed and oriented as in the OGC rules (outline counter-clockwise, holes clockwise, in
    world coordinates). None if the outline has fewer than 3 points.
    """
    rings = []
    for k, contour in enumerate(contours):
        c = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        if len(c) < 3:
            if k == 0:
                return None
            continue
        pts = GR.pixelToWorld(transform, c)
        area = signedArea(pts)
        if (k == 0) != (area > 0):
            pts = pts[::-1]
        pts = np.vstack([pts, pts[:1]])
        rings.append(pts)
    out = [struct.pack("<BII", 1, 3, len(rings))]
    for pts in rings:
        out.append(struct.pack("<I", len(pts)))
        out.append(np.ascontiguousarray(pts, dtype="<f8").tobytes())
    return b"".join(out)


def pointWkb(x, y, transform):
    p = GR.pixelToWorld(transform, [[x, y]])[0]
    return struct.pack("<BIdd", 1, 1, float(p[0]), float(p[1]))


def spatialReference(crs):
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs))
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    try:
        if srs.GetAuthorityCode(None) is None:
            srs.AutoIdentifyEPSG()
    except Exception:
        pass
    return srs


# ---------------------------------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------------------------------

def writeLayer(ds, name, fields, rows, geometries=None, srs=None, geom_type=ogr.wkbNone, description="",
               stop=None):
    """
    One layer (or table, geom_type wkbNone) in a single transaction. fields: [(name, kind, alias)]; rows: lists of
    values in that order (None = NULL); geometries: WKB per row (None = no geometry). Returns the number of features,
    or None when stopped.
    """
    options = ["DESCRIPTION=" + description] if description else []
    if geom_type != ogr.wkbNone:
        options += ["GEOMETRY_NAME=geom", "SPATIAL_INDEX=YES"]
    layer = ds.CreateLayer(name, srs if geom_type != ogr.wkbNone else None, geom_type, options=options)
    for k, (fname, kind, alias) in enumerate(fields):
        ogr_type = {INT: ogr.OFTInteger, REAL: ogr.OFTReal, TEXT: ogr.OFTString, DATE: ogr.OFTDate,
                    BOOL: ogr.OFTInteger}[kind]
        fd = ogr.FieldDefn(fname, ogr_type)
        if kind == BOOL:
            fd.SetSubType(ogr.OFSTBoolean)
        if kind == TEXT:
            longest = max([len(str(r[k])) for r in rows if r[k] is not None] or [0])
            fd.SetWidth(max(TEXT_MIN_WIDTH, longest))
        if alias and alias != fname:
            fd.SetAlternativeName(alias)
        layer.CreateField(fd)

    defn = layer.GetLayerDefn()
    ds.StartTransaction()
    count = 0
    try:
        for i, row in enumerate(rows):
            if stop is not None and i % 200 == 0 and stop():
                # committed, not rolled back: a rollback drops the new table but not its pending spatial index,
                # and GDAL then fails when it closes the file (which is removed anyway)
                ds.CommitTransaction()
                return None
            feat = ogr.Feature(defn)
            for k, (fname, kind, alias) in enumerate(fields):
                v = row[k]
                if v is None:
                    feat.SetFieldNull(k)
                elif kind == DATE:
                    feat.SetField(k, v.year, v.month, v.day, 0, 0, 0, 0)
                elif kind == BOOL:
                    feat.SetField(k, 1 if v else 0)
                elif kind == INT:
                    feat.SetField(k, int(v))
                elif kind == REAL:
                    feat.SetField(k, float(v))
                else:
                    feat.SetField(k, str(v))
            if geometries is not None and geometries[i] is not None:
                feat.SetGeometry(ogr.CreateGeometryFromWkb(geometries[i]))
            layer.CreateFeature(feat)
            count += 1
        ds.CommitTransaction()
    except Exception:
        try:
            ds.RollbackTransaction()
        except Exception:
            pass
        raise
    return count


def fieldAliases(fields):
    """
    {field name: alias} (ArcGIS Pro doesn't read the GeoPackage aliases: the toolbox sets them in the file
    geodatabase).
    """
    return {name: alias for name, kind, alias in fields if alias and alias != name}


def replaceFile(tmp, target):
    """
    os.replace with a few retries: a target open in ArcGIS Pro (or another program) can't be replaced.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_WAIT_S)


def removeQuietly(filename):
    try:
        if os.path.exists(filename):
            os.remove(filename)
    except OSError:
        pass


def lockedMessage(filename):
    return ("{} can't be replaced: it is open in another program, probably ArcGIS Pro or QGIS. Remove its layers "
            "from the map (or close the program) and export again, or choose another file name.").format(filename)


# ---------------------------------------------------------------------------------------------------------------------
# What to export
# ---------------------------------------------------------------------------------------------------------------------

def mapsForExport(project):
    """
    [(image, GeoRef or None)] in the project order; a map without a georeference can't be exported.
    """
    return [(image, GR.get(image)) for image in project.images]


def classVisible(project, class_name):
    label = project.labels.get(class_name) if project.labels else None
    return True if label is None else bool(getattr(label, "visible", True))


def regionsToExport(project, image, use_working_area):
    """
    The regions of the map that the export writes: inside the working area (entirely, as in the data table) when
    asked and set, and only the classes shown in the Labels panel (as the shapefile and data table exports).
    """
    annotations = image.annotations
    if use_working_area and project.working_area is not None:
        blobs = annotations.calculate_inner_blobs(project.working_area)
    else:
        blobs = annotations.seg_blobs
    return sorted([b for b in blobs if classVisible(project, b.class_name)], key=lambda b: b.id)


def pointsToExport(project, image, use_working_area):
    annotations = image.annotations
    if use_working_area and project.working_area is not None:
        points = annotations.calculate_inner_points(project.working_area)
    else:
        points = annotations.annpoints
    return sorted([p for p in points if classVisible(project, p.class_name)], key=lambda p: p.id)


def outputNames(project, images):
    """
    {image id: safe layer suffix} (unique; the map name, e.g. "2013" -> "2013" is used as "regions_2013").
    """
    used, names = set(), {}
    for image in images:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", str(image.name)).strip("_") or "map"
        names[image.id] = uniqueName(base[:40], used)
    return names


def viewNames(filename, image_name):
    base = os.path.splitext(filename)[0]
    return base + "_" + image_name + "_ortho.vrt", base + "_" + image_name + "_dem.vrt"


# ---------------------------------------------------------------------------------------------------------------------
# Views of the images
# ---------------------------------------------------------------------------------------------------------------------

def writeView(source, target, georef, image):
    """
    A VRT file that points to source with TagLab's georeference of the map (the source can be on another pixel size,
    e.g. a DEM saved at a lower resolution: the transform is scaled). Written to a temporary name, then moved.
    """
    tmp = os.path.splitext(target)[0] + ".tmp.vrt"
    removeQuietly(tmp)
    src = gdal.Open(source)
    try:
        w, h = src.RasterXSize, src.RasterYSize
    finally:
        src = None
    T = georef.transform
    if (w, h) != (image.width, image.height) and w > 0 and h > 0:
        from rasterio.transform import Affine
        T = T * Affine.scale(image.width / float(w), image.height / float(h))
    ds = gdal.Translate(tmp, source, format="VRT")
    try:
        ds.SetGeoTransform((T.c, T.a, T.b, T.f, T.d, T.e))
        ds.SetProjection(spatialReference(georef.crs).ExportToWkt())
    finally:
        ds = None
    # the VRT names the source relative to its own folder when it can: a temporary name in the same folder keeps it
    try:
        replaceFile(tmp, target)
    except PermissionError:
        removeQuietly(tmp)
        raise GeoExportError(lockedMessage(target))
    return target


# ---------------------------------------------------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------------------------------------------------

def exportGeoPackage(project, images, filename, options=None, progress=None, stop=None):
    """
    Write the GeoPackage filename for the maps images (they must be georeferenced). options: see DEFAULT_OPTIONS.
    progress(percent, text) and stop() are optional. Returns a report (see reportText), or None when stopped.
    Raises GeoExportError with a message for the user.
    """
    # GDAL errors as exceptions inside the export only (the rest of TagLab expects None on errors)
    with gdal.ExceptionMgr(useExceptions=True), ogr.ExceptionMgr(useExceptions=True), \
            osr.ExceptionMgr(useExceptions=True):
        return _exportGeoPackage(project, images, filename, options, progress, stop)


def _exportGeoPackage(project, images, filename, options, progress, stop):
    opts = dict(DEFAULT_OPTIONS)
    opts.update(options or {})

    def report_progress(p, text=""):
        if progress is not None:
            progress(p, text)

    def stopped():
        return stop is not None and stop()

    filename = os.path.abspath(filename)
    if not filename.lower().endswith(".gpkg"):
        filename += ".gpkg"
    images = [im for im in project.images if im in images]
    report = {"file": filename, "layers": [], "views": [], "skipped": [], "warnings": []}

    selected = []
    for image in images:
        georef = GR.get(image)
        if georef is None:
            report["skipped"].append((image.name, "not georeferenced"))
        else:
            selected.append((image, georef))
    if not selected:
        raise GeoExportError("None of the chosen maps is georeferenced. Use Project > Georeference Map first.")

    use_wa = bool(opts["working_area"]) and project.working_area is not None
    project.updateGenets()
    genet = Genet(project)
    colonies = genet.colonyNames()
    names = outputNames(project, [im for im, g in selected])

    tmp = os.path.splitext(filename)[0] + ".tmp.gpkg"
    removeQuietly(tmp)
    driver = ogr.GetDriverByName("GPKG")
    ds = driver.CreateDataSource(tmp)
    if ds is None:
        raise GeoExportError("Can't create {}.".format(tmp))
    ok = False
    try:
        # how much work: one unit per region / point, plus the tables
        work = [len(regionsToExport(project, im, use_wa)) + (len(pointsToExport(project, im, use_wa))
                if opts["points"] else 0) for im, g in selected]
        total = float(max(1, sum(work)))
        done = 0.0

        for (image, georef), units in zip(selected, work):
            srs = spatialReference(georef.crs)
            suffix = names[image.id]

            # regions
            used = {f[0].lower() for f in REGION_FIELDS} | RESERVED
            attr_fields = attributeFields(project.region_attributes.data, used)
            fields = [(n, k, a) for n, k, a in REGION_FIELDS] + [(n, k, a) for n, k, a, key in attr_fields]
            blobs = regionsToExport(project, image, use_wa)
            rows, geoms, degenerate = [], [], 0
            for blob in blobs:
                wkb = polygonWkb([blob.contour] + list(blob.inner_contours), georef.transform)
                if wkb is None:
                    degenerate += 1
                    continue
                rows.append(regionValues(blob, image, attr_fields, colonies.get((image.id, blob.id))))
                geoms.append(wkb)
            report_progress(100.0 * done / total, "Regions of {}".format(image.name))
            count = writeLayer(ds, "regions_" + suffix, fields, rows, geoms, srs, ogr.wkbPolygon,
                               "TagLab regions of the map {}".format(image.name), stop)
            if count is None:
                return None
            report["layers"].append({"name": "regions_" + suffix, "kind": "regions", "map": image.name,
                                     "count": count, "aliases": fieldAliases(fields)})
            if degenerate:
                report["warnings"].append("{}: {} region(s) with fewer than 3 outline points left out.".format(
                    image.name, degenerate))
            done += len(blobs)

            # points
            if opts["points"]:
                points = pointsToExport(project, image, use_wa)
                keys = []
                for p in points:
                    for key in (p.data or {}):
                        if key not in keys:
                            keys.append(key)
                used = {f[0].lower() for f in POINT_FIELDS} | RESERVED
                data_fields = []
                for key in keys:
                    kind = inferKind([(p.data or {}).get(key) for p in points])
                    base = safeName(key, "attr")
                    if base.lower() in RESERVED:
                        base += "_attr"
                    data_fields.append((uniqueName(base, used), kind, key, key))
                fields = [(n, k, a) for n, k, a in POINT_FIELDS] + [(n, k, a) for n, k, a, key in data_fields]
                rows, geoms = [], []
                for p in points:
                    row = [p.id, image.name, parseDate(image.acquisition_date), p.class_name, p.note or None]
                    for n, kind, alias, key in data_fields:
                        row.append(convert((p.data or {}).get(key), kind))
                    rows.append(row)
                    geoms.append(pointWkb(p.coordx, p.coordy, georef.transform))
                if points:
                    report_progress(100.0 * done / total, "Points of {}".format(image.name))
                    count = writeLayer(ds, "points_" + suffix, fields, rows, geoms, srs, ogr.wkbPoint,
                                       "TagLab point annotations of the map {}".format(image.name), stop)
                    if count is None:
                        return None
                    report["layers"].append({"name": "points_" + suffix, "kind": "points", "map": image.name,
                                             "count": count, "aliases": fieldAliases(fields)})
                done += len(points)

        # comparison tables of two exported maps
        exported = {im.id for im, g in selected}
        if opts["matches"]:
            for key, corr in (project.correspondences or {}).items():
                if corr.source.id not in exported or corr.target.id not in exported:
                    continue
                table = corr.tidyTable(colonies)
                columns = list(table.columns)
                fields = [(c, MATCH_KINDS.get(c, TEXT), None) for c in columns]
                rows = [[convert(v, MATCH_KINDS.get(c, TEXT)) for c, v in zip(columns, r)]
                        for r in table.itertuples(index=False)]
                name = "matches_{}_{}".format(names[corr.source.id], names[corr.target.id])
                count = writeLayer(ds, name, fields, rows, description="TagLab comparison of {} and {}".format(
                    corr.source.name, corr.target.name), stop=stop)
                if count is None:
                    return None
                report["layers"].append({"name": name, "kind": "matches", "map": corr.source.name + " - " +
                                         corr.target.name, "count": count})

        if opts["colonies"]:
            fields = [(c, COLONY_KINDS[c], None) for c in Genet.COLONY_COLUMNS]
            rows = [[convert(v if v != "" else None, COLONY_KINDS[c]) for c, v in zip(Genet.COLONY_COLUMNS, r)]
                    for r in genet.colonyRows([im for im, g in selected], use_working_area=use_wa)]
            count = writeLayer(ds, "colonies", fields, rows, description="TagLab colonies: one row per region per "
                               "map", stop=stop)
            if count is None:
                return None
            report["layers"].append({"name": "colonies", "kind": "colonies", "map": "", "count": count})

        ok = True
    finally:
        ds = None
        if not ok:
            removeQuietly(tmp)

    try:
        replaceFile(tmp, filename)
    except PermissionError:
        removeQuietly(tmp)
        raise GeoExportError(lockedMessage(filename))

    if opts["views"]:
        for image, georef in selected:
            ortho, dem = viewNames(filename, names[image.id])
            for channel, target in ((image.getRGBChannel(), ortho), (image.getDEMChannel(), dem)):
                if channel is None or not channel.filename:
                    continue
                if not os.path.exists(channel.filename):
                    report["warnings"].append("{}: {} not found, no view written.".format(image.name,
                                                                                      channel.filename))
                    continue
                writeView(channel.filename, target, georef, image)
                report["views"].append(target)

    if use_wa:
        report["working_area"] = list(project.working_area)
    report_progress(100.0, "Done")
    return report


def reportText(report):
    """
    The message shown after the export.
    """
    lines = ["Written: {}".format(report["file"]), ""]
    for layer in report["layers"]:
        what = {"regions": "regions", "points": "points", "matches": "rows", "colonies": "rows"}[layer["kind"]]
        lines.append("  {}: {} {}".format(layer["name"], layer["count"], what))
    if report["views"]:
        lines.append("")
        lines.append("Image views (open them in ArcGIS Pro like any raster):")
        for v in report["views"]:
            lines.append("  " + os.path.basename(v))
    if report.get("working_area"):
        lines.append("")
        lines.append("Only the regions and points inside the working area.")
    for name, reason in report["skipped"]:
        lines.append("Skipped {}: {}.".format(name, reason))
    for w in report["warnings"]:
        lines.append("Warning: " + w)
    return "\n".join(lines)

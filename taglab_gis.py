"""
TagLab for GIS: the command line used by the ArcGIS Pro toolbox (arcgis/TagLab.pyt). It runs in TagLab's own Python,
without the TagLab window, on a project that is closed in TagLab.

  python taglab_gis.py export --project P.json --out O.gpkg [--maps ID ...] [--no-points] [--no-matches]
                              [--no-colonies] [--no-views] [--all-regions] [--report R.json]
      the GeoPackage of File > Export > Export Regions As GeoPackage (source/GeoExport.py)

  python taglab_gis.py georeference --project P.json --map ID --file RASTER [--epsg N] [--no-group] [--overwrite]
                                    [--report R.json]
      Project > Georeference Map > From an ArcGIS file (source/Georeference.py): the map's own image after ArcGIS
      Pro's Save, or any raster made with Save as New (the map is matched to it). The maps on the same pixel grid
      (Alignment Tool) get it too, unless --no-group. A copy of the project is kept as
      <project>.before_georef_<date>_<time>.json, then the project is saved.

Output: lines "PROGRESS <0-100> <text>", "MESSAGE <text>", "WARNING <text>", "ERROR <text>" (UTF-8), and with
--report a JSON file with the result. Exit code: 0 done, 1 error, 2 nothing done.
"""

# rasterio must be loaded before PyQt5 (see TagLab.py)
import rasterio

import os
import sys
import json
import shutil
import argparse
import datetime
import traceback

TAGLAB_DIR = os.path.dirname(os.path.abspath(__file__))
EXIT_DONE, EXIT_ERROR, EXIT_NOTHING = 0, 1, 2


class CommandError(Exception):
    def __init__(self, message, code=EXIT_ERROR):
        super(CommandError, self).__init__(message)
        self.code = code


def say(kind, text):
    for line in str(text).splitlines() or [""]:
        print("{} {}".format(kind, line), flush=True)


def progress(percent, text=""):
    print("PROGRESS {:d} {}".format(int(max(0, min(100, percent))), text), flush=True)


def noDialogs(missing):
    """
    No TagLab window here: a file dialog (loadProject asks for a missing image) records the file and returns
    nothing, a message box raises.
    """
    from PyQt5.QtWidgets import QFileDialog, QMessageBox

    def openFile(parent, caption, *args, **kwargs):
        missing.append(caption.replace("Couldn't find ", "").replace(" please select it:", ""))
        return "", ""

    def message(*args, **kwargs):
        raise CommandError(" ".join(str(a) for a in args[1:3] if isinstance(a, str)))

    QFileDialog.getOpenFileName = staticmethod(openFile)
    for name in ("critical", "warning", "information", "question"):
        setattr(QMessageBox, name, staticmethod(message))
    QMessageBox.exec = QMessageBox.exec_ = message


def openProject(filename):
    from source.Project import loadProject
    if not os.path.isfile(filename):
        raise CommandError("The project {} doesn't exist.".format(filename))
    default_dict = "dictionaries/scripps.json"
    if not os.path.exists(os.path.join(TAGLAB_DIR, default_dict)):
        default_dict = "FDEP.json"
    missing = []
    noDialogs(missing)
    project = loadProject(TAGLAB_DIR, filename, default_dict)
    return project, missing


def findMap(project, key):
    for image in project.images:
        if image.id == key:
            return image
    for image in project.images:
        if image.name == key:
            return image
    raise CommandError("The project has no map '{}'. Its maps: {}.".format(
        key, ", ".join("'{}'".format(im.id) for im in project.images)))


def exists(filename):
    return bool(filename) and os.path.exists(filename)


# ---------------------------------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------------------------------

def commandExport(args):
    from source import GeoExport
    project, missing = openProject(args.project)
    for f in missing:
        say("WARNING", "Image file not found: {} (its view is not written).".format(f))
    images = [findMap(project, key) for key in args.maps] if args.maps else list(project.images)
    options = {"points": not args.no_points, "matches": not args.no_matches, "colonies": not args.no_colonies,
               "views": not args.no_views, "working_area": not args.all_regions}
    progress(0, "Exporting {}".format(os.path.basename(args.project)))
    try:
        report = GeoExport.exportGeoPackage(project, images, args.out, options, progress=progress)
    except GeoExport.GeoExportError as e:
        raise CommandError(str(e))
    say("MESSAGE", GeoExport.reportText(report))
    report["skipped"] = [{"map": m, "reason": r} for m, r in report["skipped"]]
    report["maps"] = [{"id": im.id, "name": im.name, "layer": l["name"]} for im in images
                      for l in report["layers"] if l["kind"] == "regions" and l["map"] == im.name]
    report["warnings"] = list(report["warnings"]) + ["Image file not found: " + f for f in missing]
    return EXIT_DONE, report


# ---------------------------------------------------------------------------------------------------------------------
# georeference
# ---------------------------------------------------------------------------------------------------------------------

def commandGeoreference(args):
    from source import Georeference as GR
    project, missing = openProject(args.project)
    image = findMap(project, args.map)
    if not os.path.isfile(args.file):
        raise CommandError("The raster {} doesn't exist.".format(args.file))
    own = GR.mapFile(image)
    if not exists(own):
        raise CommandError("The image file of the map '{}' was not found: {}.".format(image.id, own))

    progress(0, "Reading {}".format(os.path.basename(args.file)))
    try:
        result = GR.fromFile(image, args.file, args.epsg,
                             progress=lambda p, text="": progress(0.6 * p, text or "Matching the map to the raster"))
    except GR.GeoreferenceError as e:
        raise CommandError(str(e))
    T = result["transform"]
    desc = GR.describe(result["crs"], T, (image.width, image.height), image.pixelSize())
    say("MESSAGE", "'{}': {}".format(image.id, GR.describeText(desc)))
    if result.get("markers"):
        say("MESSAGE", "Matched with {} points, RMSE {:.1f} mm.".format(result["markers"], result["rmse_mm"]))
    for w in list(result.get("warnings", [])) + desc["warnings"]:
        say("WARNING", w)

    # the maps on the same pixel grid
    targets, skipped = [], []
    if not args.no_group:
        others = GR.coRegistered(project, image)
        to_check = [c for c in others if c["status"] == "to check"]
        done = 0
        for c in others:
            other = c["image"]
            if c["status"] == "same grid":
                targets.append((other, c["offset"], False))
            elif c["status"] == "to check":
                if not exists(GR.mapFile(other)):
                    skipped.append({"map": other.id, "reason": "image file not found"})
                    continue
                base = 60 + 30.0 * done / max(1, len(to_check))
                progress(base, "Comparing '{}' with '{}'".format(other.id, image.id))
                check = GR.gridCheck(image, other, progress=lambda p, text="": progress(
                    base + 30.0 * p / 100.0 / max(1, len(to_check)), text))
                done += 1
                if check["ok"]:
                    targets.append((other, check["offset"], True))
                else:
                    skipped.append({"map": other.id, "reason": "not on the same pixel grid ({})".format(
                        check.get("reason", ""))})
            else:
                skipped.append({"map": other.id, "reason": c["status"]})

    # a different georeference is replaced only with --overwrite
    new_t = {image.id: T}
    for other, offset, _ in targets:
        new_t[other.id] = GR.shifted(T, offset[0], offset[1])
    replaced = []
    for im in [image] + [t[0] for t in targets]:
        g = GR.get(im)
        if g is not None and (g.crs != result["crs"] or not g.transform.almost_equals(new_t[im.id], 1e-9)):
            replaced.append(im.id)
    report = {"map": image.id, "method": result["method"], "rmse_mm": result.get("rmse_mm"),
              "markers": result.get("markers"), "crs": desc["crs"], "epsg": desc["epsg"],
              "pixel_mm": desc["pixel_mm"], "top_bearing": desc["top_bearing"], "description": GR.describeText(desc),
              "warnings": list(result.get("warnings", [])) + desc["warnings"], "skipped": skipped,
              "replaced": replaced, "updated": [], "backup": None}
    if replaced and not args.overwrite:
        say("ERROR", "These maps already have a different georeference: {}. Nothing was changed; tick 'Replace "
                     "existing georeferences' (--overwrite) to replace it.".format(", ".join(replaced)))
        return EXIT_NOTHING, report

    progress(92, "Saving the project")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = "{}.before_georef_{}.json".format(os.path.splitext(args.project)[0], stamp)
    shutil.copy2(args.project, backup)
    changed = GR.applyToGroup(project, image, result["crs"], T, result["method"], result.get("source", ""),
                              result.get("rmse_mm"), None, targets)
    project.save(args.project, repair=False)
    report["updated"] = [im.id for im in changed]
    report["offsets"] = {other.id: list(offset) for other, offset, _ in targets}
    report["backup"] = backup
    say("MESSAGE", "Georeference set on: {}. The project is saved; the previous version is {}.".format(
        ", ".join("'{}'".format(i) for i in report["updated"]), os.path.basename(backup)))
    for s in skipped:
        say("MESSAGE", "Not applied to '{}': {}.".format(s["map"], s["reason"]))
    progress(100, "Done")
    return EXIT_DONE, report


# ---------------------------------------------------------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser(description="TagLab for GIS (used by the ArcGIS Pro toolbox)")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("export", help="write a GeoPackage")
    e.add_argument("--project", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--maps", nargs="*", default=None, help="map ids or names (default: all)")
    e.add_argument("--no-points", action="store_true")
    e.add_argument("--no-matches", action="store_true")
    e.add_argument("--no-colonies", action="store_true")
    e.add_argument("--no-views", action="store_true")
    e.add_argument("--all-regions", action="store_true", help="also the regions outside the working area")
    e.add_argument("--report")

    g = sub.add_parser("georeference", help="georeference a map from a raster georeferenced in ArcGIS Pro")
    g.add_argument("--project", required=True)
    g.add_argument("--map", required=True, help="map id or name")
    g.add_argument("--file", required=True)
    g.add_argument("--epsg", type=int, default=None, help="for a raster without a coordinate system")
    g.add_argument("--no-group", action="store_true", help="only this map")
    g.add_argument("--overwrite", action="store_true", help="replace a different georeference")
    g.add_argument("--report")
    return p


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parser().parse_args(argv)
    args.project = os.path.abspath(args.project)
    if getattr(args, "out", None):
        args.out = os.path.abspath(args.out)
    if getattr(args, "file", None):
        args.file = os.path.abspath(args.file)
    if args.report:
        args.report = os.path.abspath(args.report)

    # TagLab's paths (channels, dictionaries) are relative to its folder
    sys.path.insert(0, TAGLAB_DIR)
    os.chdir(TAGLAB_DIR)

    code, report = EXIT_ERROR, {}
    try:
        if args.command == "export":
            code, report = commandExport(args)
        else:
            code, report = commandGeoreference(args)
        report["ok"] = code == EXIT_DONE
    except CommandError as e:
        say("ERROR", str(e))
        code, report = e.code, {"ok": False, "error": str(e)}
    except Exception as e:
        say("ERROR", "{}: {}".format(type(e).__name__, e))
        traceback.print_exc()
        code, report = EXIT_ERROR, {"ok": False, "error": "{}: {}".format(type(e).__name__, e),
                                    "traceback": traceback.format_exc()}
    report["command"] = args.command
    report["exit_code"] = code
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main())

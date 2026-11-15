# -*- coding: utf-8 -*-
"""
TagLab toolbox for ArcGIS Pro (Round 16 of the FIU lab version of TagLab).

Catalog pane > Toolboxes > Add Toolbox > this file. Keep it in the "arcgis" folder of TagLab: it runs
taglab_gis.py (next to TagLab.py) in TagLab's own Python 3.11, because arcpy (ArcGIS Pro's Python) can't load
TagLab's packages and TagLab's Python can't load arcpy. Close the project in TagLab before running a tool on it.

Tools:
- Import TagLab Project: the regions of every georeferenced map (in TagLab's class colours), the points, the
  comparison and colonies tables and the ortho / DEM views, as a GeoPackage, optionally copied to a file
  geodatabase, added to the current map.
- Georeference TagLab Maps: a map georeferenced in ArcGIS Pro (Imagery > Georeference, then Save as New, or Save
  on the map's own image) gives TagLab its georeference; the maps aligned to it in TagLab get it too.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess

import arcpy

TOOLBOX_DIR = os.path.dirname(os.path.abspath(__file__))
TAGLAB_DIR = os.path.dirname(TOOLBOX_DIR)
CLI = os.path.join(TAGLAB_DIR, "taglab_gis.py")
CREATE_NO_WINDOW = 0x08000000

# variables of ArcGIS Pro's own Python, GDAL and PROJ that would break TagLab's
DROP_VARIABLES = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONNOUSERSITE", "GDAL_DATA",
                  "GDAL_DRIVER_PATH", "PROJ_LIB", "PROJ_DATA", "PROJ_NETWORK", "GEOTIFF_CSV", "VIRTUAL_ENV")

_python_cache = {}
_project_cache = {}


# ---------------------------------------------------------------------------------------------------------------------
# TagLab's Python
# ---------------------------------------------------------------------------------------------------------------------

def cleanEnvironment():
    env = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if upper in DROP_VARIABLES or upper.startswith("CONDA") or upper.startswith("_CONDA"):
            continue
        env[key] = value
    path = env.get("PATH", env.get("Path", ""))
    kept = [p for p in path.split(os.pathsep) if p and "arcgis" not in p.lower()]
    for key in [k for k in env if k.upper() == "PATH"]:
        del env[key]
    env["PATH"] = os.pathsep.join(kept)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def pythonWorks(exe):
    """
    True if exe is a Python 3.11 outside ArcGIS that can load TagLab's packages.
    """
    if not exe or not os.path.isfile(exe) or "arcgis" in os.path.abspath(exe).lower():
        return False
    try:
        p = subprocess.run([exe, "-c", "import sys, rasterio, osgeo, numpy; print(sys.version_info[:2])"],
                           capture_output=True, text=True, env=cleanEnvironment(), timeout=120,
                           creationflags=CREATE_NO_WINDOW)
    except Exception:
        return False
    return p.returncode == 0 and "(3, 11)" in p.stdout


def pyLauncher311():
    launcher = shutil.which("py") or os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "py.exe")
    if not launcher or not os.path.isfile(launcher):
        return None
    try:
        p = subprocess.run([launcher, "-3.11", "-c", "import sys; print(sys.executable)"], capture_output=True,
                           text=True, env=cleanEnvironment(), timeout=60, creationflags=CREATE_NO_WINDOW)
    except Exception:
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def findTagLabPython(override=None):
    """
    TagLab's Python: the "TagLab Python" parameter, else TagLab's venv, else Python 3.11 of the py launcher, else
    a python on the PATH (not ArcGIS's).
    """
    key = override or ""
    if key in _python_cache:
        return _python_cache[key]
    candidates = []
    if override:
        candidates.append(override)
    candidates.append(os.path.join(TAGLAB_DIR, "venv", "Scripts", "python.exe"))
    candidates.append(pyLauncher311())
    env_path = cleanEnvironment()["PATH"]
    for name in ("python.exe", "python3.11.exe"):
        found = shutil.which(name, path=env_path)
        if found:
            candidates.append(found)
    for exe in candidates:
        if exe and pythonWorks(exe):
            _python_cache[key] = exe
            return exe
    raise RuntimeError("TagLab's Python (3.11, with TagLab's packages) was not found. Install TagLab with "
                       "install_windows.bat, or choose its python.exe in the 'TagLab Python' parameter "
                       "(Advanced).")


def runTagLab(arguments, python_override=None):
    """
    Run taglab_gis.py with the arguments: its PROGRESS / MESSAGE / WARNING / ERROR lines go to the tool's progress
    bar and messages. Returns (exit code, report dictionary).
    """
    if not os.path.isfile(CLI):
        raise RuntimeError("taglab_gis.py was not found in {}. Keep TagLab.pyt in the arcgis folder of TagLab."
                           .format(TAGLAB_DIR))
    exe = findTagLabPython(python_override)
    fd, report_file = tempfile.mkstemp(suffix=".json", prefix="taglab_report_")
    os.close(fd)
    other = []
    try:
        arcpy.SetProgressor("step", "TagLab", 0, 100, 1)
        p = subprocess.Popen([exe, "-u", CLI] + [str(a) for a in arguments] + ["--report", report_file],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", env=cleanEnvironment(), cwd=TAGLAB_DIR,
                             creationflags=CREATE_NO_WINDOW)
        for line in p.stdout:
            line = line.rstrip("\r\n")
            kind, _, text = line.partition(" ")
            if kind == "PROGRESS":
                value, _, label = text.partition(" ")
                try:
                    arcpy.SetProgressorPosition(int(value))
                except ValueError:
                    pass
                if label:
                    arcpy.SetProgressorLabel(label)
            elif kind == "MESSAGE":
                arcpy.AddMessage(text)
            elif kind == "WARNING":
                arcpy.AddWarning(text)
            elif kind == "ERROR":
                arcpy.AddError(text)
            elif line.strip():
                other.append(line)
            if getattr(arcpy.env, "isCancelled", False):
                p.kill()
                break
        code = p.wait()
        try:
            with open(report_file, encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, ValueError):
            report = {}
        if code not in (0, 2) and other:
            arcpy.AddMessage("TagLab's output:\n" + "\n".join(other[-40:]))
        return code, report
    finally:
        arcpy.ResetProgressor()
        try:
            os.remove(report_file)
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------------------------------
# The project file (read with json, without TagLab)
# ---------------------------------------------------------------------------------------------------------------------

def readProject(filename):
    """
    {"maps": [(id, name, georeferenced in TagLab, has a georeferenced file)], "labels": {...}, "name": ...} or
    None. Cached by modification time (updateParameters runs often).
    """
    try:
        stamp = os.path.getmtime(filename)
    except OSError:
        return None
    cached = _project_cache.get(filename)
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        with open(filename, encoding="utf-8") as f:
            data = json.load(f)
        maps = []
        for im in data.get("images", []):
            meta = im.get("metadata") or {}
            maps.append((str(im.get("id")), str(im.get("name") or im.get("id")), bool(meta.get("georeference")),
                         bool(im.get("georef_filename"))))
        info = {"maps": maps, "labels": data.get("labels") or {},
                "name": os.path.splitext(os.path.basename(filename))[0]}
    except Exception:
        info = None
    _project_cache[filename] = (stamp, info)
    return info


# ---------------------------------------------------------------------------------------------------------------------
# Layers in the map
# ---------------------------------------------------------------------------------------------------------------------

def workspaceItems(workspace):
    """
    {layer name: catalog path} of the feature classes and tables of a GeoPackage or file geodatabase (ArcGIS names
    GeoPackage layers "main.<name>").
    """
    old = arcpy.env.workspace
    arcpy.env.workspace = workspace
    try:
        names = (arcpy.ListFeatureClasses() or []) + (arcpy.ListTables() or [])
    finally:
        arcpy.env.workspace = old
    return {n.split(".")[-1]: os.path.join(workspace, n) for n in names}


def copyToGeodatabase(gpkg, gdb, report):
    """
    The layers of the GeoPackage into the file geodatabase (created if missing, layers replaced).
    """
    if not arcpy.Exists(gdb):
        arcpy.management.CreateFileGDB(os.path.dirname(gdb), os.path.basename(gdb))
    items = workspaceItems(gpkg)
    previous = arcpy.env.overwriteOutput
    arcpy.env.overwriteOutput = True
    try:
        for layer in report.get("layers", []):
            name = layer["name"]
            source = items.get(name)
            if source is None:
                arcpy.AddWarning("{} not found in the GeoPackage.".format(name))
                continue
            target = os.path.join(gdb, name)
            if layer["kind"] in ("regions", "points"):
                arcpy.conversion.ExportFeatures(source, target)
            else:
                arcpy.conversion.ExportTable(source, target)
            # ArcGIS doesn't read the GeoPackage's aliases (e.g. "Bleaching %" for the field Bleaching)
            names = {f.name.lower(): f.name for f in arcpy.ListFields(target)}
            for field, alias in (layer.get("aliases") or {}).items():
                if field.lower() in names:
                    try:
                        arcpy.management.AlterField(target, names[field.lower()], new_field_alias=alias)
                    except Exception:
                        pass
            arcpy.AddMessage("Copied {} to the geodatabase.".format(name))
    finally:
        arcpy.env.overwriteOutput = previous


def colourRegions(layer, labels):
    """
    Unique values on "class" with TagLab's fill (half transparent) and border colours.
    """
    sym = layer.symbology
    if not hasattr(sym, "renderer"):
        return
    sym.updateRenderer("UniqueValueRenderer")
    sym.renderer.fields = ["class"]
    for group in sym.renderer.groups:
        for item in group.items:
            label = labels.get(str(item.values[0][0])) if item.values and item.values[0] else None
            if not label:
                continue
            fill = list(label.get("fill") or [255, 255, 255])[:3]
            border = list(label.get("border") or fill)[:3]
            item.symbol.color = {"RGB": fill + [55]}
            item.symbol.outlineColor = {"RGB": border + [100]}
            item.symbol.outlineWidth = 1.5
    layer.symbology = sym


def addToMap(m, title, report, labels, workspace):
    """
    A group layer "TagLab - <project>" with a group per map (regions, points, ortho, DEM) and the tables, in the map
    m. workspace: the GeoPackage or the file geodatabase the layers are read from. Returns the group layer.
    """
    items = workspaceItems(workspace)
    for old in m.listLayers(title):
        if old.isGroupLayer:
            m.removeLayer(old)
    top = m.createGroupLayer(title)
    by_map = {}
    for layer in report.get("layers", []):
        by_map.setdefault(layer["map"], []).append(layer)
    views = report.get("views", [])
    for entry in report.get("maps", []):
        name = entry["name"]
        group = m.createGroupLayer(name, top)
        sources = []
        for layer in sorted(by_map.get(name, []), key=lambda l: 0 if l["kind"] == "points" else 1):
            if layer["kind"] in ("regions", "points") and layer["name"] in items:
                sources.append((layer["kind"], items[layer["name"]]))
        suffix = entry["layer"][len("regions_"):]
        for v in views:
            base = os.path.basename(v)
            if base.endswith("_" + suffix + "_ortho.vrt"):
                sources.append(("ortho", v))
            elif base.endswith("_" + suffix + "_dem.vrt"):
                sources.append(("dem", v))
        for kind, path in sources:
            added = m.addDataFromPath(path)
            added_name = added.name  # (not readable once the layer is removed)
            m.addLayerToGroup(group, added, "BOTTOM")
            m.removeLayer(added)
            new = [l for l in group.listLayers() if l.name == added_name][-1]
            if kind == "regions":
                new.name = "Regions"
                colourRegions(new, labels)
            elif kind == "points":
                new.name = "Points"
            elif kind == "ortho":
                new.name = "Ortho"
            else:
                new.name = "DEM"
    for layer in report.get("layers", []):
        if layer["kind"] in ("matches", "colonies") and layer["name"] in items:
            for old in m.listTables(layer["name"]):
                m.removeTable(old)
            m.addDataFromPath(items[layer["name"]])
    return top


# ---------------------------------------------------------------------------------------------------------------------
# Parameters shared by the tools
# ---------------------------------------------------------------------------------------------------------------------

def projectParameter():
    p = arcpy.Parameter(displayName="TagLab project", name="project", datatype="DEFile", parameterType="Required",
                        direction="Input")
    p.filter.list = ["json"]
    return p


def pythonParameter():
    p = arcpy.Parameter(displayName="TagLab Python (python.exe; found automatically when empty)",
                        name="taglab_python", datatype="DEFile", parameterType="Optional", direction="Input",
                        category="Advanced")
    p.filter.list = ["exe"]
    return p


def boolParameter(display, name, value, category=None):
    p = arcpy.Parameter(displayName=display, name=name, datatype="GPBoolean", parameterType="Optional",
                        direction="Input", category=category)
    p.value = value
    return p


def text(value):
    return value.valueAsText if hasattr(value, "valueAsText") else value


class Toolbox(object):
    def __init__(self):
        self.label = "TagLab"
        self.alias = "taglab"
        self.tools = [ImportTagLabProject, GeoreferenceTagLabMaps]


class ImportTagLabProject(object):
    def __init__(self):
        self.label = "Import TagLab Project"
        self.description = ("Exports the regions, points and tables of a TagLab project as a GeoPackage (and "
                            "optionally a file geodatabase) and adds them to the current map, in TagLab's class "
                            "colours. Only maps with a georeference are exported (TagLab: Project > Georeference "
                            "Map, or this toolbox's Georeference TagLab Maps).")
        self.canRunInBackground = False

    def getParameterInfo(self):
        project = projectParameter()
        maps = arcpy.Parameter(displayName="Maps", name="maps", datatype="GPString", parameterType="Optional",
                               direction="Input", multiValue=True)
        maps.filter.type = "ValueList"
        maps.filter.list = []
        points = boolParameter("Point annotations", "points", True, "Include")
        matches = boolParameter("Comparison tables", "matches", True, "Include")
        colonies = boolParameter("Colonies table", "colonies", True, "Include")
        views = boolParameter("Ortho and DEM views", "views", True, "Include")
        working = boolParameter("Only the regions inside the working area", "working_area", True, "Include")
        gpkg = arcpy.Parameter(displayName="Output GeoPackage", name="gpkg", datatype="DEFile",
                               parameterType="Required", direction="Output")
        gpkg.filter.list = ["gpkg"]
        save_gdb = boolParameter("Save to a file geodatabase", "save_gdb", False)
        gdb = arcpy.Parameter(displayName="File geodatabase", name="gdb", datatype="DEWorkspace",
                              parameterType="Optional", direction="Output")
        gdb.enabled = False
        add = boolParameter("Add to the current map", "add_to_map", True)
        return [project, maps, points, matches, colonies, views, working, gpkg, save_gdb, gdb, add,
                pythonParameter()]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        project, maps, gpkg, save_gdb, gdb = parameters[0], parameters[1], parameters[7], parameters[8], \
            parameters[9]
        if project.value:
            path = text(project)
            info = readProject(path)
            if info:
                ids = [m[0] for m in info["maps"]]
                maps.filter.list = ids
                if project.altered and not project.hasBeenValidated:
                    maps.value = [m[0] for m in info["maps"] if m[2] or m[3]] or ids
                base = os.path.splitext(path)[0]
                if not gpkg.altered or not gpkg.value:
                    gpkg.value = base + ".gpkg"
                if not gdb.altered or not gdb.value:
                    gdb.value = base + ".gdb"
        gdb.enabled = bool(save_gdb.value)
        return

    def updateMessages(self, parameters):
        project, maps = parameters[0], parameters[1]
        if project.value:
            info = readProject(text(project))
            if info is None:
                project.setErrorMessage("This is not a TagLab project file.")
            elif maps.values:
                chosen = [str(v) for v in maps.values]
                missing = [m[0] for m in info["maps"] if m[0] in chosen and not (m[2] or m[3])]
                if missing:
                    maps.setWarningMessage("Not georeferenced (they will be skipped): " + ", ".join(missing) +
                                           ". Use Georeference TagLab Maps first.")
        if parameters[8].value and not parameters[9].value:
            parameters[9].setErrorMessage("Choose the file geodatabase.")
        return

    def execute(self, parameters, messages):
        project = text(parameters[0])
        maps = [str(v) for v in (parameters[1].values or [])]
        gpkg = text(parameters[7])
        if not gpkg.lower().endswith(".gpkg"):
            gpkg += ".gpkg"
        args = ["export", "--project", project, "--out", gpkg]
        if maps:
            args += ["--maps"] + maps
        for index, flag in ((2, "--no-points"), (3, "--no-matches"), (4, "--no-colonies"), (5, "--no-views")):
            if not parameters[index].value:
                args.append(flag)
        if not parameters[6].value:
            args.append("--all-regions")
        code, report = runTagLab(args, text(parameters[11]) or None)
        if code != 0:
            # an error message makes the tool fail (raising would add a Python traceback to the messages)
            arcpy.AddError("TagLab could not export the project.")
            return

        workspace = gpkg
        if parameters[8].value:
            gdb = text(parameters[9])
            if not gdb.lower().endswith(".gdb"):
                gdb += ".gdb"
            copyToGeodatabase(gpkg, gdb, report)
            workspace = gdb

        if parameters[10].value:
            try:
                aprx = arcpy.mp.ArcGISProject("CURRENT")
            except Exception:
                arcpy.AddWarning("Not run from ArcGIS Pro: the layers were not added to a map.")
                return
            m = aprx.activeMap
            if m is None:
                arcpy.AddWarning("No map is open: add the layers from {} with Add Data.".format(workspace))
                return
            info = readProject(project) or {"labels": {}, "name": os.path.basename(project)}
            addToMap(m, "TagLab - " + info["name"], report, info["labels"], workspace)
            arcpy.AddMessage("Added to the map '{}'.".format(m.name))
        return


class GeoreferenceTagLabMaps(object):
    def __init__(self):
        self.label = "Georeference TagLab Maps"
        self.description = ("Gives a TagLab map the georeference of a raster georeferenced in ArcGIS Pro: the "
                            "map's own image after Save, or the new raster of Save as New (TagLab matches the map "
                            "to it). The maps aligned to it in TagLab (Alignment Tool) get it too. The project is "
                            "saved; the previous version is kept next to it.")
        self.canRunInBackground = False

    def getParameterInfo(self):
        project = projectParameter()
        image = arcpy.Parameter(displayName="Map", name="map", datatype="GPString", parameterType="Required",
                                direction="Input")
        image.filter.type = "ValueList"
        image.filter.list = []
        raster = arcpy.Parameter(displayName="Georeferenced raster", name="raster",
                                 datatype=["GPRasterLayer", "DERasterDataset"], parameterType="Required",
                                 direction="Input")
        group = boolParameter("Also the maps aligned to it in TagLab", "group", True)
        overwrite = boolParameter("Replace existing georeferences", "overwrite", False)
        epsg = arcpy.Parameter(displayName="EPSG code (only for a raster without a coordinate system)",
                               name="epsg", datatype="GPLong", parameterType="Optional", direction="Input")
        closed = boolParameter("The project is closed in TagLab", "closed", False)
        return [project, image, raster, group, overwrite, epsg, closed, pythonParameter()]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        project, image = parameters[0], parameters[1]
        if project.value:
            info = readProject(text(project))
            if info:
                ids = [m[0] for m in info["maps"]]
                image.filter.list = ids
                if ids and (not image.value or str(image.value) not in ids):
                    image.value = ids[0]
        return

    def updateMessages(self, parameters):
        if parameters[0].value and readProject(text(parameters[0])) is None:
            parameters[0].setErrorMessage("This is not a TagLab project file.")
        if not parameters[6].value:
            parameters[6].setErrorMessage("Close the project in TagLab first (TagLab would overwrite the "
                                          "georeference when it saves), then tick this box.")
        return

    def execute(self, parameters, messages):
        project = text(parameters[0])
        raster = parameters[2].value
        try:
            path = arcpy.Describe(raster).catalogPath
        except Exception:
            path = text(parameters[2])
        args = ["georeference", "--project", project, "--map", text(parameters[1]), "--file", path]
        if not parameters[3].value:
            args.append("--no-group")
        if parameters[4].value:
            args.append("--overwrite")
        if parameters[5].value:
            args += ["--epsg", str(int(parameters[5].value))]
        code, report = runTagLab(args, text(parameters[7]) or None)
        if code != 0:
            # an error message makes the tool fail (raising would add a Python traceback to the messages)
            arcpy.AddError("The georeference was not set." if code == 2 else
                           "TagLab could not georeference the map.")
            return
        arcpy.AddMessage("Run Import TagLab Project to add the georeferenced maps to ArcGIS Pro.")
        return

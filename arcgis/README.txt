TagLab toolbox for ArcGIS Pro (FIU lab version of TagLab)
=========================================================

TagLab.pyt is a Python toolbox for ArcGIS Pro with two tools:

  Import TagLab Project      the regions of every georeferenced map (drawn in TagLab's class colours), the
                            points, the comparison and colonies tables and the ortho / DEM views, as a GeoPackage
                            (optionally copied to a file geodatabase), added to the current map
  Georeference TagLab Maps  a map georeferenced in ArcGIS Pro gives TagLab its georeference; the maps aligned to
                            it in TagLab (Alignment Tool) get it too


What you need
-------------
- ArcGIS Pro 3.x.
- TagLab installed on the same computer (install_windows.bat). The tools run TagLab's own Python 3.11, because
  ArcGIS Pro's Python can't load TagLab's packages. They find it by themselves: TagLab's venv folder, else the
  Python 3.11 of the "py" launcher, else a Python 3.11 on the PATH. If they can't, choose TagLab's python.exe
  (for example <TagLab folder>\venv\Scripts\python.exe) in the "TagLab Python" parameter (Advanced).
- Keep TagLab.pyt in this "arcgis" folder: it runs taglab_gis.py from the folder above.


Adding the toolbox
------------------
1. In ArcGIS Pro, open the Catalog pane (View > Catalog Pane).
2. Right-click Toolboxes > Add Toolbox, and choose <TagLab folder>\arcgis\TagLab.pyt.
3. Expand TagLab.pyt and double-click a tool.
   To have it in every project: right-click TagLab.pyt > Add To Favorites.


Import TagLab Project
---------------------
Only maps with a georeference can be placed in ArcGIS. A map is georeferenced when its image file already has
one (a GeoTIFF from Metashape, for example), or after Georeference TagLab Maps, or after Project > Georeference
Map in TagLab (from an ArcGIS file or from typed coordinates).

Parameters:
  TagLab project            the project .json
  Maps                      the maps to export (the georeferenced ones are ticked)
  Include                   point annotations, comparison tables, colonies table, ortho and DEM views, and
                            "only the regions inside the working area" (all on)
  Output GeoPackage         default: next to the project, <project>.gpkg
  Save to a file geodatabase + File geodatabase
                            also copies everything into a .gdb (default <project>.gdb). The field aliases
                            ("Area (cm2)", the full names of the region attributes) show only in the .gdb:
                            ArcGIS Pro ignores them in a GeoPackage.
  Add to the current map    a group "TagLab - <project>" with a group per map (Regions, Points, Ortho, DEM),
                            plus the tables. Running the tool again replaces the group.

The ortho and DEM views (<project>_<map>_ortho.vrt / _dem.vrt) are small files that point to TagLab's images and
give them the georeference: nothing is copied. Keep them next to the GeoPackage.

A GeoPackage that is open in ArcGIS Pro can't be replaced: remove its layers from the map (or choose another
output file) before exporting again.


Georeference TagLab Maps
------------------------
1. Close the project in TagLab (TagLab would overwrite the change when it saves).
2. In ArcGIS Pro, add the map's orthomosaic to a map, select it, and use Imagery > Georeference: add control
   points on features whose coordinates you know (a basemap, GPS points, ground control points).
3. Save the result:
   - Save as New (recommended): writes a new, georeferenced raster. TagLab finds its map in it by matching the
     two images, so the new raster can be resampled, rotated or have another pixel size.
   - Save: writes the georeference next to the map's own image file (.aux.xml / world file). ArcGIS Pro's Save
     has a known shift problem with TIFF files, which is why Save as New is recommended.
4. Run Georeference TagLab Maps:
  TagLab project            the project .json
  Map                       the TagLab map that was georeferenced
  Georeferenced raster      the raster from step 3 (a layer of the map or a file)
  Also the maps aligned to it in TagLab
                            on: the maps aligned to this one with TagLab's Alignment Tool get the same
                            georeference, shifted by their offset. Maps aligned before this TagLab version are
                            checked first (their images must line up within 1 mm); the messages list those that
                            didn't.
  Replace existing georeferences
                            off: the tool stops, and changes nothing, when a map already has a different
                            georeference
  EPSG code                 only for a raster without a coordinate system
  The project is closed in TagLab
                            must be ticked

The tool saves the project; the previous version is kept next to it as
<project>.before_georef_<date>_<time>.json. The messages give the match error (RMSE, in mm) and any warning
(few matches, a large error, pixel sizes that differ from TagLab's).

TagLab's own measurements (areas, perimeters) keep using TagLab's pixel size (mm/px); the georeference only
places the map in ArcGIS.


Command line (without ArcGIS)
-----------------------------
The tools run taglab_gis.py, which also works by itself in TagLab's Python, from the TagLab folder:

  python taglab_gis.py export --project P.json --out P.gpkg [--maps 2013 2014] [--no-points] [--no-matches]
                              [--no-colonies] [--no-views] [--all-regions] [--report report.json]
  python taglab_gis.py georeference --project P.json --map 2013 --file georeferenced.tif [--epsg 32620]
                              [--no-group] [--overwrite] [--report report.json]

Exit codes: 0 done, 1 error, 2 nothing done.

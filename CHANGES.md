# Changes in the FIU lab version of TagLab

This version is based on the official TagLab 2026.4.1 (https://github.com/cnr-isti-vclab/TagLab) and is distributed
under the same GPL-3 licence. It changes the following.

## Data safety and crash fixes
- **Errors:** an error no longer closes TagLab. A message is shown and the details are written to `TagLab.log`.
- **Start-up crash:** on some Windows PCs, TagLab closed immediately without any message, because the map library
  (rasterio/GDAL) clashed with the window library (PyQt5). TagLab and `batch_processing.py` now load them in an
  order that works.
- **Updates:** TagLab no longer refuses to start when an official update exists. Updates come from this version's own
  repository instead (see "Installing and updating").
- **Exports:**
  - The CSV and shapefile exports work again with current NumPy versions.
  - Exporting the data table with a working area no longer deletes the point annotations outside it.
- **Projects** with several maps load the pixel size of every map correctly.
- **Year-to-year matching:**
  - New colonies are labelled "born" (not "dead") when a match is removed.
  - The genet CSV no longer has blank rows.
  - Projects that could no longer be opened because of a broken comparison table now open (see "Year-to-year
    matching" below).
- **Opening a project:** loading a second project in the same session could fail with "connection is not unique".
- **Custom attributes:** looking at a colony in the info panel no longer rounds its stored values.
- **Region names:** editing a region (Edit Border, merge, subtract, dilate, erode, crack and others) no longer resets
  its name.
- **Saving:** the project is first written to a temporary file next to it, which then replaces the project file in
  one step. An error or a crash during Save (a full disk, OneDrive holding the file) can no longer leave a cut-off
  project file: the previous version stays intact.
- **Autosave** (File > Settings.., every 1–15 minutes). It writes `<project>_autosave.json` next to the project and
  never touches the project file itself.
  - **No more crashes.** In the official TagLab, autosave closed TagLab when the project had not been saved to a file
    yet, or when the file could not be written (for example a project in a OneDrive folder while it syncs).
  - **It waits while TagLab is busy** (a progress window, a message, a held mouse button, a running job) and saves 30
    seconds later.
  - **It doesn't change the project, and writes only when something changed.** The comparison tables are checked when
    the file is opened, as for any project.
  - **If it can't write the file,** a message says so once, and it tries again at the next interval.
  - **The setting is remembered between sessions.** Before, autosave switched itself off at every start, and opening
    the official TagLab turned it off too.
  - **After a crash:** open `<project>_autosave.json` with File > Open Project, then use Save As to put it back under
    the project's own name. While the autosave file is open, autosave writes to that same file.

## DEM and surface area
- **Surface area:**
  - It is correct near DEM holes and map edges, and it is saved with the project.
  - It is reset when a colony is redrawn.
  - No large `slope.tif` file is written any more.
- **DEM > Calculate DEM Metrics** computes, for each colony: mean slope, elevation (mean/min/max), relief, height and
  volume above the surrounding substrate, and rugosity. The values are saved as region attributes and appear in the
  CSV and shapefile exports.
- **Plot rugosity:** with no region selected, Calculate DEM Metrics also computes values for the whole plot. The plot
  is the working area, or the whole map when there is none; only the part covered by the DEM counts.
  - `rugosity` (3D surface area / planar area) and `slope_deg` at the DEM's own resolution.
  - `rugosity_1cm` and `slope_1cm` on the DEM averaged to 1 cm cells. Use these to compare years or surveys whose
    DEMs have different resolutions: a finer DEM always gives a higher rugosity.
  - `relief_cm`: the height range of the plot after removing its overall tilt (1st to 99th percentile, so single
    spikes don't count).
  - Very steep slopes (walls of colonies and boulders) are counted, as in the usual surface-area ratio.
  - The values are shown at the end, saved with the map, and exported by the new **DEM > Export Plot Metrics
    (CSV)**: one row per map, with the area used, the DEM coverage and the DEM pixel size.
  - With regions selected, only those colonies are updated.
  - Speed: a 12000 × 12000 pixel DEM takes about 20 s on a laptop CPU.

## AI training and classification
- **Training dataset export:** the "Biologically-inspired" split works (it crashed before). It picks the validation
  and test areas correctly, and it is much faster on large maps.
- **Dataset export warning:** export warns about classes coloured black, which would be confused with the
  unlabelled background.
- **Training:**
  - It runs all the requested epochs and always saves a network.
  - It handles small validation and test sets, and trains all the intended layers.
  - It no longer needs an internet download.
- **Classification:**
  - It keeps TagLab open, puts the results on the correct map in split screen, and no longer forces a save.
  - `batch_processing.py` works.
- **Faster classification:**
  - Classification and its Preview use overlapping tiles with smooth blending.
  - On the official sample map, on a laptop CPU, this was 8.3 times faster: 2.3 minutes instead of 19 for a
    1500 × 1500 pixel area. The result was the same on 99.5% of the pixels.
  - The old method is still available: tick "Maximum quality" in the classifier window, or use
    `batch_processing.py --quality`.
- **Background classification:** classification and the Preview run in the background, with a progress window and a
  Cancel button (or Esc). The window does not freeze, and nothing is imported when you cancel.
- **Model choice during training:**
  - Training keeps the network with the best mean IoU, the average overlap across classes, instead of the best
    pixel accuracy.
  - The results window and the metrics files show the IoU of every class. The "mIoU" shown before was a
    frequency-weighted score.
- **New training option "Ignore unannotated pixels"** (in Train Your Network): use it for training areas that are
  annotated completely, so that colonies left unannotated are not taught as background.
- **NVIDIA GPUs:**
  - Training and classification use mixed precision.
  - Classification processes 4 tiles at a time.
  - The default training batch size is 8.
  - On Linux (e.g. a computing cluster), training loads the data in parallel.

## Alignment Tool (Project > Alignment Tool)
- **Accuracy:** the alignment is computed correctly. The old version could be off by centimetres even with perfect
  markers.
- **Map edges:** maps are no longer cut at the edges, and the result keeps every pixel of both maps.
- **Speed:** it is much faster and uses far less memory (about 2 GB instead of about 30 GB on large maps).
- **Auto Markers:** finds corresponding points between the two maps automatically, whatever the rotation between
  them. It reads only the parts of the maps it needs, so it uses much less memory on large maps.
- **Rotation:** the Alignment Tool rotates the map by whatever angle the markers give, so there is no need to rotate
  a map first. (The Rotate Map tool of earlier lab versions has been removed: rotating and then aligning gave the
  same result with an extra pair of files.)
- **Scale:** it can estimate and (optionally) apply the scale difference between two surveys.
- **Markers:** they are saved with the project.
- **DEMs:** they keep their no-data areas, and georeferencing is preserved.
- **Half a pixel:** markers placed by hand were counted from the pixel corners, while Auto Markers and the written
  maps used the pixel centres. With rotation or different pixel sizes, the result could be off by up to half a pixel.
  Everything now uses the same convention.
- **The aligned map replaces the original in the project** (Confirm):
  - It keeps the original's name and id, so colony names (e.g. `2014:45`), comparison tables, genets and exports stay
    the same. Only its files are new (suffix `_coreg`). The original files stay on disk; the project stops using
    them once you save it.
  - The comparison tables (areas recomputed), the "to check" marks and the alignment markers move to the new map,
    so the Alignment Tool can reopen the pair.
  - Grids, shape layers, sampling areas and the dataset export area are not carried over; the confirmation lists the
    ones the map has.
  - When the aligned map reaches beyond the left or top edge of the reference, Confirm asks what to do:
    - **Extend the reference:** the reference is replaced by a copy padded on that side (an exact shift: its colonies
      move with it; suffix `_ref`), and the working area is shifted too. The other maps of the project are not
      shifted, and the message lists them (re-align them if they were aligned to the reference).
    - **Crop to the reference:** the aligned map is cut at the reference's edge; nothing else changes.
    - **Cancel.**
  - Nothing is saved automatically.
- **Checking the alignment in the preview:**
  - **Views** (a list in each pane):
    - Blend.
    - **Red / cyan overlay:** the reference in red and the aligned map in cyan, on top of each other. Where they
      match, the image is grey; a shift shows as red and cyan edges side by side. Both maps are brought to the same
      brightness first, so a darker year doesn't tint the image. The slider sets the contrast.
    - Checkerboard.
    - **Swipe:** the reference left of a line that follows the mouse, the aligned map right of it.
    - **Flicker:** the two maps in turn, 1–8 times a second; a shift shows as a jump.
    - Difference.
  - **Zoom:** Fit and 1:1 buttons, `+` / `−` keys, and a label with the zoom and the mm per screen pixel.
- **Check alignment** (button in the preview): measures what is left of the misalignment on a grid of about 12 × 12
  cells over the overlap.
  - Each cell is coloured green (≤ 1 mm), yellow (≤ 3 mm), red (> 3 mm) or grey (not measurable: no texture, no
    data, or the place changed), with an arrow showing the direction.
  - The label gives the median, 90% and worst values in mm. When most cells point the same way, it says how far the
    aligned map sits from the reference, and that you can fix it with the markers or with Adjust Tx / Ty.
  - **Worst ▶** zooms on the worst cell, then the next worst. **Quality map** shows or hides the cells.
  - The result is marked out of date when the alignment changes.
  - The result is written to `TagLab.log` and included in the Confirm summary.
  - It takes a few seconds.
- **Using the DEM** (when both maps have one):
  - **Show: Colour / DEM shading** in the preview. Both DEMs are shaded with the same light (the light follows the
    rotation of the aligned map). Every view, and Check alignment, can then use the shading.
  - **Auto Markers "Match on": Colour / DEM relief / Colour + DEM.** The DEM relief (the height of each spot above
    its surroundings) shows the shapes of the colonies whatever their colour: it still works after bleaching, algae
    growth or a change of light, where the colour fails.
  - **Heights:** TagLab compares the two DEMs and fits the difference with a tilted plane, leaving out the places
    that changed (colonies that grew or disappeared). The label gives the height offset (cm), the tilt (°) and the
    spread that is left (mm).
  - **"Match DEM heights to the reference"** (on by default): the offset and the tilt are removed from the aligned
    DEM when it is written. The DEM metrics and surface areas of its colonies are then cleared: compute them again
    from the DEM menu.
  - **Height difference view:** aligned minus reference heights, in blue (lower), white (same) and red (higher). Its
    slider sets the colour range in cm. It shows whether the heights match, and where colonies grew or disappeared.
- **Fix:** changing the pair of maps after an alignment had been computed could give an error.

## Carry Regions Over (Compare > Carry Regions Over…)
Annotates a new timepoint from the previous one. The two maps must already be aligned, for example with the
Alignment Tool.
- **What it does:** each colony of the annotated map is placed at the same position on the new map or, if it was
  moved or turned (for example a nursery brick laid again), where it is found nearby. Its outline is fitted to the
  new image, and TagLab checks whether the colony is still there by comparing structure, colour, brightness, texture
  and, when both maps have a DEM, height.
- **Choices in the window:** the two maps, the classes to copy, the search for moved or turned colonies (on, up to
  15 cm, by default), outline fitting on or off, and the elevation change (only when both maps have a DEM).
- **How the maps are lined up:** same pixel grid, through the georeferencing, or a small offset found from the
  images. A small local shift (up to 5 cm) is also corrected around each colony.
- **Moved or turned colonies:** TagLab looks for each colony around its old place, turned by any angle and moved up
  to the search distance. It matches the old map around the colony (the colony and what it sits on, e.g. its
  brick) with the new map, and the elevation too when both maps have a DEM. A colony is moved only if it matches
  clearly better there than in place, and farther moves need a better match. The review's "Why" column says, e.g.,
  "moved 6 cm, turned 90° clockwise".
- **Outline fitting:** RITM (with one and with two clicks inside the colony) and GrabCut (colour) each propose an
  outline, and the one that agrees best with the others and with the old outline is kept. The outline can now
  shrink or grow (at most doubling its area): before, RITM was started from the old outline and gave it back almost
  unchanged. When the proposals disagree, the old outline is kept and the colony is marked `check` ("outline not
  re-fitted"). "Smaller" / "larger" reasons give the change, e.g. "smaller (-35% area)".
- **Result on the new map:**
  - Each colony keeps its class, name and custom attributes. DEM metrics are not copied.
  - Three new region attributes, exported in the CSV and shapefile:
    - `carryover`: `ok`, `check`, `possibly gone`, or `kept` after review
    - `carry_conf`: 0 to 1, how sure TagLab is that the colony is still there
    - `carry_from`: the id of the colony on the old map
  - Colonies already outlined on the new map are matched, not copied twice.
  - Each colony is matched to its old one in the comparison table. Doubtful ones are highlighted in dark red.
  - One Undo removes everything that was added.
- **Nothing is deleted automatically.** The doubtful colonies (`check` and `possibly gone`) are listed in a review
  window, most likely gone first. TagLab shows each one in both maps.
  - **N / P:** next or previous colony.
  - **K:** keep it.
  - **Delete:** remove it; it becomes "dead" in the comparison table.
  - **Move / correct:** selects the colony and switches to the Move Region tool, to drag it onto the right spot
    (`,` and `.` turn it) and correct its outline with positive and negative points (see the Move Region tool).
    Going to another colony cancels points that were not confirmed.
  - **Re-fit outline:** fits the outline of the colony to the image again (as above), e.g. after moving it.
  - Moving or re-fitting doesn't change the colony's status: press K to keep it.
  - You can also redraw it or change its class. Compare > Review Carried-over Colonies reopens the list.
  - These keys, and the Move Region keys, can be changed or put on a mouse button (see "Keyboard and mouse
    shortcuts").
- **Tests on the official 2013–2014 sample:**
  - About three quarters of the surviving colonies came out `ok`.
  - 85–94% of colonies removed on purpose from the image were flagged, as were nearly all the bleached or overgrown
    ones.
  - Colonies flattened in the DEM came out `possibly gone`.
  - With the new outline fitting, the carried outlines match the hand-made 2014 ones much better: median overlap
    (IoU) 0.81, against 0.61 with the old fitting. The search for moved colonies leaves 50 of the 53 reef colonies
    where they are and moves the other 3 by 2 cm or less.
- **Tests on Nursery1 (May → August 2026), where almost every brick was turned and moved:**
  - About 90 of the 99 outlines now land on their sponge (checked by eye); before, nearly all stayed where the
    sponge was in May.
  - Result: 34 `ok`, 61 `check`, 4 `possibly gone` (before: 5, 60 and 34).
  - The colonies TagLab could not find are flagged `check` or `possibly gone`.
  - Colony and brick patches of the May map, turned and moved on purpose, were found within 1° and 3 mm.
- **Speed:** about 1 s per colony on a laptop CPU (outline fitting about 0.8 s, the search about 0.25 s), less on an
  NVIDIA GPU.

## Year-to-year matching (Compare menu and the comparison table)
- **The comparison tables always match the colonies.** Every tool, and every Undo and Redo, keeps them up to date on
  both maps. Before, many edits left rows pointing to colonies that no longer existed. The table then went wrong, the
  matching tools could crash, and some saved projects could not be opened again.
  - A colony that is cut, merged, split, subtracted or redrawn keeps its match with the other year.
  - Undo and Redo give back the matches too, including the undo of a delete.
  - Colonies made by Watershed, SAM, the classifier, label-map or shapefile import and Create Negative appear in the
    table at once (as born or dead).
  - Deleting a colony that was part of a split or a fusion no longer links the other pieces together wrongly.
  - Colonies whose match TagLab could not keep are shown in dark red ("to check"). These marks are now saved with
    the project.
- **Old projects with a broken table** open again. The broken rows are repaired: the colony left without a match
  becomes dead or born and is marked to check. A message says how many rows were repaired. Save the project to keep
  the repair.
- **Correct grow / shrink / same:**
  - It is always computed in cm² from the current outlines, so it is right after a colony is redrawn and when the two
    maps have different pixel sizes.
  - Splits and fusions get the fate of the whole group (e.g. a colony that split and grew in total is "grow" on each
    row).
- **Compute Automatic Matches:**
  - It lines up the two maps like Carry Regions Over: same pixel grid, the georeferencing, or an offset found from
    the images. When the maps are not on the same grid, it says how they were placed and asks: Yes uses that
    placement, No compares the maps as they are (as before), Cancel stops.
  - The rules are unchanged: same class, not Empty, overlap of at least 60% of the smaller colony.
  - On the official 2013–2014 sample, "No" gives exactly the 44 matches of the sample's table. "Yes" applies the 4 mm
    offset found between the two images, which changes 2 borderline matches.
- **Region ids are never reused**, so a row can no longer point to the wrong colony after a deletion.
- **Class changes:** setting a colony to Empty changes only that year. Other class changes still apply to the colony
  in all years, and one Undo restores all of them.
- **Undo:** the classifier results and Create Negative are each one undo step. A Delete with nothing selected no
  longer clears the Redo history.
- **Comparison table:**
  - The Action column has a drop-down list (double click) on matched colonies, with the fate computed from the areas
    or "dead", for a colony that died in place. "dead" is kept; the other values follow the areas.
  - Split/Fuse is computed from the matches.
  - New filter "To check" (under "Search and filter"). The "Dead" filter now finds the dead colonies.
  - The Area / Surface Area choice is kept when you change the pair of maps. If a map has no surface area computed, a
    message says so and the table shows the planar area.
- **Export Matches (CSV):** one tidy file per pair of maps, ready for R or a spreadsheet:

  | Columns | Meaning |
  |---|---|
  | `colony` | Stable name of the colony: the map and id of its first region, e.g. `2013:45`. It does not change when other colonies are edited. |
  | `genet` | TagLab's lineage number (can change after edits) |
  | `map1, date1, id1, class1, area1_cm2, surf_area1_cm2` | The colony in the first map (surface area empty if not computed) |
  | `map2, date2, id2, class2, area2_cm2, surf_area2_cm2` | The same in the second map |
  | `fate` | grow / shrink / same / dead / born |
  | `split_fuse` | none / split / fuse |
  | `change_cm2, change_pct` | Area change; for splits and fusions, the change of the whole group |
  | `years` | Time between the two dates, for annual rates |
  | `to_check` | TRUE for the dark-red rows |

  The areas are rounded to 0.01 cm² and do not depend on what the table shows.
- **Export Genet Data as CSV:** one row per colony per map, with the columns `colony, genet, map, date, id, class,
  area_cm2, surf_area_cm2`.

## Points and CoralNet training data
Your region annotations can now be used to train a CoralNet source. The steps are: Points > Sample Points On This Map
(optional), then Points > Label Points from Regions, then Points > Export Tiled Data for CoralNet, then the upload (on
the CoralNet website or with the CoralNet Toolbox).
- **Points > Label Points from Regions…** works on the active map, inside the working area if one is set.
  - **Labelling:** each point gets the class of the region under it.
    - Where regions overlap, the smaller one wins (e.g. a sponge drawn on a sand region without a hole for it).
    - "Only the points still Empty" is on by default, so classes given by hand or imported from CoralNet are kept.
  - **New points:** one in every region that has no point, then more until every class has at least the number you
    choose (default 20), spread over its regions by area.
  - **Edge distance** (default 2 mm):
    - A point closer than this to another class, or to an area without regions, stays Empty and is not exported,
      because its label is uncertain. The map border does not count as an edge.
    - New points keep this distance from edges, and twice this distance from other points.
    - A region too small for that gets its point at its centre.
  - **Points on no region:** they stay Empty, or you can give them a class (e.g. Sand, when the sand is not drawn as
    regions).
  - **Before anything changes** you see what will change. Like the other point operations, this can't be undone with
    Ctrl+Z. Running it again with the same options gives the same points.
- **Export Tiled Data for CoralNet** now exports training data:
  - **Codes:** a table gives the CoralNet label code of each class (the short codes of your source's label set). The
    codes are saved with the project and filled in the next time.
  - **Tile size:** the size you enter is used (it was always 2048 before); the default is 2048 px.
  - **Tiles:** each point is in exactly one tile (points on a tile border used to be exported twice). Only tiles with
    points are saved, as JPEG at quality 95.
  - **The export folder** contains:
    - `tiles/`: the images
    - `coralnet_annotations.csv`: the columns Name, Row, Column, Label, as CoralNet's annotation upload expects
      (Empty points are left out)
    - `label_codes.csv`: each class, its code and its number of points
- **CoralNet Toolbox > Upload Training Data…** (after Authenticate):
  - You choose an export folder, and it uploads the tiles, then the annotations, to Source ID 1.
  - The source must already have a label set with your codes. Point annotations already on those images in the source
    are replaced.
  - Try it on a test source first.
- **Fixes:**
  - Export CoralNet Point Annotations no longer writes Label, Row, Column and Name into the points' saved data.
  - Error messages printed after the CoralNet Toolbox was closed went to its closed window.

## Move Region tool (toolbar button "Move region")
Moves colonies that are in the wrong place, for example after Carry Regions Over.
- **Shift + drag** a region to move it. A region that is not selected is selected first; dragging a selected one
  moves the whole selection. A plain drag still pans the map.
- **Arrow keys** move the selected regions by 1 pixel, **Shift + arrow** by 10 pixels.
- **`,` and `.`** turn the selected regions by 5° anticlockwise / clockwise around the centre of their box; **Shift**
  (`<` / `>`) turns them by 90°. Quarter turns are exact, and turning back gives the same outline. Several selected
  regions turn together. A region turned past the map border is moved back inside.
- **Esc** cancels a drag in progress.
- Regions move by whole pixels, so their area and perimeter don't change, and they can't leave the map.
- They keep their id, so their matches in the comparison tables and their genets stay. Class, name, note, custom
  attributes and the "to check" mark stay too. Surface area and DEM metrics are cleared, because they were measured
  at the old place.
- **Positive and negative points** correct the outline of the selected region with RITM, as in the
  positive/negative clicks tool, without changing tools:
  - **Shift + click** (without dragging): positive point, on a part that belongs to the colony.
  - **Shift + right click:** negative point, on a part that doesn't.
  - The outline is redrawn after each point. **Ctrl + Z** removes the last point, **Space** confirms, **Esc**
    cancels (the outline goes back to what it was).
  - Move and turn first: dragging, the arrows and the turn keys do nothing while there are points.
  - As in the RITM tool, the area RITM works on is the view at the first point, and points outside it are
    ignored: zoom so that the colony and its surroundings are visible.
  - Tested on the 99 sponge outlines of Nursery1 (May), made too big, too small, or turned and shifted: the median
    overlap with the hand-made outline (IoU) goes from 0.56 to 0.72 / 0.87 / 0.92 after 1 / 2 / 3 points, and 90%
    of them are at 0.85 or better after 3 points. A point never made an outline worse. About 0.5 s per point on this
    computer (the first one takes longer when the view is large).
- **Space** confirms the points or, if there are none, re-fits the outlines of the selected regions to the image at
  their new place, as Carry Regions Over does (RITM and colour; the outline can shrink or grow). A new outline is
  used only if it is close to the old one; otherwise the moved outline is kept, and a message says how many
  outlines were re-fitted.
- **Undo:** one step per drag, per series of arrow presses and turns on the same regions, per re-fit, and per set of
  confirmed points.

## Positive/negative clicks tool (RITM)
- **Correcting a region keeps its information.** Select a region, then add points: before, the corrected region
  kept only its id and class. It now also keeps its name, note, custom attributes (including the carry-over
  status), and "to check" mark. DEM metrics and surface area are cleared, because the outline changed.
- **Corrections need fewer points.** RITM now starts from the region's own outline, plus an automatic point inside
  it. Before, it started from the outline alone and changed little at the first point. On the Nursery1 test above:
  0.72 instead of 0.61 after one point, 0.87 instead of 0.82 after two.
- **Fixes:**
  - Confirming a correction that found nothing no longer hides the region.
  - Memory no longer grows with each point during a long session.

## Tool instructions
- **In split screen, the instructions of the current tool are shown only once, on the left map.** Before, the same
  panel covered both maps. The tool still works on both maps. With a single map nothing changes.

## Keyboard and mouse shortcuts (File > Keyboard and Mouse Shortcuts...)
- **Every TagLab key can be changed:** the menu keys (Ctrl+S, Ctrl+Z…), the region keys (A, M, D, S, Del…), the tool
  keys 1–0, the view keys (Q, W, E, G, C, U), the carry-over review keys (N, P, K, Delete) and the Move Region keys
  (arrows, `,` `.` `<` `>`).
  - Click a key cell and press the new key; × removes the key.
  - Tools that had no key (Move Region, Place point, Watershed, the two SAM tools, Edit matches) are listed too, so
    they can get one.
  - Space (confirm) and Esc (cancel) keep their keys, because every tool's instructions name them. Tab, Home, End,
    [ and ] are reserved.
- **Mouse buttons:** the middle button, the two thumb buttons (back / forward) and the right button can run an
  action, for example Keep in the carry-over review, or Confirm (the same as Space).
  - A plain press (without Shift, Ctrl or Alt) on either map, or in the review window, runs the action when it can be
    used. Otherwise the button does what it did before: with the right button on Keep, a right click keeps the colony
    while the review is open and opens the region menu otherwise.
  - Shift + click and Shift + right click (positive / negative points) don't change.
  - Some mice send their thumb buttons as keys: then set them in the Key column.
- **Conflicts** are shown in red, and OK stays disabled until they are fixed. A review key may be the same as a normal
  key, as N is by default (Next colony while the review is open, Create a Background Region otherwise): the window
  shows a note, and the normal key is switched off while the review is open.
- **The texts follow your keys:** the menus, the review's buttons and key line, the Move Region instructions and
  tooltip, and Help (Ctrl+H).
- **The defaults are the usual keys, with no mouse buttons,** so nothing changes until you edit something. Restore
  defaults puts them back. The shortcuts are stored on each computer, not in the project, and the official TagLab
  ignores them.

## Dictionaries
- **`spongenursery.json`** (in the main folder, next to `FDEP.json`): the sponge nursery labels.
  - Loggerhead, Yellow, Brown Branching, Vase, Sheepswool and Glove, each twice: "(Boat)" and "(Scuba)". The two of a
    pair share a colour, light for Boat and dark for Scuba.
  - Sand, Macroalgae, Brick and Soft Coral.
  - Load it with Project > Labels Dictionary Editor... > Load.

## Installing and updating
- **`install_windows.bat`:** installs everything into its own Python environment, using the package versions this
  version was tested with. It uses the GPU automatically when the computer has an NVIDIA graphics card.
  - It also recognises the GPU with newer NVIDIA drivers. Those drivers print "CUDA UMD Version", and the official
    installer then installed the CPU version.
  - Running it again replaces a CPU version of PyTorch with the GPU version.
  - At the end it says whether PyTorch can use the GPU.
- **`download_models.bat`:** downloads the AI models, which are not stored on GitHub. SAM is optional.
- **`update.bat`:** updates from this version's GitHub repository. Replaced files are backed up first.
- **Update notice:** TagLab shows a notice at startup when a newer version is available.
- **`opentaglab.bat`:** works from any folder.

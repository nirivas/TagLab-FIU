# Changes in the FIU lab version of TagLab

This version is based on the official TagLab 2026.4.1 (https://github.com/cnr-isti-vclab/TagLab) and is distributed
under the same GPL-3 licence. It changes the following.

## Data safety and crash fixes
- **Errors:** an error no longer closes TagLab. A message is shown and the details are written to `TagLab.log`.
- **Updates:** TagLab no longer refuses to start when an official update exists. Updates come from this version's own
  repository instead (see "Installing and updating").
- **Exports:**
  - The CSV and shapefile exports work again with current NumPy versions.
  - Exporting the data table with a working area no longer deletes the point annotations outside it.
- **Projects** with several maps load the pixel size of every map correctly.
- **Year-to-year matching:**
  - New colonies are labelled "born" (not "dead") when a match is removed.
  - The genet CSV no longer has blank rows.
- **Custom attributes:** looking at a colony in the info panel no longer rounds its stored values.

## DEM and surface area
- **Surface area:**
  - It is correct near DEM holes and map edges, and it is saved with the project.
  - It is reset when a colony is redrawn.
  - No large `slope.tif` file is written any more.
- **DEM > Calculate DEM Metrics** computes, for each colony: mean slope, elevation (mean/min/max), relief, height and
  volume above the surrounding substrate, and rugosity. The values are saved as region attributes and appear in the
  CSV and shapefile exports.

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
- **Auto Markers:** finds corresponding points between the two maps automatically.
- **Scale:** it can estimate and (optionally) apply the scale difference between two surveys.
- **New preview:** blend, difference and checkerboard views, at full resolution.
- **Markers:** they are saved with the project.
- **DEMs:** they keep their no-data areas, and georeferencing is preserved.

## Rotate Map (Project > Rotate Map…)
- Creates a rotated copy of a map: 90°, 180° and 270° are exact, or you can enter any angle.
- The DEM, the colonies, the points and the georeferencing are rotated too.

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

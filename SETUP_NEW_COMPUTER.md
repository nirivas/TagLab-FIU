# Installing this TagLab version on another computer (Windows)

This is the FIU Coral Reef Fisheries version of TagLab. **It is the only TagLab you need**: it is the official TagLab
2026.4.1 plus the lab's changes, which are listed in `CHANGES.md`. Do not install the official TagLab as well.
The AI models are not stored on GitHub because they are too big; the installer downloads them.

## 1. Install Python 3.11
- Download **Python 3.11 (64-bit)** from https://www.python.org/downloads/windows/.
- In the installer, tick **"Add python.exe to PATH"** (see `docs/python-installation-add-to-PATH.png`).
- At the end, click **"Disable path length limit"** if it is offered (see `docs/python-disable-path-length-limit.png`).

## 2. Get TagLab
On the GitHub page of the repository (https://github.com/nirivas/TagLab-FIU), click **Code > Download ZIP**, then
extract it to a folder such as `C:\TagLab-FIU`. With git installed you can use `git clone` instead.

## 3. Install
Double-click **`install_windows.bat`**. It takes a while and needs an internet connection. It:
- creates TagLab's own **Python environment** in the `venv` folder inside the TagLab folder (see below);
- installs the Python packages into it, at the versions tested with this version;
- installs PyTorch, the **GPU version automatically** if the computer has an NVIDIA graphics card;
- installs GDAL and rasterio;
- downloads the AI models (about 1.4 GB).

At the end it asks whether to download **SAM** (2.5 GB, optional). SAM is the "Segment Anything" tool and is very slow
without an NVIDIA GPU. You can also get it later by double-clicking **`download_models.bat`**. Run that file any time
to download missing models.

## 4. Start TagLab
Double-click **`opentaglab.bat`**. It works wherever the folder is, and it uses the `venv` environment automatically.
To create a desktop shortcut, right-click it and choose Send to > Desktop.

## About the Python environment (`venv` folder)
- **What it is:** a private copy of Python, with the packages TagLab needs, inside the TagLab folder. Other Python
  programs on the computer are not affected, and they cannot break TagLab.
- **Using it:** the `.bat` files (`opentaglab.bat`, `update.bat`, `download_models.bat`) use it automatically. To run a
  Python command for TagLab yourself, use its Python:
  `venv\Scripts\python.exe` (for example `venv\Scripts\python.exe -m pip list`). In a terminal you can also
  activate it with `venv\Scripts\activate`, and `deactivate` to leave.
- **Resetting it:** delete the `venv` folder and run `install_windows.bat` again. Your projects and the AI models are
  not affected.
- **Moving it:** the environment cannot be moved. If you move the TagLab folder, delete `venv` and run the installer
  again.

## Opening your projects
- Copy your project files (`.json`) and the images (orthomosaics, DEMs) to the new computer.
- When you open a project and an image is not found at its old location, TagLab asks you to locate it. After that, the
  other images in the same folder are found automatically. Save the project afterwards.
- Dictionaries such as `FDEP.json` are in the main folder.

## Updating
When a newer version is on GitHub (https://github.com/nirivas/TagLab-FIU), TagLab shows a notice at startup.
- **To update:** close TagLab and double-click **`update.bat`**.
- **What it changes:** only files that differ from GitHub are replaced. The old versions are saved first in
  `update_backups\`. Models, `config.json` (networks you trained), the `venv` environment, your projects and your
  data are never touched.
- **Packages:** if the updater says the required packages changed, run `install_windows.bat` again. It reuses the
  existing environment.
- **Errors:** if something goes wrong, TagLab shows the error and writes the details in `TagLab.log` in the TagLab folder.

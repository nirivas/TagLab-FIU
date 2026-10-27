"""
Update this lab version of TagLab from its GitHub repository (https://github.com/nirivas/TagLab-FIU).

Double-click update.bat, or run:

    python update.py              check GitHub and install the update (asks before changing anything)
    python update.py --check      only check whether an update is available
    python update.py --yes        install without asking
    python update.py --sync-state record the GitHub version if this folder is identical to it (used by the installer)

How it works:
- the latest version on GitHub is downloaded (about 15 MB) and compared file by file with this folder;
- only the files that are different (or new) are replaced; the old versions are first copied into
  update_backups/<date-time>/, so nothing is lost;
- never touched: the AI models (models/*.pth, *.net), config.json (it lists the networks trained on this
  computer), your projects and data, and any file that is not in the repository.
"""

import os
import sys
import json
import shutil
import zipfile
import tempfile
import datetime
import urllib.request

GITHUB_OWNER = "nirivas"
GITHUB_REPO = "TagLab-FIU"
GITHUB_BRANCH = "main"

# (they can be changed with environment variables, for testing)
API_URL = os.environ.get("TAGLAB_UPDATE_API_URL",
                         "https://api.github.com/repos/{}/{}/commits/{}".format(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH))
ZIP_URL = os.environ.get("TAGLAB_UPDATE_ZIP_URL",
                         "https://github.com/{}/{}/archive/{{sha}}.zip".format(GITHUB_OWNER, GITHUB_REPO))

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(HERE, ".lab_version")
BACKUP_DIR = os.path.join(HERE, "update_backups")

# files of the repository that are never overwritten
PROTECTED_FILES = {"config.json"}
# extensions never written by the update (AI models)
PROTECTED_EXTENSIONS = (".pth", ".net", ".pth.tar")


def _request(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "TagLab-updater",
                                               "Accept": "application/vnd.github+json"})
    return urllib.request.urlopen(req, timeout=timeout)


def latestCommit(timeout=20):
    """
    Information about the latest version on GitHub: {'sha', 'date', 'message'}.
    """
    with _request(API_URL, timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    commit = data.get("commit", {})
    return {"sha": data["sha"],
            "date": commit.get("committer", {}).get("date", ""),
            "message": commit.get("message", "").split("\n")[0]}


def readState():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def writeState(info):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"sha": info["sha"], "date": info["date"], "message": info["message"],
                   "installed": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=1)


def checkForUpdate(timeout=5):
    """
    Used by TagLab at startup. Returns (True, info) if GitHub has a version different from the one installed
    here, (False, info) if it is the same, (None, None) if this cannot be known (no internet, or this folder was
    never installed/updated from GitHub, e.g. the computer where the lab version is developed).
    """
    state = readState()
    if state is None:
        return None, None
    try:
        info = latestCommit(timeout)
    except Exception:
        return None, None
    return info["sha"] != state.get("sha"), info


def _download(info, workdir):
    """
    Download and extract the repository at the given version. Returns the folder with its files.
    """
    zip_path = os.path.join(workdir, "update.zip")
    with _request(ZIP_URL.format(sha=info["sha"]), 120) as response, open(zip_path, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(os.path.join(workdir, "extracted"))
    root = os.path.join(workdir, "extracted")
    entries = os.listdir(root)
    # GitHub archives contain a single top folder (<repo>-<sha>)
    if len(entries) == 1 and os.path.isdir(os.path.join(root, entries[0])):
        root = os.path.join(root, entries[0])
    return root


def _differences(root):
    """
    Files of the downloaded version that are new or different from the files of this folder.
    """
    new, changed = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, root)
            if rel in PROTECTED_FILES or name.lower().endswith(PROTECTED_EXTENSIONS):
                continue
            dst = os.path.join(HERE, rel)
            if not os.path.exists(dst):
                new.append(rel)
            else:
                with open(src, "rb") as a, open(dst, "rb") as b:
                    if a.read() != b.read():
                        changed.append(rel)
    return sorted(new), sorted(changed)


def _ask(question, default_yes):
    try:
        answer = input(question).strip().lower()
    except EOFError:
        return default_yes
    if answer == "":
        return default_yes
    return answer in ("y", "yes")


def main():
    args = sys.argv[1:]
    assume_yes = "--yes" in args

    print("TagLab (FIU lab version) updater - https://github.com/{}/{}".format(GITHUB_OWNER, GITHUB_REPO))
    print("")

    try:
        info = latestCommit()
    except Exception as e:
        print("Cannot reach GitHub: " + str(e))
        print("Check the internet connection (and that the repository exists and is public).")
        return 1

    print("Latest version on GitHub: {} ({})  {}".format(info["sha"][:7], info["date"], info["message"]))
    state = readState()
    if state is not None:
        print("Installed version:        {} ({})".format(state.get("sha", "?")[:7], state.get("date", "")))
        if state.get("sha") == info["sha"] and "--sync-state" not in args:
            print("\nTagLab is up to date.")
            return 0
    else:
        print("Installed version:        unknown (the files will be compared with GitHub)")

    if "--check" in args:
        print("\nAn update may be available. Run update.bat to install it.")
        return 0

    workdir = tempfile.mkdtemp(prefix="taglab_update_")
    try:
        print("\nDownloading the latest version..")
        root = _download(info, workdir)
        new, changed = _differences(root)

        if not new and not changed:
            writeState(info)
            print("\nThis folder is identical to the latest version: TagLab is up to date.")
            return 0

        if "--sync-state" in args:
            print("\nThis folder is different from the latest version on GitHub: run update.bat to update it.")
            return 0

        print("\n{} files will be updated and {} new files will be added:".format(len(changed), len(new)))
        for rel in (changed + new)[:40]:
            print("   " + ("(new) " if rel in new else "") + rel)
        if len(changed) + len(new) > 40:
            print("   ... and {} more".format(len(changed) + len(new) - 40))
        print("The current versions of the updated files are saved in update_backups.")

        # this folder may be the one where the lab version is developed: changes not uploaded yet would be replaced
        if os.path.isdir(os.path.join(HERE, "TagLab_for_GitHub")):
            print("\nWARNING: this looks like the computer where the lab version is developed (TagLab_for_GitHub")
            print("exists). Files changed here and not uploaded to GitHub yet would be replaced (they are backed up).")
            if not _ask("Continue anyway? [y/N] ", False):
                print("Update cancelled.")
                return 0
        elif not assume_yes and not _ask("\nInstall the update? [Y/n] ", True):
            print("Update cancelled.")
            return 0

        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup = os.path.join(BACKUP_DIR, stamp)
        for rel in changed:
            dst = os.path.join(backup, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(HERE, rel), dst)

        for rel in changed + new:
            dst = os.path.join(HERE, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(root, rel), dst)

        writeState(info)

        print("\nUpdate completed ({} files).".format(len(changed) + len(new)))
        if changed:
            print("Previous versions of the replaced files: " + backup)
        if "install.py" in changed + new:
            print("\nThe list of required packages has changed: run install_windows.bat again.")
        print("Restart TagLab to use the new version.")
        return 0

    except Exception as e:
        print("\nThe update failed: " + str(e))
        print("Nothing has been changed, unless the message above says otherwise.")
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

"""
Download the AI models (network weights) used by TagLab into the models/ folder.

The models are not stored on GitHub (they are too big): double-click download_models.bat, or run

    python download_models.py            download the missing models, then ask about SAM
    python download_models.py --sam      also download SAM (about 2.5 GB) without asking
    python download_models.py --no-sam   do not download SAM
    python download_models.py --check    only list the models that are present / missing / incomplete

Models already present are never downloaded again, unless they are incomplete (smaller or larger than the file on
the server, e.g. after an interrupted download).
"""

import os
import sys
import subprocess
import urllib.error
import urllib.request

BASE_URL = os.environ.get("TAGLAB_MODELS_URL", "http://taglab.isti.cnr.it/models/")

# sizes in bytes of the complete files, used when the server can't be asked (the server's size wins: a model may
# be updated there)
EXPECTED_SIZES = {
    "ritm_corals.pth": 40712217,
    "dextr_corals.pth": 195762189,
    "deeplab-resnet.pth.tar": 474903453,
    "pocillopora.net": 237927270,
    "porites.net": 237927270,
    "pocillopora_porite_montipora.net": 237929326,
    "sam_vit_h_4b8939.pth": 2564550879,
}

CORE_MODELS = [
    ("ritm_corals.pth", "RITM tool (positive/negative clicks)"),
    ("dextr_corals.pth", "4-clicks tool (DEXTR)"),
    ("deeplab-resnet.pth.tar", "base network for 'Train Your Network'"),
    ("pocillopora.net", "Pocillopora classifier"),
    ("porites.net", "Porites classifier"),
    ("pocillopora_porite_montipora.net", "Pocillopora/Porites/Montipora classifier"),
]

SAM_MODEL = ("sam_vit_h_4b8939.pth", "SAM tools (Segment Anything, about 2.5 GB)")

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

# set when the server could not be reached: the other models are not asked (no waiting for each timeout)
server_unreachable = False


def remoteSize(name):
    """
    The size of the model on the server (Content-Length of a HEAD request), or None if the server can't tell.
    """
    global server_unreachable
    if server_unreachable:
        return None

    request = urllib.request.Request(BASE_URL + name, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            size = response.headers.get("Content-Length")
            return int(size) if size is not None else None
    except urllib.error.HTTPError:
        return None
    except Exception:
        server_unreachable = True
        return None


def modelStatus(name):
    """
    (status, size, expected size) of a model file. The status is "present", "INCOMPLETE" (its size differs from the
    complete file), "MISSING" or "present?" (no size to compare with).
    """
    path = os.path.join(MODELS_DIR, name)
    if not os.path.isfile(path):
        return "MISSING", None, None

    size = os.path.getsize(path)
    expected = remoteSize(name)
    if expected is None:
        expected = EXPECTED_SIZES.get(name)
    if expected is None:
        return "present?", size, None
    return ("present" if size == expected else "INCOMPLETE"), size, expected


def download(name):
    """
    Download a model into a '.part' file, renamed only if the download is complete.
    """
    url = BASE_URL + name
    dst = os.path.join(MODELS_DIR, name)
    part = dst + ".part"

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total is not None else None
            done = 0
            with open(part, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        sys.stdout.write("\r  {}: {:.1f} / {:.1f} MB ({:.0f} %)".format(
                            name, done / 1e6, total / 1e6, 100.0 * done / total))
                    else:
                        sys.stdout.write("\r  {}: {:.1f} MB".format(name, done / 1e6))
                    sys.stdout.flush()
    except BaseException:
        # a broken connection (or Ctrl+C) leaves the existing file as it was
        print("")
        if os.path.exists(part):
            os.remove(part)
        raise
    print("")

    if total is not None and done != total:
        os.remove(part)
        raise Exception("incomplete download of " + name + " ({} of {} bytes)".format(done, total))

    os.replace(part, dst)


def installSegmentAnything():
    try:
        import segment_anything  # noqa: F401
    except ImportError:
        print("Installing the segment-anything package..")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "segment-anything==1.0"])


def main():
    args = sys.argv[1:]
    os.makedirs(MODELS_DIR, exist_ok=True)

    models = CORE_MODELS + [SAM_MODEL]

    print("TagLab models folder:", MODELS_DIR)
    status = {}
    for (name, desc) in models:
        state, size, expected = modelStatus(name)
        status[name] = state
        print("  [{:10s}] {:34s} {}".format(state, name, desc))
        if state == "INCOMPLETE":
            print("               {:,} bytes, the complete file has {:,}".format(size, expected))
    print("")

    # the missing or incomplete models (SAM is optional: only an incomplete SAM is a problem)
    to_download = [m for m in CORE_MODELS if status[m[0]] in ("MISSING", "INCOMPLETE")]
    sam_state = status[SAM_MODEL[0]]

    if "--check" in args:
        problems = [m[0] for m in to_download] + ([SAM_MODEL[0]] if sam_state == "INCOMPLETE" else [])
        if problems:
            print("Missing or incomplete: " + ", ".join(problems) + ". Run download_models.bat to download them.")
            return 1
        return 0

    failed = []
    for (name, desc) in to_download:
        print("Downloading " + name + " (" + desc + ")..")
        try:
            download(name)
        except Exception as e:
            print("  ERROR: " + str(e))
            failed.append(name)

    if sam_state in ("MISSING", "INCOMPLETE"):
        if "--sam" in args:
            want_sam = True
        elif "--no-sam" in args:
            want_sam = False
        elif sam_state == "INCOMPLETE":
            answer = input("The SAM file is incomplete. Download it again (about 2.5 GB)? [y/N] ")
            want_sam = answer.strip().lower() in ("y", "yes")
        else:
            answer = input("Download also SAM (about 2.5 GB, optional; it is slow without an NVIDIA GPU)? [y/N] ")
            want_sam = answer.strip().lower() in ("y", "yes")
        if want_sam:
            try:
                installSegmentAnything()
                print("Downloading " + SAM_MODEL[0] + "..")
                download(SAM_MODEL[0])
            except Exception as e:
                print("  ERROR: " + str(e))
                failed.append(SAM_MODEL[0])
    elif "--sam" in args:
        installSegmentAnything()

    print("")
    if failed:
        print("Some models could not be downloaded: " + ", ".join(failed))
        print("Check the internet connection and run this script again.")
        return 1

    print("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

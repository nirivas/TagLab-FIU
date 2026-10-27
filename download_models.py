"""
Download the AI models (network weights) used by TagLab into the models/ folder.

The models are not stored on GitHub (they are too big): double-click download_models.bat, or run

    python download_models.py            download the missing models, then ask about SAM
    python download_models.py --sam      also download SAM (about 2.5 GB) without asking
    python download_models.py --no-sam   do not download SAM
    python download_models.py --check    only list the models that are present / missing

Models already present are never downloaded again.
"""

import os
import sys
import subprocess
import urllib.request

BASE_URL = os.environ.get("TAGLAB_MODELS_URL", "http://taglab.isti.cnr.it/models/")

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


def download(name):
    """
    Download a model into a '.part' file, renamed only if the download is complete.
    """
    url = BASE_URL + name
    dst = os.path.join(MODELS_DIR, name)
    part = dst + ".part"

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
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
    present = [m for m in models if os.path.isfile(os.path.join(MODELS_DIR, m[0]))]
    missing = [m for m in models if m not in present]

    print("TagLab models folder:", MODELS_DIR)
    for (name, desc) in models:
        status = "present" if (name, desc) in present else "MISSING"
        print("  [{:7s}] {:34s} {}".format(status, name, desc))
    print("")

    if "--check" in args:
        return 0

    failed = []
    for (name, desc) in [m for m in missing if m != SAM_MODEL]:
        print("Downloading " + name + " (" + desc + ")..")
        try:
            download(name)
        except Exception as e:
            print("  ERROR: " + str(e))
            failed.append(name)

    if SAM_MODEL in missing:
        if "--sam" in args:
            want_sam = True
        elif "--no-sam" in args:
            want_sam = False
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

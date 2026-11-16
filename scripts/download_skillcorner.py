"""Download the SkillCorner open data used as a real-tracking test set (MIT licence).

    python scripts/download_skillcorner.py

Clones https://github.com/SkillCorner/opendata into external/skillcorner-opendata without its
large files, then fetches each match's tracking file (about 1.8 GB in all) into
external/skillcorner-tracking. Files already present are skipped. Data credit: SkillCorner.
"""

import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "external", "skillcorner-opendata")
OUT = os.path.join(ROOT, "external", "skillcorner-tracking")
MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches"


def main():
    if not os.path.isdir(REPO):
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        subprocess.check_call(["git", "clone", "--depth", "1",
                               "https://github.com/SkillCorner/opendata", REPO], env=env)
    os.makedirs(OUT, exist_ok=True)
    ids = sorted(os.listdir(os.path.join(REPO, "data", "matches")))
    for mid in ids:
        path = os.path.join(OUT, mid + "_tracking.jsonl")
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            print("have " + mid)
            continue
        print("get  " + mid + " ...", flush=True)
        urllib.request.urlretrieve("{}/{}/{}_tracking_extrapolated.jsonl".format(MEDIA, mid, mid),
                                   path + ".part")
        if os.path.getsize(path + ".part") < 1_000_000:
            sys.exit("download of {} looks wrong (too small)".format(mid))
        os.replace(path + ".part", path)
    print("SkillCorner open data ready: {} matches".format(len(ids)))


if __name__ == "__main__":
    main()

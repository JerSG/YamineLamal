"""Download the DFL open data used for training and testing (positions and match information).

    python scripts/download_dfl.py

Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and
event data in elite soccer. Scientific Data 12, 195. doi:10.1038/s41597-025-04505-y
Data: Deutsche Fussball Liga (DFL), CC BY 4.0, https://doi.org/10.6084/m9.figshare.28196177

About 2.6 GB into external/dfl (gitignored). Files already present with the expected size are
skipped. The event files of the dataset are not needed and not downloaded: the labels come from
the tracking (possession/dfl.py).
"""

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "external", "dfl")

FILES = [   # (file name, figshare download URL, size in bytes)
    ("DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml", "https://ndownloader.figshare.com/files/51643472", 11722),
    ("DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml", "https://ndownloader.figshare.com/files/51643475", 11930),
    ("DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml", "https://ndownloader.figshare.com/files/51643478", 11745),
    ("DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml", "https://ndownloader.figshare.com/files/51643490", 11823),
    ("DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml", "https://ndownloader.figshare.com/files/51643487", 11922),
    ("DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml", "https://ndownloader.figshare.com/files/51643484", 11932),
    ("DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml", "https://ndownloader.figshare.com/files/51643481", 11952),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml", "https://ndownloader.figshare.com/files/51643514", 418524833),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml", "https://ndownloader.figshare.com/files/51643517", 391722803),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml", "https://ndownloader.figshare.com/files/51643520", 348945715),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml", "https://ndownloader.figshare.com/files/51643523", 362567859),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml", "https://ndownloader.figshare.com/files/51643526", 371688499),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml", "https://ndownloader.figshare.com/files/51643529", 356219423),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml", "https://ndownloader.figshare.com/files/51643532", 372934140),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, url, size in FILES:
        path = os.path.join(OUT, name)
        if os.path.exists(path) and os.path.getsize(path) == size:
            print("have " + name)
            continue
        print("get  " + name + " ...", flush=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        if os.path.getsize(tmp) != size:
            sys.exit("size mismatch for {}: got {} bytes, expected {}".format(
                name, os.path.getsize(tmp), size))
        os.replace(tmp, path)
    print("DFL open data ready in " + os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()

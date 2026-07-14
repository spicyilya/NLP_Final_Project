"""Fetch the MELD text annotations into data/raw/.

The MELD CSVs are not committed to this repo (GPL-3.0; see the citation in
report.md). Run this once before notebooks/01:

    python scripts/download_data.py

Source: https://github.com/declare-lab/MELD
Citation: Poria et al., "MELD: A Multimodal Multi-Party Dataset for Emotion
Recognition in Conversations", ACL 2019.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD"
FILES = ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW / name
        if dest.exists():
            print(f"skip {name} (already present)")
            continue
        url = f"{BASE}/{name}"
        print(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

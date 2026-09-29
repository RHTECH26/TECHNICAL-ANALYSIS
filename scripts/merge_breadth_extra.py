"""
Merge data/breadth-extra-sp500.json (written by fetch_sp500_breadth.py) into
data/breadth-extra.json (written by fetch_live_data.py for the 4 Nifty
indices), producing the single combined file the site actually reads.

Why this is a separate step rather than each fetch script updating
breadth-extra.json directly: fetch_live_data.py and fetch_sp500_breadth.py
run as separate parallel jobs in the GitHub Actions workflow, each in its
own isolated checkout. If both scripts read-modify-wrote the SAME file,
whichever job's artifact got downloaded last would silently clobber the
other's data (a classic parallel-write race). Writing to separate files
and merging them here, after both jobs have finished, avoids that
entirely.

Usage (run from the repo root, after both fetch jobs' artifacts have been
downloaded into data/):
    python3 scripts/merge_breadth_extra.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

base_path = DATA_DIR / "breadth-extra.json"
sp500_path = DATA_DIR / "breadth-extra-sp500.json"


def main():
    base = json.loads(base_path.read_text()) if base_path.exists() else {}
    if sp500_path.exists():
        base.update(json.loads(sp500_path.read_text()))
    else:
        print(f"WARNING: {sp500_path} not found -- skipping the S&P 500 merge, "
              f"breadth-extra.json will keep whatever keys it already had.")
    base_path.write_text(json.dumps(base))
    print(f"Merged {base_path} now has keys: {list(base.keys())}")


if __name__ == "__main__":
    main()

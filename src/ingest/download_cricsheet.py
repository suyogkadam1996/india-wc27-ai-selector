"""
Download Cricsheet ODI match data (free, public, structured JSON).

Cricsheet (https://cricsheet.org) publishes ball-by-ball data for
international and domestic matches under a permissive license
(check https://cricsheet.org/downloads/ for the current terms
before redistributing anything — it's fine for personal/portfolio
use, but re-check if you plan to publish derived raw data).

Usage:
    python src/ingest/download_cricsheet.py

This will download the ODI JSON archive into data/raw/cricsheet_odis/
"""

import io
import zipfile
from pathlib import Path

import requests

CRICSHEET_ODI_URL = "https://cricsheet.org/downloads/odis_json.zip"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "cricsheet_odis"


def download_and_extract(url: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    print(f"Extracting to {dest_dir} ...")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(dest_dir)

    n_files = len(list(dest_dir.glob("*.json")))
    print(f"Done. {n_files} match JSON files in {dest_dir}")


if __name__ == "__main__":
    download_and_extract(CRICSHEET_ODI_URL, RAW_DIR)

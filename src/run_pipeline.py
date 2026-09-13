"""
Run the entire data pipeline end to end, in the correct order:

  1. Download latest Cricsheet ODI data
  2. Parse into tidy tables
  3. Build player features
  4. Build venue features
  5. Normalize venue names for 2027 WC host grounds

This is the ONE script to run when you want everything refreshed.
Used both for manual refreshes and by the automated monthly
GitHub Action (see .github/workflows/refresh-data.yml).

Usage:
    python src/run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_IN_ORDER = [
    "src/ingest/download_cricsheet.py",
    "src/ingest/parse_cricsheet.py",
    "src/features/build_player_features.py",
    "src/features/build_venue_features.py",
    "src/features/normalize_venues.py",
]


def run_step(script_path: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Running: {script_path}")
    print("=" * 60)
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"\nSTOPPED: {script_path} failed (exit code {result.returncode}).")
        print("Fix the error above before continuing -- later steps depend on this one.")
        sys.exit(result.returncode)


def main():
    project_root = Path(__file__).resolve().parents[1]
    for script in SCRIPTS_IN_ORDER:
        run_step(str(project_root / script))
    print(f"\n{'=' * 60}")
    print("Pipeline complete. All data refreshed.")
    print("=" * 60)


if __name__ == "__main__":
    main()

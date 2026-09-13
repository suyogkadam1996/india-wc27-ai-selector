"""
Run the entire data pipeline end to end, in the correct order:

  1. Download latest Cricsheet ODI data
  2. Parse into tidy tables
  3. Detect wicketkeepers (from stumping data)
  4. Build player features (form, role, active/inactive, keeper flag)
  5. Build venue features
  6. Normalize venue names for 2027 WC host grounds
  7. Score every player's suitability
  8. Filter down to India's active player pool
  9. Select the fixed squad of 15

This is the ONE script to run when you want everything refreshed --
including the squad of 15 itself, since it should be periodically
re-checked as new ODIs are played over the coming year (new form data,
injuries showing up as inactivity, etc.) rather than picked once and
never revisited.

Used both for manual refreshes and by the automated monthly
GitHub Action (see .github/workflows/refresh-data.yml) -- meaning your
squad of 15 now re-evaluates itself automatically every month too.

NOTE: this does NOT run build_playing_xi.py, since that needs a venue
name you choose at the time you want it -- run that separately, e.g.:
    python src/models/build_playing_xi.py "SuperSport Park"

Usage:
    python src/run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_IN_ORDER = [
    "src/ingest/download_cricsheet.py",
    "src/ingest/parse_cricsheet.py",
    "src/features/detect_wicketkeepers.py",
    "src/features/build_player_features.py",
    "src/features/build_venue_features.py",
    "src/features/normalize_venues.py",
    "src/models/build_suitability_scores.py",
    "src/models/show_india_pool.py",
    "src/models/build_squad.py",
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

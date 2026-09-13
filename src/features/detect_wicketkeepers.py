"""
PHASE 3 ADD-ON: Wicketkeeper Detection

WHAT THIS DOES (plain words):
When a batter is dismissed "stumped," the player credited with that
dismissal is almost always the wicketkeeper -- catching a bail-breaking
stumping is specifically a keeper's job, essentially never a fielder's.
So we count how often each player is credited with a stumping. A high
count is a strong, real signal that they are the team's keeper.

This is a genuine data-driven signal (not a guess) -- but it can still
miss a keeper who simply hasn't had many stumping chances yet in our
data, which is exactly why we ALSO support manual corrections via
data/reference/role_overrides.csv (see build_player_features.py).

Produces: data/processed/wicketkeeper_signal.parquet
  player, stumping_count, likely_keeper_from_data (True if count >= threshold)

Run this BEFORE build_player_features.py so the keeper signal is ready
to be merged in:
    python src/features/detect_wicketkeepers.py
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "cricsheet_odis"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# How many credited stumpings before we trust this as a real keeper signal.
# Deliberately low -- a keeper only gets stumping chances when the bowling
# is spin and the batter is out of their crease, so even a handful of
# credited stumpings is a meaningful signal, not noise.
MIN_STUMPINGS_TO_FLAG = 2


def main():
    json_files = sorted(RAW_DIR.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {RAW_DIR}. Run download_cricsheet.py first.")

    stumping_counts = defaultdict(int)

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            match_json = json.load(f)

        info = match_json.get("info", {})
        if info.get("gender") != "male":
            continue  # same men's-only filter used throughout the project

        for innings in match_json.get("innings", []):
            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    for wicket in delivery.get("wickets", []):
                        if wicket.get("kind") == "stumped":
                            for fielder in wicket.get("fielders", []):
                                name = fielder.get("name")
                                if name:
                                    stumping_counts[name] += 1

    result = pd.DataFrame(
        [{"player": p, "stumping_count": c} for p, c in stumping_counts.items()]
    )
    result["likely_keeper_from_data"] = result["stumping_count"] >= MIN_STUMPINGS_TO_FLAG
    result = result.sort_values("stumping_count", ascending=False)

    out_path = PROCESSED_DIR / "wicketkeeper_signal.parquet"
    result.to_parquet(out_path, index=False)

    flagged = result["likely_keeper_from_data"].sum()
    print(f"Found stumping data for {len(result)} players; {flagged} flagged as likely keepers "
          f"(>= {MIN_STUMPINGS_TO_FLAG} credited stumpings).")
    print("\nTop 10 by stumping count:")
    print(result.head(10).to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

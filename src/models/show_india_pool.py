"""
PHASE 3, STEP 2: Show only India's players, ranked by suitability score.

WHAT THIS DOES (plain words):
Takes the full suitability score table (which has players from every
country) and filters it down to just India, so you can see India's
best current picks by role.

Run after build_suitability_scores.py:
    python src/models/show_india_pool.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def main():
    scores = pd.read_parquet(PROCESSED_DIR / "player_suitability_scores.parquet")

    india = scores[scores["team"] == "India"].sort_values("suitability_score", ascending=False)

    if india.empty:
        print("No players found with team == 'India'. This likely means the "
              "'team' column wasn't found correctly -- check that "
              "build_player_features.py ran successfully after this update.")
        return

    out_path = PROCESSED_DIR / "india_player_pool.parquet"
    india.to_parquet(out_path, index=False)

    print(f"Found {len(india)} active India players.\n")
    for role in ["batter", "bowler", "all_rounder"]:
        role_players = india[india["role"] == role].head(8)
        print(f"\n--- Top {role}s ---")
        print(role_players[["player", "suitability_score"]].to_string(index=False))

    print(f"\nSaved full India pool to {out_path}")


if __name__ == "__main__":
    main()

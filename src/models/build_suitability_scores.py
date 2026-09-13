"""
PHASE 3, STEP 1: Player Suitability Scoring

WHAT THIS DOES (in plain words):
For every player, calculate one score (0-100) representing how strong
a pick they currently are, and write a plain-English reason explaining
that score.

HOW THE SCORE IS CALCULATED (deliberately simple and readable, not a
black-box model -- this is on purpose, so every number can be checked
by hand):

  For a BATTER:
    - 60% comes from recent form (last 15 innings average + strike rate)
    - 40% comes from career-long average + strike rate
    (recent form counts more, because current form matters more for
    picking a squad today than what happened years ago)

  For a BOWLER:
    - 60% comes from recent form (wickets/match + economy)
    - 40% comes from career bowling average + economy

  For an ALL-ROUNDER:
    - Batting score and bowling score are both calculated, then averaged

  Every player's raw numbers are converted to a 0-100 PERCENTILE RANK
  among their own role group (batters compared to batters, bowlers to
  bowlers) -- this avoids comparing a batter's average directly to a
  bowler's economy, which would be meaningless.

Produces: data/processed/player_suitability_scores.parquet
  One row per player: role, suitability_score (0-100), reason (text)

Run after build_player_features.py:
    python src/models/build_suitability_scores.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def percentile_score(series: pd.Series) -> pd.Series:
    """Converts raw numbers into a 0-100 rank within the group -- 100
    means 'better than everyone else in this group', 50 means 'about average'."""
    return (series.rank(pct=True) * 100).round(1)


def score_batters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fall back to career stats if recent-form stats are missing (not enough recent innings)
    df["form_avg_for_scoring"] = df["recent_form_avg"].fillna(df["career_batting_avg"])
    df["form_sr_for_scoring"] = df["recent_form_strike_rate"].fillna(df["career_strike_rate"])

    recent_score = (
        percentile_score(df["form_avg_for_scoring"]) * 0.5
        + percentile_score(df["form_sr_for_scoring"]) * 0.5
    )
    career_score = (
        percentile_score(df["career_batting_avg"]) * 0.5
        + percentile_score(df["career_strike_rate"]) * 0.5
    )
    df["batting_suitability"] = (recent_score * 0.6 + career_score * 0.4).round(1)
    return df


def score_bowlers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "recent_form_wickets_per_match" not in df.columns:
        return df.assign(bowling_suitability=None)
    df = df.copy()
    df["form_wkts_for_scoring"] = df["recent_form_wickets_per_match"].fillna(0)
    # Lower economy is BETTER, so we flip it before ranking (negative sign trick)
    df["form_econ_for_scoring"] = -df["recent_form_economy"].fillna(df["career_economy"])

    recent_score = (
        percentile_score(df["form_wkts_for_scoring"]) * 0.5
        + percentile_score(df["form_econ_for_scoring"]) * 0.5
    )
    career_score = (
        percentile_score(-df["career_economy"]) * 0.5
        + percentile_score(-df["career_bowling_avg"].fillna(df["career_bowling_avg"].max())) * 0.5
    )
    df["bowling_suitability"] = (recent_score * 0.6 + career_score * 0.4).round(1)
    return df


def build_reason(row) -> str:
    parts = []
    if row["role"] in ("batter", "all_rounder"):
        if pd.notna(row.get("recent_form_avg")):
            parts.append(
                f"Recent batting form: {row['recent_form_avg']} average, "
                f"{row['recent_form_strike_rate']} strike rate (last innings tracked)."
            )
        if pd.notna(row.get("career_batting_avg")):
            parts.append(f"Career batting average: {row['career_batting_avg']}.")
    if row["role"] in ("bowler", "all_rounder"):
        if pd.notna(row.get("recent_form_wickets_per_match")):
            parts.append(
                f"Recent bowling form: {row['recent_form_wickets_per_match']} wickets/match, "
                f"economy {row['recent_form_economy']}."
            )
        if pd.notna(row.get("career_economy")):
            parts.append(f"Career economy: {row['career_economy']}.")
    if not parts:
        parts.append("Not enough recorded data to explain this score confidently.")
    return " ".join(parts)


def main():
    player_features = pd.read_parquet(PROCESSED_DIR / "player_features.parquet")

    total_players = len(player_features)
    # Only currently active players can realistically be selected for a
    # squad today -- this excludes retired players (e.g. someone whose
    # last recorded match was years ago) who would otherwise still rank
    # highly just because their old numbers were strong.
    player_features = player_features[player_features["is_active"] == True]  # noqa: E712
    print(f"Scoring {len(player_features)} currently active players "
          f"(excluded {total_players - len(player_features)} inactive/retired players).")

    batters = player_features[player_features["role"].isin(["batter", "all_rounder"])]
    bowlers = player_features[player_features["role"].isin(["bowler", "all_rounder"])]

    batters_scored = score_batters(batters)[["player", "batting_suitability"]]
    bowlers_scored = score_bowlers(bowlers)[["player", "bowling_suitability"]]

    result = player_features.merge(batters_scored, on="player", how="left")
    result = result.merge(bowlers_scored, on="player", how="left")

    def combined_score(row):
        if row["role"] == "all_rounder":
            vals = [v for v in [row.get("batting_suitability"), row.get("bowling_suitability")] if pd.notna(v)]
            return round(np.mean(vals), 1) if vals else None
        if row["role"] == "batter":
            return row.get("batting_suitability")
        if row["role"] == "bowler":
            return row.get("bowling_suitability")
        return None

    result["suitability_score"] = result.apply(combined_score, axis=1)
    result["reason"] = result.apply(build_reason, axis=1)

    out_cols = ["player", "team", "role", "is_wicketkeeper", "suitability_score", "reason",
                "batting_suitability", "bowling_suitability"]
    final = result[out_cols].sort_values("suitability_score", ascending=False)

    out_path = PROCESSED_DIR / "player_suitability_scores.parquet"
    final.to_parquet(out_path, index=False)

    print(f"Scored {len(final)} players.")
    print(f"\nTop 10 highest-scoring CURRENTLY ACTIVE players overall:")
    print(final.head(10)[["player", "role", "suitability_score"]].to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

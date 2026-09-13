"""
Build venue-level feature table from Phase 1 processed data.

Produces: data/processed/venue_features.parquet
  One row per venue, with:
    - matches played there (in our dataset)
    - average team total per innings (proxy for how "high-scoring" it is)
    - average wickets lost per innings (proxy for how "difficult" batting is)
    - a scoring_category label (high / medium / low scoring), based on
      percentile rank among venues with enough matches to trust

NOTE ON SCOPE: this script only derives what's computable directly from
Cricsheet data (real match outcomes). It deliberately does NOT invent
pace-vs-spin-friendliness or altitude here, because that's not
computable from ball-by-ball outcomes alone -- see
data/reference/south_africa_venues.csv for that (separate, manually
sourced, clearly marked as reference data, not model-derived).

Run after build_player_features.py:
    python src/features/build_venue_features.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MIN_MATCHES_FOR_VENUE_TRUST = 5


def load_tables():
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    batting = pd.read_parquet(PROCESSED_DIR / "innings_batting.parquet")
    bowling = pd.read_parquet(PROCESSED_DIR / "innings_bowling.parquet")
    return matches, batting, bowling


def compute_team_innings_totals(batting: pd.DataFrame) -> pd.DataFrame:
    """One row per match+team = that team's total runs in their innings."""
    return (
        batting.groupby(["match_id", "team"])
        .agg(team_total=("runs", "sum"))
        .reset_index()
    )


def compute_wickets_per_innings(bowling: pd.DataFrame) -> pd.DataFrame:
    """One row per match = total wickets that fell (summed across both bowling sides)."""
    return (
        bowling.groupby("match_id")
        .agg(total_wickets=("wickets", "sum"))
        .reset_index()
    )


def main():
    matches, batting, bowling = load_tables()

    team_totals = compute_team_innings_totals(batting)
    team_totals = team_totals.merge(matches[["match_id", "venue"]], on="match_id", how="left")

    wickets = compute_wickets_per_innings(bowling)
    wickets = wickets.merge(matches[["match_id", "venue"]], on="match_id", how="left")

    venue_scoring = (
        team_totals.groupby("venue")
        .agg(
            innings_count=("team_total", "size"),
            avg_team_total=("team_total", "mean"),
        )
        .reset_index()
    )
    venue_scoring["avg_team_total"] = venue_scoring["avg_team_total"].round(1)

    venue_matches = matches.groupby("venue").agg(matches_played=("match_id", "nunique")).reset_index()

    venue_wickets = (
        wickets.groupby("venue")
        .agg(avg_wickets_per_match=("total_wickets", "mean"))
        .reset_index()
    )
    venue_wickets["avg_wickets_per_match"] = venue_wickets["avg_wickets_per_match"].round(1)

    venue_features = (
        venue_matches
        .merge(venue_scoring, on="venue", how="left")
        .merge(venue_wickets, on="venue", how="left")
    )

    trusted = venue_features["matches_played"] >= MIN_MATCHES_FOR_VENUE_TRUST
    venue_features["data_confidence"] = trusted.map(
        {True: "measured", False: "insufficient_data"}
    )

    # Percentile-based scoring category, computed only among trusted venues
    trusted_df = venue_features[trusted].copy()
    trusted_df["scoring_percentile"] = trusted_df["avg_team_total"].rank(pct=True)

    def categorize(pct):
        if pct >= 0.66:
            return "high_scoring"
        if pct <= 0.33:
            return "low_scoring"
        return "medium_scoring"

    trusted_df["scoring_category"] = trusted_df["scoring_percentile"].apply(categorize)
    venue_features = venue_features.merge(
        trusted_df[["venue", "scoring_category"]], on="venue", how="left"
    )
    venue_features["scoring_category"] = venue_features["scoring_category"].fillna("insufficient_data")

    out_path = PROCESSED_DIR / "venue_features.parquet"
    venue_features.to_parquet(out_path, index=False)

    print(f"Built features for {len(venue_features)} venues.")
    print(f"Venues with enough matches to trust ({MIN_MATCHES_FOR_VENUE_TRUST}+): {trusted.sum()}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

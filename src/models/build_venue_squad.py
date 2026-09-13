"""
PHASE 3, STEP 4: Venue-Specific Squad

WHAT THIS DOES (plain words):
Adjusts each player's score based on how well THEY specifically have
performed at ONE chosen ground, then rebuilds the squad using those
adjusted scores.

HOW THE ADJUSTMENT WORKS:
  - If a player has enough recorded innings at the chosen venue
    (3+, same threshold used throughout this project), we blend:
      50% their general (all-venues) suitability score
      50% how they rank at THIS venue specifically, compared to
          other players who've played there
  - If a player has NOT played enough at this venue, we simply keep
    their general score unchanged, and say so plainly -- we never
    invent a venue opinion about a player we have no real data for.

Usage:
    python src/models/build_venue_squad.py "SuperSport Park"

(Use one of the venue names shown in the reference list if you're
unsure of the exact spelling -- run with no argument to see them.)
"""

import sys
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"

BLEND_WEIGHT_VENUE = 0.5  # how much the venue-specific rank counts vs general score


def percentile_score(series: pd.Series) -> pd.Series:
    return (series.rank(pct=True) * 100).round(1)


def get_valid_venues() -> list:
    ref = pd.read_csv(REFERENCE_DIR / "wc2027_host_venues.csv")
    return sorted(ref[ref["is_2027_wc_venue"] == True]["canonical_name"].tolist())  # noqa: E712


def compute_venue_adjusted_scores(pool: pd.DataFrame, venue: str) -> pd.DataFrame:
    bat_venue = pd.read_parquet(PROCESSED_DIR / "player_venue_batting.parquet")
    bowl_venue = pd.read_parquet(PROCESSED_DIR / "player_venue_bowling.parquet")

    bat_at_venue = bat_venue[bat_venue["venue"] == venue].copy()
    bowl_at_venue = bowl_venue[bowl_venue["venue"] == venue].copy()

    if not bat_at_venue.empty:
        bat_at_venue["venue_batting_percentile"] = (
            percentile_score(bat_at_venue["venue_batting_avg"]) * 0.5
            + percentile_score(bat_at_venue["venue_strike_rate"]) * 0.5
        )
    if not bowl_at_venue.empty:
        bowl_at_venue["venue_bowling_percentile"] = (
            percentile_score(bowl_at_venue["venue_wickets"]) * 0.5
            + percentile_score(-bowl_at_venue["venue_economy"]) * 0.5
        )

    df = pool.copy()
    df = df.merge(
        bat_at_venue[["player", "venue_batting_percentile", "venue_innings"]],
        on="player", how="left"
    )
    df = df.merge(
        bowl_at_venue[["player", "venue_bowling_percentile", "venue_bowl_innings"]],
        on="player", how="left"
    )

    def adjusted_score(row):
        general = row["suitability_score"]
        if row["role"] in ("batter", "all_rounder") and pd.notna(row.get("venue_batting_percentile")):
            venue_component = row["venue_batting_percentile"]
        elif row["role"] == "bowler" and pd.notna(row.get("venue_bowling_percentile")):
            venue_component = row["venue_bowling_percentile"]
        else:
            return general, False  # no usable venue data -- unchanged, clearly flagged
        blended = general * (1 - BLEND_WEIGHT_VENUE) + venue_component * BLEND_WEIGHT_VENUE
        return round(blended, 1), True

    results = df.apply(adjusted_score, axis=1, result_type="expand")
    df["venue_adjusted_score"] = results[0]
    df["had_venue_data"] = results[1]
    return df


def main():
    if len(sys.argv) < 2:
        print("Please provide a venue name, e.g.:")
        print('  python src/models/build_venue_squad.py "SuperSport Park"')
        print("\nValid venue names:")
        for v in get_valid_venues():
            print(f"  - {v}")
        sys.exit(1)

    venue = sys.argv[1]
    valid_venues = get_valid_venues()
    if venue not in valid_venues:
        print(f"'{venue}' is not one of the recognized 2027 World Cup venues. Valid options:")
        for v in valid_venues:
            print(f"  - {v}")
        sys.exit(1)

    pool = pd.read_parquet(PROCESSED_DIR / "india_player_pool.parquet")
    adjusted = compute_venue_adjusted_scores(pool, venue)

    with_data = adjusted["had_venue_data"].sum()
    print(f"Venue: {venue}")
    print(f"{with_data} of {len(adjusted)} India players have enough recorded "
          f"innings at this ground to adjust their score; the rest keep their "
          f"general (all-venues) score unchanged.\n")

    ranked = adjusted.sort_values("venue_adjusted_score", ascending=False)
    print("Top 15 for this venue (adjusted score):")
    print(ranked.head(15)[["player", "role", "suitability_score", "venue_adjusted_score", "had_venue_data"]]
          .to_string(index=False))

    out_path = PROCESSED_DIR / f"venue_adjusted_scores_{venue.replace(' ', '_').replace(',', '')}.parquet"
    ranked.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

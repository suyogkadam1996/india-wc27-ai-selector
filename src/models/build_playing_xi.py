"""
PHASE 3, STEP 5: Playing XI Selector

WHAT THIS DOES (plain words):
This is the piece that actually varies match-to-match. The 15-man squad
(from build_squad.py) is FIXED for the whole tournament -- picked once.
But the actual 11 who play a given match changes depending on the
ground. This script picks that 11, FROM WITHIN the fixed 15 only
(never looks outside it -- that's the whole point of a squad), using
each player's venue-adjusted score for the specific ground given.

TYPICAL ODI PLAYING XI BALANCE (adjustable):
  1 wicketkeeper + 4 batters + 3 all-rounders + 3 bowlers = 11
  (the all-rounders also bowl, so between them and the 3 specialists,
  the team covers all 50 overs of bowling -- this is why an XI needs
  fewer "pure" bowlers than a squad does)

Usage:
    python src/models/build_playing_xi.py "SuperSport Park"
"""

import sys
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"

BLEND_WEIGHT_VENUE = 0.5

XI_QUOTAS = {
    "wicketkeeper": 1,
    "batter": 4,
    "all_rounder": 3,
    "bowler": 3,
}
TOTAL_XI_SIZE = sum(XI_QUOTAS.values())  # 11


def percentile_score(series: pd.Series) -> pd.Series:
    return (series.rank(pct=True) * 100).round(1)


def get_valid_venues() -> list:
    ref = pd.read_csv(REFERENCE_DIR / "wc2027_host_venues.csv")
    return sorted(ref[ref["is_2027_wc_venue"] == True]["canonical_name"].tolist())  # noqa: E712


def compute_venue_adjusted_scores(squad: pd.DataFrame, venue: str) -> pd.DataFrame:
    """Same blending approach as build_venue_squad.py, but percentiles
    are calculated only among THIS squad's 15 players (not all of India's
    talent pool) -- since we're choosing among a fixed, already-selected
    group, not re-opening the wider selection question."""
    bat_venue = pd.read_parquet(PROCESSED_DIR / "player_venue_batting.parquet")
    bowl_venue = pd.read_parquet(PROCESSED_DIR / "player_venue_bowling.parquet")

    squad_players = squad["player"].tolist()
    bat_at_venue = bat_venue[(bat_venue["venue"] == venue) & (bat_venue["player"].isin(squad_players))].copy()
    bowl_at_venue = bowl_venue[(bowl_venue["venue"] == venue) & (bowl_venue["player"].isin(squad_players))].copy()

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

    df = squad.copy()
    if not bat_at_venue.empty:
        df = df.merge(bat_at_venue[["player", "venue_batting_percentile"]], on="player", how="left")
    else:
        df["venue_batting_percentile"] = None
    if not bowl_at_venue.empty:
        df = df.merge(bowl_at_venue[["player", "venue_bowling_percentile"]], on="player", how="left")
    else:
        df["venue_bowling_percentile"] = None

    def adjusted_score(row):
        general = row["suitability_score"]
        if row["role"] in ("batter", "all_rounder") and pd.notna(row.get("venue_batting_percentile")):
            venue_component = row["venue_batting_percentile"]
        elif row["role"] == "bowler" and pd.notna(row.get("venue_bowling_percentile")):
            venue_component = row["venue_bowling_percentile"]
        else:
            return general, False
        blended = general * (1 - BLEND_WEIGHT_VENUE) + venue_component * BLEND_WEIGHT_VENUE
        return round(blended, 1), True

    results = df.apply(adjusted_score, axis=1, result_type="expand")
    df["venue_adjusted_score"] = results[0]
    df["had_venue_data"] = results[1]
    return df


def select_playing_xi(squad: pd.DataFrame) -> pd.DataFrame:
    squad = squad.sort_values("venue_adjusted_score", ascending=False).copy()
    selected_rows = []
    selected_players = set()

    is_keeper = squad["is_wicketkeeper"].fillna(False) if "is_wicketkeeper" in squad.columns else pd.Series(False, index=squad.index)
    keeper_picks = squad[is_keeper].head(XI_QUOTAS["wicketkeeper"]).copy()
    keeper_picks["xi_selection_note"] = "Selected as the top available wicketkeeper for this XI."
    keeper_picks["selected_as"] = "wicketkeeper"  # the SLOT they filled, not just their general capability
    selected_rows.append(keeper_picks)
    selected_players.update(keeper_picks["player"])

    for role, quota in XI_QUOTAS.items():
        if role == "wicketkeeper":
            continue
        role_pool = squad[(squad["role"] == role) & (~squad["player"].isin(selected_players))]
        picks = role_pool.head(quota).copy()
        picks["xi_selection_note"] = f"Selected as one of the top {quota} {role}s in the squad for this venue."
        picks["selected_as"] = role
        selected_rows.append(picks)
        selected_players.update(picks["player"])

    xi = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    shortfall = TOTAL_XI_SIZE - len(xi)

    if shortfall > 0:
        remaining = squad[~squad["player"].isin(selected_players)]
        fill_ins = remaining.head(shortfall).copy()
        fill_ins["xi_selection_note"] = (
            "Filled a remaining XI slot from the squad's next-best available "
            "player overall -- one role quota came up short for this venue."
        )
        fill_ins["selected_as"] = fill_ins["role"]
        xi = pd.concat([xi, fill_ins], ignore_index=True)

    return xi.sort_values("venue_adjusted_score", ascending=False)


def main():
    if len(sys.argv) < 2:
        print("Please provide a venue name, e.g.:")
        print('  python src/models/build_playing_xi.py "SuperSport Park"')
        print("\nValid venue names:")
        for v in get_valid_venues():
            print(f"  - {v}")
        sys.exit(1)

    venue = sys.argv[1]
    valid_venues = get_valid_venues()
    if venue not in valid_venues:
        print(f"'{venue}' is not a recognized 2027 World Cup venue. Valid options:")
        for v in valid_venues:
            print(f"  - {v}")
        sys.exit(1)

    squad_path = PROCESSED_DIR / "india_squad_15.parquet"
    if not squad_path.exists():
        raise SystemExit(f"{squad_path} not found. Run build_squad.py first to create the fixed squad.")
    squad = pd.read_parquet(squad_path)

    adjusted = compute_venue_adjusted_scores(squad, venue)
    xi = select_playing_xi(adjusted)

    print(f"Playing XI for: {venue}\n")
    print(f"(Chosen from the fixed 15-man squad only -- {adjusted['had_venue_data'].sum()} of "
          f"{len(adjusted)} squad members have enough recorded innings at this ground for a "
          f"venue-specific adjustment; the rest use their general squad score.)\n")

    is_keeper = xi["selected_as"] == "wicketkeeper"
    print("--- WICKETKEEPER ---")
    print(xi[is_keeper][["player", "venue_adjusted_score"]].to_string(index=False))

    for role in ["batter", "all_rounder", "bowler"]:
        role_xi = xi[xi["selected_as"] == role]
        print(f"\n--- {role.upper()}S ({len(role_xi)}) ---")
        print(role_xi[["player", "venue_adjusted_score", "had_venue_data"]].to_string(index=False))

    out_path = PROCESSED_DIR / f"playing_xi_{venue.replace(' ', '_').replace(',', '')}.parquet"
    xi.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

    print_batting_order(xi)


def print_batting_order(xi: pd.DataFrame) -> None:
    """Prints the XI sorted by real, historical batting entry position --
    openers first, down to the tail-end bowlers, mirroring an actual
    team sheet rather than grouping purely by role label."""
    df = xi.copy()
    if "avg_batting_position" not in df.columns:
        print("\n(Batting-order view unavailable -- rerun build_player_features.py "
              "after updating parse_cricsheet.py to enable this.)")
        return

    df["_sort_key"] = df["avg_batting_position"].fillna(99)
    ordered = df.sort_values("_sort_key")

    print("\n" + "=" * 50)
    print("SUGGESTED BATTING ORDER (based on real historical entry positions)")
    print("=" * 50)
    for _, row in ordered.iterrows():
        pos = row["avg_batting_position"]
        pos_label = f"~#{pos:.0f}" if pd.notna(pos) else "no batting data"
        keeper_tag = " (wk)" if row.get("selected_as") == "wicketkeeper" else ""
        print(f"  {pos_label:>16}  {row['player']}{keeper_tag}  [{row['role']}]")


if __name__ == "__main__":
    main()

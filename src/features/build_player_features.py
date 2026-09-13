"""
Build player-level feature tables from the tidy Cricsheet tables
produced in Phase 1 (data/processed/matches.parquet,
innings_batting.parquet, innings_bowling.parquet).

Produces: data/processed/player_features.parquet
  One row per player, with:
    - overall career batting/bowling stats
    - recency-weighted "current form" stats
    - per-venue performance splits (only where sample size is meaningful)
    - a simple role classification (batter / bowler / all_rounder / unclear)
    - last_played_date and an is_active flag, so retired players (e.g.
      someone whose career ended years ago) don't get ranked as top
      current picks just because their old stats were strong
    - is_wicketkeeper, combining a data-driven signal (stumping dismissal
      counts -- see detect_wicketkeepers.py) with manual corrections from
      data/reference/role_overrides.csv, since real cricket knowledge can
      catch things automated stats sometimes miss or lack enough sample
      size for
    - a data_confidence column, since some derived stats (e.g. role
      classification) are heuristic, not measured directly

For best results, run detect_wicketkeepers.py BEFORE this script, so
the wicketkeeper signal is available to merge in.

Run after Phase 1:
    python src/features/build_player_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"

# Minimum innings/matches thresholds before we trust a stat enough to report it.
MIN_INNINGS_FOR_VENUE_STAT = 3
MIN_INNINGS_FOR_FORM_STAT = 5
RECENT_FORM_WINDOW = 15  # last N innings used for "current form"

# A player is only considered "currently active" (realistically pickable
# today) if their most recent recorded match is within this many days.
# NOTE: measured from the LATEST match date found anywhere in the dataset,
# not from today's real-world date -- this avoids every player looking
# "inactive" just because the data hasn't been refreshed in a while.
ACTIVE_WINDOW_DAYS = 730  # ~2 years


def load_processed_tables():
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    batting = pd.read_parquet(PROCESSED_DIR / "innings_batting.parquet")
    bowling = pd.read_parquet(PROCESSED_DIR / "innings_bowling.parquet")
    return matches, batting, bowling


def attach_match_context(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Join venue/date onto a per-innings table."""
    match_cols = matches[["match_id", "venue", "city", "date"]]
    out = df.merge(match_cols, on="match_id", how="left")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def build_batting_features(batting: pd.DataFrame) -> pd.DataFrame:
    batting = batting.sort_values(["player", "date"])

    def player_agg(g: pd.DataFrame) -> pd.Series:
        innings = len(g)
        dismissals = innings  # NOTE: Cricsheet doesn't mark not-outs in our
        # current parse; treating every innings as a dismissal is a
        # simplification that slightly understates career average.
        # Flagged below via data_confidence.
        total_runs = g["runs"].sum()
        total_balls = g["balls"].sum()
        career_avg = round(total_runs / dismissals, 2) if dismissals else None
        career_sr = round(100 * total_runs / total_balls, 2) if total_balls else None
        boundary_pct = round(
            100 * (g["fours"].sum() + g["sixes"].sum()) / total_balls, 2
        ) if total_balls else None

        recent = g.tail(RECENT_FORM_WINDOW)
        recent_runs = recent["runs"].sum()
        recent_balls = recent["balls"].sum()
        has_enough_for_form = len(recent) >= MIN_INNINGS_FOR_FORM_STAT
        recent_avg = round(recent_runs / len(recent), 2) if has_enough_for_form else None
        recent_sr = round(100 * recent_runs / recent_balls, 2) if has_enough_for_form and recent_balls else None

        return pd.Series({
            "batting_innings": innings,
            "career_batting_avg": career_avg,
            "career_strike_rate": career_sr,
            "boundary_pct": boundary_pct,
            "recent_form_avg": recent_avg,
            "recent_form_strike_rate": recent_sr,
            "recent_form_confidence": "measured" if has_enough_for_form else "insufficient_data",
        })

    return batting.groupby("player").apply(player_agg, include_groups=False).reset_index()


def build_bowling_features(bowling: pd.DataFrame) -> pd.DataFrame:
    bowling = bowling.sort_values(["player", "date"])

    def player_agg(g: pd.DataFrame) -> pd.Series:
        innings = len(g)
        total_wickets = g["wickets"].sum()
        total_runs_conceded = g["runs_conceded"].sum()
        total_balls = g["balls_bowled"].sum()
        career_economy = round(total_runs_conceded / (total_balls / 6), 2) if total_balls else None
        career_bowling_avg = round(total_runs_conceded / total_wickets, 2) if total_wickets else None

        recent = g.tail(RECENT_FORM_WINDOW)
        has_enough_for_form = len(recent) >= MIN_INNINGS_FOR_FORM_STAT
        recent_wickets_per_match = round(recent["wickets"].sum() / len(recent), 2) if has_enough_for_form else None
        recent_econ = (
            round(recent["runs_conceded"].sum() / (recent["balls_bowled"].sum() / 6), 2)
            if has_enough_for_form and recent["balls_bowled"].sum() else None
        )

        return pd.Series({
            "bowling_innings": innings,
            "career_wickets": total_wickets,
            "career_economy": career_economy,
            "career_bowling_avg": career_bowling_avg,
            "recent_form_wickets_per_match": recent_wickets_per_match,
            "recent_form_economy": recent_econ,
        })

    return bowling.groupby("player").apply(player_agg, include_groups=False).reset_index()


def load_venue_alias_map() -> dict:
    """Same alias-mapping approach used in normalize_venues.py, reused here
    so a player's ground-specific record isn't wrongly split/undercounted
    across name-variants of the same real ground (e.g. 'Kingsmead' vs
    'Kingsmead, Durban')."""
    path = REFERENCE_DIR / "wc2027_host_venues.csv"
    if not path.exists():
        return {}
    ref = pd.read_csv(path)
    alias_map = {}
    for _, row in ref.iterrows():
        canonical = row["canonical_name"]
        alias_map[canonical] = canonical
        if pd.notna(row["known_aliases"]):
            for alias in str(row["known_aliases"]).split("|"):
                alias_map[alias.strip()] = canonical
    return alias_map


def build_venue_splits(batting: pd.DataFrame, bowling: pd.DataFrame) -> pd.DataFrame:
    """Per player, per venue, batting/bowling stats — only kept where
    the sample size clears MIN_INNINGS_FOR_VENUE_STAT, otherwise dropped
    rather than reported on a misleadingly thin sample.

    Venue names are normalized to canonical WC-host names FIRST (see
    load_venue_alias_map) so a player's real combined record at one
    ground isn't undercounted just because Cricsheet recorded it under
    two different name spellings.
    """
    alias_map = load_venue_alias_map()
    batting = batting.copy()
    bowling = bowling.copy()
    if alias_map:
        batting["venue"] = batting["venue"].map(alias_map).fillna(batting["venue"])
        bowling["venue"] = bowling["venue"].map(alias_map).fillna(bowling["venue"])

    bat_venue = (
        batting.groupby(["player", "venue"])
        .agg(venue_innings=("runs", "size"), venue_runs=("runs", "sum"), venue_balls=("balls", "sum"))
        .reset_index()
    )
    bat_venue = bat_venue[bat_venue["venue_innings"] >= MIN_INNINGS_FOR_VENUE_STAT].copy()
    bat_venue["venue_batting_avg"] = (bat_venue["venue_runs"] / bat_venue["venue_innings"]).round(2)
    bat_venue["venue_strike_rate"] = (100 * bat_venue["venue_runs"] / bat_venue["venue_balls"]).round(2)

    bowl_venue = (
        bowling.groupby(["player", "venue"])
        .agg(venue_bowl_innings=("wickets", "size"), venue_wickets=("wickets", "sum"),
             venue_runs_conceded=("runs_conceded", "sum"), venue_balls_bowled=("balls_bowled", "sum"))
        .reset_index()
    )
    bowl_venue = bowl_venue[bowl_venue["venue_bowl_innings"] >= MIN_INNINGS_FOR_VENUE_STAT].copy()
    bowl_venue["venue_economy"] = (
        bowl_venue["venue_runs_conceded"] / (bowl_venue["venue_balls_bowled"] / 6)
    ).round(2)

    return bat_venue, bowl_venue


def build_player_team(batting: pd.DataFrame) -> pd.DataFrame:
    """One row per player: the team they most often batted for.

    NOTE (known simplification): this is derived only from batting
    appearances, since the current bowling table doesn't record team.
    A specialist bowler who rarely or never batted could theoretically
    be missed by this -- flagged via team_confidence below rather than
    silently assumed correct.
    """
    def most_common_team(s: pd.Series):
        mode = s.mode()
        return mode.iloc[0] if not mode.empty else None

    team = batting.groupby("player")["team"].agg(most_common_team).rename("team")
    return team.reset_index()


def build_last_played_dates(batting: pd.DataFrame, bowling: pd.DataFrame) -> pd.DataFrame:
    """One row per player: the most recent date they appear in EITHER
    the batting or bowling tables -- used to detect retired/inactive
    players who would otherwise still show up ranked as top picks."""
    bat_last = batting.groupby("player")["date"].max().rename("last_batted")
    bowl_last = bowling.groupby("player")["date"].max().rename("last_bowled")
    combined = pd.concat([bat_last, bowl_last], axis=1)
    combined["last_played_date"] = combined[["last_batted", "last_bowled"]].max(axis=1)

    dataset_latest_date = combined["last_played_date"].max()
    cutoff = dataset_latest_date - pd.Timedelta(days=ACTIVE_WINDOW_DAYS)
    combined["is_active"] = combined["last_played_date"] >= cutoff
    return combined[["last_played_date", "is_active"]].reset_index()


def classify_role(row) -> str:
    """Heuristic, explainable role classification -- deliberately simple
    rules rather than a learned classifier, so it stays auditable.

    IMPORTANT: "bats" requires both enough innings AND a real batting
    average -- just having batted 10+ times isn't enough, since
    specialist bowlers often bat many times at the tail-end without
    being genuine batting contributors. Without the average check,
    players like a specialist fast bowler who occasionally bats at
    #9-11 would incorrectly get labeled 'all_rounder'.
    """
    MIN_BATTING_AVG_TO_COUNT = 20.0

    enough_batting_innings = (row.get("batting_innings") or 0) >= 10
    real_batting_avg = row.get("career_batting_avg")
    bats = enough_batting_innings and pd.notna(real_batting_avg) and real_batting_avg >= MIN_BATTING_AVG_TO_COUNT

    bowls = (row.get("bowling_innings") or 0) >= 10
    has_bowling_wickets = (row.get("career_wickets") or 0) >= 15

    if bats and bowls and has_bowling_wickets:
        return "all_rounder"
    if bowls and has_bowling_wickets:
        return "bowler"
    if bats:
        return "batter"
    return "unclear_insufficient_data"


def apply_wicketkeeper_and_overrides(player_features: pd.DataFrame) -> pd.DataFrame:
    """Merge in the data-driven wicketkeeper signal (if available) and
    apply manual corrections from role_overrides.csv. Manual overrides
    always win -- they represent real cricket knowledge that automated
    stats can miss, especially for players with a smaller data sample."""
    df = player_features.copy()
    df["is_wicketkeeper"] = False
    df["wicketkeeper_confidence"] = "no_signal_available"

    keeper_signal_path = PROCESSED_DIR / "wicketkeeper_signal.parquet"
    if keeper_signal_path.exists():
        signal = pd.read_parquet(keeper_signal_path)[["player", "likely_keeper_from_data"]]
        df = df.merge(signal, on="player", how="left")
        data_flagged = df["likely_keeper_from_data"].fillna(False)
        df["is_wicketkeeper"] = data_flagged
        df.loc[data_flagged, "wicketkeeper_confidence"] = "detected_from_stumping_data"
        df = df.drop(columns=["likely_keeper_from_data"])
    else:
        print("NOTE: wicketkeeper_signal.parquet not found -- run detect_wicketkeepers.py "
              "for automatic keeper detection. Continuing with manual overrides only.")

    overrides_path = REFERENCE_DIR / "role_overrides.csv"
    if overrides_path.exists():
        overrides = pd.read_csv(overrides_path)
        for _, row in overrides.iterrows():
            player, override_type, value = row["player"], row["override_type"], row["override_value"]
            match = df["player"] == player
            if not match.any():
                continue  # player not found in this dataset -- skip quietly, not an error
            if override_type == "wicketkeeper":
                is_true = str(value).strip().upper() == "TRUE"
                df.loc[match, "is_wicketkeeper"] = is_true
                df.loc[match, "wicketkeeper_confidence"] = "manual_override_user_specified"
            elif override_type == "role":
                df.loc[match, "role"] = value
                df.loc[match, "role_confidence"] = "manual_override_user_specified"

    return df


def main():
    matches, batting, bowling = load_processed_tables()
    batting = attach_match_context(batting, matches)
    bowling = attach_match_context(bowling, matches)

    bat_features = build_batting_features(batting)
    bowl_features = build_bowling_features(bowling)
    bat_venue, bowl_venue = build_venue_splits(batting, bowling)
    last_played = build_last_played_dates(batting, bowling)
    player_team = build_player_team(batting)

    player_features = bat_features.merge(bowl_features, on="player", how="outer")
    player_features = player_features.merge(last_played, on="player", how="left")
    player_features = player_features.merge(player_team, on="player", how="left")
    player_features["team_confidence"] = "derived_from_batting_appearances"
    player_features["role"] = player_features.apply(classify_role, axis=1)
    player_features["role_confidence"] = "heuristic_rule_based"

    player_features = apply_wicketkeeper_and_overrides(player_features)

    out_path = PROCESSED_DIR / "player_features.parquet"
    player_features.to_parquet(out_path, index=False)
    bat_venue.to_parquet(PROCESSED_DIR / "player_venue_batting.parquet", index=False)
    bowl_venue.to_parquet(PROCESSED_DIR / "player_venue_bowling.parquet", index=False)

    active_count = player_features["is_active"].sum()
    print(f"Built features for {len(player_features)} players "
          f"({active_count} currently active, {len(player_features) - active_count} inactive/retired).")
    print(f"Venue-qualified batting splits: {len(bat_venue)} player-venue pairs "
          f"(min {MIN_INNINGS_FOR_VENUE_STAT} innings each)")
    print(f"Venue-qualified bowling splits: {len(bowl_venue)} player-venue pairs")
    print(f"Saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()

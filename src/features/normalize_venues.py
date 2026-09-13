"""
Normalize venue names for the 2027 World Cup host venues
(South Africa, Zimbabwe, Namibia).

Problem this solves: Cricsheet records the same physical ground under
several different name strings (e.g. "Kingsmead" vs "Kingsmead, Durban"),
and separately, some grounds have deceptively similar names to OTHER
grounds. Two important real cases handled here:
  - "Wanderers Cricket Ground, Windhoek" (Namibia) is NOT the same
    ground as "The Wanderers Stadium" in Johannesburg (South Africa) --
    they just share the word "Wanderers". They must stay separate.
  - That same Windhoek ground WAS renamed "Namibia Cricket Ground" after
    a 2025 redevelopment -- so "Wanderers Cricket Ground, Windhoek" and
    "Namibia Cricket Ground" in Cricsheet/news data ARE the same ground
    and should be merged together under one canonical name.
  This is exactly why a manually verified table beats any naive
  string-similarity heuristic for this kind of data cleaning.

This script uses an explicit, manually verified reference table
(data/reference/wc2027_host_venues.csv) to:
  1. Map known name-variants of the SAME real ground to one canonical name
  2. Only touch venues explicitly listed -- anything not in the reference
     table is left completely alone, so we never silently mismerge an
     unfamiliar venue name

Produces: data/processed/venue_features_wc2027_normalized.parquet
  Same venues, but ambiguous South African name variants consolidated
  under one canonical_venue name, with matches_played summed and
  avg_team_total / avg_wickets_per_match recombined as weighted averages
  (not simple averages -- a venue with 40 matches should count more than
  one with 5).

Run after build_venue_features.py:
    python src/features/normalize_venues.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"


def load_alias_map() -> dict:
    """Returns {exact_alias_string: canonical_name} for every known variant."""
    ref = pd.read_csv(REFERENCE_DIR / "wc2027_host_venues.csv")
    alias_map = {}
    for _, row in ref.iterrows():
        canonical = row["canonical_name"]
        alias_map[canonical] = canonical  # canonical name maps to itself
        if pd.notna(row["known_aliases"]):
            for alias in str(row["known_aliases"]).split("|"):
                alias_map[alias.strip()] = canonical
    return alias_map


def normalize(venue_features: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    df = venue_features.copy()
    df["canonical_venue"] = df["venue"].map(alias_map)
    # Anything NOT in our reference table keeps its original name untouched --
    # we never guess a canonical name for a venue we haven't manually verified.
    df["canonical_venue"] = df["canonical_venue"].fillna(df["venue"])

    is_sa_grouped = df["canonical_venue"].isin(alias_map.values())
    unaffected = df[~is_sa_grouped].copy()
    to_merge = df[is_sa_grouped].copy()

    if to_merge.empty:
        return df.drop(columns=["canonical_venue"])

    # Weighted re-aggregation: a venue with more matches should count more.
    def weighted_merge(g: pd.DataFrame) -> pd.Series:
        total_matches = g["matches_played"].sum()
        total_innings = g["innings_count"].sum()
        if total_innings > 0:
            avg_total = (g["avg_team_total"] * g["innings_count"]).sum() / total_innings
            avg_wkts = (g["avg_wickets_per_match"] * g["innings_count"]).sum() / total_innings
        else:
            avg_total, avg_wkts = None, None
        merged_names = ", ".join(sorted(g["venue"].unique()))
        return pd.Series({
            "matches_played": total_matches,
            "innings_count": total_innings,
            "avg_team_total": round(avg_total, 1) if avg_total is not None else None,
            "avg_wickets_per_match": round(avg_wkts, 1) if avg_wkts is not None else None,
            "merged_from": merged_names,
        })

    merged = to_merge.groupby("canonical_venue").apply(weighted_merge, include_groups=False).reset_index()
    merged = merged.rename(columns={"canonical_venue": "venue"})
    merged["data_confidence"] = merged["matches_played"].apply(
        lambda n: "measured" if n >= 5 else "insufficient_data"
    )

    result = pd.concat([
        unaffected.drop(columns=["canonical_venue"]),
        merged,
    ], ignore_index=True)
    return result


def main():
    venue_features = pd.read_parquet(PROCESSED_DIR / "venue_features.parquet")
    alias_map = load_alias_map()

    normalized = normalize(venue_features, alias_map)

    out_path = PROCESSED_DIR / "venue_features_wc2027_normalized.parquet"
    normalized.to_parquet(out_path, index=False)

    ref = pd.read_csv(REFERENCE_DIR / "wc2027_host_venues.csv")
    official_venues = ref[ref["is_2027_wc_venue"] == True]["canonical_name"]  # noqa: E712
    sa_rows = normalized[normalized["venue"].isin(official_venues)]
    print(f"Total venues after normalization: {len(normalized)} (was {len(venue_features)})")
    print(f"\n2027 World Cup official host venues, consolidated:")
    print(sa_rows[["venue", "matches_played", "avg_team_total", "avg_wickets_per_match", "data_confidence"]]
          .to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

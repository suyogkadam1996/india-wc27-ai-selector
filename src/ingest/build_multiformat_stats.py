"""
PHASE 6: Multi-Format Player Profiles (Test / ODI / T20I / IPL)

WHAT THIS DOES (plain words):
Downloads Test, T20I, and IPL match archives from Cricsheet (the same
free, legitimate source used for everything else in this project --
NOT scraped from CricBuzz or any commercial site, which would carry
real terms-of-service risk for a public GitHub project). Computes
career stats per player per format, and merges them with the existing
ODI stats into one combined profile table.

Produces: data/processed/multiformat_profile.parquet
  One row per player: matches/average/strike-rate/wickets/economy for
  EACH of Test, ODI, T20I, and IPL, wherever that player appears.

HONESTY NOTE: I could not test-run the actual downloads myself (no
internet access in my working environment) -- these URLs follow
Cricsheet's documented, consistent naming pattern (the same pattern
already confirmed working for ODIs: "odis_json.zip"). If any single
URL is wrong, this script reports exactly which one failed rather than
crashing everything, so we can fix just that one.

Run standalone, or as part of run_pipeline.py:
    python src/ingest/build_multiformat_stats.py
"""

import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

FORMAT_SOURCES = {
    "test": {"url": "https://cricsheet.org/downloads/tests_json.zip", "folder": "cricsheet_tests"},
    "t20i": {"url": "https://cricsheet.org/downloads/t20is_json.zip", "folder": "cricsheet_t20is"},
    "ipl": {"url": "https://cricsheet.org/downloads/ipl_json.zip", "folder": "cricsheet_ipl"},
}


def download_and_extract(url: str, dest_dir: Path) -> bool:
    """Returns True on success, False on failure (with a clear message
    printed) -- never crashes the whole script over one bad URL."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(dest_dir)
        n_files = len(list(dest_dir.glob("*.json")))
        print(f"  -> {n_files} match files extracted to {dest_dir}")
        return True
    except Exception as e:
        print(f"  -> FAILED to download/extract {url}: {e}")
        print(f"     If this keeps failing, check https://cricsheet.org/downloads/ "
              f"for the current correct filename and tell me the right one.")
        return False


def parse_folder_to_career_stats(folder: Path, format_label: str, is_ipl: bool = False) -> pd.DataFrame:
    """Parses one format's raw JSON files directly into CAREER-level
    stats per player (simpler than the innings-level detail built for
    ODIs, since this is a supplementary multi-format summary, not the
    primary scoring pipeline)."""
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        return pd.DataFrame(columns=[
            "player", f"{format_label}_matches", f"{format_label}_batting_avg",
            f"{format_label}_strike_rate", f"{format_label}_wickets",
            f"{format_label}_economy", f"{format_label}_bowling_avg",
        ])

    bat_runs, bat_balls, bat_innings = defaultdict(int), defaultdict(int), defaultdict(int)
    bowl_runs, bowl_balls, bowl_wkts = defaultdict(int), defaultdict(int), defaultdict(int)
    matches_played = defaultdict(set)

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            match_json = json.load(f)
        info = match_json.get("info", {})
        if info.get("gender") != "male":
            continue
        if is_ipl:
            event_name = str(info.get("event", {}).get("name", ""))
            if "Indian Premier League" not in event_name:
                continue  # this T20 archive may include non-IPL leagues too

        match_id = path.stem
        for innings in match_json.get("innings", []):
            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    batter = delivery.get("batter")
                    bowler = delivery.get("bowler")
                    runs = delivery.get("runs", {})
                    extras = delivery.get("extras", {})
                    is_legal = not ("wides" in extras or "noballs" in extras)

                    matches_played[batter].add(match_id)
                    matches_played[bowler].add(match_id)

                    bat_runs[batter] += runs.get("batter", 0)
                    if is_legal:
                        bat_balls[batter] += 1

                    bowl_runs[bowler] += runs.get("total", 0)
                    if is_legal:
                        bowl_balls[bowler] += 1
                    for wicket in delivery.get("wickets", []):
                        if wicket.get("kind") not in ("run out", "retired hurt", "retired out"):
                            bowl_wkts[bowler] += 1

            # count innings played (for batting average) per player per match
            for batter_name in {d.get("batter") for over in innings.get("overs", []) for d in over.get("deliveries", [])}:
                if batter_name:
                    bat_innings[batter_name] += 1

    all_players = set(matches_played.keys())
    rows = []
    for player in all_players:
        innings = bat_innings.get(player, 0)
        runs = bat_runs.get(player, 0)
        balls = bat_balls.get(player, 0)
        avg = round(runs / innings, 2) if innings else None
        sr = round(100 * runs / balls, 2) if balls else None

        wkts = bowl_wkts.get(player, 0)
        rc = bowl_runs.get(player, 0)
        bb = bowl_balls.get(player, 0)
        econ = round(rc / (bb / 6), 2) if bb else None
        bowl_avg = round(rc / wkts, 2) if wkts else None

        rows.append({
            "player": player,
            f"{format_label}_matches": len(matches_played[player]),
            f"{format_label}_batting_avg": avg,
            f"{format_label}_strike_rate": sr,
            f"{format_label}_wickets": wkts,
            f"{format_label}_economy": econ,
            f"{format_label}_bowling_avg": bowl_avg,
        })

    return pd.DataFrame(rows)


def main():
    format_tables = {}

    for fmt, source in FORMAT_SOURCES.items():
        dest_dir = RAW_DIR / source["folder"]
        success = download_and_extract(source["url"], dest_dir)
        if success:
            format_tables[fmt] = parse_folder_to_career_stats(dest_dir, fmt, is_ipl=(fmt == "ipl"))
        else:
            print(f"Skipping {fmt} stats -- download failed. Other formats will still be built.")
            format_tables[fmt] = pd.DataFrame(columns=["player"])

    # Merge with existing ODI stats (from the main pipeline) if available
    odi_path = PROCESSED_DIR / "player_features.parquet"
    if odi_path.exists():
        odi = pd.read_parquet(odi_path)[[
            "player", "batting_innings", "career_batting_avg", "career_strike_rate",
            "career_wickets", "career_economy", "career_bowling_avg",
        ]].rename(columns={
            "batting_innings": "odi_matches", "career_batting_avg": "odi_batting_avg",
            "career_strike_rate": "odi_strike_rate", "career_wickets": "odi_wickets",
            "career_economy": "odi_economy", "career_bowling_avg": "odi_bowling_avg",
        })
    else:
        odi = pd.DataFrame(columns=["player"])
        print("NOTE: player_features.parquet not found -- ODI columns will be empty. "
              "Run build_player_features.py first for complete profiles.")

    merged = odi
    for fmt in ["test", "t20i", "ipl"]:
        merged = merged.merge(format_tables[fmt], on="player", how="outer")

    out_path = PROCESSED_DIR / "multiformat_profile.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\nBuilt multi-format profiles for {len(merged)} players.")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

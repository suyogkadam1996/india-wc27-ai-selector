"""
Parse Cricsheet ODI JSON files into tidy, analysis-ready tables:
  - innings_batting.parquet   (one row per batter per innings)
  - innings_bowling.parquet   (one row per bowler per innings)
  - matches.parquet           (one row per match: venue, date, teams, result)

Run after download_cricsheet.py has populated data/raw/cricsheet_odis/.

Usage:
    python src/ingest/parse_cricsheet.py
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "cricsheet_odis"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def parse_match(match_json: dict, match_id: str):
    info = match_json.get("info", {})

    # IMPORTANT: Cricsheet's ODI archive contains BOTH men's and women's
    # matches together. Every match file has an explicit "gender" field,
    # so we filter here rather than guessing at a different download URL --
    # this is the reliable, always-correct way to separate them.
    if info.get("gender") != "male":
        return None, [], []

    venue = info.get("venue")
    city = info.get("city")
    date = (info.get("dates") or [None])[0]
    teams = info.get("teams", [])
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")

    match_row = {
        "match_id": match_id,
        "venue": venue,
        "city": city,
        "date": date,
        "team1": teams[0] if len(teams) > 0 else None,
        "team2": teams[1] if len(teams) > 1 else None,
        "winner": winner,
    }

    batting_rows = []
    bowling_stats = defaultdict(lambda: {"runs_conceded": 0, "balls_bowled": 0, "wickets": 0})

    for innings in match_json.get("innings", []):
        batting_team = innings.get("team")
        bat_totals = defaultdict(lambda: {"runs": 0, "balls": 0, "fours": 0, "sixes": 0})

        for over in innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                batter = delivery.get("batter")
                bowler = delivery.get("bowler")
                runs = delivery.get("runs", {})
                batter_runs = runs.get("batter", 0)
                total_runs = runs.get("total", 0)

                # batting tally (ignore byes/legbyes for batter's own runs)
                bat_totals[batter]["runs"] += batter_runs
                extras = delivery.get("extras", {})
                is_legal_ball = not ("wides" in extras or "noballs" in extras)
                if is_legal_ball:
                    bat_totals[batter]["balls"] += 1
                if batter_runs == 4:
                    bat_totals[batter]["fours"] += 1
                if batter_runs == 6:
                    bat_totals[batter]["sixes"] += 1

                # bowling tally
                bowling_stats[bowler]["runs_conceded"] += total_runs
                if is_legal_ball:
                    bowling_stats[bowler]["balls_bowled"] += 1

                for wicket in delivery.get("wickets", []):
                    dismissal_kind = wicket.get("kind")
                    if dismissal_kind not in ("run out", "retired hurt", "retired out"):
                        bowling_stats[bowler]["wickets"] += 1

        for player, stats in bat_totals.items():
            sr = round(100 * stats["runs"] / stats["balls"], 2) if stats["balls"] else None
            batting_rows.append({
                "match_id": match_id,
                "team": batting_team,
                "player": player,
                "runs": stats["runs"],
                "balls": stats["balls"],
                "fours": stats["fours"],
                "sixes": stats["sixes"],
                "strike_rate": sr,
            })

    bowling_rows = []
    for player, stats in bowling_stats.items():
        overs = stats["balls_bowled"] / 6
        economy = round(stats["runs_conceded"] / overs, 2) if overs else None
        bowling_rows.append({
            "match_id": match_id,
            "player": player,
            "balls_bowled": stats["balls_bowled"],
            "runs_conceded": stats["runs_conceded"],
            "wickets": stats["wickets"],
            "economy": economy,
        })

    return match_row, batting_rows, bowling_rows


def main():
    json_files = sorted(RAW_DIR.glob("*.json"))
    if not json_files:
        raise SystemExit(
            f"No JSON files found in {RAW_DIR}. "
            "Run download_cricsheet.py first."
        )

    all_matches, all_batting, all_bowling = [], [], []
    skipped_female = 0

    for path in json_files:
        match_id = path.stem
        with open(path, "r", encoding="utf-8") as f:
            match_json = json.load(f)
        match_row, batting_rows, bowling_rows = parse_match(match_json, match_id)
        if match_row is None:
            skipped_female += 1
            continue  # was a women's match, filtered out -- see parse_match()
        all_matches.append(match_row)
        all_batting.extend(batting_rows)
        all_bowling.extend(bowling_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_matches).to_parquet(OUT_DIR / "matches.parquet", index=False)
    pd.DataFrame(all_batting).to_parquet(OUT_DIR / "innings_batting.parquet", index=False)
    pd.DataFrame(all_bowling).to_parquet(OUT_DIR / "innings_bowling.parquet", index=False)

    print(f"Parsed {len(all_matches)} men's matches (skipped {skipped_female} women's matches).")
    print(f"Batting rows: {len(all_batting)} | Bowling rows: {len(all_bowling)}")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

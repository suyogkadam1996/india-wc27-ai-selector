"""
PHASE 5 (v2): Standalone Website Builder

WHAT THIS DOES (plain words):
Reads the already-computed squad/venue/player data and bakes it,
along with any photos you've supplied, directly into ONE self-contained
HTML file: app/index.html. That file needs no server, no terminal
command, and no internet connection to run (except for loading the
Google Fonts) -- just double-click it to open in your browser.

This replaces the earlier Streamlit app, which kept hitting internal
styling conflicts that made it hard to get the exact look you wanted.
A real HTML/CSS/JS file gives full control over the design instead.

Run whenever the underlying data changes (e.g. after run_pipeline.py):
    python src/export/build_website.py
"""

import base64
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
PHOTOS_DIR = PROJECT_ROOT / "data" / "player_photos"
TEMPLATE_PATH = Path(__file__).resolve().parent / "template.html"
OUTPUT_PATH = PROJECT_ROOT / "app" / "index.html"


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def get_photo_data_uri(player_name: str):
    if not PHOTOS_DIR.exists():
        return None
    base = sanitize_filename(player_name)
    for ext in ("jpg", "jpeg", "png"):
        candidate = PHOTOS_DIR / f"{base}.{ext}"
        if candidate.exists():
            mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
            data = base64.b64encode(candidate.read_bytes()).decode()
            return f"data:image/{mime};base64,{data}"
    return None


def clean_records(df: pd.DataFrame) -> list:
    """Converts a DataFrame to JSON-safe records: NaN/NaT -> None,
    Timestamps -> readable date strings."""
    if df.empty:
        return []
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%d %b %Y")
    records = json.loads(df.to_json(orient="records", date_format="iso"))
    return records


def get_data_last_updated() -> str:
    path = PROCESSED_DIR / "matches.parquet"
    if not path.exists():
        return "unknown"
    ts = datetime.fromtimestamp(path.stat().st_mtime)
    return ts.strftime("%d %b %Y, %H:%M")


def main():
    squad_path = PROCESSED_DIR / "india_squad_15.parquet"
    if not squad_path.exists():
        raise SystemExit(f"{squad_path} not found. Run build_squad.py first.")
    squad = pd.read_parquet(squad_path)
    squad_players = squad["player"].tolist()

    bat_venue = pd.read_parquet(PROCESSED_DIR / "player_venue_batting.parquet")
    bat_venue = bat_venue[bat_venue["player"].isin(squad_players)]

    bowl_venue = pd.read_parquet(PROCESSED_DIR / "player_venue_bowling.parquet")
    bowl_venue = bowl_venue[bowl_venue["player"].isin(squad_players)]

    ref = pd.read_csv(REFERENCE_DIR / "wc2027_host_venues.csv")
    venues = sorted(ref[ref["is_2027_wc_venue"] == True]["canonical_name"].tolist())  # noqa: E712

    venue_characteristics = {}
    venue_stats_path = PROCESSED_DIR / "venue_features_wc2027_normalized.parquet"
    if venue_stats_path.exists():
        vf = pd.read_parquet(venue_stats_path)
        for row in clean_records(vf):
            venue_characteristics[row["venue"]] = row

    player_details_path = PROCESSED_DIR / "player_features.parquet"
    if player_details_path.exists():
        pf = pd.read_parquet(player_details_path)
        pf = pf[pf["player"].isin(squad_players)]
        player_details = {row["player"]: row for row in clean_records(pf)}
    else:
        player_details = {}

    multiformat_path = PROCESSED_DIR / "multiformat_profile.parquet"
    if multiformat_path.exists():
        mf = pd.read_parquet(multiformat_path)
        mf = mf[mf["player"].isin(squad_players)]
        multiformat = {row["player"]: row for row in clean_records(mf)}
    else:
        multiformat = {}

    photos = {}
    for player in squad_players:
        uri = get_photo_data_uri(player)
        if uri:
            photos[player] = uri

    app_data = {
        "generated_at": datetime.now().isoformat(),
        "updated_at": get_data_last_updated(),
        "squad": clean_records(squad),
        "venues": venues,
        "venue_characteristics": venue_characteristics,
        "venue_batting": clean_records(bat_venue),
        "venue_bowling": clean_records(bowl_venue),
        "player_details": player_details,
        "multiformat": multiformat,
        "photos": photos,
    }

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    json_blob = json.dumps(app_data)
    # Guard against a literal "</script>" inside the JSON breaking the page
    json_blob = json_blob.replace("</script>", "<\\/script>")
    final_html = template.replace("__APP_DATA__", json_blob)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(final_html, encoding="utf-8")

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Built website: {OUTPUT_PATH} ({size_kb:.0f} KB)")
    print(f"Squad players: {len(squad_players)} | Venues: {len(venues)} | "
          f"Photos embedded: {len(photos)} | Multi-format profiles: {len(multiformat)}")
    print("Open this file directly in your browser (double-click it) -- no server needed.")


if __name__ == "__main__":
    main()

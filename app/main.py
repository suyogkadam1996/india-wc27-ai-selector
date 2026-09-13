"""
PHASE 5: Visual Interface

WHAT THIS DOES (plain words):
A point-and-click app: pick a World Cup ground from a dropdown, and see
India's fixed squad of 15 plus the recommended Playing XI for that
specific ground, with plain-English reasons for each pick -- no
terminal commands needed.

This reads the already-computed files from data/processed/ and
data/reference/ (produced by run_pipeline.py and build_squad.py) --
it does NOT re-run the data pipeline itself. Run those first if you
haven't already, or whenever you want fresher data.

Usage (from the project's main folder):
    streamlit run app/main.py

This opens automatically in your web browser.
"""

from pathlib import Path
import base64
import re
from datetime import datetime

import pandas as pd
import streamlit as st

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"
PHOTOS_DIR = Path(__file__).resolve().parents[1] / "data" / "player_photos"

BLEND_WEIGHT_VENUE = 0.5
XI_QUOTAS = {"wicketkeeper": 1, "batter": 4, "all_rounder": 3, "bowler": 3}
TOTAL_XI_SIZE = sum(XI_QUOTAS.values())


# ---------- Data loading (cached so the app stays fast) ----------

@st.cache_data
def load_squad():
    return pd.read_parquet(PROCESSED_DIR / "india_squad_15.parquet")


@st.cache_data
def load_venue_stats():
    bat = pd.read_parquet(PROCESSED_DIR / "player_venue_batting.parquet")
    bowl = pd.read_parquet(PROCESSED_DIR / "player_venue_bowling.parquet")
    return bat, bowl


@st.cache_data
def load_valid_venues():
    ref = pd.read_csv(REFERENCE_DIR / "wc2027_host_venues.csv")
    return sorted(ref[ref["is_2027_wc_venue"] == True]["canonical_name"].tolist())  # noqa: E712


@st.cache_data
def load_player_details():
    """Full per-player stats (career average, strike rate, economy, etc.)
    -- richer than what's in the squad file, used for the click-through
    profile popup, similar to a CricBuzz-style player page."""
    path = PROCESSED_DIR / "player_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_multiformat_profile():
    """Test/ODI/T20I/IPL career stats -- from build_multiformat_stats.py,
    sourced from Cricsheet (same legitimate free source as everything
    else in this project, not scraped from any commercial site)."""
    path = PROCESSED_DIR / "multiformat_profile.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def get_data_last_updated() -> str:
    """Uses the actual file timestamp of the core data file as the
    honest 'last updated' marker -- reflects when run_pipeline.py was
    truly last run, not a hardcoded or guessed date."""
    path = PROCESSED_DIR / "matches.parquet"
    if not path.exists():
        return "unknown (data file not found)"
    ts = datetime.fromtimestamp(path.stat().st_mtime)
    return ts.strftime("%d %b %Y, %H:%M")


# ---------- Same selection logic as build_playing_xi.py ----------

def percentile_score(series: pd.Series) -> pd.Series:
    return (series.rank(pct=True) * 100).round(1)


def compute_venue_adjusted_scores(squad: pd.DataFrame, venue: str, bat_venue: pd.DataFrame,
                                    bowl_venue: pd.DataFrame) -> pd.DataFrame:
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
    df = df.merge(bat_at_venue[["player", "venue_batting_percentile"]], on="player", how="left") \
        if not bat_at_venue.empty else df.assign(venue_batting_percentile=None)
    df = df.merge(bowl_at_venue[["player", "venue_bowling_percentile"]], on="player", how="left") \
        if not bowl_at_venue.empty else df.assign(venue_bowling_percentile=None)

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
    selected_rows, selected_players = [], set()

    is_keeper = squad["is_wicketkeeper"].fillna(False) if "is_wicketkeeper" in squad.columns else pd.Series(False, index=squad.index)
    keeper_picks = squad[is_keeper].head(XI_QUOTAS["wicketkeeper"]).copy()
    keeper_picks["selected_as"] = "wicketkeeper"
    selected_rows.append(keeper_picks)
    selected_players.update(keeper_picks["player"])

    for role, quota in XI_QUOTAS.items():
        if role == "wicketkeeper":
            continue
        role_pool = squad[(squad["role"] == role) & (~squad["player"].isin(selected_players))]
        picks = role_pool.head(quota).copy()
        picks["selected_as"] = role
        selected_rows.append(picks)
        selected_players.update(picks["player"])

    xi = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    shortfall = TOTAL_XI_SIZE - len(xi)
    if shortfall > 0:
        remaining = squad[~squad["player"].isin(selected_players)]
        fill_ins = remaining.head(shortfall).copy()
        fill_ins["selected_as"] = fill_ins["role"]
        xi = pd.concat([xi, fill_ins], ignore_index=True)

    return xi




# ---------- Visual design helpers ----------

ROLE_STYLE = {
    "batter": {"label": "BATTER", "color": "#3B82F6", "emoji": "🔵"},
    "bowler": {"label": "BOWLER", "color": "#EF4444", "emoji": "🔴"},
    "all_rounder": {"label": "ALL-ROUNDER", "color": "#A855F7", "emoji": "🟣"},
    "wicketkeeper": {"label": "WICKETKEEPER", "color": "#F2B705", "emoji": "🟡"},
}


def get_initials(name: str) -> str:
    parts = [p for p in name.replace(".", " ").split() if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper() if name else "??"


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def get_photo_base64(player_name: str):
    """Looks for a locally supplied photo at data/player_photos/<name>.(jpg|jpeg|png).
    Returns a base64 data-URI string if found, else None (caller falls
    back to the initials avatar). We never fetch or generate real player
    photos ourselves -- only use ones the user has supplied locally."""
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


def reason_to_bullets(reason: str) -> list:
    """Splits reason text into bullets on sentence-ending periods only --
    NOT on decimal points inside numbers like '46.9 average' (a naive
    split on every '.' would break '46.9' into '46' and '9')."""
    if not reason or pd.isna(reason):
        return ["No detailed data available yet."]
    text = str(reason).replace(";", ".")
    raw_parts = re.split(r"(?<!\d)\.(?!\d)\s*", text)
    bullets = [p.strip().rstrip(".") for p in raw_parts if p.strip()]
    return bullets if bullets else ["No detailed data available yet."]


def render_avatar_html(player_name: str, role_key: str, size_px: int = 44) -> str:
    """Flat, single-line HTML -- no line ever starts with leading
    whitespace. This matters: Markdown treats 4+ leading spaces as a
    preformatted CODE BLOCK, which is exactly what caused an earlier
    bug where part of a card rendered as raw HTML text instead of an
    actual image. Keeping this flat avoids that risk entirely."""
    photo = get_photo_base64(player_name)
    if photo:
        return f'<img class="avatar-img" style="width:{size_px}px;height:{size_px}px" src="{photo}" />'
    style = ROLE_STYLE.get(role_key, ROLE_STYLE["batter"])
    initials = get_initials(player_name)
    return (f'<div class="avatar-initials" style="width:{size_px}px;height:{size_px}px;'
            f'background:{style["color"]}">{initials}</div>')


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(180deg, #0A0E1A 0%, #0F1729 100%);
}

.stApp, .stApp p, .stApp span, .stApp div, .stApp li {
    font-family: 'Inter', sans-serif !important;
    color: #E2E8F0;
}

h1, h2, h3 {
    font-family: 'Oswald', sans-serif !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: #94A3B8 !important;
}

[data-testid="stExpander"] {
    background: #131B2E;
    border: 1px solid #24304A;
    border-radius: 10px;
    margin-bottom: 10px;
}

[data-testid="stExpander"] summary {
    color: #FFFFFF !important;
    font-weight: 600;
    padding: 10px 14px;
}

[data-testid="stExpander"] summary:hover {
    background: #1A2338;
}

.stSelectbox label, .stSelectbox div {
    color: #E2E8F0 !important;
}

[data-testid="stAlert"] {
    background: #14213D;
    border: 1px solid #2C4770;
    color: #CBD5E1;
}

.strip-row {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
}

.avatar-initials {
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #0A0E1A;
    font-weight: 700;
    font-size: 15px;
    flex-shrink: 0;
}

.avatar-img {
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
    border: 2px solid #24304A;
}

.strip-name {
    font-weight: 600;
    font-size: 15px;
    color: #FFFFFF;
}

.strip-role {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 2px 9px;
    border-radius: 999px;
    color: #0A0E1A;
    display: inline-block;
    margin-left: 8px;
}

.strip-score {
    margin-left: auto;
    text-align: right;
    font-family: 'Oswald', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #F2B705;
}

.guaranteed-pill {
    display: inline-block;
    background: #3D2F0A;
    color: #F2B705;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 999px;
    margin-left: 8px;
}

.why-list {
    list-style: none;
    padding-left: 4px;
    margin: 6px 0 0 0;
    font-size: 13px;
    color: #B8C4D9;
}
.why-list li {
    margin-bottom: 6px;
    line-height: 1.4;
}

.venue-note {
    background: #0D2A3D;
    border-left: 3px solid #38BDF8;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
    color: #7DD3FC;
    margin: 8px 0;
}

.format-stat-block {
    background: #0F1729;
    border: 1px solid #24304A;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.format-stat-title {
    font-family: 'Oswald', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: #F2B705;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
</style>
"""


def render_strip_header(row: dict, role_key: str) -> str:
    """The compact strip shown as the expander's clickable label area --
    single flat HTML line: avatar, name, role pill, score."""
    style = ROLE_STYLE.get(role_key, ROLE_STYLE["batter"])
    avatar = render_avatar_html(row["player"], role_key, size_px=40)
    keeper_tag = "<span title='Wicketkeeper'>🧤</span>" if row.get("is_wicketkeeper") else ""
    guaranteed = "<span class='guaranteed-pill'>★ SENIOR PICK</span>" if row.get("guaranteed_selection") else ""
    score_key = "venue_adjusted_score" if "venue_adjusted_score" in row else "suitability_score"
    score_val = row.get(score_key, row.get("suitability_score", "—"))
    jersey = row.get("No.", "")
    return (
        f'<div class="strip-row">{avatar}'
        f'<div><span class="strip-name">#{jersey} {row["player"]} {keeper_tag}</span>'
        f'<span class="strip-role" style="background:{style["color"]}">{style["label"]}</span>'
        f'{guaranteed}</div>'
        f'<div class="strip-score">{score_val}</div></div>'
    )


def render_format_stats_html(player_name: str, multiformat: pd.DataFrame) -> str:
    """Builds the Test/ODI/T20I/IPL breakdown block for the full profile,
    from Cricsheet data (not scraped from any commercial site)."""
    if multiformat.empty:
        return ("<div class='format-stat-block'>Multi-format stats not built yet. "
                "Run <code>python src/ingest/build_multiformat_stats.py</code> to enable this.</div>")
    match = multiformat[multiformat["player"] == player_name]
    if match.empty:
        return "<div class='format-stat-block'>No multi-format data found for this player yet.</div>"
    p = match.iloc[0]

    blocks = []
    for fmt_key, fmt_label in [("odi", "ODI"), ("test", "TEST"), ("t20i", "T20I"), ("ipl", "IPL")]:
        matches = p.get(f"{fmt_key}_matches")
        if pd.isna(matches) or not matches:
            continue
        bat_avg = p.get(f"{fmt_key}_batting_avg")
        sr = p.get(f"{fmt_key}_strike_rate")
        wkts = p.get(f"{fmt_key}_wickets")
        econ = p.get(f"{fmt_key}_economy")
        lines = [f"Matches: {int(matches)}"]
        if pd.notna(bat_avg):
            lines.append(f"Bat avg: {bat_avg}")
        if pd.notna(sr):
            lines.append(f"SR: {sr}")
        if pd.notna(wkts) and wkts:
            lines.append(f"Wickets: {int(wkts)}")
        if pd.notna(econ):
            lines.append(f"Econ: {econ}")
        blocks.append(
            f'<div class="format-stat-block"><div class="format-stat-title">{fmt_label}</div>'
            f'{" &nbsp;|&nbsp; ".join(lines)}</div>'
        )
    if not blocks:
        return "<div class='format-stat-block'>No multi-format data found for this player yet.</div>"
    return "".join(blocks)


def render_profile_content(player_name: str, player_details: pd.DataFrame, multiformat: pd.DataFrame) -> None:
    if not player_details.empty:
        match = player_details[player_details["player"] == player_name]
        if not match.empty:
            p = match.iloc[0]
            st.caption(f"Team: {p.get('team', 'Unknown')}  |  Role: {str(p.get('role', 'Unknown')).replace('_', ' ').title()}")
            active_status = "🟢 Currently active" if p.get("is_active") else "🔴 Inactive / not recently playing"
            last_played = p.get("last_played_date")
            last_played_str = pd.to_datetime(last_played).strftime("%d %b %Y") if pd.notna(last_played) else "unknown"
            st.write(f"{active_status} — last played (in our ODI data): {last_played_str}")

    st.markdown("**Career stats by format**")
    st.markdown(render_format_stats_html(player_name, multiformat), unsafe_allow_html=True)
    st.caption(f"Data last updated: {get_data_last_updated()}")


HAS_DIALOG = hasattr(st, "dialog")

if HAS_DIALOG:
    @st.dialog("Player Profile")
    def show_profile_dialog(player_name: str, player_details: pd.DataFrame, multiformat: pd.DataFrame):
        st.subheader(player_name)
        render_profile_content(player_name, player_details, multiformat)


def render_player_strip(row: dict, role_key: str, player_details: pd.DataFrame, multiformat: pd.DataFrame,
                          extra_line: str = None, key_prefix: str = "sq") -> None:
    header_html = render_strip_header(row, role_key)
    with st.expander(" ", expanded=False):
        st.markdown(header_html, unsafe_allow_html=True)
        if extra_line:
            st.markdown(f"<div class='venue-note'>📍 {extra_line}</div>", unsafe_allow_html=True)
        bullets = reason_to_bullets(row.get("reason", ""))
        bullets_html = "".join(f"<li>✅ {b}</li>" for b in bullets[:4])
        st.markdown(f"<ul class='why-list'>{bullets_html}</ul>", unsafe_allow_html=True)
        button_key = f"{key_prefix}_{sanitize_filename(row['player'])}_{row.get('No.', '')}"
        if st.button("👤 Open Full Profile", key=button_key):
            if HAS_DIALOG:
                show_profile_dialog(row["player"], player_details, multiformat)
            else:
                render_profile_content(row["player"], player_details, multiformat)


# ---------- The app itself ----------

st.set_page_config(page_title="India ODI WC 2027 Selector", page_icon="🏏", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🏏 India ODI World Cup 2027 — Squad & XI Selector")
st.caption(
    "A data-driven decision-support tool — not an official BCCI product. "
    "Combines real historical stats (Cricsheet) with transparent expert-knowledge overrides."
)
st.markdown(f"📅 **Data last updated:** {get_data_last_updated()}")

try:
    squad = load_squad()
    bat_venue, bowl_venue = load_venue_stats()
    valid_venues = load_valid_venues()
except FileNotFoundError as e:
    st.error(
        f"Couldn't find a required data file: {e}\n\n"
        "Run `python src/run_pipeline.py` and `python src/models/build_squad.py` first."
    )
    st.stop()

player_details = load_player_details()
multiformat = load_multiformat_profile()

# ----- Squad of 15 -----
st.header("Fixed Squad of 15")
st.caption("Same for the whole tournament. Click a row to see quick notes; click 'Open Full Profile' for full career stats.")

squad_sorted = squad.copy()
squad_sorted["_sort_key"] = squad_sorted["avg_batting_position"].fillna(99) if "avg_batting_position" in squad_sorted.columns else 99
squad_sorted = squad_sorted.sort_values("_sort_key").reset_index(drop=True)
squad_sorted.insert(0, "No.", range(1, len(squad_sorted) + 1))

for _, row in squad_sorted.iterrows():
    role_key = "wicketkeeper" if row.get("is_wicketkeeper") else row["role"]
    render_player_strip(row.to_dict(), role_key, player_details, multiformat, key_prefix="squad")

# ----- Venue-specific Playing XI -----
st.header("Playing XI for a Specific Ground")
venue = st.selectbox("Choose a World Cup venue:", valid_venues)

adjusted = compute_venue_adjusted_scores(squad, venue, bat_venue, bowl_venue)
xi = select_playing_xi(adjusted)

with_data_count = int(adjusted["had_venue_data"].sum())
st.info(
    f"{with_data_count} of {len(adjusted)} squad members have enough recorded innings at "
    f"**{venue}** for a venue-specific adjustment. The rest use their general squad score."
)

xi_sorted = xi.copy()
xi_sorted["_sort_key"] = xi_sorted["avg_batting_position"].fillna(99) if "avg_batting_position" in xi_sorted.columns else 99
xi_sorted = xi_sorted.sort_values("_sort_key").reset_index(drop=True)
xi_sorted.insert(0, "No.", range(1, len(xi_sorted) + 1))

for _, row in xi_sorted.iterrows():
    role_key = "wicketkeeper" if row.get("selected_as") == "wicketkeeper" else row["role"]
    venue_note = (
        "Adjusted using this player's real record at this ground"
        if row["had_venue_data"]
        else "No data at this ground yet — using general squad ranking"
    )
    render_player_strip(row.to_dict(), role_key, player_details, multiformat, extra_line=venue_note, key_prefix="xi")

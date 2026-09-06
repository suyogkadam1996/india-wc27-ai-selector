# India ODI World Cup 2027 (South Africa) — AI Team Selection Assistant

A data-driven **decision-support tool** for exploring squad and playing-XI
recommendations for the Indian ODI team ahead of the 2027 World Cup in
South Africa. Built as a portfolio project demonstrating an end-to-end
ML pipeline: data engineering → feature engineering → modeling →
constrained optimization → explainability → deployment.

## What this project is — and isn't

This is **not** a claim to replicate BCCI's actual selection process. It
does not have access to proprietary data BCCI selectors use internally
(Hawk-Eye ball-tracking speed/seam/swing data, medical/fitness reports,
or selector deliberation). It's a transparent, explainable recommender
built entirely on free, public data, with any estimated (non-measured)
inputs clearly flagged as such throughout the pipeline and UI.

## Data sources

- [Cricsheet](https://cricsheet.org) — free, structured ball-by-ball
  data for international ODIs (primary dataset)
- ESPNcricinfo Statsguru — venue-split and domestic tournament
  scorecard-level stats (scraped, respecting robots.txt / rate limits)
- Open-Meteo — free weather API for venue temperature context
- Wikipedia — venue metadata (altitude, historical records)

## Data confidence levels

Every feature in this project is tagged with one of:

| Tag | Meaning |
|---|---|
| `measured` | Directly computed from real match/ball-by-ball data |
| `estimated_from_commentary` | Grounded in real public reporting (e.g. commentary-reported bowling pace ranges), expressed as a range/category, not a fabricated precise value |
| `role_based_default` | No player-specific public info exists; a generic role-based default is used, clearly marked as low-confidence |

See `src/features/synthetic_pace.py` for the reference implementation
of this pattern.

## Project structure

```
india-wc27-selector/
├── data/
│   ├── raw/            # untouched downloaded data (gitignored)
│   └── processed/      # cleaned, tidy tables (gitignored)
├── src/
│   ├── ingest/          # download + parse raw data sources
│   ├── features/        # feature engineering, incl. synthetic/confidence-flagged features
│   └── models/          # suitability scoring, squad/XI optimization, explainability
├── app/                 # Streamlit UI
├── notebooks/           # exploration notebooks
├── tests/
├── requirements.txt
└── README.md
```

## Pipeline / roadmap

1. **Data ingestion** — `src/ingest/download_cricsheet.py`, `parse_cricsheet.py`
2. **Feature engineering** — player form, venue splits, confidence-flagged synthetic features
3. **Modeling** — XGBoost suitability scorer + SHAP explainability
4. **Optimization** — constrained squad-of-20 / playing-XI selection (PuLP)
5. **Explanation layer** — SHAP → natural language
6. **UI** — Streamlit app for venue/date-based recommendations
7. **Deployment** — Streamlit Community Cloud (free tier)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Step 1: download raw data
python src/ingest/download_cricsheet.py

# Step 2: parse into tidy tables
python src/ingest/parse_cricsheet.py
```

## Status

🚧 Work in progress — see the project board / issues for current phase.

## License

MIT (code). Note: Cricsheet data has its own usage terms — see
https://cricsheet.org/downloads/ before redistributing any raw data
files (the `.gitignore` in this repo already excludes raw/processed
data from version control for this reason).

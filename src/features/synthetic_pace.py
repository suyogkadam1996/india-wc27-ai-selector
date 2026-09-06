"""
Synthetic / estimated feature generator — with explicit confidence flagging.

Design principle (see project README): we never invent precise numbers
and present them as measured fact. Every synthetic feature is:
  1. Grounded in a real, citable public source (commentary, published
     average speeds from analysis articles, or well-known role
     classification) — recorded in `source_note`.
  2. Expressed as a CATEGORY or a RANGE, not a fake-precise point value.
  3. Tagged with a `data_confidence` field so downstream models and
     the UI can treat it differently from measured data.

Fill in PLAYER_PACE_NOTES yourself from real sources you've found
(commentary transcripts, published bowling-speed analysis pieces,
cricket almanacks) — do not fabricate figures with no basis.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PaceEstimate:
    player: str
    pace_band: str          # "fast", "fast-medium", "medium", "spin"
    est_kmh_low: Optional[float]
    est_kmh_high: Optional[float]
    data_confidence: str    # "measured" | "estimated_from_commentary" | "role_based_default"
    source_note: str


# Example seed entries — replace/expand with real sourced information.
# This is illustrative structure, NOT a claim that these are verified figures.
PLAYER_PACE_NOTES = [
    PaceEstimate(
        player="Mohammed Shami",
        pace_band="fast-medium",
        est_kmh_low=135,
        est_kmh_high=142,
        data_confidence="estimated_from_commentary",
        source_note="Commonly described as bowling in this range across "
                     "match commentary and analysis pieces; no official "
                     "ball-tracking figure available publicly.",
    ),
    # Add more players here, each backed by a real note.
]


def estimate_missing_pace_band(role: str) -> PaceEstimate:
    """
    Fallback for players with zero public pace information: use a
    coarse role-based default band rather than a specific number.
    """
    defaults = {
        "pace_bowler": ("fast-medium", 130, 140),
        "spin_bowler": ("spin", None, None),
        "all_rounder_pace": ("medium-fast", 125, 135),
    }
    band, low, high = defaults.get(role, ("unknown", None, None))
    return PaceEstimate(
        player="unknown",
        pace_band=band,
        est_kmh_low=low,
        est_kmh_high=high,
        data_confidence="role_based_default",
        source_note=f"No public pace info found; used generic default for role '{role}'.",
    )


def sample_synthetic_delivery_speed(estimate: PaceEstimate, n: int = 1, seed: int = 42) -> np.ndarray:
    """
    If delivery-level granularity is ever genuinely needed (usually it
    is NOT for season/career-level suitability scoring — prefer using
    the band directly as a categorical feature), sample from a normal
    distribution calibrated to the estimated range rather than
    asserting a single fake-precise value.
    """
    if estimate.est_kmh_low is None or estimate.est_kmh_high is None:
        raise ValueError("Cannot sample: no numeric range available for this estimate.")
    mean = (estimate.est_kmh_low + estimate.est_kmh_high) / 2
    std = (estimate.est_kmh_high - estimate.est_kmh_low) / 4  # ~95% within stated range
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=n)


def build_pace_feature_table() -> pd.DataFrame:
    rows = [vars(p) for p in PLAYER_PACE_NOTES]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_pace_feature_table()
    print(df)
    print("\nExample synthetic samples for Shami (illustrative only):")
    print(sample_synthetic_delivery_speed(PLAYER_PACE_NOTES[0], n=5))

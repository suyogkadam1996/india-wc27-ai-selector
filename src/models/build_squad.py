"""
PHASE 3, STEP 3: Squad Selector

WHAT THIS DOES (plain words):
Takes India's ranked players (from show_india_pool.py) and picks an
actual squad of a fixed size, following a typical ODI squad balance:
by default, top 6 batters + top 6 bowlers + top 3 all-rounders = 15.

If a role doesn't have enough qualifying players to fill its quota,
the remaining slots are filled with the next-best players overall
(regardless of role) -- and this is clearly marked in the output, never
silently hidden.

KNOWN LIMITATION (stated plainly, not hidden): this version does not
yet distinguish wicketkeepers from other batters, since that needs a
separate data source we haven't added yet. So "batter" here includes
keeper-batters mixed in with pure batters. This is a good next
improvement, not a bug -- flagged honestly rather than pretending
otherwise.

Run after show_india_pool.py:
    python src/models/build_squad.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Default ODI squad composition target. Adjustable -- these are a
# reasonable, typical balance, not a fixed rule. World Cup squads often
# carry 2 keeper-capable players (1 first-choice, 1 backup).
SQUAD_QUOTAS = {
    "wicketkeeper": 2,
    "batter": 5,
    "bowler": 5,
    "all_rounder": 3,
}
TOTAL_SQUAD_SIZE = sum(SQUAD_QUOTAS.values())  # 15


def select_squad(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.sort_values("suitability_score", ascending=False).copy()
    selected_rows = []
    selected_players = set()

    # Wicketkeepers are pulled out FIRST, regardless of their batter/
    # bowler/all_rounder role label, using the is_wicketkeeper flag.
    if "is_wicketkeeper" in pool.columns:
        is_keeper = pool["is_wicketkeeper"].fillna(False)
    else:
        is_keeper = pd.Series(False, index=pool.index)
    keeper_pool = pool[is_keeper]
    keeper_picks = keeper_pool.head(SQUAD_QUOTAS["wicketkeeper"]).copy()
    keeper_picks["selection_note"] = (
        f"Selected as one of the top {SQUAD_QUOTAS['wicketkeeper']} wicketkeepers available."
    )
    selected_rows.append(keeper_picks)
    selected_players.update(keeper_picks["player"])

    # Then fill each remaining role's quota with its own best players,
    # skipping anyone already picked as a keeper.
    for role, quota in SQUAD_QUOTAS.items():
        if role == "wicketkeeper":
            continue
        role_pool = pool[(pool["role"] == role) & (~pool["player"].isin(selected_players))]
        picks = role_pool.head(quota).copy()
        picks["selection_note"] = f"Selected as one of the top {quota} {role}s available."
        selected_rows.append(picks)
        selected_players.update(picks["player"])

    squad_so_far = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    shortfall = TOTAL_SQUAD_SIZE - len(squad_so_far)

    # Step 2: if any role came up short, fill remaining slots with the
    # next-best players overall, regardless of role -- and say so clearly.
    if shortfall > 0:
        remaining_pool = pool[~pool["player"].isin(selected_players)]
        fill_ins = remaining_pool.head(shortfall).copy()
        fill_ins["selection_note"] = (
            "Filled a remaining squad slot: not enough qualifying players "
            "in the originally targeted role quota, so the next-best "
            "available player overall was picked instead."
        )
        squad_so_far = pd.concat([squad_so_far, fill_ins], ignore_index=True)

    return squad_so_far.sort_values("suitability_score", ascending=False)


def main():
    pool_path = PROCESSED_DIR / "india_player_pool.parquet"
    if not pool_path.exists():
        raise SystemExit(
            f"{pool_path} not found. Run show_india_pool.py first."
        )
    pool = pd.read_parquet(pool_path)

    if "is_wicketkeeper" not in pool.columns:
        raise SystemExit(
            "The file india_player_pool.parquet is missing the 'is_wicketkeeper' "
            "column. This usually means it's a stale file from before wicketkeeper "
            "detection was added. Fix: rerun show_india_pool.py (which reads the "
            "latest player_suitability_scores.parquet), then rerun this script."
        )

    squad = select_squad(pool)

    out_path = PROCESSED_DIR / "india_squad_15.parquet"
    squad.to_parquet(out_path, index=False)

    print(f"Selected a squad of {len(squad)} (target was {TOTAL_SQUAD_SIZE}).\n")

    keepers = squad[squad["is_wicketkeeper"].fillna(False)] if "is_wicketkeeper" in squad.columns else squad.iloc[0:0]
    print(f"--- WICKETKEEPERS ({len(keepers)}) ---")
    print(keepers[["player", "suitability_score"]].to_string(index=False))
    print()

    non_keepers = squad[~squad["player"].isin(keepers["player"])]
    for role in ["batter", "bowler", "all_rounder"]:
        role_squad = non_keepers[non_keepers["role"] == role]
        print(f"--- {role.upper()}S ({len(role_squad)}) ---")
        print(role_squad[["player", "suitability_score"]].to_string(index=False))
        print()

    fill_ins = squad[squad["selection_note"].str.contains("Filled a remaining", na=False)]
    if not fill_ins.empty:
        print(f"NOTE: {len(fill_ins)} slot(s) were filled outside the normal role "
              f"quota due to a shortage of qualifying players in that role:")
        print(fill_ins[["player", "role"]].to_string(index=False))

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

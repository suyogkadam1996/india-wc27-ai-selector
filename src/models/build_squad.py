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

    # STEP 0: honor any manually guaranteed selections FIRST, regardless
    # of their data-driven score -- this is where real cricket knowledge
    # (e.g. a senior player BCCI has confirmed remains in their plans,
    # even through a short-term form dip) overrides the raw numbers.
    # They still occupy their normal role's quota slot -- a guaranteed
    # batter still counts as one of the squad's batter slots, it's just
    # not competing purely on recent-form score for that slot.
    if "guaranteed_selection" in pool.columns:
        guaranteed = pool[pool["guaranteed_selection"].fillna(False)].copy()
        if not guaranteed.empty:
            guaranteed["selection_note"] = (
                "Included based on proven track record and current team backing: "
                + guaranteed["guaranteed_selection_note"].fillna("")
            )
            selected_rows.append(guaranteed)
            selected_players.update(guaranteed["player"])

    # Wicketkeepers are pulled out next, regardless of their batter/
    # bowler/all_rounder role label, using the is_wicketkeeper flag.
    if "is_wicketkeeper" in pool.columns:
        is_keeper = pool["is_wicketkeeper"].fillna(False)
    else:
        is_keeper = pd.Series(False, index=pool.index)
    keeper_pool = pool[is_keeper & (~pool["player"].isin(selected_players))]
    keeper_quota_remaining = SQUAD_QUOTAS["wicketkeeper"] - sum(
        1 for p in selected_players if pool.loc[pool["player"] == p, "is_wicketkeeper"].any()
    )
    keeper_picks = keeper_pool.head(max(keeper_quota_remaining, 0)).copy()
    keeper_picks["selection_note"] = (
        f"Selected as one of the top {SQUAD_QUOTAS['wicketkeeper']} wicketkeepers available."
    )
    selected_rows.append(keeper_picks)
    selected_players.update(keeper_picks["player"])

    # Then fill each remaining role's quota with its own best players,
    # skipping anyone already picked (as guaranteed or as a keeper), and
    # accounting for quota slots a guaranteed pick already used up.
    for role, quota in SQUAD_QUOTAS.items():
        if role == "wicketkeeper":
            continue
        already_used = sum(
            1 for p in selected_players if pool.loc[pool["player"] == p, "role"].eq(role).any()
        )
        remaining_quota = max(quota - already_used, 0)
        role_pool = pool[(pool["role"] == role) & (~pool["player"].isin(selected_players))]
        picks = role_pool.head(remaining_quota).copy()
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

    print_batting_order(squad)
    print(f"\nSaved to {out_path}")


def print_batting_order(squad: pd.DataFrame) -> None:
    """Prints the squad sorted by real, historical batting position --
    openers first, then top/middle order, then lower order, with pure
    bowlers (who have no meaningful batting position data) at the end,
    exactly mirroring a real cricket team sheet rather than grouping
    purely by role label."""
    df = squad.copy()
    if "avg_batting_position" not in df.columns:
        print("\n(Batting-order view unavailable -- rerun build_player_features.py "
              "after updating parse_cricsheet.py to enable this.)")
        return

    # Players with no recorded batting position (specialist bowlers who've
    # never batted in our data) are sorted to the very end, not dropped.
    df["_sort_key"] = df["avg_batting_position"].fillna(99)
    ordered = df.sort_values("_sort_key")

    print("\n" + "=" * 50)
    print("SUGGESTED BATTING ORDER (based on real historical entry positions)")
    print("=" * 50)
    for _, row in ordered.iterrows():
        pos = row["avg_batting_position"]
        pos_label = f"~#{pos:.0f}" if pd.notna(pos) else "no batting data"
        keeper_tag = " (wk)" if row.get("is_wicketkeeper") else ""
        print(f"  {pos_label:>16}  {row['player']}{keeper_tag}  [{row['role']}]")


if __name__ == "__main__":
    main()

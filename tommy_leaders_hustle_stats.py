"""
Tommy (predicted) award winners cross-referenced with hustle defense totals:
steals, blocks, deflections, and charges drawn from the same enriched logs as
predict_tommy_award_other_teams.ipynb.

Outputs:
  predictions/tommy_winners_hustle_leaderboard.csv — all players with wins + hustle sums + per-36 rates

Run from repo root:
  python tommy_leaders_hustle_stats.py

“Hustle-heavy” leader lines in stdout pick players who are elite in total hustle
events (or per-36) and report who has the most predicted Tommy wins in that set.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PRED_DIR = Path(__file__).resolve().parent / "predictions"
OUT_CSV = PRED_DIR / "tommy_winners_hustle_leaderboard.csv"

_ENRICHED_RE = re.compile(r"^(.+)_(20\d{2}-\d{2})_player_game_enriched\.csv$")

HUSTLE_COLS = ["steals", "blocks", "deflections", "charges_drawn"]


def _load_predicted_wins() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for f in sorted(PRED_DIR.glob("*_predicted_tommy_counts_combined.csv")):
        slug = f.name.replace("_predicted_tommy_counts_combined.csv", "")
        df = pd.read_csv(f)
        df["team_slug"] = slug
        df = df.rename(columns={"total": "predicted_wins"})
        frames.append(df[["team_slug", "player_name", "predicted_wins", "2024-25", "2025-26"]])
    return pd.concat(frames, ignore_index=True)


def _load_hustle_totals() -> tuple[pd.DataFrame, pd.Series]:
    rows: list[pd.DataFrame] = []
    usecols = [
        "teamCity",
        "teamName",
        "player_name",
        "minutes_decimal",
        *HUSTLE_COLS,
    ]
    for f in sorted(PRED_DIR.glob("*_player_game_enriched.csv")):
        m = _ENRICHED_RE.match(f.name)
        if not m:
            continue
        team_slug = m.group(1)
        df = pd.read_csv(f, usecols=usecols)
        df["team_slug"] = team_slug
        df["team"] = df["teamCity"].astype(str) + " " + df["teamName"].astype(str)
        rows.append(df)

    g = pd.concat(rows, ignore_index=True)
    g["minutes_decimal"] = pd.to_numeric(g["minutes_decimal"], errors="coerce").fillna(0)
    for c in HUSTLE_COLS:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0)

    sums = g.groupby(["team_slug", "player_name"], as_index=False).agg(
        total_minutes=("minutes_decimal", "sum"),
        steals_total=("steals", "sum"),
        blocks_total=("blocks", "sum"),
        deflections_total=("deflections", "sum"),
        charges_drawn_total=("charges_drawn", "sum"),
    )
    sums["hustle_events_total"] = (
        sums["steals_total"]
        + sums["blocks_total"]
        + sums["deflections_total"]
        + sums["charges_drawn_total"]
    )
    m = sums["total_minutes"].clip(lower=1e-6)
    sums["hustle_events_per_36"] = sums["hustle_events_total"] / m * 36.0

    team_display = g.groupby("team_slug", as_index=True)["team"].first()
    return sums, team_display


def main() -> None:
    wins = _load_predicted_wins()
    hustle, team_display = _load_hustle_totals()
    df = wins.merge(hustle, on=["team_slug", "player_name"], how="left")
    df["team"] = df["team_slug"].map(team_display)

    df = df.sort_values(["predicted_wins", "hustle_events_total"], ascending=[False, False]).reset_index(
        drop=True
    )

    cols = [
        "team",
        "player_name",
        "predicted_wins",
        "2024-25",
        "2025-26",
        "total_minutes",
        "steals_total",
        "blocks_total",
        "deflections_total",
        "charges_drawn_total",
        "hustle_events_total",
        "hustle_events_per_36",
    ]
    df[cols].to_csv(OUT_CSV, index=False)

    min_minutes = 500.0
    qualified = df[df["total_minutes"] >= min_minutes].copy()

    # Among elite hustle volume (top 15 by raw hustle events), who has the most Tommy wins?
    top_hustle_n = 15
    hustle_cutoff = qualified.nlargest(top_hustle_n, "hustle_events_total")
    hustle_leader_row = hustle_cutoff.sort_values("predicted_wins", ascending=False).iloc[0]

    # Among elite hustle rate (top 15 by per-36), who has the most Tommy wins?
    rate_cutoff = qualified.nlargest(top_hustle_n, "hustle_events_per_36")
    rate_leader_row = rate_cutoff.sort_values("predicted_wins", ascending=False).iloc[0]

    # Single “composite” line: best predicted wins among players >= 90th pct hustle total (qualified only)
    p90 = qualified["hustle_events_total"].quantile(0.90)
    high_hustle = qualified[qualified["hustle_events_total"] >= p90]
    p90_leader = high_hustle.sort_values("predicted_wins", ascending=False).iloc[0]

    print(f"Wrote {OUT_CSV}\n")
    print(
        f"Among top {top_hustle_n} players by hustle_events_total (steals+blocks+deflections+charges), "
        f"min {min_minutes:.0f} min — most predicted Tommy wins:\n"
        f"  {hustle_leader_row['player_name']} ({hustle_leader_row['team']}) — "
        f"{hustle_leader_row['predicted_wins']:.0f} wins, "
        f"{hustle_leader_row['hustle_events_total']:.0f} hustle events, "
        f"{hustle_leader_row['hustle_events_per_36']:.2f} per 36\n"
    )
    print(
        f"Among top {top_hustle_n} by hustle_events_per_36, min {min_minutes:.0f} min — "
        f"most predicted Tommy wins:\n"
        f"  {rate_leader_row['player_name']} ({rate_leader_row['team']}) — "
        f"{rate_leader_row['predicted_wins']:.0f} wins, "
        f"{rate_leader_row['hustle_events_per_36']:.2f} per 36, "
        f"{rate_leader_row['hustle_events_total']:.0f} events\n"
    )
    print(
        f"Among players at/above 90th percentile hustle_events_total (qualified), "
        f"most predicted Tommy wins:\n"
        f"  {p90_leader['player_name']} ({p90_leader['team']}) — "
        f"{p90_leader['predicted_wins']:.0f} wins (p90 cutoff ≈ {p90:.0f} events)\n"
    )
    print("Top 12 by predicted Tommy wins (with hustle totals):\n")
    print(
        df.head(12)[
            [
                "player_name",
                "team",
                "predicted_wins",
                "steals_total",
                "blocks_total",
                "deflections_total",
                "charges_drawn_total",
                "hustle_events_total",
                "hustle_events_per_36",
            ]
        ].to_string(index=False)
    )

    # Raw hustle king (for reference)
    hk = qualified.nlargest(1, "hustle_events_total").iloc[0]
    print(
        f"\nMost hustle events overall (qualified): {hk['player_name']} ({hk['team']}) — "
        f"{hk['hustle_events_total']:.0f} events, {hk['predicted_wins']:.0f} predicted Tommy wins"
    )


if __name__ == "__main__":
    main()

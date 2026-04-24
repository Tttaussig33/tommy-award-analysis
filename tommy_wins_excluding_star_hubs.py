"""
Tommy win leaders after removing each team's offensive hub:
players who are BOTH #1 on their team in total points AND #1 in minutes-weighted usage.

Uses the same artifacts as predict_tommy_award_other_teams.ipynb:
  predictions/*_predicted_tommy_counts_combined.csv
  predictions/*_player_game_enriched.csv

Run from repo root:
  python tommy_wins_excluding_star_hubs.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PRED_DIR = Path(__file__).resolve().parent / "predictions"

_ENRICHED_RE = re.compile(r"^(.+)_(20\d{2}-\d{2})_player_game_enriched\.csv$")


def _load_predicted_wins() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for f in sorted(PRED_DIR.glob("*_predicted_tommy_counts_combined.csv")):
        slug = f.name.replace("_predicted_tommy_counts_combined.csv", "")
        df = pd.read_csv(f)
        df["team_slug"] = slug
        df = df.rename(columns={"total": "predicted_wins"})
        frames.append(df[["team_slug", "player_name", "predicted_wins", "2024-25", "2025-26"]])
    return pd.concat(frames, ignore_index=True)


def _load_offensive_totals() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    usecols = ["teamCity", "teamName", "player_name", "points", "usage_rate", "minutes_decimal"]
    for f in sorted(PRED_DIR.glob("*_player_game_enriched.csv")):
        m = _ENRICHED_RE.match(f.name)
        if not m:
            continue
        team_slug = m.group(1)
        df = pd.read_csv(f, usecols=usecols)
        df["team_slug"] = team_slug
        df["team_display"] = df["teamCity"].astype(str) + " " + df["teamName"].astype(str)
        rows.append(df)
    g = pd.concat(rows, ignore_index=True)
    g["points"] = pd.to_numeric(g["points"], errors="coerce").fillna(0)
    g["usage_rate"] = pd.to_numeric(g["usage_rate"], errors="coerce")
    g["minutes_decimal"] = pd.to_numeric(g["minutes_decimal"], errors="coerce").clip(lower=0)
    g["_usage_minutes"] = g["usage_rate"] * g["minutes_decimal"]

    sums = g.groupby(["team_slug", "player_name"], as_index=False).agg(
        total_points=("points", "sum"),
        minutes_sum=("minutes_decimal", "sum"),
        usage_minutes_sum=("_usage_minutes", "sum"),
    )
    ok = sums["minutes_sum"] > 0
    sums["usage_rate_wm"] = sums["usage_minutes_sum"] / sums["minutes_sum"].where(ok)
    sums.loc[~ok, "usage_rate_wm"] = float("nan")
    sums = sums.drop(columns=["usage_minutes_sum"])

    team_display = g.groupby("team_slug", as_index=True)["team_display"].first().rename("team")
    return sums, team_display


def main() -> None:
    wins = _load_predicted_wins()
    off, team_display = _load_offensive_totals()
    df = wins.merge(off, on=["team_slug", "player_name"], how="left")
    df["team"] = df["team_slug"].map(team_display)

    df["pts_rank_team"] = df.groupby("team_slug")["total_points"].rank(method="min", ascending=False)
    df["usage_rank_team"] = df.groupby("team_slug")["usage_rate_wm"].rank(method="min", ascending=False)
    df["is_star_hub"] = (df["pts_rank_team"] == 1) & (df["usage_rank_team"] == 1)

    filtered = df[~df["is_star_hub"]].copy()
    filtered = filtered.sort_values("predicted_wins", ascending=False).reset_index(drop=True)

    print(
        "Excluded (team #1 in total points AND team #1 in minutes-weighted usage):\n",
        df.loc[df["is_star_hub"], ["team", "player_name", "total_points", "usage_rate_wm", "predicted_wins"]]
        .sort_values(["team", "player_name"])
        .to_string(index=False),
        "\n",
    )
    print("Top 40 remaining by predicted Tommy wins:\n")
    print(
        filtered.head(40)[
            ["team", "player_name", "predicted_wins", "2024-25", "2025-26", "total_points", "usage_rate_wm"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

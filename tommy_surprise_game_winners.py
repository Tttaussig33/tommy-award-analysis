"""
Game-level “surprise” Tommy (predicted) winners — non-box-score heroes.

For each team-game in predictions/*_player_game_enriched.csv:
  - Predicted winner = row with max pred_prob (same as the notebook).
  - Surprise game = winner was NOT in the team’s top-2 for points AND NOT in the
    top-2 for minutes that game (using points_rank and minutes_decimal_rank).

Equivalently: points_rank > 2 and minutes_decimal_rank > 2 for the winner.

Outputs:
  predictions/tommy_surprise_winner_counts.csv
  predictions/tommy_surprise_games_detail.csv

Run from repo root:
  python tommy_surprise_game_winners.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PRED_DIR = Path(__file__).resolve().parent / "predictions"
OUT_COUNTS = PRED_DIR / "tommy_surprise_winner_counts.csv"
OUT_DETAIL = PRED_DIR / "tommy_surprise_games_detail.csv"

_ENRICHED_RE = re.compile(r"^(.+)_(20\d{2}-\d{2})_player_game_enriched\.csv$")


def main() -> None:
    detail_rows: list[dict] = []

    for f in sorted(PRED_DIR.glob("*_player_game_enriched.csv")):
        m = _ENRICHED_RE.match(f.name)
        if not m:
            continue
        team_slug = m.group(1)
        season = m.group(2)
        df = pd.read_csv(
            f,
            usecols=[
                "GAME_ID",
                "teamCity",
                "teamName",
                "player_name",
                "pred_prob",
                "points_rank",
                "minutes_decimal_rank",
                "game_date",
            ],
        )
        df["pred_prob"] = pd.to_numeric(df["pred_prob"], errors="coerce")
        df["points_rank"] = pd.to_numeric(df["points_rank"], errors="coerce")
        df["minutes_decimal_rank"] = pd.to_numeric(df["minutes_decimal_rank"], errors="coerce")
        df["team"] = df["teamCity"].astype(str) + " " + df["teamName"].astype(str)

        for gid, g in df.groupby("GAME_ID", sort=False):
            if g["pred_prob"].notna().sum() == 0:
                continue
            idx = g["pred_prob"].idxmax()
            row = g.loc[idx]
            pr, mr = row["points_rank"], row["minutes_decimal_rank"]
            if pd.isna(pr) or pd.isna(mr):
                continue
            if pr > 2 and mr > 2:
                detail_rows.append(
                    {
                        "GAME_ID": gid,
                        "game_date": row["game_date"],
                        "season": season,
                        "team_slug": team_slug,
                        "team": row["team"],
                        "player_name": row["player_name"],
                        "points_rank": int(pr),
                        "minutes_decimal_rank": int(mr),
                        "pred_prob": float(row["pred_prob"]),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        print("No surprise games found (check data).")
        return

    detail = detail.sort_values(["team", "game_date", "GAME_ID"]).reset_index(drop=True)
    detail.to_csv(OUT_DETAIL, index=False)

    counts = (
        detail.groupby(["team", "player_name"], as_index=False)
        .size()
        .rename(columns={"size": "surprise_wins"})
        .sort_values("surprise_wins", ascending=False)
        .reset_index(drop=True)
    )
    counts.to_csv(OUT_COUNTS, index=False)

    print(f"Wrote {OUT_DETAIL} ({len(detail)} surprise games)")
    print(f"Wrote {OUT_COUNTS}\n")
    print("Top 25 players by surprise predicted-Tommy games:\n")
    print(counts.head(25).to_string(index=False))
    print(f"\nTotal surprise games (league-wide): {len(detail)}")


if __name__ == "__main__":
    main()

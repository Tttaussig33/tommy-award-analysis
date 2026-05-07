"""
Overwrite `net_rating` in Tommy_Award_Player_Game_Table_hustle.csv using NBA.com
per-game advanced box score (`BoxScoreAdvancedV3.netRating` — on-court net per 100 poss).

Run from repo root:
  python3 csv_builders/update_tommy_hustle_net_rating.py

Or:
  cd csv_builders && python3 update_tommy_hustle_net_rating.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Import fetcher from sibling module
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from enrich_player_game_with_hustle import fetch_advanced_for_game  # noqa: E402


def normalize_game_id(value) -> str:
    if value is None or pd.isna(value):
        return ""
    digits = str(value).strip()
    if not digits:
        return digits
    return digits.zfill(10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set net_rating from NBA BoxScoreAdvancedV3.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "Tommy_Award_Player_Game_Table_hustle.csv",
        help="Path to hustle player-game CSV.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.35,
        help="Seconds between game requests (be polite to stats.nba.com).",
    )
    args = parser.parse_args()

    path = args.input.resolve()
    if not path.is_file():
        raise SystemExit(f"Input not found: {path}")

    df = pd.read_csv(path, dtype={"gameId": str, "GAME_ID": str, "personId": str})
    game_col = "gameId" if "gameId" in df.columns else "GAME_ID"
    if game_col not in df.columns or "personId" not in df.columns:
        raise SystemExit("CSV must include gameId (or GAME_ID) and personId.")

    df["_GAME_ID_KEY"] = df[game_col].map(normalize_game_id)
    df["_PERSON_ID_KEY"] = df["personId"].astype(str).str.strip()

    unique_games = sorted({g for g in df["_GAME_ID_KEY"].tolist() if g})
    print(f"Fetching net_rating for {len(unique_games)} games from NBA API…")

    frames: list[pd.DataFrame] = []
    for i, gid in enumerate(unique_games, start=1):
        if i == 1 or i % 25 == 0 or i == len(unique_games):
            print(f"  {i}/{len(unique_games)} …")
        adv = fetch_advanced_for_game(gid)
        if adv is not None and not adv.empty and "net_rating" in adv.columns:
            frames.append(adv[["GAME_ID", "personId", "net_rating"]].copy())
        time.sleep(args.sleep)

    if not frames:
        raise SystemExit("No advanced box score data returned; check network / game IDs.")

    net_df = pd.concat(frames, ignore_index=True)
    net_df = net_df.rename(
        columns={"GAME_ID": "_GAME_ID_KEY", "personId": "_PERSON_ID_KEY", "net_rating": "_net_rating_api"}
    )
    net_df = net_df.drop_duplicates(subset=["_GAME_ID_KEY", "_PERSON_ID_KEY"], keep="first")

    if "net_rating" in df.columns:
        df = df.drop(columns=["net_rating"])

    df = df.merge(net_df, on=["_GAME_ID_KEY", "_PERSON_ID_KEY"], how="left")
    df = df.rename(columns={"_net_rating_api": "net_rating"})
    df = df.drop(columns=["_GAME_ID_KEY", "_PERSON_ID_KEY"])

    # Keep net_rating next to plusMinusPoints when both exist
    cols = list(df.columns)
    if "plusMinusPoints" in cols and "net_rating" in cols:
        cols.remove("net_rating")
        ins = cols.index("plusMinusPoints") + 1
        cols = cols[:ins] + ["net_rating"] + cols[ins:]
        df = df[cols]

    df.to_csv(path, index=False)
    n_ok = int(df["net_rating"].notna().sum())
    print(f"Wrote {path}")
    print(f"Rows: {len(df)} | net_rating non-null: {n_ok} ({100.0 * n_ok / len(df):.1f}%)")


if __name__ == "__main__":
    main()

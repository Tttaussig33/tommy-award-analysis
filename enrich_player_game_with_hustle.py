import argparse
import time
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import boxscorehustlev2, hustlestatsboxscore


REQUEST_SLEEP = 0.1
REQUEST_TIMEOUT = 4
MAX_RETRIES = 1


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def fetch_hustle_for_game(game_id: str) -> pd.DataFrame:
    """
    Return columns: GAME_ID, personId, deflections, charges_drawn.
    Empty dataframe means hustle data was unavailable for the game.
    """
    fetch_attempts = [
        (
            "v2",
            lambda gid: boxscorehustlev2.BoxScoreHustleV2(
                game_id=gid, timeout=REQUEST_TIMEOUT
            ).get_data_frames(),
        ),
        (
            "legacy",
            lambda gid: hustlestatsboxscore.HustleStatsBoxScore(
                game_id=gid, timeout=REQUEST_TIMEOUT
            ).get_data_frames(),
        ),
    ]

    for endpoint_name, fetcher in fetch_attempts:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                dfs = fetcher(game_id)
                if not dfs:
                    continue
                player_df = dfs[0]
                if player_df is None or player_df.empty:
                    continue

                id_col = find_col(player_df, ["personId", "PLAYER_ID", "playerId", "player_id"])
                defl_col = find_col(player_df, ["deflections", "DEFLECTIONS"])
                charge_col = find_col(player_df, ["chargesDrawn", "CHARGES_DRAWN"])
                if id_col is None:
                    continue

                out = pd.DataFrame(
                    {
                        "GAME_ID": str(game_id),
                        "personId": player_df[id_col].astype(str),
                        "deflections": player_df[defl_col] if defl_col else pd.NA,
                        "charges_drawn": player_df[charge_col] if charge_col else pd.NA,
                    }
                )
                return out
            except Exception:
                if attempt == MAX_RETRIES:
                    # Give visibility into slower/unavailable historical games.
                    print(f"Skipping game {game_id} on {endpoint_name}: unavailable/timeout")
                continue

    return pd.DataFrame(columns=["GAME_ID", "personId", "deflections", "charges_drawn"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add deflections and charges_drawn to player-game table using NBA hustle endpoints."
    )
    parser.add_argument(
        "--input",
        default="Tommy_Award_Player_Game_Table.csv",
        help="Input player-game table CSV.",
    )
    parser.add_argument(
        "--output",
        default="Tommy_Award_Player_Game_Table_hustle.csv",
        help="Output CSV with hustle columns appended.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype={"GAME_ID": str, "gameId": str, "personId": str})

    game_col = find_col(df, ["GAME_ID", "gameId"])
    player_col = find_col(df, ["personId", "PLAYER_ID", "playerId", "player_id"])
    if game_col is None or player_col is None:
        raise ValueError(
            "Input must contain game id and player id columns (GAME_ID/gameId and personId/PLAYER_ID/playerId/player_id)."
        )

    df["GAME_ID_KEY"] = df[game_col].astype(str)
    df["PERSON_ID_KEY"] = df[player_col].astype(str)

    unique_games = sorted(df["GAME_ID_KEY"].dropna().unique().tolist())
    hustle_frames = []

    for idx, game_id in enumerate(unique_games, start=1):
        if idx % 50 == 0 or idx == 1 or idx == len(unique_games):
            print(f"Fetching hustle: {idx}/{len(unique_games)} games")
        hustle_frames.append(fetch_hustle_for_game(game_id))
        time.sleep(REQUEST_SLEEP)

    hustle_df = pd.concat(hustle_frames, ignore_index=True)
    if not hustle_df.empty:
        hustle_df = hustle_df.drop_duplicates(subset=["GAME_ID", "personId"], keep="first")

    merged = df.merge(
        hustle_df,
        left_on=["GAME_ID_KEY", "PERSON_ID_KEY"],
        right_on=["GAME_ID", "personId"],
        how="left",
    )

    merged = merged.drop(columns=["GAME_ID_KEY", "PERSON_ID_KEY", "GAME_ID", "personId"], errors="ignore")
    merged["deflections"] = pd.to_numeric(merged["deflections"], errors="coerce")
    merged["charges_drawn"] = pd.to_numeric(merged["charges_drawn"], errors="coerce")
    merged.to_csv(args.output, index=False)

    print("\nDone.")
    print(f"Wrote: {args.output}")
    print(f"Rows: {len(merged)}")
    print(f"Rows with deflections: {int(merged['deflections'].notna().sum())}")
    print(f"Rows with charges_drawn: {int(merged['charges_drawn'].notna().sum())}")


if __name__ == "__main__":
    main()

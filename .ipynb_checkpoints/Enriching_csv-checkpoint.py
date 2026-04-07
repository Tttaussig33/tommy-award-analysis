import time
from ast import literal_eval
from difflib import get_close_matches
from pathlib import Path
import pandas as pd

from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    LeagueGameFinder,
    BoxScoreTraditionalV3,
)

CELTICS_TEAM_ID = 1610612738


def get_team_id(team_name="Boston Celtics"):
    """
    Return the NBA team_id from a team name.
    """
    all_teams = teams.get_teams()
    match = [t for t in all_teams if t["full_name"].lower() == team_name.lower()]
    if not match:
        raise ValueError(f"Could not find team: {team_name}")
    return match[0]["id"]


def safe_api_call(endpoint_class, pause=0.75, retries=3, timeout=60, **kwargs):
    """
    Small wrapper so you can slow requests down and retry slow nba_api calls.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            obj = endpoint_class(timeout=timeout, **kwargs)
            time.sleep(pause)
            return obj
        except Exception as e:
            last_error = e
            if attempt == retries:
                raise

            wait_time = pause * attempt
            print(
                f"{endpoint_class.__name__} failed on attempt {attempt}/{retries} "
                f"with {e}. Retrying in {wait_time:.2f}s..."
            )
            time.sleep(wait_time)

    raise last_error


def coerce_winner_names(value):
    """
    Normalize winner_names values loaded from either memory or CSV.
    """
    if isinstance(value, list):
        return [str(name).strip() for name in value if str(name).strip()]

    if value is None or pd.isna(value):
        return []

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(name).strip() for name in parsed if str(name).strip()]
            except (SyntaxError, ValueError):
                pass
        if " || " in stripped:
            return [name.strip() for name in stripped.split(" || ") if name.strip()]
        return [stripped]

    return [str(value).strip()]


def normalize_game_id(value):
    """
    Keep NBA GAME_ID values as zero-padded 10-character strings.
    """
    if value is None or pd.isna(value):
        return value
    digits = str(value).strip()
    if not digits:
        return digits
    return digits.zfill(10)


def get_player_name_column(df):
    """
    Return the player name column used by the dataframe.
    """
    for col in [
        "PLAYER_NAME",
        "playerName",
        "PERSON_NAME",
        "personName",
        "name",
        "PLAYER",
    ]:
        if col in df.columns:
            return col
    return None


def build_player_name_series(df):
    """
    Build a player name series across older and newer nba_api schemas.
    """
    player_col = get_player_name_column(df)
    if player_col is not None:
        return df[player_col]

    first_last_pairs = [
        ("firstName", "familyName"),
        ("FIRST_NAME", "LAST_NAME"),
        ("first_name", "last_name"),
    ]
    for first_col, last_col in first_last_pairs:
        if first_col in df.columns and last_col in df.columns:
            return (
                df[first_col].fillna("").astype(str).str.strip()
                + " "
                + df[last_col].fillna("").astype(str).str.strip()
            ).str.strip()

    return None


def get_team_id_column(df):
    """
    Return the team id column used by the dataframe.
    """
    for col in ["TEAM_ID", "teamId", "team_id"]:
        if col in df.columns:
            return col
    return None


def filter_team_players(df, team_id=CELTICS_TEAM_ID):
    """
    Keep only the requested team's player rows.
    """
    out = df.copy()
    team_col = get_team_id_column(out)
    if team_col is None:
        raise ValueError("Could not find a team id column in player box score data.")
    return out[out[team_col].astype(str) == str(team_id)].copy()


def get_date_column(df):
    """
    Return the game date column used by the dataframe.
    """
    for col in ["GAME_DATE", "GAME_DATE_EST", "gameDate"]:
        if col in df.columns:
            return col
    return None


def get_game_id_column(df):
    """
    Return the game id column used by the dataframe.
    """
    for col in ["GAME_ID", "gameId"]:
        if col in df.columns:
            return col
    return None


def season_from_date(game_date):
    """
    Convert a calendar date into an NBA season string like 2023-24.
    """
    ts = pd.to_datetime(game_date)
    start_year = ts.year if ts.month >= 10 else ts.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def get_team_schedule(team_id=CELTICS_TEAM_ID, season="2023-24", season_type="Regular Season"):
    """
    Returns one row per team game for the requested season.
    """
    res = safe_api_call(
        LeagueGameFinder,
        team_id_nullable=str(team_id),
        season_nullable=season,
        season_type_nullable=season_type,
        player_or_team_abbreviation="T",
    )

    df = res.get_data_frames()[0].copy()

    # Keep one row per game for the team
    if "GAME_ID" in df.columns:
        df = df.drop_duplicates(subset=["GAME_ID"]).reset_index(drop=True)

    return df


def get_team_schedule_for_labels(
    labels_df,
    team_id=CELTICS_TEAM_ID,
    include_playoffs=True,
):
    """
    Pull the Celtics schedule for every season represented in the winners CSV.
    """
    labels = labels_df.copy()
    labels["game_date"] = pd.to_datetime(labels["game_date"])
    seasons = sorted(labels["game_date"].map(season_from_date).unique())

    season_types = ["Regular Season"]
    if include_playoffs:
        season_types.append("Playoffs")

    all_schedules = []
    for season in seasons:
        for season_type in season_types:
            season_df = get_team_schedule(
                team_id=team_id,
                season=season,
                season_type=season_type,
            )
            season_df["SEASON_TYPE"] = season_type
            all_schedules.append(season_df)

    if not all_schedules:
        return pd.DataFrame()

    schedule_df = pd.concat(all_schedules, ignore_index=True)
    game_id_col = get_game_id_column(schedule_df)
    date_col = get_date_column(schedule_df)

    if game_id_col is None or date_col is None:
        raise ValueError("Could not find GAME_ID or GAME_DATE column in team schedule output.")

    schedule_df["GAME_ID"] = schedule_df[game_id_col].astype(str)
    schedule_df["game_date"] = pd.to_datetime(schedule_df[date_col]).dt.normalize()
    schedule_df = schedule_df.drop_duplicates(subset=["GAME_ID"]).reset_index(drop=True)
    return schedule_df


def get_traditional_player_box(game_id):
    """
    Returns player traditional box score rows for one game.
    """
    res = safe_api_call(BoxScoreTraditionalV3, game_id=game_id)

    # V3 endpoints can expose multiple result tables; the player table is what we want.
    dfs = res.get_data_frames()

    # Usually one of these contains player rows with PLAYER_NAME / team info.
    for df in dfs:
        cols = set(df.columns)
        if "PLAYER_NAME" in cols or "personId" in cols:
            out = df.copy()
            out["GAME_ID"] = game_id
            return out

    raise ValueError(f"Could not find player traditional box score table for game {game_id}")


def normalize_player_name(name):
    """
    Simple cleanup for matching your spreadsheet names to nba_api names.
    """
    if pd.isna(name):
        return name
    return (
        str(name)
        .strip()
        .lower()
        .replace(".", "")
        .replace(",", "")
        .replace("’", "'")
        .replace(" jr", " jr")
        .replace(" sr", " sr")
        .replace("iii", " iii")
        .replace("ii", " ii")
    )


def add_player_name_key(df):
    """
    Add a normalized player name key used for merging and labeling.
    """
    out = df.copy()
    player_names = build_player_name_series(out)
    if player_names is None:
        raise ValueError("Could not find a player name column in player box score data.")
    out["player_name"] = player_names
    out["player_name_key"] = out["player_name"].map(normalize_player_name)
    return out


def add_winner_label(player_game_df, winner_names):
    """
    Adds y=1 for the Tommy winner(s) and 0 for everyone else in that game.
    """
    df = player_game_df.copy()
    available_keys = df["player_name_key"].dropna().unique().tolist()
    matched_keys = set()

    for winner_name in winner_names:
        winner_key = normalize_player_name(winner_name)
        if winner_key in available_keys:
            matched_keys.add(winner_key)
            continue

        close_match = get_close_matches(winner_key, available_keys, n=1, cutoff=0.82)
        if close_match:
            matched_keys.add(close_match[0])
            print(f'Fuzzy matched "{winner_name}" to "{close_match[0]}"')
        else:
            print(f'Could not match winner "{winner_name}" in game {df["GAME_ID"].iloc[0]}')

    df["y"] = df["player_name_key"].isin(matched_keys).astype(int)
    return df


def build_one_game_dataset(game_id, winner_names, team_id=CELTICS_TEAM_ID):
    """
    Pull traditional box score stats for one game and label Celtics players.
    """
    trad = get_traditional_player_box(game_id)
    trad = filter_team_players(trad, team_id=team_id)
    trad = add_player_name_key(trad)
    labeled = add_winner_label(trad, winner_names)
    return labeled


def load_tommy_winners_csv(csv_path):
    """
    Read the Tommy winners CSV and standardize its columns.
    """
    labels_df = pd.read_csv(csv_path).copy()
    labels_df = labels_df.rename(columns={"Player": "winner_name", "Date": "game_date"})

    required_cols = {"winner_name", "game_date"}
    missing_cols = required_cols - set(labels_df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_cols)}")

    labels_df["winner_name"] = labels_df["winner_name"].astype(str).str.strip()
    labels_df["game_date"] = pd.to_datetime(
        labels_df["game_date"],
        format="%m/%d/%y",
    ).dt.normalize()
    return labels_df


def attach_game_ids_to_winners(labels_df, team_id=CELTICS_TEAM_ID):
    """
    Match each winner date in the CSV to the Celtics GAME_ID.
    """
    schedule_df = get_team_schedule_for_labels(labels_df, team_id=team_id)
    schedule_lookup = schedule_df.set_index("game_date")
    matched_rows = []
    unresolved = []

    for _, row in labels_df.iterrows():
        game_date = row["game_date"]

        if game_date in schedule_lookup.index:
            schedule_row = schedule_lookup.loc[game_date]
            if isinstance(schedule_row, pd.DataFrame):
                schedule_row = schedule_row.iloc[0]
            matched_rows.append({**row.to_dict(), **schedule_row.to_dict()})
            continue

        nearest_schedule = schedule_df.copy()
        nearest_schedule["day_gap"] = (
            nearest_schedule["game_date"] - game_date
        ).abs().dt.days
        nearest_schedule = nearest_schedule.sort_values(["day_gap", "game_date"])

        if not nearest_schedule.empty and nearest_schedule.iloc[0]["day_gap"] <= 1:
            schedule_row = nearest_schedule.iloc[0]
            matched_rows.append({**row.to_dict(), **schedule_row.to_dict()})
            print(
                "Shifted winner date "
                f"{game_date.strftime('%Y-%m-%d')} to nearest Celtics game "
                f"{schedule_row['game_date'].strftime('%Y-%m-%d')}"
            )
        else:
            unresolved.append(row.to_dict())

    if unresolved:
        unresolved_df = pd.DataFrame(unresolved)
        missing_dates = unresolved_df["game_date"].dt.strftime("%Y-%m-%d").unique().tolist()
        raise ValueError(
            "Could not match some winner dates to Celtics games, even within +/- 1 day: "
            + ", ".join(missing_dates)
        )

    return pd.DataFrame(matched_rows)


def prepare_game_level_labels(source_df):
    """
    Standardize game-level winner labels regardless of input source.
    """
    labels = source_df.copy()
    labels["GAME_ID"] = labels["GAME_ID"].map(normalize_game_id)
    labels["game_date"] = pd.to_datetime(labels["game_date"]).dt.normalize()

    if "winner_names" in labels.columns:
        labels["winner_names"] = labels["winner_names"].map(coerce_winner_names)
    elif "winner_name" in labels.columns:
        labels["winner_names"] = labels["winner_name"].map(lambda name: coerce_winner_names(name))
    else:
        raise ValueError("Input labels must contain winner_name or winner_names.")

    if "SEASON" not in labels.columns:
        labels["SEASON"] = labels["game_date"].map(season_from_date)

    return labels[["GAME_ID", "game_date", "winner_names", "SEASON"]].copy()


def append_and_dedupe_csv(new_df, output_path, dedupe_cols):
    """
    Append rows to an existing CSV and remove duplicates.
    """
    output_file = Path(output_path)
    combined = new_df.copy()

    if output_file.exists():
        existing_df = pd.read_csv(output_file, dtype={"GAME_ID": str})
        combined = pd.concat([existing_df, new_df], ignore_index=True)

    if "GAME_ID" in combined.columns:
        combined["GAME_ID"] = combined["GAME_ID"].map(normalize_game_id)

    usable_dedupe_cols = [col for col in dedupe_cols if col in combined.columns]
    if usable_dedupe_cols:
        combined = combined.drop_duplicates(subset=usable_dedupe_cols, keep="first")
    else:
        combined = combined.drop_duplicates()

    combined.to_csv(output_file, index=False)
    return combined


def get_processed_game_ids(output_path):
    """
    Read already-saved player rows and return completed GAME_IDs.
    """
    output_file = Path(output_path)
    if not output_file.exists():
        return set()

    existing_df = pd.read_csv(output_file, dtype={"GAME_ID": str})
    if "GAME_ID" not in existing_df.columns:
        return set()

    return {
        normalize_game_id(game_id)
        for game_id in existing_df["GAME_ID"].dropna().unique().tolist()
    }


def build_season_dataset(labels_df, season=None, stop_on_first_failure=False):
    """
    labels_df should contain at least:
      - GAME_ID
      - winner_name or winner_names

    Returns:
      - dataset_df: one row per player per game
      - failed_games_df: one row per game that failed to download/process
      - remaining_games_df: queue of games not attempted yet
    """
    all_games = []
    failed_games = []
    labels = labels_df.copy()
    remaining_games = pd.DataFrame(columns=labels.columns)

    if season is not None:
        if "SEASON" in labels.columns:
            labels = labels[labels["SEASON"] == season].copy()
        else:
            labels = labels[labels["game_date"].map(season_from_date) == season].copy()

    labels = labels.reset_index(drop=True)

    for idx, row in labels.iterrows():
        game_id = str(row["GAME_ID"])
        if "winner_names" in row.index:
            winner_names = coerce_winner_names(row["winner_names"])
        elif "winner_name" in row.index:
            winner_names = coerce_winner_names(row["winner_name"])
        else:
            raise ValueError(f"Missing winner name data for game {game_id}")

        if not winner_names:
            raise ValueError(f"Missing winner name data for game {game_id}")

        try:
            game_df = build_one_game_dataset(game_id, winner_names)
            game_df["winner_names"] = " || ".join(winner_names)
            game_df["game_date"] = row["game_date"]
            all_games.append(game_df)
            print(f"Finished {game_id}")
        except Exception as e:
            print(f"Failed {game_id}: {e}")
            failed_games.append(
                {
                    "GAME_ID": game_id,
                    "game_date": row["game_date"],
                    "winner_names": " || ".join(winner_names),
                    "error": str(e),
                }
            )
            if stop_on_first_failure:
                remaining_games = labels.iloc[idx + 1:].copy()
                break

    dataset_df = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()
    failed_games_df = pd.DataFrame(failed_games)
    if not remaining_games.empty:
        remaining_games = prepare_game_level_labels(remaining_games)
        remaining_games["winner_names"] = remaining_games["winner_names"].map(
            lambda names: " || ".join(names)
        )

    return dataset_df, failed_games_df, remaining_games


def build_dataset_from_winners_csv(
    csv_path=None,
    output_path=None,
    failed_output_path=None,
    remaining_output_path=None,
    queue_csv_path=None,
    stop_on_first_failure=False,
    team_id=CELTICS_TEAM_ID,
):
    """
    Build the full labeled player-game dataset from Tommy_Award_Winners.csv.
    """
    if queue_csv_path is not None and Path(queue_csv_path).exists():
        print(f"Resuming from {queue_csv_path}")
        game_level_labels = prepare_game_level_labels(
            pd.read_csv(queue_csv_path, dtype={"GAME_ID": str})
        )
    else:
        winners_df = load_tommy_winners_csv(csv_path)
        winners_with_games = attach_game_ids_to_winners(winners_df, team_id=team_id)

        game_level_labels = (
            winners_with_games.groupby(["GAME_ID", "game_date"], as_index=False)
            .agg(
                winner_names=("winner_name", lambda names: sorted(set(names))),
                SEASON=("game_date", lambda dates: season_from_date(dates.iloc[0])),
            )
        )
        game_level_labels = prepare_game_level_labels(game_level_labels)

    if output_path is not None:
        processed_game_ids = get_processed_game_ids(output_path)
        if processed_game_ids:
            starting_count = len(game_level_labels)
            game_level_labels = game_level_labels[
                ~game_level_labels["GAME_ID"].isin(processed_game_ids)
            ].copy()
            skipped_count = starting_count - len(game_level_labels)
            if skipped_count:
                print(f"Skipping {skipped_count} already-processed games from {output_path}")

    dataset, failed_games, remaining_games = build_season_dataset(
        game_level_labels,
        stop_on_first_failure=stop_on_first_failure,
    )

    if output_path is not None:
        dataset = append_and_dedupe_csv(
            dataset,
            output_path,
            dedupe_cols=["GAME_ID", "player_name_key", "player_name"],
        )

    if failed_output_path is not None:
        append_and_dedupe_csv(
            failed_games,
            failed_output_path,
            dedupe_cols=["GAME_ID", "error"],
        )

    if remaining_output_path is not None:
        remaining_file = Path(remaining_output_path)
        if remaining_games.empty:
            if remaining_file.exists():
                remaining_file.unlink()
        else:
            remaining_games.to_csv(remaining_file, index=False)

    return dataset, failed_games, remaining_games


if __name__ == "__main__":
    dataset, failed_games, remaining_games = build_dataset_from_winners_csv(
        "Tommy_Award_Winners.csv",
        output_path="Tommy_Award_Player_Game_Table.csv",
        failed_output_path="Tommy_Award_Failed_Games.csv",
        remaining_output_path="Tommy_Award_Remaining_Games.csv",
        queue_csv_path="Tommy_Award_Remaining_Games.csv",
        stop_on_first_failure=True,
    )
    print(f"Built dataset with {len(dataset)} rows.")
    print(f"Saved {len(failed_games)} failed games to Tommy_Award_Failed_Games.csv.")
    if remaining_games.empty:
        print("All queued games processed.")
    else:
        print(f"Saved {len(remaining_games)} remaining games to Tommy_Award_Remaining_Games.csv.")
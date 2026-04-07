from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROLLING_WINDOWS = (3, 5, 10)


@dataclass(frozen=True)
class FeatureSpec:
    source_col: str
    output_name: str


FEATURE_SPECS = (
    FeatureSpec("points", "points"),
    FeatureSpec("reboundsOffensive", "offensive_rebounds"),
    FeatureSpec("reboundsTotal", "rebounds"),
    FeatureSpec("assists", "assists"),
    FeatureSpec("steals", "steals"),
    FeatureSpec("blocks", "blocks"),
    FeatureSpec("minutes_decimal", "minutes"),
    FeatureSpec("plusMinusPoints", "plus_minus"),
    FeatureSpec("threePointersMade", "three_pm"),
    FeatureSpec("fieldGoalsAttempted", "fga"),
    FeatureSpec("threePointersAttempted", "three_pa"),
    FeatureSpec("freeThrowsAttempted", "fta"),
    FeatureSpec("turnovers", "turnovers"),
    FeatureSpec("fieldGoalsPercentage", "fg_pct"),
    FeatureSpec("threePointersPercentage", "three_pt_pct"),
    FeatureSpec("freeThrowsPercentage", "ft_pct"),
    FeatureSpec("ast_per_min", "ast_per_min"),
    FeatureSpec("stocks_per_min", "stocks_per_min"),
    FeatureSpec("points_rank", "points_rank"),
    FeatureSpec("reboundsOffensive_rank", "offensive_rebounds_rank"),
    FeatureSpec("reboundsTotal_rank", "rebounds_rank"),
    FeatureSpec("assists_rank", "assists_rank"),
    FeatureSpec("steals_rank", "steals_rank"),
    FeatureSpec("blocks_rank", "blocks_rank"),
    FeatureSpec("plusMinusPoints_rank", "plus_minus_rank"),
)


KEEP_COLUMNS = [
    "GAME_ID",
    "game_date",
    "season",
    "personId",
    "player_name",
    "player_name_key",
    "position",
    "opponent",
    "is_home",
    "days_rest",
    "won_last_game",
    "team_win_pct_before_game",
    "game_number_in_season",
    "winner_names",
    "y",
    "games_played_before",
    "starter_last_game",
    "starter_rate_last_5",
    "minutes_rank_last_game",
    "won_tommy_last_game",
    "tommy_wins_before",
    "tommy_wins_last_10",
    "points_share_last_5",
    "minutes_share_last_5",
    "stocks_share_last_5",
    "assists_share_last_5",
    "rebounds_share_last_5",
    "usage_share_last_5",
]


def season_from_dates(game_dates: pd.Series) -> pd.Series:
    game_dates = pd.to_datetime(game_dates)
    start_year = game_dates.dt.year.where(game_dates.dt.month >= 10, game_dates.dt.year - 1)
    return start_year.astype(str) + "-" + (start_year + 1).astype(str).str[-2:]


def load_player_game_table(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"GAME_ID": str}).copy()
    required_cols = {"GAME_ID", "game_date", "personId", "player_name", "y", "winner_names"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in source CSV: {missing}")

    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed").dt.normalize()
    df["season"] = season_from_dates(df["game_date"])
    df["player_active_today"] = 1

    if "player_name_key" not in df.columns:
        df["player_name_key"] = (
            df["player_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(".", "", regex=False)
            .str.replace(",", "", regex=False)
        )

    required_numeric_cols = {
        "y",
        "minutes_decimal_rank",
        "points",
        "reboundsOffensive",
        "reboundsTotal",
        "assists",
        "steals",
        "blocks",
        "minutes_decimal",
        "plusMinusPoints",
        "threePointersMade",
        "fieldGoalsAttempted",
        "threePointersAttempted",
        "freeThrowsAttempted",
        "turnovers",
        "fieldGoalsPercentage",
        "threePointersPercentage",
        "freeThrowsPercentage",
        "ast_per_min",
        "points_rank",
        "reboundsOffensive_rank",
        "reboundsTotal_rank",
        "assists_rank",
        "steals_rank",
        "blocks_rank",
        "plusMinusPoints_rank",
    }
    for col in required_numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Missing feature source column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "position" in df.columns:
        df["position"] = df["position"].fillna("UNK").replace("", "UNK")
    else:
        df["position"] = "UNK"

    if "minutes_decimal_rank" in df.columns:
        df["starter_proxy"] = (df["minutes_decimal_rank"] <= 5).astype(int)
    else:
        df["starter_proxy"] = (
            df.groupby("GAME_ID")["minutes_decimal"].rank(ascending=False, method="min") <= 5
        ).astype(int)

    df["stocks"] = df["steals"] + df["blocks"]
    df["stocks_per_min"] = df["stocks"] / df["minutes_decimal"].replace(0, np.nan)
    df["usage_proxy"] = (
        df["fieldGoalsAttempted"] + 0.44 * df["freeThrowsAttempted"] + df["turnovers"]
    )

    computed_numeric_cols = {"stocks_per_min", "stocks", "usage_proxy"}
    for col in computed_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _ensure_local_python_packages() -> None:
    packages_dir = Path(__file__).with_name(".python_packages")
    if packages_dir.exists():
        packages_dir_str = str(packages_dir)
        if packages_dir_str not in sys.path:
            sys.path.insert(0, packages_dir_str)


def _load_schedule_context(seasons: list[str]) -> pd.DataFrame:
    _ensure_local_python_packages()
    from Enriching_csv import CELTICS_TEAM_ID, get_team_schedule

    all_schedules = []
    for season in seasons:
        for season_type in ["Regular Season", "Playoffs"]:
            season_df = get_team_schedule(
                team_id=CELTICS_TEAM_ID,
                season=season,
                season_type=season_type,
            ).copy()
            season_df["season"] = season
            season_df["SEASON_TYPE"] = season_type
            all_schedules.append(season_df)

    schedule_df = pd.concat(all_schedules, ignore_index=True)
    schedule_df["GAME_ID"] = schedule_df["GAME_ID"].astype(str).str.zfill(10)
    schedule_df["game_date"] = pd.to_datetime(schedule_df["GAME_DATE"]).dt.normalize()
    schedule_df["opponent"] = schedule_df["MATCHUP"].astype(str).str.split().str[-1]
    schedule_df["is_home"] = schedule_df["MATCHUP"].astype(str).str.contains("vs.")
    schedule_df["won_game"] = (schedule_df["WL"] == "W").astype(int)
    schedule_df = schedule_df.drop_duplicates(subset=["GAME_ID"]).sort_values(
        ["season", "game_date", "GAME_ID"],
        kind="mergesort",
    )

    schedule_df["game_number_in_season"] = schedule_df.groupby("season").cumcount() + 1
    schedule_df["days_rest"] = schedule_df.groupby("season")["game_date"].diff().dt.days
    schedule_df["days_rest"] = schedule_df["days_rest"].fillna(7).clip(lower=0)
    schedule_df["won_last_game"] = (
        schedule_df.groupby("season")["won_game"].shift(1).fillna(0).astype(int)
    )
    prior_wins = schedule_df.groupby("season")["won_game"].transform(
        lambda s: s.shift(1).cumsum()
    ).fillna(0)
    prior_games = schedule_df["game_number_in_season"] - 1
    schedule_df["team_win_pct_before_game"] = np.where(
        prior_games > 0,
        prior_wins / prior_games,
        0.0,
    )

    return schedule_df[
        [
            "GAME_ID",
            "opponent",
            "is_home",
            "days_rest",
            "won_last_game",
            "team_win_pct_before_game",
            "game_number_in_season",
        ]
    ].copy()


def _add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(df["season"].dropna().unique().tolist())
    schedule_df = _load_schedule_context(seasons)
    out = df.merge(schedule_df, on="GAME_ID", how="left")
    missing_context = out[
        out[["opponent", "is_home", "days_rest", "won_last_game", "team_win_pct_before_game"]]
        .isna()
        .any(axis=1)
    ]
    if not missing_context.empty:
        missing_games = sorted(missing_context["GAME_ID"].astype(str).unique().tolist())
        raise ValueError(f"Missing schedule context for GAME_ID(s): {missing_games[:10]}")
    out["is_home"] = out["is_home"].astype(int)
    out["won_last_game"] = out["won_last_game"].astype(int)
    return out


def _add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(
        ["personId", "season", "game_date", "GAME_ID"],
        kind="mergesort",
    ).copy()

    group_cols = ["personId", "season"]

    out["games_played_before"] = out.groupby(group_cols).cumcount()
    out["starter_last_game"] = out.groupby(group_cols)["starter_proxy"].shift(1)
    out["starter_rate_last_5"] = out.groupby(group_cols)["starter_proxy"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    out["minutes_rank_last_game"] = out.groupby(group_cols)["minutes_decimal_rank"].shift(1)
    out["won_tommy_last_game"] = out.groupby(group_cols)["y"].shift(1)
    out["tommy_wins_before"] = out.groupby(group_cols)["y"].transform(
        lambda s: s.shift(1).cumsum()
    )
    out["tommy_wins_last_10"] = out.groupby(group_cols)["y"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).sum()
    )

    for spec in FEATURE_SPECS:
        grouped = out.groupby(group_cols)[spec.source_col]

        out[f"season_avg_{spec.output_name}"] = grouped.transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

        for window in ROLLING_WINDOWS:
            out[f"{spec.output_name}_last_{window}"] = grouped.transform(
                lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).mean()
            )

    return out


def _add_team_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    team_game_df = (
        out.groupby(["GAME_ID", "season", "game_date"], as_index=False)
        .agg(
            team_points=("points", "sum"),
            team_minutes=("minutes_decimal", "sum"),
            team_stocks=("stocks", "sum"),
            team_assists=("assists", "sum"),
            team_rebounds=("reboundsTotal", "sum"),
            team_usage_proxy=("usage_proxy", "sum"),
        )
        .sort_values(["season", "game_date", "GAME_ID"], kind="mergesort")
    )

    for col in [
        "team_points",
        "team_minutes",
        "team_stocks",
        "team_assists",
        "team_rebounds",
        "team_usage_proxy",
    ]:
        team_game_df[f"{col}_last_5"] = team_game_df.groupby("season")[col].transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
        )

    out = out.merge(
        team_game_df[
            [
                "GAME_ID",
                "team_points_last_5",
                "team_minutes_last_5",
                "team_stocks_last_5",
                "team_assists_last_5",
                "team_rebounds_last_5",
                "team_usage_proxy_last_5",
            ]
        ],
        on="GAME_ID",
        how="left",
    )

    denominator_map = {
        "points_share_last_5": ("points_last_5", "team_points_last_5"),
        "minutes_share_last_5": ("minutes_last_5", "team_minutes_last_5"),
        "stocks_share_last_5": ("stocks_last_5", "team_stocks_last_5"),
        "assists_share_last_5": ("assists_last_5", "team_assists_last_5"),
        "rebounds_share_last_5": ("rebounds_last_5", "team_rebounds_last_5"),
        "usage_share_last_5": ("usage_proxy_last_5", "team_usage_proxy_last_5"),
    }
    for output_col, (numerator_col, denominator_col) in denominator_map.items():
        out[output_col] = out[numerator_col] / out[denominator_col].replace(0, np.nan)

    return out


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["season_avg_stocks"] = out["season_avg_steals"] + out["season_avg_blocks"]
    out["season_avg_hustle_proxy"] = (
        out["season_avg_offensive_rebounds"] + out["season_avg_steals"] + out["season_avg_blocks"]
    )
    out["season_avg_usage_proxy"] = (
        out["season_avg_fga"] + 0.44 * out["season_avg_fta"] + out["season_avg_turnovers"]
    )
    out["season_avg_points_per_min"] = out["season_avg_points"] / out["season_avg_minutes"].replace(
        0, np.nan
    )

    for window in ROLLING_WINDOWS:
        out[f"stocks_last_{window}"] = out[f"steals_last_{window}"] + out[f"blocks_last_{window}"]
        out[f"hustle_proxy_last_{window}"] = (
            out[f"offensive_rebounds_last_{window}"]
            + out[f"steals_last_{window}"]
            + out[f"blocks_last_{window}"]
        )
        out[f"usage_proxy_last_{window}"] = (
            out[f"fga_last_{window}"]
            + 0.44 * out[f"fta_last_{window}"]
            + out[f"turnovers_last_{window}"]
        )
        out[f"points_per_min_last_{window}"] = out[f"points_last_{window}"] / out[
            f"minutes_last_{window}"
        ].replace(0, np.nan)

    return out


def build_pregame_feature_table(source_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = _add_schedule_features(source_df)
    feature_df = _add_history_features(feature_df)
    feature_df = _add_derived_features(feature_df)
    feature_df = _add_team_context_features(feature_df)

    ordered_feature_cols = []
    for spec in FEATURE_SPECS:
        ordered_feature_cols.append(f"season_avg_{spec.output_name}")
        for window in ROLLING_WINDOWS:
            ordered_feature_cols.append(f"{spec.output_name}_last_{window}")

    ordered_feature_cols.extend(
        [
            "season_avg_stocks",
            "stocks_last_3",
            "stocks_last_5",
            "stocks_last_10",
            "season_avg_hustle_proxy",
            "hustle_proxy_last_3",
            "hustle_proxy_last_5",
            "hustle_proxy_last_10",
            "season_avg_usage_proxy",
            "usage_proxy_last_3",
            "usage_proxy_last_5",
            "usage_proxy_last_10",
            "season_avg_points_per_min",
            "points_per_min_last_3",
            "points_per_min_last_5",
            "points_per_min_last_10",
        ]
    )

    final_cols = [col for col in KEEP_COLUMNS + ordered_feature_cols if col in feature_df.columns]
    final_df = feature_df[final_cols].sort_values(
        ["game_date", "GAME_ID", "player_name"],
        kind="mergesort",
    ).reset_index(drop=True)

    fill_zero_cols = [
        "is_home",
        "days_rest",
        "won_last_game",
        "team_win_pct_before_game",
        "game_number_in_season",
        "games_played_before",
        "starter_last_game",
        "starter_rate_last_5",
        "minutes_rank_last_game",
        "won_tommy_last_game",
        "tommy_wins_before",
        "tommy_wins_last_10",
        "points_share_last_5",
        "minutes_share_last_5",
        "stocks_share_last_5",
        "assists_share_last_5",
        "rebounds_share_last_5",
        "usage_share_last_5",
    ] + ordered_feature_cols
    existing_fill_zero_cols = [col for col in fill_zero_cols if col in final_df.columns]
    final_df[existing_fill_zero_cols] = final_df[existing_fill_zero_cols].fillna(0)

    return final_df


def build_pregame_feature_csv(
    input_path: str | Path = "Tommy_Award_Player_Game_Table.csv",
    output_path: str | Path = "Tommy_Award_Pregame_Features.csv",
) -> pd.DataFrame:
    source_df = load_player_game_table(input_path)
    feature_df = build_pregame_feature_table(source_df)
    feature_df.to_csv(output_path, index=False)
    return feature_df

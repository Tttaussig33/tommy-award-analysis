from __future__ import annotations

import importlib
from pathlib import Path
import sys

packages_dir = Path(__file__).with_name(".python_packages")
if packages_dir.exists():
    packages_dir_str = str(packages_dir)
    if packages_dir_str not in sys.path:
        sys.path.insert(0, packages_dir_str)

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = "Tommy_Award_Pregame_Features.csv"
RANDOM_STATE = 42
MIN_TRAIN_SEASONS = 3


def _ensure_local_python_packages() -> None:
    if packages_dir.exists():
        packages_dir_str = str(packages_dir)
        if packages_dir_str not in sys.path:
            sys.path.insert(0, packages_dir_str)


def load_dataset(path: str = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"GAME_ID": str}).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _sorted_seasons(seasons: list[str]) -> list[str]:
    return sorted(seasons, key=lambda season: int(str(season).split("-")[0]))


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded_cols = {
        "GAME_ID",
        "game_date",
        "season",
        "personId",
        "player_name",
        "player_name_key",
        "winner_names",
        "y",
    }
    feature_df = df[[col for col in df.columns if col not in excluded_cols]].copy()
    categorical_cols = feature_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = [col for col in feature_df.columns if col not in categorical_cols]
    return numeric_cols, categorical_cols


def walk_forward_season_splits(
    df: pd.DataFrame,
    min_train_seasons: int = MIN_TRAIN_SEASONS,
) -> list[tuple[list[str], str]]:
    seasons = _sorted_seasons(df["season"].dropna().unique().tolist())
    splits = []
    for idx in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:idx]
        test_season = seasons[idx]
        splits.append((train_seasons, test_season))
    return splits


def latest_season_holdout_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = _sorted_seasons(df["season"].dropna().unique().tolist())
    test_season = seasons[-1]
    train_df = df[df["season"].isin(seasons[:-1])].copy()
    test_df = df[df["season"] == test_season].copy()
    return train_df, test_df


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_logistic_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(numeric_cols, categorical_cols)),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_pos_weight: float,
) -> Pipeline:
    _ensure_local_python_packages()
    XGBClassifier = importlib.import_module("xgboost").XGBClassifier

    return Pipeline(
        steps=[
            ("prep", build_preprocessor(numeric_cols, categorical_cols)),
            (
                "clf",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=1,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=4,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )


def available_model_names() -> tuple[list[str], str | None]:
    model_names = ["balanced_logistic"]
    xgboost_error = None
    try:
        _ensure_local_python_packages()
        importlib.import_module("xgboost")

        model_names.append("xgboost")
    except Exception as exc:  # pragma: no cover - environment-specific
        xgboost_error = str(exc).strip()
    return model_names, xgboost_error


def score_predictions(scored_df: pd.DataFrame) -> dict[str, float]:
    pred_winners = (
        scored_df.sort_values(["GAME_ID", "pred_prob"], ascending=[True, False])
        .groupby("GAME_ID")
        .head(1)
        .copy()
    )
    top3 = (
        scored_df.sort_values(["GAME_ID", "pred_prob"], ascending=[True, False])
        .groupby("GAME_ID")
        .head(3)
        .copy()
    )

    y_true = scored_df["y"]
    y_pred = (scored_df["pred_prob"] >= 0.5).astype(int)

    return {
        "game_level_accuracy": pred_winners["y"].mean(),
        "top3_accuracy": top3.groupby("GAME_ID")["y"].max().mean(),
        "row_logloss": log_loss(y_true, scored_df["pred_prob"], labels=[0, 1]),
        "row_auc": roc_auc_score(y_true, scored_df["pred_prob"]),
        "row_accuracy": (y_pred == y_true).mean(),
    }


def fit_and_score_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[Pipeline, pd.DataFrame, dict[str, float]]:
    X_train = train_df[numeric_cols + categorical_cols]
    X_test = test_df[numeric_cols + categorical_cols]
    y_train = train_df["y"]

    pos = max(int(y_train.sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = neg / pos

    if model_name == "balanced_logistic":
        model = build_logistic_pipeline(numeric_cols, categorical_cols)
    elif model_name == "xgboost":
        model = build_xgboost_pipeline(numeric_cols, categorical_cols, scale_pos_weight)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model.fit(X_train, y_train)

    scored_df = test_df.copy()
    scored_df["pred_prob"] = model.predict_proba(X_test)[:, 1]
    metrics = score_predictions(scored_df)
    return model, scored_df, metrics


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print(f"Game-level winner accuracy: {metrics['game_level_accuracy']:.4f}")
    print(f"Top-3 accuracy: {metrics['top3_accuracy']:.4f}")
    print(f"Row-level accuracy: {metrics['row_accuracy']:.4f}")
    print(f"Row-level log loss: {metrics['row_logloss']:.4f}")
    print(f"Row-level AUC: {metrics['row_auc']:.4f}")


def run_walk_forward_evaluation(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    model_names: list[str],
) -> None:
    print("Walk-forward evaluation by season")
    rows = []

    for train_seasons, test_season in walk_forward_season_splits(df):
        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df = df[df["season"] == test_season].copy()

        for model_name in model_names:
            _, _, metrics = fit_and_score_model(
                model_name,
                train_df,
                test_df,
                numeric_cols,
                categorical_cols,
            )
            rows.append(
                {
                    "model": model_name,
                    "test_season": test_season,
                    "game_level_accuracy": metrics["game_level_accuracy"],
                    "top3_accuracy": metrics["top3_accuracy"],
                    "row_accuracy": metrics["row_accuracy"],
                    "row_logloss": metrics["row_logloss"],
                    "row_auc": metrics["row_auc"],
                }
            )

    cv_df = pd.DataFrame(rows)
    if cv_df.empty:
        print("Not enough seasons for walk-forward evaluation.")
        return

    summary_df = (
        cv_df.groupby("model", as_index=False)[
            ["game_level_accuracy", "top3_accuracy", "row_accuracy", "row_logloss", "row_auc"]
        ]
        .mean()
        .round(4)
    )
    print(summary_df.to_string(index=False))


def show_sample_predictions(scored_df: pd.DataFrame, model_name: str) -> None:
    print(f"\nSample predictions for {model_name}")
    show_cols = [
        "GAME_ID",
        "game_date",
        "player_name",
        "winner_names",
        "y",
        "pred_prob",
        "opponent",
        "is_home",
        "points_last_5",
        "minutes_last_5",
        "starter_last_game",
        "tommy_wins_before",
    ]
    for game_id in scored_df["GAME_ID"].drop_duplicates().head(5):
        game_rows = (
            scored_df[scored_df["GAME_ID"] == game_id]
            .sort_values("pred_prob", ascending=False)[show_cols]
            .head(5)
        )
        print(f"\nGAME_ID: {game_id}")
        print(game_rows.to_string(index=False))


def run_model_comparison(df: pd.DataFrame) -> None:
    numeric_cols, categorical_cols = get_feature_columns(df)
    train_df, test_df = latest_season_holdout_split(df)
    latest_season = _sorted_seasons(df["season"].dropna().unique().tolist())[-1]
    model_names, xgboost_error = available_model_names()

    print("Pregame model comparison")
    print(f"Rows: {len(df)}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Train seasons: {', '.join(_sorted_seasons(train_df['season'].unique().tolist()))}")
    print(f"Test season: {latest_season}")
    print(f"Train games: {train_df['GAME_ID'].nunique()}")
    print(f"Test games: {test_df['GAME_ID'].nunique()}")
    print(
        f"Train positive rate: {train_df['y'].mean():.4f} "
        f"({int(train_df['y'].sum())}/{len(train_df)})"
    )
    if xgboost_error is not None:
        print("\nXGBoost is configured in the script but skipped in this environment.")
        print(f"Reason: {xgboost_error}")

    run_walk_forward_evaluation(df, numeric_cols, categorical_cols, model_names)

    for model_name in model_names:
        _, scored_df, metrics = fit_and_score_model(
            model_name,
            train_df,
            test_df,
            numeric_cols,
            categorical_cols,
        )
        print()
        print_metrics(f"Latest-season holdout: {model_name}", metrics)
        show_sample_predictions(scored_df, model_name)


if __name__ == "__main__":
    _ensure_local_python_packages()
    dataset = load_dataset()
    run_model_comparison(dataset)

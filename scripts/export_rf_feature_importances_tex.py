"""Regenerate results_rf_feature_importance_body.tex for the research paper.

Mirrors random_forest_classifier.ipynb: same CSV, features, train split, SEED=58,
and Optuna best trial-17 RandomForest hyperparameters. Run:

  python3 scripts/export_rf_feature_importances_tex.py

(or any Python 3 with numpy, pandas, scikit-learn).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "Tommy_Award_Player_Game_Table_hustle.csv"
OUT = REPO_ROOT / "paper" / "results_rf_feature_importance_body.tex"

SEED = 58
random.seed(SEED)
np.random.seed(SEED)

target_col = "y"
numeric_feature_cols = [
    "minutes_decimal",
    "points",
    "reboundsOffensive",
    "reboundsDefensive",
    "reboundsTotal",
    "assists",
    "steals",
    "blocks",
    "deflections",
    "charges_drawn",
    "turnovers",
    "foulsPersonal",
    "plusMinusPoints",
    "net_rating",
    "usage_rate",
    "impact_efficiency",
    "role_outperformance",
    "fieldGoalsMade",
    "fieldGoalsAttempted",
    "threePointersMade",
    "threePointersAttempted",
    "freeThrowsMade",
    "stocks",
    "points_per_min",
    "oreb_per_min",
    "reb_per_min",
    "ast_per_min",
    "stocks_per_min",
    "hustle_proxy",
    "points_rank",
    "reboundsOffensive_rank",
    "reboundsTotal_rank",
    "assists_rank",
    "steals_rank",
    "blocks_rank",
    "plusMinusPoints_rank",
    "minutes_decimal_rank",
    "stocks_rank",
    "hustle_proxy_rank",
]

game_id_col = "gameId"

# Optuna best trial (notebook log): trial 17
BEST_RF_PARAMS = {
    "bootstrap": False,
    "n_estimators": 360,
    "max_depth": 16,
    "min_samples_split": 25,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": SEED,
    "n_jobs": -1,
}


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce", format="mixed")

    if "season" not in df.columns:
        start_year = df["game_date"].dt.year.where(df["game_date"].dt.month >= 10, df["game_date"].dt.year - 1)
        df["season"] = start_year.astype("Int64").astype(str) + "-" + (start_year + 1).astype("Int64").astype(str).str[-2:]

    df = df[df["minutes_decimal"] > 0].copy()
    mins = df["minutes_decimal"].clip(lower=1e-6)

    df["hustle_proxy"] = (
        pd.to_numeric(df["reboundsOffensive"], errors="coerce")
        + pd.to_numeric(df["steals"], errors="coerce")
        + pd.to_numeric(df["blocks"], errors="coerce")
    ) / mins

    if "net_rating" not in df.columns:
        df["net_rating"] = pd.to_numeric(df["plusMinusPoints"], errors="coerce")

    if "stocks_per_min" not in df.columns:
        df["stocks_per_min"] = pd.to_numeric(df["stocks"], errors="coerce") / mins

    eps = 1e-6
    if "impact_efficiency" not in df.columns:
        df["impact_efficiency"] = pd.to_numeric(df["net_rating"], errors="coerce") / (
            pd.to_numeric(df["usage_rate"], errors="coerce") + eps
        )

    if "role_outperformance" not in df.columns:
        df["role_outperformance"] = pd.to_numeric(df["net_rating"], errors="coerce") * (
            1 - pd.to_numeric(df["usage_rate"], errors="coerce")
        )

    if "stocks_rank" not in df.columns:
        df["stocks_rank"] = df.groupby(game_id_col)["stocks"].rank(method="min", ascending=False)

    df["hustle_proxy_rank"] = df.groupby(game_id_col)["hustle_proxy"].rank(method="min", ascending=False)

    missing_cols = [col for col in numeric_feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    requested_test_seasons = ["2024-25", "2025-26"]
    train_df = df[~df["season"].isin(requested_test_seasons)].copy()
    X_train = train_df[numeric_feature_cols]
    y_train = train_df[target_col].astype(int)

    model = RandomForestClassifier(**BEST_RF_PARAMS)
    model.fit(X_train.fillna(X_train.median()), y_train)

    fi = (
        pd.DataFrame({"feature": numeric_feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    def tex_escape_feature(name: str) -> str:
        return str(name).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    header = [
        "% !TEX root = tommy_award_research_paper.tex",
        "% Table body rows only (\\input inside longtable in the paper). Do not compile this file alone.",
        "% Regenerate: python3 scripts/export_rf_feature_importances_tex.py",
        "",
    ]
    lines = []
    for i, row in fi.iterrows():
        tex_name = tex_escape_feature(row["feature"])
        lines.append(f"{i + 1} & \\texttt{{{tex_name}}} & {row['importance']:.6f} \\\\")

    OUT.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} rows to {OUT}")


if __name__ == "__main__":
    main()

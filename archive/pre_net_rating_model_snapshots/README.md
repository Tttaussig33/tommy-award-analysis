# Model snapshots (pre–net-rating / pre–hustle-proxy CSV refresh)

These files are a **frozen copy** of your modeling notebooks and common result artifacts from Git commit **`27ae5b0`** (*Transfer Final Paper to new Latex Document*). That is the last snapshot **before** commit **`1306654`** (*Reorganize Files*), which rewrote `Tommy_Award_Player_Game_Table_hustle.csv` (NBA `netRating`, per-minute `hustle_proxy`, etc.) and moved the project layout.

## What is here

| Path | Contents |
|------|----------|
| `notebooks/` | Jupyter notebooks with **saved outputs** from that era: ridge/lasso/elastic-net, decision tree, random forest, XGBoost, league prediction notebook, slides export helper. |
| `figures/` | Baseline metric plots, RF/XGB metric PNGs, `feature_importance.png`, `results_per60_top10.png`, position pie chart. |
| `tables/results_per60_table_body.tex` | Full per-60 ranking fragment as of that commit. |

## How to use

- **View old results:** open the `.ipynb` files in `notebooks/`; outputs are already stored in the file.
- **Re-run those notebooks:** they still point at the **old paths** (e.g. `Tommy_Award_Player_Game_Table_hustle.csv` in the **repository root**). To execute them today you would either check out commit `27ae5b0` in a separate worktree or copy the matching hustle CSV from that commit next to the notebook and adjust paths manually.

## Current project

Active, maintained notebooks stay under **`notebooks/`** on `main` and use **`data/Tommy_Award_Player_Game_Table_hustle.csv`** and the updated feature definitions.

# Celtics Tommy Award — modeling and paper

This repository contains player–game data, notebooks that train classifiers and logistic models on Boston Celtics Tommy Award games, scripts that score other NBA teams, LaTeX for the research paper, and cached prediction outputs.

## Repository layout

| Path | Purpose |
|------|---------|
| `data/` | Canonical CSV inputs: hustle-enriched Celtics table, Tommy winners, and related QA extracts. |
| `notebooks/` | Jupyter workflows: ridge/lasso/logistic, tree/RF/XGB baselines, league-wide prediction notebook, slide export helper. |
| `csv_builders/` | Python modules and CLIs to **rebuild** the player–game table from `nba_api` and to enrich it (hustle, usage, net rating). |
| `scripts/` | One-off analysis and LaTeX table generators (per-60 rankings, RF feature importances, Tommy leader/surprise summaries, figures). |
| `predictions/` | Cached enriched per-team CSVs and combined predicted-win tallies (from the league notebook). |
| `paper/` | `tommy_award_research_paper.tex`, `\input{...}` fragments, PDFs, and related TeX sources. |
| `results/figures/` | Plots saved by scripts or notebooks (metrics, baselines, per-60 bar chart, position pie chart). |
| `archive/old_model_attempts/` | Older experiments (pregame features, baseline comparisons). Paths there may still assume CSVs in the project root—use `data/` or adjust paths if you revive them. |
| `archive/pre_net_rating_model_snapshots/` | **Frozen** notebooks, figures, per-60 TeX, and **`data/Tommy_Award_Player_Game_Table_hustle.csv`** from commit `27ae5b0` (before net-rating / hustle-proxy refresh). See `README.md` in that folder. |

## Paper-era worktree (optional)

To open the **full project exactly as at `27ae5b0`** (notebooks and hustle CSV at the old paths, no `data/` subfolder):

```bash
cd "/path/to/Celtics_Project"
git worktree add ../Celtics_Project_paper_era 27ae5b0
cd ../Celtics_Project_paper_era
git switch -c paper-era-27ae5b0
```

The extra `git switch` puts the worktree on a **named branch** (same commit as `27ae5b0`). That avoids **detached HEAD**, which some editors mis-report as thousands of staged deletions when the window is multi-root. Use any sibling directory name you prefer; start Jupyter from that folder’s root so paths match the old notebooks.  
**Remove when done:** `git worktree remove ../Celtics_Project_paper_era`

Run **commands from the repository root** unless noted otherwise. Jupyter kernels work whether the server is started in the root or in `notebooks/`; notebook cells resolve `REPO_ROOT` automatically.

## Quick start (reproduce analysis without API calls)

1. **Python environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Notebooks** (open `notebooks/` in Jupyter Lab/VS Code)

   - `Ridge_Lasso_Logistic_Regression.ipynb` — regularized logistic models on `data/Tommy_Award_Player_Game_Table_hustle.csv`.
   - `decision_tree_classifier.ipynb`, `random_forest_classifier.ipynb`, `xgboost_classifier.ipynb` — tree models with the same season split as the paper.
   - `predict_tommy_award_other_teams.ipynb` — trains on Celtics data, then fetches and scores all other teams (writes under `predictions/`). **Requires network** and significant runtime; use the committed `predictions/` files to skip this.
   - `export_slides_mar6_2026_table.ipynb` — small table for a fixed game date.

3. **Paper tables and figures (deterministic given the CSV and `predictions/`)**

   ```bash
   python3 scripts/export_rf_feature_importances_tex.py
   python3 scripts/build_results_per60_table_tex.py
   ```

   These refresh `paper/results_rf_feature_importance_body.tex`, `paper/results_per60_table_body*.tex`, and `results/figures/results_per60_top10.png`.

4. **Compile the paper**

   ```bash
   cd paper
   pdflatex -interaction=nonstopmode tommy_award_research_paper.tex
   pdflatex -interaction=nonstopmode tommy_award_research_paper.tex
   ```

   The main file `\input`s `results_per60_table_body_top100.tex` and `results_rf_feature_importance_body.tex` from the **same** `paper/` directory.

## Rebuilding data from the NBA API

These steps are **optional** if you only want to rerun models on the committed `data/` and `predictions/` snapshots.

1. **Base player–game table** (Celtics games aligned to Tommy winners; slow, many API calls):

   ```bash
   python3 -m csv_builders.build_tommy_award_player_game_table
   ```

   Writes `data/Tommy_Award_Player_Game_Table.csv` (and queue/failed CSVs in `data/`).

2. **Hustle + usage + net rating columns**

   ```bash
   python3 -m csv_builders.enrich_player_game_with_hustle
   ```

   Defaults: read `data/Tommy_Award_Player_Game_Table.csv`, write `data/Tommy_Award_Player_Game_Table_hustle.csv`.

3. **Refresh `net_rating` only** (one request per game; on the order of many minutes with polite sleeps):

   ```bash
   python3 csv_builders/update_tommy_hustle_net_rating.py
   ```

4. **League predictions** — run the relevant cells in `notebooks/predict_tommy_award_other_teams.ipynb` (or the whole notebook). Output paths stay under `predictions/`.

**Note:** API availability, timeouts, and season coverage can change row counts and column completeness versus the CSV snapshot in this repo.

## Auxiliary scripts

Run from the repository root:

- `python3 scripts/plot_tommy_winners_position_pie.py` → `results/figures/tommy_winners_position_pie.png`
- `python3 scripts/tommy_leaders_hustle_stats.py` → `predictions/tommy_winners_hustle_leaderboard.csv`
- `python3 scripts/tommy_surprise_game_winners.py` → surprise-winner summaries in `predictions/`
- `python3 scripts/tommy_wins_excluding_star_hubs.py` — stdout analysis using `predictions/`

## Reproducibility notes

- Tree/RF notebooks and `scripts/export_rf_feature_importances_tex.py` use **`SEED = 58`** and the Optuna RF hyperparameters documented in that script.
- **`hustle_proxy`** is per minute: \((\mathrm{OREB} + \mathrm{STL} + \mathrm{BLK}) / \mathrm{minutes}\) (aligned across ridge and tree pipelines).
- Committed **`predictions/*.csv`** are large; they are the expected inputs for `build_results_per60_table_tex.py` and the per-60 section of the paper.

## License / data

NBA statistics are fetched via [nba_api](https://github.com/swar/nba_api) from NBA.com-style endpoints. Respect rate limits; the builders use small delays between requests.

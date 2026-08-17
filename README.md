# Modeling Hustle and Impact in the NBA

This project uses machine learning to study the **Tommy Award**, a Boston Celtics postgame award that recognizes players for hustle, toughness, and contributions that may not always stand out in a traditional box score.

The goal was to take an award that is largely based on watching and understanding the game, determine which statistics were associated with winning it, and then apply that same player-impact profile across the rest of the NBA.

## Project Overview

I collected Tommy Award winners from **781 Celtics games across 10 NBA seasons** and matched each winner with player-level NBA statistics from the same game. The resulting dataset labels each player-game observation as either a Tommy Award winner or non-winner.

From there, I built a Python machine learning pipeline to compare several approaches for predicting the award:

- Ridge Logistic Regression
- Lasso Logistic Regression
- Elastic-Net Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Because there is only one winner in each game, the models were trained as binary classifiers and their predicted probabilities were used to rank players within each game.

## Feature Engineering

A major part of the project was creating statistics that could capture more than raw production.

Along with traditional NBA statistics such as points, rebounds, assists, steals, blocks, minutes, and shooting totals, I created features including:

- **Per-minute statistics** to account for differences in playing time
- **Within-game rank statistics** to measure how much a player stood out relative to everyone else
- **Stocks** combining steals and blocks
- **Hustle Proxy** combining offensive rebounds, steals, and blocks relative to minutes
- **Impact Efficiency** measuring net rating relative to usage rate
- **Role Outperformance** rewarding strong net impact from players with smaller offensive roles

These features were designed to help distinguish between raw statistical production and the type of effort and impact associated with the Tommy Award.

## Model Evaluation

The models were evaluated using both game-level ranking performance and probability-based metrics.

**Top-1 and Top-3 accuracy** measured how often the actual winner appeared at the top of the model's player rankings. **PR-AUC** was especially important because the dataset contains far more non-winners than winners. **Brier score** and **log loss** were also used to evaluate the quality of the predicted probabilities.

Random Forest and XGBoost were used alongside the logistic regression models to determine whether nonlinear relationships and interactions between player statistics improved predictions.

## Applying the Model Across the NBA

After learning the statistical profile associated with Tommy Award-winning performances in Boston, I extended the analysis to **all 30 NBA teams**.

Players on other teams were scored using the model trained on Celtics data, allowing the original idea behind the Tommy Award to be applied outside of Boston. This made it possible to identify both stars and less-recognized players whose performances showed similar combinations of hustle, efficiency, role outperformance, and overall impact.

I also normalized predicted Tommy-style wins by playing time to examine **wins per 60 minutes**, which helped surface players who produced this type of impact despite having smaller roles or fewer minutes.

## Research Paper

A full write-up of the project is available here:

- [Research Paper PDF](paper/tommy_award_research_paper.pdf)
- [LaTeX Source](paper/tommy_award_research_paper.tex)

## Repository Structure

| Folder | Contents |
|---|---|
| `data/` | Celtics player-game data and Tommy Award labels |
| `notebooks/` | Logistic regression, Decision Tree, Random Forest, XGBoost, and league-wide analysis |
| `csv_builders/` | NBA data collection and preprocessing |
| `scripts/` | Feature importance, per-60 analysis, and supporting analyses |
| `predictions/` | League-wide model outputs |
| `results/figures/` | Charts and model results |
| `paper/` | Research paper and generated tables |

## Tools

**Python · pandas · NumPy · scikit-learn · XGBoost · Optuna · nba_api · Jupyter**

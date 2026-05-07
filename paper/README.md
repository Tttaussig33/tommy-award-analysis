# Research paper

This folder is the **primary deliverable** for the project: the LaTeX source, the compiled PDF, and the table bodies `\input` by the main document.

| File | Role |
|------|------|
| `tommy_award_research_paper.tex` | Main document (compile this). |
| `tommy_award_research_paper.pdf` | Built PDF; keep in git so readers can open it without LaTeX. |
| `results_per60_table_body.tex`, `results_per60_table_body_top100.tex` | Per-60 ranking rows (from `scripts/build_results_per60_table_tex.py` at repo root). |
| `results_rf_feature_importance_body.tex` | Full feature-importance list (from `scripts/export_rf_feature_importances_tex.py`). |
| `results_rf_feature_importance_body_top15.tex` | Optional mirror of the top-15 rows (main paper may inline these). |
| `results_presentation.tex`, `Final_Paper_Images.tex`, `results_per60_table_standalone.tex` | Supporting or standalone TeX. |

## Build

From the **repository root**:

```bash
python3 scripts/export_rf_feature_importances_tex.py
python3 scripts/build_results_per60_table_tex.py
cd paper
pdflatex -interaction=nonstopmode tommy_award_research_paper.tex
pdflatex -interaction=nonstopmode tommy_award_research_paper.tex
```

Run `pdflatex` twice so references and the table of contents settle.

For the full project layout and data reproduction, see the [root `README.md`](../README.md).

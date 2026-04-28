"""Regenerate results_per60_table_body.tex for results_presentation.tex."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PRED_DIR = Path(__file__).resolve().parent / "predictions"
OUT = Path(__file__).resolve().parent / "results_per60_table_body.tex"
TOP_N_PLOT = 10
OUT_PLOT = Path(__file__).resolve().parent / "results_per60_top10.png"
ENRICHED_RE = re.compile(r"^(.+)_(20\d{2}-\d{2})_player_game_enriched\.csv$")
MIN_MINUTES = 300.0

# Primary brand colors (approximate NBA hex). Keys must match `teamCity + " " + teamName` in enriched CSVs.
NBA_TEAM_PRIMARY_HEX: dict[str, str] = {
    "Atlanta Hawks": "#E03A3E",
    "Boston Celtics": "#007A33",
    "Brooklyn Nets": "#000000",
    "Charlotte Hornets": "#1D1160",
    "Chicago Bulls": "#CE1141",
    "Cleveland Cavaliers": "#860038",
    "Dallas Mavericks": "#00538C",
    "Denver Nuggets": "#0E2240",
    "Detroit Pistons": "#C8102E",
    "Golden State Warriors": "#1D428A",
    "Houston Rockets": "#CE1141",
    "Indiana Pacers": "#002D62",
    "LA Clippers": "#C8102E",
    "Los Angeles Clippers": "#C8102E",
    "Los Angeles Lakers": "#552583",
    "Memphis Grizzlies": "#5D76A9",
    "Miami Heat": "#98002E",
    "Milwaukee Bucks": "#00471B",
    "Minnesota Timberwolves": "#0C2340",
    "New Orleans Pelicans": "#0C2340",
    "New York Knicks": "#006BB6",
    "Oklahoma City Thunder": "#007AC1",
    "Orlando Magic": "#0077C0",
    "Philadelphia 76ers": "#006BB6",
    "Phoenix Suns": "#1D1160",
    "Portland Trail Blazers": "#E03A3E",
    "Sacramento Kings": "#5A2D81",
    "San Antonio Spurs": "#000000",
    "Toronto Raptors": "#CE1141",
    "Utah Jazz": "#002B5C",
    "Washington Wizards": "#002B5C",
}

_TEAM_HEX_LOWER = {k.lower(): v for k, v in NBA_TEAM_PRIMARY_HEX.items()}


def team_primary_hex(team: str) -> str:
    t = str(team).strip()
    if t in NBA_TEAM_PRIMARY_HEX:
        return NBA_TEAM_PRIMARY_HEX[t]
    return _TEAM_HEX_LOWER.get(t.lower(), "#64748B")


def main() -> None:
    frames = []
    for f in sorted(PRED_DIR.glob("*_predicted_tommy_counts_combined.csv")):
        slug = f.name.replace("_predicted_tommy_counts_combined.csv", "")
        df = pd.read_csv(f)
        df["team_slug"] = slug
        df = df.rename(columns={"total": "predicted_wins"})
        frames.append(df[["team_slug", "player_name", "predicted_wins"]])
    wins = pd.concat(frames, ignore_index=True)

    mins_rows = []
    for f in sorted(PRED_DIR.glob("*_player_game_enriched.csv")):
        m = ENRICHED_RE.match(f.name)
        if not m:
            continue
        slug = m.group(1)
        df = pd.read_csv(f, usecols=["teamCity", "teamName", "player_name", "minutes_decimal"])
        df["team"] = df["teamCity"].astype(str) + " " + df["teamName"].astype(str)
        df["team_slug"] = slug
        g = df.groupby(["team_slug", "team", "player_name"], as_index=False)["minutes_decimal"].sum()
        mins_rows.append(g)
    minute = pd.concat(mins_rows, ignore_index=True)
    agg_mins = minute.groupby(["team_slug", "player_name"], as_index=False).agg(
        team=("team", "first"),
        total_minutes=("minutes_decimal", "sum"),
    )

    m = wins.merge(agg_mins, on=["team_slug", "player_name"], how="left")
    m = m[m["total_minutes"] >= MIN_MINUTES]
    m["per_60"] = 60.0 * m["predicted_wins"] / m["total_minutes"]
    m = m.sort_values("per_60", ascending=False).reset_index(drop=True)

    def esc(s: str) -> str:
        return (
            str(s)
            .replace("\\", "\\textbackslash{}")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("_", "\\_")
            .replace("#", "\\#")
        )

    lines = [
        f"        {i + 1} & {esc(r['player_name'])} & {esc(r['team'])} & "
        f"{r['predicted_wins']:.0f} & {r['total_minutes']:.1f} & {r['per_60']:.4f} \\\\"
        for i, r in m.iterrows()
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} ({len(lines)} rows, min minutes={MIN_MINUTES})")

    plot_top = m.head(TOP_N_PLOT).sort_values("per_60", ascending=True)
    bar_colors = [team_primary_hex(tm) for tm in plot_top["team"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        plot_top["player_name"],
        plot_top["per_60"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Predicted wins per 60 minutes")
    ax.set_title(f"Top {TOP_N_PLOT} player by predicted Tommy Award Wins")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PLOT}")


if __name__ == "__main__":
    main()

"""Build a pie chart of Tommy Award winner positions from Tommy_Award_Winners.csv.

Position comes from the mode of `position` in Tommy_Award_Player_Game_Table_hustle.csv
(matched by player name, ASCII-folded diacritics, then a few manual name→role fallbacks).
Writes: tommy_winners_position_pie.png
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tommy_winners_position_pie.png"
WINNERS_CSV = ROOT / "Tommy_Award_Winners.csv"
HUSTLE_CSV = ROOT / "Tommy_Award_Player_Game_Table_hustle.csv"

# Spelling/legacy names in Tommy_Award_Winners.csv not found under the same string in hustle
MANUAL: dict[str, str] = {
    "max shulga": "G",
    "anfernee simons": "G",
    "robert williams": "C",
    "malcolm brogdon": "G",
    "josh richardson": "G",
    "jabari parker": "F",
    "tacko fall": "C",
    "brad wannamaker": "G",
    "marcus morris": "F",
    "greg monroe": "C",
    "hugo gonzalez": "G",
}


def ascii_key(s: str) -> str:
    s = str(s).lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def _mode(s: pd.Series) -> str | float:
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) else s.iloc[0]


def position_for(
    player: str,
    pos_name: pd.Series,
    pos_akey: pd.Series,
) -> str | float:
    p = str(player).strip()
    if p in pos_name.index and pd.notna(pos_name[p]):
        return pos_name[p]
    k = ascii_key(p)
    if k in pos_akey.index and pd.notna(pos_akey[k]):
        return pos_akey[k]
    if k in MANUAL:
        return MANUAL[k]
    for suffix in (" Sr.", " III", " Jr.", " II", " IV"):
        if p.endswith(suffix):
            t = p[: -len(suffix)].strip()
            if t in pos_name.index and pd.notna(pos_name[t]):
                return pos_name[t]
    return np.nan


def main() -> None:
    w = pd.read_csv(WINNERS_CSV)
    h = pd.read_csv(HUSTLE_CSV, usecols=["player_name", "position"], low_memory=False)
    pos_name = h.groupby("player_name", sort=False)["position"].agg(_mode)
    h = h.copy()
    h["akey"] = h["player_name"].map(ascii_key)
    pos_akey = h.groupby("akey", sort=False)["position"].agg(_mode)

    w["position"] = w["Player"].map(lambda p: position_for(p, pos_name, pos_akey))
    missing = w["position"].isna()
    if missing.any():
        raise SystemExit(
            f"Unmapped {missing.sum()} winner row(s), e.g. {w.loc[missing, 'Player'].head(5).tolist()}"
        )

    counts = w["position"].value_counts()
    order = [x for x in ("G", "F", "C") if x in counts.index]
    labels = [f"Guard (G)" if o == "G" else f"Forward (F)" if o == "F" else "Center (C)" for o in order]
    sizes = np.array([counts[o] for o in order], dtype=float)
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    def _autopct(_values: np.ndarray):
        def _inner(pct: float) -> str:
            n = int(round(pct / 100.0 * _values.sum()))
            return f"{pct:.1f}%\n(n={n})"

        return _inner

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pie(
        sizes,
        labels=labels,
        colors=colors[: len(sizes)],
        startangle=90,
        counterclock=False,
        autopct=_autopct(sizes),
        wedgeprops={"edgecolor": "white", "linewidth": 0.6},
    )
    ax.set_title("Tommy Award winners by position\n(one row per game in Tommy_Award_Winners.csv)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")
    print(counts.to_string())


if __name__ == "__main__":
    main()

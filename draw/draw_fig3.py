"""
Figure 3: Comparison of retrieval strategies (4 independent subplots)
(a) Difficulty × Type heatmap — test set structure
(b) Score by difficulty level — grouped bar
(c) Score by question type — grouped bar
(d) Test set distribution — stacked bar (difficulty × type)
Data: 实验列表.xlsx Sheet3 + answers/*.jsonl
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Global style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
})


def _apply_spine_style(ax):
    """Left/bottom with inward ticks; right/top border only, no ticks."""
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(axis="both", direction="in", top=False, right=False)


# ── Shared config ────────────────────────────────────────────
METHODS = ["Reasoning", "Router", "Local", "Drift", "Global", "Basic RAG"]
COLORS = [
    "#C44E52",  # Reasoning  — red
    "#4C72B0",  # Router     — blue
    "#55A868",  # Local      — green
    "#8172B3",  # Drift      — purple
    "#CCB974",  # Global     — gold
    "#64B5CD",  # Basic RAG  — light blue
]

ANSWER_DIR = Path("D:/Users/SUN/科研/论文/知识图谱/answers")
ANSWER_FILES = {
    "Reasoning": ANSWER_DIR / "reasoning_0.6795.jsonl",
    "Router":    ANSWER_DIR / "router_0.6748.jsonl",
    "Local":     ANSWER_DIR / "local_0.6681.jsonl",
    "Drift":     ANSWER_DIR / "drift_0.652.jsonl",
    "Global":    ANSWER_DIR / "global_0.6255.jsonl",
    "Basic RAG": ANSWER_DIR / "basic_rag_0.4827.jsonl",
}

OUTPUT_DIR = Path("../visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared constants ─────────────────────────────────────────
DIFF_CATS = ["L1", "L2", "L3", "L4"]
TYPE_ORDER = ["Fact Retrieval", "Numerical", "Reasoning",
              "Global Summary", "Design/Discovery"]
TYPE_MERGE = {"Design/Discovery": "Design/Discovery"}  # keep as-is

# Sheet3 scores (hard merged into L1-L4), best config per method
DIFF_SCORES = np.array([
    [0.6241, 0.6565, 0.8089, 0.8239],  # Reasoning
    [0.6289, 0.6567, 0.7949, 0.7705],  # Router
    [0.6238, 0.6488, 0.8159, 0.7254],  # Local
    [0.6159, 0.6410, 0.7760, 0.6661],  # Drift
    [0.6264, 0.5977, 0.7124, 0.6627],  # Global
    [0.5722, 0.4431, 0.6165, 0.2120],  # Basic RAG
])

# ── Type-level colors (for stacked bar & heatmap) ────────────
TYPE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


# ── Helper: load per-question data from one JSONL ────────────
def _load_questions(fpath):
    """Return list of dicts with difficulty, type, overall_score."""
    items = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            score = obj.get("overall_score") or obj["scores"]["overall_score"]
            items.append({
                "difficulty": obj["difficulty"],
                "type": obj["type"],
                "domain": obj.get("domain", "Other"),
                "score": score,
            })
    return items


def _load_type_scores():
    """Return dict[method] -> dict[type] -> mean score."""
    result = {}
    for method, fpath in ANSWER_FILES.items():
        type_scores = defaultdict(list)
        for q in _load_questions(fpath):
            type_scores[q["type"]].append(q["score"])
        result[method] = {t: np.mean(v) for t, v in type_scores.items()}
    return result


def _build_cross_tab():
    """Build Difficulty × Type count matrix from any JSONL (all share same Qs)."""
    questions = _load_questions(next(iter(ANSWER_FILES.values())))
    mat = np.zeros((len(TYPE_ORDER), len(DIFF_CATS)), dtype=int)
    for q in questions:
        if q["type"] in TYPE_ORDER and q["difficulty"] in DIFF_CATS:
            r = TYPE_ORDER.index(q["type"])
            c = DIFF_CATS.index(q["difficulty"])
            mat[r, c] += 1
    return mat


def _save(fig, name):
    for fmt in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{name}.{fmt}", dpi=300, bbox_inches="tight")
    print(f"Saved {name}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  fig3a — Difficulty × Type heatmap (test set structure)
# ══════════════════════════════════════════════════════════════
def draw_fig3a():
    mat = _build_cross_tab()  # (n_types × n_diffs)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = LinearSegmentedColormap.from_list("wh_rd", ["#FFFFFF", "#C44E52"])
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0)

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if val > mat.max() * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=24, fontweight="bold", color=color)

    ax.set_xticks(np.arange(len(DIFF_CATS)))
    ax.set_xticklabels(DIFF_CATS)
    ax.set_yticks(np.arange(len(TYPE_ORDER)))
    ax.set_yticklabels(TYPE_ORDER)
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Question Type")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Count", rotation=270, labelpad=20)

    ax.tick_params(axis="both", direction="in", top=False, right=False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    _save(fig, "fig3a_heatmap")


# ══════════════════════════════════════════════════════════════
#  fig3b — Score by difficulty level (grouped bar)
# ══════════════════════════════════════════════════════════════
def draw_fig3b():
    fig, ax = plt.subplots(figsize=(10, 7))

    n_m = len(METHODS)
    x = np.arange(len(DIFF_CATS))
    bar_w = 0.78 / n_m

    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        offset = bar_w * (i - (n_m - 1) / 2)
        ax.bar(x + offset, DIFF_SCORES[i], bar_w,
               label=method, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(DIFF_CATS)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(fontsize=16, frameon=True, loc="upper left",
              edgecolor="black", fancybox=False)
    _apply_spine_style(ax)
    _save(fig, "fig3b_difficulty")


# ══════════════════════════════════════════════════════════════
#  fig3c — Score by question type (grouped bar)
# ══════════════════════════════════════════════════════════════
def draw_fig3c():
    type_data = _load_type_scores()

    score_mat = np.zeros((len(METHODS), len(TYPE_ORDER)))
    for i, method in enumerate(METHODS):
        for j, qtype in enumerate(TYPE_ORDER):
            score_mat[i, j] = type_data[method].get(qtype, 0.0)

    fig, ax = plt.subplots(figsize=(14, 7))

    n_m = len(METHODS)
    x = np.arange(len(TYPE_ORDER))
    bar_w = 0.78 / n_m

    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        offset = bar_w * (i - (n_m - 1) / 2)
        ax.bar(x + offset, score_mat[i], bar_w,
               label=method, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Question Type")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_ORDER, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(fontsize=16, frameon=True, loc="upper left",
              edgecolor="black", fancybox=False, ncol=2)
    _apply_spine_style(ax)
    _save(fig, "fig3c_type")


# ══════════════════════════════════════════════════════════════
#  fig3d — Test set distribution (stacked bar: difficulty × type)
# ══════════════════════════════════════════════════════════════
def draw_fig3d():
    mat = _build_cross_tab()  # (n_types × n_diffs)

    fig, ax = plt.subplots(figsize=(9, 7))

    x = np.arange(len(DIFF_CATS))
    bottom = np.zeros(len(DIFF_CATS))

    for i, (qtype, color) in enumerate(zip(TYPE_ORDER, TYPE_COLORS)):
        vals = mat[i, :]  # counts for this type across L1-L4
        ax.bar(x, vals, 0.55, bottom=bottom, label=qtype,
               color=color, edgecolor="white", linewidth=0.5)
        # Annotate non-zero segments
        for j, v in enumerate(vals):
            if v > 0:
                ax.text(x[j], bottom[j] + v / 2, str(v),
                        ha="center", va="center", fontsize=18,
                        fontweight="bold", color="white")
        bottom += vals

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Number of Questions")
    ax.set_xticks(x)
    ax.set_xticklabels(DIFF_CATS)
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 120, 20))
    ax.legend(fontsize=14, frameon=True, loc="upper right",
              edgecolor="black", fancybox=False)
    _apply_spine_style(ax)
    _save(fig, "fig3d_distribution")


if __name__ == "__main__":
    draw_fig3a()
    draw_fig3b()
    draw_fig3c()
    draw_fig3d()

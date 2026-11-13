"""Draw the held-out results chart for the README (light + dark versions).

    python scripts/plot_results.py

Reads artifacts/<task>/<model>/evaluation.json and writes docs/media/results-{light,dark}.png.
Dots are test-set macro-F1; whiskers are the 1,000-round bootstrap 95% interval.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = [  # (folder, label, added in the HingRoBERTa/ensemble update?)
    ("tfidf", "TF-IDF + logistic regression", False),
    ("transformer", "XLM-RoBERTa", False),
    ("ensemble", "TF-IDF + XLM-RoBERTa", True),
    ("hingroberta", "HingRoBERTa", True),
    ("ensemble-hing", "TF-IDF + HingRoBERTa", True),
]
TASKS = [("hot", "HOT · offensive language (3 classes, n=443)"),
         ("thar", "THAR · religious hate (2 classes, n=1,730)")]
THEMES = {  # validated with the dataviz palette script (categorical slots 1-2, both modes)
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1", new="#2a78d6", old="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#33332f", new="#3987e5", old="#d95926"),
}


def load(task, folder):
    path = ROOT / "artifacts" / task / folder / "evaluation.json"
    e = json.loads(path.read_text(encoding="utf-8"))
    return e["macro_f1"], e["macro_f1_ci95"]


def draw(mode):
    t = THEMES[mode]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, facecolor=t["surface"])
    for ax, (task, title) in zip(axes, TASKS):
        ax.set_facecolor(t["surface"])
        rows = [(label, new, *load(task, folder)) for folder, label, new in MODELS]
        best = max(r[2] for r in rows)
        for y, (label, new, f1, (lo, hi)) in enumerate(rows):
            color = t["new"] if new else t["old"]
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([f1], [y], s=80, marker="o" if new else "s", color=color,
                       edgecolors=t["surface"], linewidths=2, zorder=3)
            if f1 == best or label == "XLM-RoBERTa":  # selective direct labels: baseline and best only
                ax.annotate(f"{f1:.3f}", (hi, y), xytext=(6, 0), textcoords="offset points",
                            va="center", color=t["ink"], fontweight="bold" if f1 == best else "normal")
        ax.set_title(title, loc="left", color=t["ink"], fontsize=11, pad=10)
        ax.set_xlim(0.66, 0.89)
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
        ax.tick_params(colors=t["ink2"], length=0)
        ax.set_xlabel("Test macro-F1 (dot) with 95% bootstrap interval", color=t["ink2"])
    axes[0].set_yticks(range(len(MODELS)), [m[1] for m in MODELS], color=t["ink"])
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], marker="s", linestyle="-", color=t["old"], markersize=8, label="Earlier models"),
               plt.Line2D([], [], marker="o", linestyle="-", color=t["new"], markersize=8, label="Added: Hinglish encoder + ensembles")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, labelcolor=t["ink"], bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "docs" / "media" / f"results-{mode}.png"
    fig.savefig(out, dpi=160, facecolor=t["surface"])
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    for mode in THEMES:
        draw(mode)

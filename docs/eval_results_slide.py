"""Build the corrected evaluation-results slide as a PNG, matching the
layout of the original marketing slide but using the real numbers from
eval/baselines/hallucination_compare.json and eval/baselines/baseline_v5_1_stage_aware.json.

Run:  python docs/eval_results_slide.py
Out:  docs/eval_results_slide.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

FIG_W_IN, FIG_H_IN = 17.0, 10.0
DPI = 144

INK = "#0B1418"
TEAL = "#0E7C8C"
GREEN = "#2E7D5B"
GREY_BAR = "#B8BEC2"
GREY_TRACK = "#E8ECEE"
SUB = "#5E6B72"
MUTE = "#9AA3A8"
BG = "#F4F7F9"

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.add_patch(Rectangle((0, 0), 1, 1, color="white"))
ax.add_patch(Rectangle((0, 0.94), 1, 0.06, color=BG))

ax.text(0.030, 0.965, "18 · EVALUATION", color=SUB, fontsize=11, weight="bold")
ax.text(0.970, 0.965, "BASELINE LLM vs DOCUMED AI", color=SUB, fontsize=11,
        ha="right", weight="bold")

ax.text(0.040, 0.885, "RESULTS", color=TEAL, fontsize=12, weight="bold")
ax.text(0.040, 0.820, "Honest pilot.", color=INK, fontsize=36, weight="bold")
ax.text(0.040, 0.755, "Where it counts, DocuMed beats baseline.",
        color=INK, fontsize=22, weight="bold")

ax.text(0.660, 0.840, "–38%", color=TEAL, fontsize=54, weight="bold")
ax.text(0.815, 0.840, "+106%", color=GREEN, fontsize=54, weight="bold")
ax.text(0.660, 0.770, "HALLUCINATION RATE", color=SUB, fontsize=10, weight="bold")
ax.text(0.660, 0.755, "(claim-mode, n=10)", color=MUTE, fontsize=9)
ax.text(0.815, 0.770, "MEAN ENTAILMENT", color=SUB, fontsize=10, weight="bold")
ax.text(0.815, 0.755, "(NLI faithfulness)", color=MUTE, fontsize=9)

card = FancyBboxPatch((0.035, 0.065), 0.930, 0.625,
                       boxstyle="round,pad=0.005,rounding_size=0.012",
                       linewidth=1, edgecolor="#D7DCE0", facecolor="white")
ax.add_patch(card)

ax.text(0.055, 0.660, "METRIC", color=TEAL, fontsize=11, weight="bold")
ax.add_patch(Rectangle((0.770, 0.659), 0.012, 0.014, color=GREY_BAR))
ax.text(0.788, 0.660, "BASELINE LLM", color=SUB, fontsize=10, weight="bold")
ax.add_patch(Rectangle((0.890, 0.659), 0.012, 0.014, color=TEAL))
ax.text(0.908, 0.660, "DOCUMED AI", color=TEAL, fontsize=10, weight="bold")

ROWS = [
    {
        "label": "Hallucination rate (claim-mode)",
        "hint": "lower is better",
        "baseline": 14.4, "documed": 8.9,
        "max": 100.0, "fmt": "{:.1f}%",
        "head_to_head": True,
    },
    {
        "label": "Mean entailment vs sources (NLI)",
        "hint": "higher is better",
        "baseline": 0.082, "documed": 0.169,
        "max": 1.0, "fmt": "{:.3f}",
        "head_to_head": True,
    },
    {
        "label": "Share high-confidence (entail ≥ 0.80)",
        "hint": "higher is better",
        "baseline": 5.1, "documed": 10.7,
        "max": 100.0, "fmt": "{:.1f}%",
        "head_to_head": True,
    },
    {
        "label": "Share hard hallucination (entail < 0.05)",
        "hint": "lower is better",
        "baseline": 87.3, "documed": 73.2,
        "max": 100.0, "fmt": "{:.1f}%",
        "head_to_head": True,
    },
    {
        "label": "Retrieval Recall@5  (best stage / weighted avg)",
        "hint": "DocuMed only — vanilla LLM has no retrieval step",
        "baseline": None, "documed": 43.3,
        "documed_secondary": 13.9,
        "max": 100.0, "fmt": "{:.1f}%",
        "head_to_head": False,
    },
    {
        "label": "Refusal accuracy on must-refuse gold (n=40)",
        "hint": "higher is better",
        "baseline": 0.825, "documed": 1.00,
        "max": 1.0, "fmt": "{:.2%}",
        "head_to_head": True,
    },
]

ROW_TOP = 0.615
ROW_DY = 0.085
BAR_X0 = 0.055
BAR_X1 = 0.760
BAR_W = BAR_X1 - BAR_X0
BAR_H = 0.014
VAL_X = 0.770
HINT_X = 0.965

for i, row in enumerate(ROWS):
    y_label = ROW_TOP - i * ROW_DY
    y_b = y_label - 0.027
    y_d = y_label - 0.050

    ax.text(0.055, y_label, row["label"], color=INK, fontsize=13.5, weight="bold")
    hint_color = SUB if row["head_to_head"] else MUTE
    ax.text(HINT_X, y_label, row["hint"], color=hint_color, fontsize=9.5,
            ha="right",
            style="italic" if not row["head_to_head"] else "normal")

    # Baseline track
    ax.add_patch(FancyBboxPatch((BAR_X0, y_b - BAR_H / 2), BAR_W, BAR_H,
                                 boxstyle="round,pad=0,rounding_size=0.007",
                                 facecolor=GREY_TRACK, edgecolor="none"))
    if row["baseline"] is not None:
        w = BAR_W * (row["baseline"] / row["max"])
        if w > 0.005:
            ax.add_patch(FancyBboxPatch((BAR_X0, y_b - BAR_H / 2), w, BAR_H,
                                         boxstyle="round,pad=0,rounding_size=0.007",
                                         facecolor=GREY_BAR, edgecolor="none"))
        ax.text(VAL_X, y_b, row["fmt"].format(row["baseline"]),
                color=SUB, fontsize=11, weight="bold", va="center")
    else:
        ax.text(BAR_X0 + 0.008, y_b, "not measured", color=MUTE,
                fontsize=10, va="center", style="italic")
        ax.text(VAL_X, y_b, "—", color=MUTE, fontsize=11, weight="bold", va="center")

    # DocuMed track
    ax.add_patch(FancyBboxPatch((BAR_X0, y_d - BAR_H / 2), BAR_W, BAR_H,
                                 boxstyle="round,pad=0,rounding_size=0.007",
                                 facecolor=GREY_TRACK, edgecolor="none"))
    if row["documed"] is not None:
        w = BAR_W * (row["documed"] / row["max"])
        if w > 0.005:
            ax.add_patch(FancyBboxPatch((BAR_X0, y_d - BAR_H / 2), w, BAR_H,
                                         boxstyle="round,pad=0,rounding_size=0.007",
                                         facecolor=TEAL, edgecolor="none"))
        label = row["fmt"].format(row["documed"])
        if "documed_secondary" in row:
            label += " / " + row["fmt"].format(row["documed_secondary"])
        ax.text(VAL_X, y_d, label, color=TEAL, fontsize=11, weight="bold", va="center")

# Caption / methodology
cap_y = 0.092
ax.text(0.055, cap_y + 0.022,
        "Hallucination & faithfulness: N = 10 head-to-head pilot · gold = coverage.jsonl · "
        "same Groq Llama 3.3 70B generated every answer in both arms · "
        "DocuMed retrieval uses Cohere Rerank · scored with cross-encoder/nli-deberta-v3-base at τ=0.50.",
        color=SUB, fontsize=9.5)
ax.text(0.055, cap_y + 0.004,
        "Refusal accuracy: N = 40 adversarial prompts (must_refuse.jsonl) · regex forbidden-pattern check · "
        "vanilla = bare Llama 3.3 70B with generic medical-assistant system prompt.",
        color=SUB, fontsize=9.5)
ax.text(0.055, cap_y - 0.014,
        "Recall@5: DocuMed N = 30 across 6 stages (baseline_v5_1_stage_aware.json) — vanilla LLM has no retrieval to score. "
        "Pilot stopped at 10/20 cases on a rate-limit error (Groq TPD or Cohere Rerank trial cap).",
        color=MUTE, fontsize=9.5, style="italic")

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results_slide.png")
fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches=None)
print(f"wrote {out}  ({FIG_W_IN*DPI:.0f}x{FIG_H_IN*DPI:.0f} px)")

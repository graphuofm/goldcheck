#!/usr/bin/env python3
"""CHI2 Figure 1: expert annotation error rate vs required depth (panel A)
and the direction of structural corrections (panel B). Real numbers from
results/error_vs_complexity.json and results/diff_taxonomy.json.
Palette #0072B2/#D55E00 validated earlier (CVD dE 21.9, all checks PASS)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results", "error_vs_complexity.json")))
T = json.load(open(os.path.join(HERE, "results", "diff_taxonomy.json")))

BLUE, VERM, INK, MUT = "#0072B2", "#D55E00", "#222222", "#666666"
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.6), dpi=200,
                           gridspec_kw={"width_ratios": [1.5, 1]})

# --- panel A: error rate by nesting depth of the CORRECT solution ---
# two strata: full error population vs the original gold, and the subset that
# survived the first (community) audit round
S = json.load(open(os.path.join(HERE, "results", "audit_survival.json")))
full = S["A_vs_original_gold"]["rate"]          # 35.1 / 60.5 / 91.7
ns = S["A_vs_original_gold"]["n"]               # 410 / 76 / 12
surv = [r[2] for r in R["buckets"]["depth"]]    # 23.4 / 38.2 / 75.0
labels = [f"no nesting\nn={ns[0] if (ns:=S['A_vs_original_gold']['n']) else 0}",
          f"1 subquery level\nn={ns[1]}", f"2+ levels\nn={ns[2]}"]
x = list(range(len(labels)))
w = 0.36
a.bar([xi - w/2 for xi in x], full, width=w, color=VERM, zorder=3,
      label="all errors (vs. original gold)")
a.bar([xi + w/2 for xi in x], surv, width=w, color=BLUE, zorder=3,
      label="errors surviving 1st audit")
for xi, (vf, vs) in enumerate(zip(full, surv)):
    a.text(xi - w/2, vf + 2.0, f"{vf:.0f}%", ha="center", fontsize=8,
           color=INK, fontweight="bold")
    a.text(xi + w/2, vs + 2.0, f"{vs:.0f}%", ha="center", fontsize=8, color=INK)
a.set_xticks(x); a.set_xticklabels(labels, fontsize=7.5, color=INK)
a.set_ylabel("SQL annotation error rate (%)", fontsize=8.5, color=INK)
a.set_ylim(0, 103)
a.legend(fontsize=6.8, frameon=False, loc="upper left",
         handlelength=1.1, borderaxespad=0.1)
a.set_title("Errors track the depth the task demands", fontsize=9, color=INK, pad=8)

# --- panel B: direction of structural corrections ---
dd = {int(k): v for k, v in T["depth_delta"].items()}
up = sum(v for k, v in dd.items() if k > 0)
dn = sum(v for k, v in dd.items() if k < 0)
bars = b.barh([1, 0], [up, dn], height=0.55, color=[VERM, BLUE], zorder=3)
b.set_yticks([1, 0])
b.set_yticklabels(["fix ADDS depth\n(expert wrote too shallow)",
                   "fix removes depth"], fontsize=7.5, color=INK)
b.set_xlim(0, 27); b.set_ylim(-0.55, 1.72)
b.set_xlabel("structural corrections (n=27)", fontsize=8, color=INK)
for y, v in ((1, up), (0, dn)):
    b.text(v + 0.6, y, str(v), va="center", fontsize=9.5, color=INK, fontweight="bold")
b.set_title("Errors err shallow, 5.8:1", fontsize=9, color=INK, pad=8)
b.text(0.98, 0.06, "binomial p = 1.5×10⁻⁴", transform=b.transAxes,
       fontsize=7.5, color=MUT, ha="right")

for ax in (a, b):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(labelsize=8, colors=MUT)
    ax.grid(axis="x" if ax is b else "y", color="#e9e9e9", lw=0.6)
    ax.set_axisbelow(True)

fig.tight_layout(pad=0.5, w_pad=2.0)
out = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(out, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"fig_error_vs_depth.{ext}"))
print("wrote paper/figures/fig_error_vs_depth.pdf (+png)")

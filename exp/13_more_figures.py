#!/usr/bin/env python3
"""
CHI2 Exp 13 -- 论文补图：把每个主要发现都配上图。

生成 6 张新图（Okabe-Ito 色盲安全配色，所有数值直接标注，灰度可读）：
  fig_depth_vs_breadth.pdf   深度 vs 表数 vs join 数的错误率并排对比（F2）
  fig_written_vs_demanded    要求深度×书写深度 3x3 矩阵（F4）
  fig_exec_consequences      错 gold 执行后的输出类别 + 各深度貌似合理率（F6）
  fig_trap_concentration     每个库的头号陷阱占该库错误的份额（F7）
  fig_loo_forest             留一库检验 + 三层错误率 bootstrap 区间（Robustness）
  fig_spider_extrapolation   BIRD/Spider 深度构成 + 预测69% vs 实测63%（外部一致性）

用法: python3 13_more_figures.py
"""
import csv, json, os
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "figures")
BLUE, VERM, GREEN, INK, MUT = "#0072B2", "#D55E00", "#009E73", "#222222", "#666666"
GRAY = "#999999"

def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(labelsize=8, colors=MUT)
    ax.grid(axis=grid_axis, color="#e9e9e9", lw=0.6)
    ax.set_axisbelow(True)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", name)

R = json.load(open(os.path.join(HERE, "results", "error_vs_complexity.json")))
S = json.load(open(os.path.join(HERE, "results", "audit_survival.json")))
EC = json.load(open(os.path.join(HERE, "results", "exec_consequences.json")))
RB = json.load(open(os.path.join(HERE, "results", "robustness.json")))
SP = json.load(open(os.path.join(HERE, "results", "spider_extrapolation.json")))

# ---------- 1. depth vs breadth ----------
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2), dpi=200, sharey=True)
panels = [("depth", "nesting depth the task demands",
           ["none", "1 level", "2+"], VERM),
          ("tables", "tables referenced", ["1", "2", "3", "4+"], BLUE),
          ("joins", "join operations", ["0", "1", "2+"], BLUE)]
for ax, (key, title, labels, color) in zip(axes, panels):
    rates = [r[2] for r in R["buckets"][key]]
    ns = [r[1] for r in R["buckets"][key]]
    x = range(len(rates))
    ax.bar(x, rates, width=0.6, color=color, zorder=3)
    for xi, (v, n) in enumerate(zip(rates, ns)):
        ax.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=8, color=INK,
                fontweight="bold" if key == "depth" else "normal")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{l}\nn={n}" for l, n in zip(labels, ns)], fontsize=7)
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.set_ylim(0, 88)
    style(ax)
axes[0].set_ylabel("annotation error rate (%)", fontsize=8.5, color=INK)
fig.tight_layout()
save(fig, "fig_depth_vs_breadth")

# ---------- 2. written vs demanded 3x3 ----------
ct = S["depth_crosstab"]
fig, ax = plt.subplots(figsize=(3.6, 2.9), dpi=200)
mat_n = [[0]*3 for _ in range(3)]
mat_e = [[0]*3 for _ in range(3)]
for k, v in ct.items():
    dd = int(k[2]); dw = int(k[6])
    mat_n[dd][dw] += v
    if k.endswith("err"):
        mat_e[dd][dw] += v
for dd in range(3):
    for dw in range(3):
        n = mat_n[dd][dw]
        if n == 0:
            ax.add_patch(plt.Rectangle((dw, dd), 1, 1, color="#f2f2f2"))
            continue
        rate = mat_e[dd][dw] / n
        ax.add_patch(plt.Rectangle((dw, dd), 1, 1,
                     color=plt.cm.Oranges(0.15 + 0.7 * rate)))
        ax.text(dw + 0.5, dd + 0.5, f"n={n}\n{100*rate:.0f}% err",
                ha="center", va="center", fontsize=7.5,
                color=INK if rate < 0.75 else "white")
labels = ["none", "1 level", "2+"]
ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(labels, fontsize=8)
ax.set_yticks([0.5, 1.5, 2.5]); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("depth the expert WROTE (original gold)", fontsize=8.5, color=INK)
ax.set_ylabel("depth the task DEMANDS", fontsize=8.5, color=INK)
ax.set_xlim(0, 3); ax.set_ylim(0, 3)
ax.set_title("Writing shallower than the task demands\nis always wrong",
             fontsize=8.5, color=INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
save(fig, "fig_written_vs_demanded")

# ---------- 3. execution consequences ----------
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.3), dpi=200,
                           gridspec_kw={"width_ratios": [1.4, 1]})
cats = [("identical-output", "identical to correct output", VERM),
        ("values-differ", "same shape, different values", BLUE),
        ("rowcount-differs", "different number of rows", BLUE),
        ("shape-differs", "different columns", BLUE),
        ("t0-error", "fails to execute", GRAY)]
vals = [EC["categories"].get(k, 0) for k, _, _ in cats]
y = range(len(cats))[::-1]
a.barh(list(y), vals, height=0.6,
       color=[c for _, _, c in cats], zorder=3)
for yi, v, (_, lab, _) in zip(y, vals, cats):
    a.text(v + 1.5, yi, f"{v} ({100*v/EC['n']:.0f}%)", va="center",
           fontsize=8, color=INK)
a.set_yticks(list(y))
a.set_yticklabels([lab for _, lab, _ in cats], fontsize=8)
a.set_xlabel(f"the {EC['n']} erroneous gold queries", fontsize=8, color=INK)
a.set_xlim(0, 118)
style(a, "x")
pb = EC["plausible_by_depth"]
ks = sorted(pb)
rates = [100 * pb[k][0] / pb[k][1] for k in ks]
b.bar(range(len(ks)), rates, width=0.55, color=GREEN, zorder=3)
for xi, (k, v) in enumerate(zip(ks, rates)):
    b.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK,
           fontweight="bold")
b.set_xticks(range(len(ks)))
b.set_xticklabels([f"none\nn={pb[k][1]}" if i == 0 else
                   (f"1 level\nn={pb[k][1]}" if i == 1 else f"2+\nn={pb[k][1]}")
                   for i, k in enumerate(ks)], fontsize=7.5)
b.set_ylabel("plausible output (%)", fontsize=8, color=INK)
b.set_ylim(0, 104)
b.set_title("invisibility is uniform in depth", fontsize=8.5, color=INK)
style(b)
fig.tight_layout()
save(fig, "fig_exec_consequences")

# ---------- 4. trap concentration ----------
tax = list(csv.DictReader(open(os.path.join(HERE, "results", "taxonomy_manual.csv"))))
uiuc = json.load(open(os.path.join(HERE, "data", "text_to_sql_benchmarks",
                                   "data", "arcwise_plat_full_with_diff.json")))
db_of = {str(e["question_id"]): e["db_id"] for e in uiuc}
per_db = defaultdict(Counter)
for r in tax:
    per_db[db_of.get(r["qid"], "?")][r["mechanism"]] += 1
rows = []
for db, c in per_db.items():
    tot = sum(c.values())
    if tot < 5:
        continue
    mech, top = c.most_common(1)[0]
    rows.append((db, tot, mech, top))
rows.sort(key=lambda x: -x[3] / x[1])
fig, ax = plt.subplots(figsize=(6.8, 2.5), dpi=200)
y = range(len(rows))[::-1]
short = {"thrombosis_prediction": "thrombosis (medical)",
         "california_schools": "california schools",
         "codebase_community": "codebase community",
         "european_football_2": "european football",
         "debit_card_specializing": "debit card",
         "student_club": "student club"}
ax.barh(list(y), [r[1] for r in rows], height=0.62, color="#d9d9d9",
        zorder=2, label="all corrected errors in this database")
ax.barh(list(y), [r[3] for r in rows], height=0.62, color=VERM, zorder=3,
        label="its single most common trap")
for yi, (db, tot, mech, top) in zip(y, rows):
    ax.text(tot + 0.3, yi, f"{top}/{tot} = {mech}", va="center", fontsize=7.5,
            color=INK)
ax.set_yticks(list(y))
ax.set_yticklabels([short.get(r[0], r[0]) for r in rows], fontsize=8)
ax.set_xlabel("corrected annotation errors (research-audit round)", fontsize=8.5,
              color=INK)
ax.set_xlim(0, 30)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
ax.set_title("Errors cluster into per-database schema traps", fontsize=9, color=INK)
style(ax, "x")
fig.tight_layout()
save(fig, "fig_trap_concentration")

# ---------- 5. robustness forest ----------
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.4), dpi=200,
                           gridspec_kw={"width_ratios": [1.2, 1]})
loo = RB["leave_one_db_out_z"]
items = sorted(loo.items(), key=lambda x: x[1])
y = range(len(items))
a.scatter([v for _, v in items], list(y), color=BLUE, s=22, zorder=3)
a.axvline(4.44, color=VERM, lw=1, ls="--")
a.text(4.5, len(items) - 0.6, "all data: z=4.44", fontsize=7.5, color=VERM)
a.axvline(1.96, color=GRAY, lw=0.8, ls=":")
a.text(2.0, -0.4, "z=1.96", fontsize=7, color=MUT)
a.set_yticks(list(y))
a.set_yticklabels([k.replace("_", " ")[:22] for k, _ in items], fontsize=7)
a.set_xlabel("trend z after excluding this database", fontsize=8.5, color=INK)
a.set_xlim(0, 5.2)
a.set_title("no single database carries the gradient", fontsize=8.5, color=INK)
style(a, "x")
ci = RB["bootstrap_ci"]
labels = ["none", "1 level", "2+"]
for i, k in enumerate(["stratum0", "stratum1", "stratum2"]):
    lo, hi = ci[k]["ci95"]
    b.plot([i, i], [lo, hi], color=BLUE, lw=2.2, zorder=3)
    b.scatter([i], [ci[k]["rate"]], color=VERM, s=30, zorder=4)
    b.text(i + 0.1, ci[k]["rate"], f"{ci[k]['rate']}%", fontsize=8, color=INK)
b.set_xticks(range(3)); b.set_xticklabels(labels, fontsize=8)
b.set_xlim(-0.5, 2.8)
b.set_ylabel("error rate, 95% bootstrap CI", fontsize=8.5, color=INK)
b.set_title("error strata fully separated (audited outcome)", fontsize=8.5, color=INK)
style(b)
fig.tight_layout()
save(fig, "fig_loo_forest")

# ---------- 6. spider extrapolation ----------
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.2), dpi=200,
                           gridspec_kw={"width_ratios": [1.2, 1]})
bird = [410 / 498, 76 / 498, 12 / 498]
spd = SP["spider_depth_dist"]
n = SP["n"]
spider = [spd.get("1", 0) / n, spd.get("2", 0) / n, spd.get("3", 0) / n]
x = range(3)
w = 0.36
a.bar([xi - w/2 for xi in x], [100 * v for v in bird], width=w, color=BLUE,
      zorder=3, label="BIRD Mini-Dev")
a.bar([xi + w/2 for xi in x], [100 * v for v in spider], width=w, color=VERM,
      zorder=3, label="Spider 2.0-Snow")
for xi, (vb, vs) in enumerate(zip(bird, spider)):
    a.text(xi - w/2, 100*vb + 2, f"{100*vb:.0f}%", ha="center", fontsize=7.5, color=INK)
    a.text(xi + w/2, 100*vs + 2, f"{100*vs:.0f}%", ha="center", fontsize=7.5, color=INK)
a.set_xticks(list(x)); a.set_xticklabels(["no nesting", "1 level", "2+ levels"], fontsize=8)
a.set_ylabel("share of gold queries (%)", fontsize=8.5, color=INK)
a.set_ylim(0, 95)
a.legend(fontsize=7.5, frameon=False)
a.set_title("two benchmarks, opposite depth profiles", fontsize=8.5, color=INK)
style(a)
b.bar([0, 1], [100 * SP["predicted_error"], 100 * SP["reported_error"]],
      width=0.5, color=[BLUE, VERM], zorder=3)
b.text(0, 100*SP["predicted_error"] + 2, f"{100*SP['predicted_error']:.1f}%",
       ha="center", fontsize=9, color=INK, fontweight="bold")
b.text(1, 100*SP["reported_error"] + 2, f"{100*SP['reported_error']:.1f}%",
       ha="center", fontsize=9, color=INK, fontweight="bold")
b.set_xticks([0, 1])
b.set_xticklabels(["predicted from\nBIRD gradient", "measured by\nindependent audit"],
                  fontsize=7.5)
b.set_ylabel("Spider annotation error rate (%)", fontsize=8, color=INK)
b.set_ylim(0, 82)
b.set_title("out-of-sample prediction", fontsize=8.5, color=INK)
style(b)
fig.tight_layout()
save(fig, "fig_spider_extrapolation")

print("all CHI2 figures written")

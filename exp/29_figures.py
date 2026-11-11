#!/usr/bin/env python3
"""ICDE figures, sized for one IEEE column (3.45 in). Every number is read from results/.
Usage: python3 29_figures.py"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)
R = lambda f: json.load(open(os.path.join(HERE, "results", f)))
BLUE, ORANGE, GREEN, PINK, SKY, GREY, YELLOW = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#7F7F7F", "#E69F00"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "axes.linewidth": 0.6})
W = 3.45


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_measurement():
    va, fd, nc = R("validity_audit.json"), R("full_dev_replication.json"), R("noncircular.json")
    panels = [("Mini-Dev (498)", va["B_headline"], nc["mini"]["by_written_depth"], va["B_diagonal_only"]),
              ("Dev held-out (1,036)", fd["heldout_not_in_minidev"]["txt"]["depth"], nc["dev_heldout"]["by_written_depth"],
               fd["heldout_not_in_minidev"]["txt"]["diag"])]
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.75), sharey=True)
    labels = ["none", "1 level", "2+"]
    for ax, (title, naive, written, diag) in zip(axes, panels):
        for s, (nm, d, col, mk) in enumerate((("depth of corrected query", naive, ORANGE, "o"),
                                             ("depth of query under review", written, BLUE, "s"),
                                             ("depth unchanged by review", diag, GREEN, "^"))):
            xs = [i + (s - 1) * 0.12 for i in range(3)]
            for x, r, n in zip(xs, d["rate"], d["n"]):
                if n == 0 or r is None:
                    continue
                k = round(r * n / 100); lo, hi = wilson(k, n)
                ax.plot([x, x], [lo, hi], color=col, lw=0.7, alpha=0.6)
                ax.plot(x, r, mk, color=col, ms=4, mfc=col if n >= 20 else "white", mew=0.9,
                        label=nm if x == xs[0] else None)
            ax.plot(xs, [r if n else None for r, n in zip(d["rate"], d["n"])], color=col, lw=0.8, alpha=0.7)
        ax.set_title(title); ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_xlabel("nesting depth")
        ax.set_ylim(0, 105); ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("% of examples changed\nby the review")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.42))
    save(fig, "fig_measurement")


def fig_flips():
    f = R("flip_rates.json")
    fig, ax = plt.subplots(figsize=(W, 1.45))
    groups = [("Mini-Dev", f["mini"]["depth"]), ("Dev held-out", f["dev_heldout"]["depth"])]
    for i, (nm, d) in enumerate(groups):
        for j, (k, n, col, lab) in enumerate(((d["flat_gain"], d["flat_n"], BLUE, "flat query made nested"),
                                              (d["nested_lose"], d["nested_n"], ORANGE, "nested query made flatter"))):
            y = i * 1.0 + (j - 0.5) * 0.36
            r = 100 * k / n; lo, hi = wilson(k, n)
            ax.barh(y, r, height=0.32, color=col, label=lab if i == 0 else None, hatch=None if j == 0 else "///", edgecolor="white", lw=0)
            ax.plot([lo, hi], [y, y], color="black", lw=0.7)
            ax.text(hi + 0.4, y, f"{r:.1f}% ({k}/{n})", va="center", fontsize=6.3)
        ax.text(-0.3, i, nm, ha="right", va="center", fontsize=7)
    ax.set_yticks([]); ax.set_xlim(0, 30); ax.set_xlabel("% of starting queries changed in depth (95% CI)")
    ax.spines["left"].set_visible(False); ax.invert_yaxis()
    ax.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_flips")


def fig_signals():
    nc = R("noncircular.json")
    fig, ax = plt.subplots(figsize=(W, 1.7))
    labels = ["0", "1", "2", "3+"]
    for s, (nm, d, col, mk) in enumerate((("Mini-Dev (lexicon built here)", nc["mini"]["by_markers"], BLUE, "s"),
                                         ("Dev held-out (lexicon frozen)", nc["dev_heldout"]["by_markers"], ORANGE, "o"))):
        xs = [i + (s - 0.5) * 0.14 for i in range(4)]
        for x, r, n in zip(xs, d["rate"], d["n"]):
            k = round(r * n / 100); lo, hi = wilson(k, n)
            ax.plot([x, x], [lo, hi], color=col, lw=0.7, alpha=0.6)
            ax.plot(x, r, mk, color=col, ms=4, mfc=col if n >= 20 else "white", mew=0.9)
            ax.text(x + 0.07, r, f"n={n}", fontsize=5.6, color=col, va="center")
        ax.plot(xs, d["rate"], color=col, lw=0.8, label=f'{nm}; trend z={d["z"]}')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_xlabel("demand words in the question")
    ax.set_ylabel("% changed by the review"); ax.set_ylim(0, 105); ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig_signals")


def fig_execution():
    fn = R("final_numbers.json"); ms = R("mutated_states.json")["summary"]
    order = [("identical rows", GREY), ("identical set, different multiplicity", "#BBBBBB"), ("same shape, different values", SKY),
             ("different row count", BLUE), ("different columns", PINK), ("old query fails", "black"), ("new query fails", "black")]
    fig, ax = plt.subplots(figsize=(W, 1.25))
    for i, (tag, nm) in enumerate((("mini", "Mini-Dev\n(201)"), ("dev_heldout", "Dev held-out\n(273)"))):
        d = fn[tag]; left = 0
        for k, col in order:
            v = d["outcomes"].get(k, 0)
            if not v:
                continue
            w = 100 * v / d["n"]
            ax.barh(i, w, left=left, color=col, height=0.6, edgecolor="white", lw=0.4, label=k if i == 0 or k == "new query fails" and v else None)
            if w >= 7:
                ax.text(left + w / 2, i, f"{w:.0f}", ha="center", va="center", fontsize=6.3, color="white" if col in (BLUE, PINK, GREY, "black") else "black")
            left += w
        ax.text(-1, i, nm, ha="right", va="center", fontsize=6.8)
    ax.set_yticks([]); ax.set_xlim(0, 100); ax.invert_yaxis(); ax.spines["left"].set_visible(False)
    ax.set_xlabel("% of changed gold queries, by the outcome of the old and new gold on the official database")
    h, l = ax.get_legend_handles_labels(); seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll.replace("old query fails", "query fails").replace("new query fails", "query fails"), hh)
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.55), fontsize=6.2)
    save(fig, "fig_execution")


def fig_routing():
    fw = R("noncircular.json")["forward"]
    fig, ax = plt.subplots(figsize=(W, 1.75))
    B = ["0.05", "0.1", "0.2", "0.3", "0.5"]
    for nm, col, mk, ls in (("random", GREY, "", "--"), ("question length", SKY, "o", "-"), ("demand words", BLUE, "s", "-"),
                            ("question+lints model", ORANGE, "^", "-")):
        key = "markers" if nm == "demand words" else nm
        ys = [fw[key]["catch_text"][b] for b in B]
        ax.plot([100 * float(b) for b in B], ys, ls, color=col, marker=mk, ms=3.2, lw=1, label=f'{nm} (AUC {fw[key]["auc_text_changed"]:.2f})')
    ax.set_xlabel("review budget (% of examples re-checked)"); ax.set_ylabel("% of changed examples\nin the re-checked set")
    ax.set_xlim(0, 52); ax.set_ylim(0, 70); ax.grid(lw=0.3, alpha=0.5); ax.legend(loc="upper left", frameon=False)
    save(fig, "fig_routing")


def fig_schemas():
    S = R("gold_effect_schema.json"); fn = R("final_numbers.json")
    a = S["mini"]["per_db"]; b = S["dev_heldout"]["per_db"]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    short = {"california_schools": "schools", "card_games": "cards", "codebase_community": "codebase", "debit_card_specializing": "debit card",
             "european_football_2": "football", "financial": "financial", "formula_1": "formula 1", "student_club": "student club",
             "superhero": "superhero", "thrombosis_prediction": "thrombosis", "toxicology": "toxicology"}
    off = {"financial": (-4, -9), "schools": (1, -8), "cards": (-9, 2.5), "football": (-11, -4.5), "codebase": (-2, 3.5), "formula 1": (-4, -6.5), "debit card": (1.2, 1.5), "toxicology": (1.5, -2.5)}
    for k in a:
        x, y = a[k]["rate"], b[k]["rate"]
        ax.plot(x, y, "o", color=BLUE, ms=4)
        dx, dy = off.get(short[k], (1.2, 1.0))
        ax.annotate(short[k], (x, y), xytext=(x + dx, y + dy), fontsize=6)
    ax.plot([0, 80], [0, 80], ls=":", color=GREY, lw=0.7)
    r = fn["per_db_spearman_mini_vs_dev"]
    ax.set_xlabel("Mini-Dev: % changed (community + audit)"); ax.set_ylabel("Dev held-out: % changed\n(maintainer review)")
    ax.set_xlim(20, 75); ax.set_ylim(0, 75); ax.grid(lw=0.3, alpha=0.5)
    ax.text(21, 70, f"Spearman rho = {r['rho']}, p = {r['p']}", fontsize=6.6)
    save(fig, "fig_schemas")


LIVE = ("topk_tie", "topk_null", "count_fanout", "empty_answer")
REVIEWS = (("mini_T0", "mini_T1", "Community pass\n(Mini-Dev T0 to T1)"), ("mini_T1", "mini_T2", "Research audit\n(Mini-Dev T1 to T2)"),
           ("dev_2024", "dev_2025", "Maintainers' review\n(Dev 2024 to 2025)"))


def _gc():
    d = R("goldcheck.json")
    by = {(r["ver"], r["qid"]): r for r in d["rows"]}
    return d, by


def fig_agreement():
    d, by = _gc(); ch = d["changes"]
    fig, ax = plt.subplots(figsize=(W, 2.05))
    groups = [("live finding", BLUE), ("latent witness only", SKY), ("no finding", GREY)]
    for i, (v, n, nm) in enumerate(REVIEWS):
        qs = [q for (vv, q) in by if vv == v]
        sets = {"live finding": [q for q in qs if any(f in by[(v, q)]["flags"] for f in LIVE)],
                "latent witness only": [q for q in qs if by[(v, q)]["flags"] == ["topk_latent"]],
                "no finding": [q for q in qs if not by[(v, q)]["flags"]]}
        for j, (g, col) in enumerate(groups):
            s = sets[g]; k = sum(ch[f"{v}|{n}|{q}"][1] for q in s); m = len(s)
            y = i + (j - 1) * 0.27; r = 100 * k / m; lo, hi = wilson(k, m)
            ax.barh(y, r, height=0.24, color=col, label=g if i == 0 else None)
            ax.plot([lo, hi], [y, y], color="black", lw=0.6)
            ax.text(hi + 1, y, f"{r:.0f}% ({k}/{m})", va="center", fontsize=5.8)
        ax.text(-1.5, i, nm, ha="right", va="center", fontsize=6.4)
    ax.set_yticks([]); ax.set_xlim(0, 100); ax.invert_yaxis(); ax.spines["left"].set_visible(False)
    ax.set_xlabel("% whose official output the later review changed")
    ax.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.38, 1.0), fontsize=6.3)
    save(fig, "fig_agreement")


def fig_fate():
    d, by = _gc()
    names = {"topk_tie": "top-k tie", "topk_null": "NULL at top", "count_fanout": "count fan-out", "empty_answer": "empty answer"}
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.45), sharey=True, gridspec_kw={"width_ratios": [1, 2.4]})
    for ax, (v, n, nm) in zip(axes, (("mini_T0", "mini_T2", "Mini-Dev T0 to T2"), ("dev_2024", "dev_2025", "Dev 2024 to 2025"))):
        for i, f in enumerate(LIVE):
            qs = [q for (vv, q) in by if vv == v and f in by[(v, q)]["flags"]]
            gone = sum(f not in by[(n, q)]["flags"] for q in qs); left = len(qs) - gone
            ax.barh(i, gone, color=BLUE, height=0.6, label="removed by the review" if i == 0 else None)
            ax.barh(i, left, left=gone + (0.4 if gone and left else 0), color=ORANGE, height=0.6, label="still present" if i == 0 else None)
            ax.text(len(qs) + 1, i, f"{gone}/{len(qs)}", va="center", fontsize=5.8)
        ax.set_title(nm, fontsize=6.8); ax.invert_yaxis() if ax is axes[0] else None
        ax.set_xlim(0, 22 if v.startswith("mini") else 64); ax.tick_params(axis="x", labelsize=6)
    axes[0].set_yticks(range(4)); axes[0].set_yticklabels([names[f] for f in LIVE], fontsize=6.4)
    fig.supxlabel("gold queries with a live finding", fontsize=6.6, y=-0.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 0.97), fontsize=6.3)
    save(fig, "fig_fate")


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    for f in (fig_measurement, fig_flips, fig_signals, fig_execution, fig_routing, fig_schemas, fig_agreement, fig_fate):
        if only and f.__name__ not in only:
            continue
        f(); print("ok", f.__name__)

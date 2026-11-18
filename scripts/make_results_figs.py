"""Additional result figures (class-wise coverage heatmap, agent budget curves, triage trade-off, decision-module accuracy).
Run from the 'new project' folder: python paper/make_results_figs.py"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)
FIG = R("paper", "figs")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
C = dict(split="#C0473F", aci="#E39B2D", weighted="#8E8E8E", rolling="#7A4FB5", maci="#2F6DB5", oracle="#1B2430",
         random="#9AA5B1", agent="#1F8A70")

# ---- (1) class-wise coverage heatmap, O1+O2 -> O3b ----
e2 = pd.read_csv(R("results_v5", "E2_shift.csv")); s = e2[(e2.source == "O1+O2") & (e2.target == "O3b")].set_index("method")
M = [("split_mondrian", "Split CP\n(Mondrian)"), ("aci_g0.005", "ACI\n(marginal)"), ("weighted_mondrian", "Weighted\nCP"),
     ("rolling_mondrian", "Rolling\nrecal.*"), ("mondrian_aci_g0.02", "Mondrian-\nACI"), ("oracle_mondrian", "Oracle")]
def pc(m): return {k: (float(v), int(n)) for k, v, n in (x.split(":") for x in s.loc[m, "per_class"].split(";"))}
base = pc("split_mondrian"); classes = sorted(base, key=lambda k: base[k][0])
Z = np.array([[pc(m).get(c, (np.nan, 0))[0] for m, _ in M] for c in classes])
fig, ax = plt.subplots(figsize=(6.6, 5.4))
im = ax.imshow(Z, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=0, vcenter=0.9, vmax=1.0), aspect="auto")
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        if np.isfinite(Z[i, j]): ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=6.4,
                                         color="white" if Z[i, j] < 0.35 else "black")
ax.set_xticks(range(len(M))); ax.set_xticklabels([l for _, l in M], fontsize=7.2)
ax.set_yticks(range(len(classes))); ax.set_yticklabels([f"{c.replace('_', ' ')} ({base[c][1]})" for c in classes], fontsize=6.8)
ax.xaxis.tick_top(); ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("class-wise coverage (nominal 0.9)")
ax.set_title("O1+O2 → O3b, class-wise coverage (rows: classes with ≥30 glitches, n in brackets; *label delay 2000)", fontsize=7.4, pad=28)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_classwise_heatmap.png"), dpi=250); plt.close(); print("heatmap")

# ---- (2) calibration agent: worst-class coverage vs label budget ----
b9 = pd.read_csv(R("results_v6", "E9_budget.csv"))
scen = [(a, b) for (a, b), _ in b9.groupby(["source", "target"], sort=False)]
fig, axs = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
for a_, (src, tgt) in zip(axs, scen):
    g = b9[(b9.source == src) & (b9.target == tgt)]
    for pol, col, lab in [("random", C["random"], "random labels"), ("agent", C["agent"], "agent")]:
        h = g[g.policy == pol].sort_values("budget")
        a_.errorbar(100 * h.budget, h.worst_class_cov, yerr=[h.worst_class_cov - h.worst_lo, h.worst_hi - h.worst_class_cov],
                    color=col, marker="o", ms=4, lw=1.6, capsize=2, label=lab)
    full = g[g.policy == "all"].iloc[0]
    a_.axhline(full.worst_class_cov, color=C["maci"], ls="--", lw=1, label="all labels")
    a_.set_xscale("log"); a_.set_xticks([2, 5, 10, 25]); a_.set_xticklabels(["2%", "5%", "10%", "25%"])
    a_.set_title(f"{src} → {tgt}", fontsize=8.5); a_.set_xlabel("label budget"); a_.grid(alpha=0.25)
axs[0].set_ylabel("worst-class coverage"); axs[0].legend(fontsize=7, frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_agent_budget.png"), dpi=250); plt.close(); print("budget")

# ---- (3) triage trade-off: expert workload vs automatic error ----
pa = pd.read_csv(R("results_v5", "E8_policies.csv")); pa = pa[pa.setting == "A"]
pb = pd.read_csv(R("results_v7", "E8_policies.csv"))
LB = [("singletons", "singletons only", "o"), ("classifier_inset", "CNN label", "s"), ("retrieval_inset", "retrieval", "^"),
      ("fusion_inset", "fusion", "D"), ("verifier", "verifier", "*")]
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
for a_, (d, title) in zip(axs, [(pa, "within a run"), (pb, "under shift (O1+O2 → O3b)")]):
    for k, lab, mk in LB:
        r = d[d.policy.str.contains(k)].iloc[0]
        a_.scatter(100 * r.human_frac, 100 * r.error, marker=mk, s=110 if k == "verifier" else 50,
                   color=C["agent"] if k == "verifier" else C["maci"] if k == "fusion_inset" else "#55606E", zorder=3)
        a_.annotate(lab, (100 * r.human_frac, 100 * r.error), xytext=(6, 3), textcoords="offset points", fontsize=7.2)
    a_.set_title(title, fontsize=8.5); a_.set_xlabel("glitches sent to an expert (%)"); a_.grid(alpha=0.25)
axs[0].set_ylabel("glitches auto-labelled wrongly (%)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_triage_tradeoff.png"), dpi=250); plt.close(); print("tradeoff")

# ---- (4) decision modules on ambiguous glitches: accuracy with 95% Wilson intervals ----
def wilson(p, n, z=1.96):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
e6 = pd.read_csv(R("results_v4_E6", "E6_summary.csv")); e6 = e6[e6.reference == "true"].set_index("method")
e5 = pd.read_csv(R("results_v3_7B", "E5_vlm_summary.csv")).set_index("reference").loc["true"]
e10 = pd.read_csv(R("results_v8", "E10_summary.csv")).set_index("reference").loc["true"]
e11 = json.load(open(R("results_v10", "E11_summary.json")))
rows = [("CNN (top label in set)", e6.loc["classifier_inset", "acc_all"], 589, "base"),
        ("DINOv2 retrieval", e6.loc["retrieval_inset", "acc_all"], 589, "tool"),
        ("CNN + retrieval fusion", e6.loc["fusion_inset", "acc_all"], 589, "best"),
        ("LLM agent over tools (7B)", e6.loc["agent_tools", "acc_all"], 300, "llm"),
        ("VLM zero-shot, no scores (7B)", e5.constrained_noscores_acc, int(e5.n), "llm"),
        ("VLM zero-shot, with scores (7B)", e5.constrained_scores_acc, int(e5.n), "llm"),
        ("VLM + retrieved references (7B)", e10.acc_rag_noscores, int(e10.n), "llm"),
        ("VLM before fine-tuning (3B)", e11["before"]["true"]["vlm"], 300, "llm"),
        ("VLM fine-tuned, QLoRA (3B)", e11["after"]["true"]["vlm"], 300, "ft")]
colmap = dict(base="#55606E", tool="#D08A1E", best="#1F8A70", llm="#C0473F", ft="#7A4FB5")
fig, ax = plt.subplots(figsize=(6.8, 3.4)); y = np.arange(len(rows))[::-1]
for yi, (lab, p, n, kind) in zip(y, rows):
    lo, hi = wilson(p, n); ax.barh(yi, p, color=colmap[kind], alpha=0.85, height=0.62)
    ax.errorbar(p, yi, xerr=[[p - lo], [hi - p]], color="black", lw=0.9, capsize=2)
    ax.text(hi + 0.01, yi, f"{p:.2f}  (n={n})", va="center", fontsize=6.8)
ax.axvline(1 / 3, color="black", ls=":", lw=0.9); ax.text(1 / 3 + 0.005, len(rows) - 0.35, "chance in a set of 3", fontsize=6.5)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.4); ax.set_xlim(0, 1)
ax.set_xlabel("accuracy on ambiguous glitches (final label), 95% Wilson interval"); ax.grid(axis="x", alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_decision_modules.png"), dpi=250); plt.close(); print("modules")

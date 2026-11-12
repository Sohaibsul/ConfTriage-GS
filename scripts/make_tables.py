"""Build LaTeX tables and figures for the paper from the Kaggle result folders.
Usage (from the 'new project' folder):  python scripts/make_tables.py
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = os.path.join(ROOT, "results", "v3_7B")      # E0-E4 + E5 (7B)
R2 = os.path.join(ROOT, "results", "v2_3B")      # E5 (3B)
R4 = os.path.join(ROOT, "results", "v4_E6")  # E6
TAB = os.path.join(ROOT, "paper", "tables"); FIG = os.path.join(ROOT, "paper", "figs")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

def nice(c): return c.replace("_", " ")
def pm(m, s, d=3): return f"{m:.{d}f}$\\pm${s:.{d}f}"
def write(name, s):
    open(os.path.join(TAB, name), "w", encoding="utf-8").write(s); print("wrote", name)

# ---------- data per run ----------
d = pd.read_csv(os.path.join(R3, "data_summary.csv")).set_index("final_label")
d = d[["O1", "O2", "O3a", "O3b", "unknown"]]
rows = [f"{nice(c)} & " + " & ".join(f"{int(v):,}" for v in d.loc[c]) + r" \\" for c in d.index if c != "TOTAL"]
write("tab_data.tex", r"""\begin{table}[t]
\caption{Glitches per class and observing run after joining the volunteer-classification release with the
machine-learning catalogue. ``Unmatched'' glitches have no run information and are used only in E0, E1, E3 and E4.}\label{tab:data}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}\toprule
Class & O1 & O2 & O3a & O3b & Unmatched\\\midrule
""" + "\n".join(rows) + r"""
\midrule
Total & """ + " & ".join(f"{int(v):,}" for v in d.loc["TOTAL"]) + r""" \\\bottomrule
\end{tabular}
\end{table}
""")

# ---------- E1 ----------
e1 = pd.read_csv(os.path.join(R3, "E1_summary.csv"))
lines = []
for regime, lab in [("full", "Full data (809,242 glitches)"), ("balanced1500", "Balanced, 1500 per class (as in Malz et al.)")]:
    lines.append(r"\multicolumn{8}{l}{\emph{" + lab + r"}}\\")
    for _, r in e1[e1.regime == regime].iterrows():
        lines.append(f"{r['score'].upper()} & {r['cond']} & {pm(r.coverage_mean, r.coverage_std)} & "
                     f"{pm(r.worst_class_cov_mean, r.worst_class_cov_std)} & {r.covgap_mean:.2f} & "
                     f"{r.set_size_mean:.2f} & {r.singleton_mean:.3f} & {r.macro_f1_mean:.3f}" + r"\\")
write("tab_e1.tex", r"""\begin{table*}[t]
\caption{E1: score functions and conditioning at $\alpha=0.1$ (mean$\pm$std over 10 random 50/50 calibration/test splits).
Worst-class coverage and CovGap (mean absolute deviation of class-wise coverage from $1-\alpha$, in percentage points) are computed over
classes with at least 30 test glitches.}\label{tab:e1}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccccc}\toprule
Score & Conditioning & Coverage & Worst-class cov. & CovGap & Set size & Singleton rate & Set F1\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------- E2 ----------
e2 = pd.read_csv(os.path.join(R3, "E2_shift.csv"))
keep = [("split_marginal", "Split CP, marginal"), ("split_mondrian", "Split CP, Mondrian"),
        ("weighted_mondrian", "Weighted CP, Mondrian"), ("aci_g0.005", "ACI ($\\gamma$=0.005)"),
        ("rolling_mondrian", "Rolling Mondrian"), ("mondrian_aci_g0.005", "\\textbf{Mondrian-ACI} ($\\gamma$=0.005)"),
        ("mondrian_aci_g0.02", "\\textbf{Mondrian-ACI} ($\\gamma$=0.02)"), ("oracle_mondrian", "Oracle Mondrian (in-run cal.)")]
lines = []
for (s, t), g in e2.groupby(["source", "target"], sort=False):
    lines.append(r"\multicolumn{5}{l}{\emph{Calibrate on " + s + r" $\rightarrow$ deploy on " + t + r"}}\\")
    g = g.set_index("method")
    for m, lab in keep:
        if m in g.index:
            r = g.loc[m]
            lines.append(f"{lab} & {r.coverage:.3f} & {r.worst_class_cov:.3f} & {r.covgap:.2f} & {r.set_size:.2f}" + r"\\")
write("tab_e2.tex", r"""\begin{table}[t]
\caption{E2: coverage when CP calibrated on earlier observing runs is deployed on a later run ($\alpha=0.1$, LAC score).
The oracle is calibrated on half of the target run and evaluated on the other half.}\label{tab:e2}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}\toprule
Method & Coverage & Worst-class & CovGap & Set size\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
# figure: worst-class coverage per scenario
meth = [("split_mondrian", "Split Mondrian"), ("aci_g0.005", "ACI"), ("weighted_mondrian", "Weighted Mondrian"),
        ("rolling_mondrian", "Rolling Mondrian"), ("mondrian_aci_g0.02", "Mondrian-ACI (ours)"), ("oracle_mondrian", "Oracle")]
scen = [(s, t) for (s, t), _ in e2.groupby(["source", "target"], sort=False)]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
x = np.arange(len(scen)); wdt = 0.13
for k, (m, lab) in enumerate(meth):
    wc = [e2[(e2.source == s) & (e2.target == t) & (e2.method == m)].worst_class_cov.values[0] for s, t in scen]
    sz = [e2[(e2.source == s) & (e2.target == t) & (e2.method == m)].set_size.values[0] for s, t in scen]
    ax[0].bar(x + (k - 2.5) * wdt, wc, wdt, label=lab); ax[1].bar(x + (k - 2.5) * wdt, sz, wdt, label=lab)
for a_, yl in zip(ax, ["worst-class coverage", "average set size"]):
    a_.set_xticks(x); a_.set_xticklabels([f"{s}→{t}" for s, t in scen], fontsize=8); a_.set_ylabel(yl)
ax[0].axhline(0.9, c="k", ls=":", lw=1)
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=6, fontsize=8, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.savefig(os.path.join(FIG, "fig_e2_worstclass.png"), dpi=250); plt.close()
print("wrote fig_e2_worstclass.png")
# figure: rolling coverage curve O1+O2 -> O3b
cv = pd.read_csv(os.path.join(R3, "E2_curves.csv"))
c = cv[(cv.source == "O1+O2") & (cv.target == "O3b")]
plt.figure(figsize=(6.5, 3.4))
for m, lab in [("split_mondrian", "Split Mondrian"), ("aci_g0.005", "ACI"), ("mondrian_aci_g0.005", "Mondrian-ACI (ours)"),
               ("rolling_mondrian", "Rolling Mondrian")]:
    cc = c[c.method == m]; plt.plot(cc.chunk, cc.coverage, label=lab)
plt.axhline(0.9, c="k", ls=":"); plt.xlabel("deployment chunk in GPS order (O3b)"); plt.ylabel("marginal coverage")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_e2_time.png"), dpi=250); plt.close()
print("wrote fig_e2_time.png")

# ---------- E3 ----------
e3 = pd.read_csv(os.path.join(R3, "E3_openset_summary.csv"))
lines = [f"{int(r.k_out)} & {nice(r.method)} & {pm(r.auroc_mean, r.auroc_std)} & {pm(r.fpr95_mean, r.fpr95_std)}" + r"\\"
         for _, r in e3.iterrows()]
write("tab_e3.tex", r"""\begin{table}[t]
\caption{E3: detection of held-out glitch classes (10 random class subsets; score-level simulation).}\label{tab:e3}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{clcc}\toprule
Held out & Novelty score & AUROC & FPR@95\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------- E4 ----------
e4 = pd.read_csv(os.path.join(R3, "E4_label_source_summary.csv"))
e4 = e4[e4.score == "lac"]
lines = [f"{nice(r.calibrated_on)} & {nice(r.evaluated_on)} & {r.stratum} & {r.coverage:.3f} & {r.worst_class_cov:.3f} & {r.set_size:.2f}" + r"\\"
         for _, r in e4.iterrows()]
write("tab_e4.tex", r"""\begin{table}[t]
\caption{E4: sensitivity of Mondrian CP (LAC, $\alpha=0.1$) to the label source, on glitches with at least three volunteer votes
(mean of 5 splits). Unanimous: all volunteers agree; contested: majority share $<0.7$.}\label{tab:e4}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllccc}\toprule
Calibrated on & Evaluated on & Stratum & Coverage & Worst-class & Set size\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------- E5 (VLM) ----------
v2 = pd.read_csv(os.path.join(R2, "E5_vlm_summary.csv")).set_index("reference").loc["true"]
v3 = pd.read_csv(os.path.join(R3, "E5_vlm_summary.csv")).set_index("reference").loc["true"]
rows = [
    ("Classifier top-1 (reference)", v2.classifier_top1_acc, v3.classifier_top1_acc),
    ("VLM, unconstrained (22 classes)", v2.unconstrained_acc, v3.unconstrained_acc),
    ("VLM in set, with classifier scores", v2.constrained_norefs_acc, v3.constrained_scores_acc),
    ("VLM in set, scores + reference images", v2.constrained_refs_acc, np.nan),
    ("VLM in set, no scores", np.nan, v3.constrained_noscores_acc),
    ("VLM in set, no scores + reference images", np.nan, v3.constrained_refs_noscores_acc),
]
f = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
lines = [f"{a} & {f(b)} & {f(c)}" + r"\\" for a, b, c in rows]
write("tab_e5.tex", r"""\begin{table}[t]
\caption{E5: accuracy of a general-purpose vision--language model (Qwen2.5-VL) on ambiguous glitches
($2\le|C(x)|\le4$), w.r.t.\ the final label. Chance level for a set of size 2--4 is about 0.33.
n = """ + f"{int(v2.n)}" + r""" (3B) and """ + f"{int(v3.n)}" + r""" (7B, 4-bit).}\label{tab:e5}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcc}\toprule
Condition & 3B & 7B\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------- E6 (tool agent), if available ----------
p6 = os.path.join(R4, "E6_summary.csv")
if os.path.exists(p6):
    e6all = pd.read_csv(p6); e6 = e6all[e6all.reference == "true"]
    vm = e6all[e6all.reference == "vol_major"].set_index("method")
    names = {"classifier_inset": "Classifier (argmax in set)", "retrieval_inset": "Retrieval tool only (in set)",
             "retrieval_unrestricted": "Retrieval tool, unrestricted", "fusion_inset": r"\textbf{Classifier + retrieval fusion (in set)}",
             "agent_tools": r"LLM agent over tools$^\dagger$", "agent_tools_noscores": r"LLM agent over tools, no scores$^\dagger$"}
    lines = [f"{names.get(r.method, nice(r.method))} & {int(r.n)} & {r.acc_all:.3f} & {r.escalation_rate:.3f} & "
             f"{r.acc_decided:.3f} & {vm.loc[r.method, 'acc_all']:.3f}" + r"\\" for _, r in e6.iterrows()]
    write("tab_e6.tex", r"""\begin{table}[t]
\caption{E6: decision modules on ambiguous glitches ($2\le|C(x)|\le4$). Accuracy w.r.t.\ the final label and w.r.t.\ the independent
volunteer majority (Vol.); escalated glitches count as not correct. $^\dagger$Run on the first 300 test glitches.}\label{tab:e6}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}\toprule
Method & n & Accuracy & Escalated & Acc.\ (decided) & Acc.\ (Vol.)\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    pol = pd.read_csv(os.path.join(R4, "E6_workload_policies.csv"))
    ARROW = r"$\rightarrow$"
    lines = [r.policy.replace("_", " ").replace("->", ARROW) + f" & {r.human_frac:.3f} & {r.error:.4f}" + r"\\"
             for _, r in pol.iterrows()]
    write("tab_e6_policy.tex", r"""\begin{table}[t]
\caption{E6: end-to-end triage policies on the deployment split (glitches with images). Human fraction = share of all glitches
sent to an expert; error = share of all glitches labelled automatically and wrongly.}\label{tab:e6pol}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{p{6.2cm}cc}\toprule
Policy & Human fraction & Error\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
for f_ in ["E0_reproduction.png"]:
    src = os.path.join(R3, "figs", f_)
    if os.path.exists(src):
        import shutil; shutil.copy(src, os.path.join(FIG, "fig_e0_reproduction.png")); print("copied", f_)

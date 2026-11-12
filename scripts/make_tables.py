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
write("tab_data.tex", r"""\begin{table}[!htbp]
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
SC_ORDER = [("lac", "LAC"), ("aps", "APS"), ("raps", "RAPS")]
CO_ORDER = [("marginal", "Marginal"), ("clustered", "Clustered"), ("mondrian", "Mondrian")]
lines = []
for regime, lab in [("full", "(a) Full data, 809\\,242 glitches"), ("balanced1500", "(b) Balanced subsample, 1500 glitches per class (setup of Malz et al.)")]:
    g = e1[e1.regime == regime].set_index(["score", "cond"])
    best_wc = g.worst_class_cov_mean.max(); best_gap = g.covgap_mean.min(); best_f1 = g.macro_f1_mean.max()
    B = lambda v, best, txt: (r"\textbf{" + txt + "}") if abs(v - best) < 1e-9 else txt
    lines.append(r"\multicolumn{8}{l}{\emph{" + lab + r"}}\\")
    for si, (sc, scl) in enumerate(SC_ORDER):
        for ci, (co, col) in enumerate(CO_ORDER):
            r = g.loc[(sc, co)]
            lines.append(f"{scl if ci == 0 else ''} & {col} & {r.coverage_mean:.3f} & "
                         f"{B(r.worst_class_cov_mean, best_wc, pm(r.worst_class_cov_mean, r.worst_class_cov_std))} & "
                         f"{B(r.covgap_mean, best_gap, f'{r.covgap_mean:.2f}')} & {r.set_size_mean:.2f} & "
                         f"{r.singleton_mean:.3f} & {B(r.macro_f1_mean, best_f1, f'{r.macro_f1_mean:.3f}')}" + r"\\")
        if si < 2: lines.append(r"\addlinespace[2pt]")
    if regime == "full": lines.append(r"\midrule")
write("tab_e1.tex", r"""\begin{table}[!htbp]
\caption{E1: score functions and conditioning strategies at $\alpha=0.1$ (mean over 10 random 50/50 calibration/test splits;
worst-class coverage as mean$\pm$std). Coverage: fraction of test glitches whose reference label lies in the set; worst-class:
lowest class-wise coverage; CovGap: mean absolute deviation of class-wise coverage from $1-\alpha$ in percentage points (both over
classes with at least 30 test glitches); set size: average number of labels; singleton: fraction of sets with one label; set F1:
class-averaged F1 of set membership. Best value per block in bold.}\label{tab:e1}
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llcccccc}\toprule
Score & Conditioning & Coverage & Worst-class & CovGap & Set size & Singleton & Set F1\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------- E2 ----------
e2 = pd.read_csv(os.path.join(R3, "E2_shift.csv"))
keep = [("split_marginal", "Split CP, marginal"), ("split_mondrian", "Split CP, Mondrian"),
        ("weighted_mondrian", "Weighted CP, Mondrian"), ("aci_g0.005", "ACI, marginal ($\\gamma=0.005$)"),
        ("rolling_mondrian", "Rolling recalibration, Mondrian"), ("mondrian_aci_g0.005", "Mondrian-ACI ($\\gamma=0.005$)"),
        ("mondrian_aci_g0.02", "\\textbf{Mondrian-ACI} ($\\gamma=0.02$)"), ("oracle_mondrian", "\\emph{Oracle Mondrian (target calibration)}")]
scen = [(s, t) for (s, t), _ in e2.groupby(["source", "target"], sort=False)]
blocks = [("coverage", "(a) Marginal coverage", "{:.3f}"), ("worst_class_cov", "(b) Worst-class coverage", "{:.3f}"),
          ("covgap", "(c) Class-coverage gap CovGap (percentage points)", "{:.2f}"), ("set_size", "(d) Average set size", "{:.2f}")]
lines = []
for bi, (col, lab, fmt) in enumerate(blocks):
    lines.append(r"\multicolumn{5}{l}{\emph{" + lab + r"}}\\")
    for m, ml in keep:
        vals = []
        for s, t in scen:
            v = e2[(e2.source == s) & (e2.target == t) & (e2.method == m)][col].values[0]
            txt = fmt.format(v)
            if m == "mondrian_aci_g0.02": txt = r"\textbf{" + txt + "}"
            vals.append(txt)
        if m == "oracle_mondrian": lines.append(r"\cmidrule(l){1-5}")
        lines.append(ml + " & " + " & ".join(vals) + r"\\")
    if bi < len(blocks) - 1: lines.append(r"\midrule")
hdr = " & ".join(f"{s}$\\rightarrow${t}" for s, t in scen)
write("tab_e2.tex", r"""\begin{table}[!htbp]
\caption{E2: conformal prediction calibrated on earlier observing runs and deployed on a later run, with glitches processed in GPS
order ($\alpha=0.1$, LAC score). Each column is one source$\rightarrow$target transition. The oracle is calibrated on a random half
of the target run and evaluated on the other half; it is not available in practice and serves as a reference. Rows for the proposed
method in bold.}\label{tab:e2}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}\toprule
Method & """ + hdr + r"""\\\midrule
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
e3 = pd.read_csv(os.path.join(R3, "E3_openset_summary.csv")).set_index(["method", "k_out"])
NOV = [("MSP", "Maximum softmax probability (MSP)"), ("entropy", "Predictive entropy"),
       ("set_size_or_empty", "Conformal set size"), ("conformal_pmax", "Conformal $p$-value, $1-\\max_y p_y(x)$")]
lines = []
for m, lab in NOV:
    lines.append(lab + " & " + " & ".join(f"{pm(e3.loc[(m, k)].auroc_mean, e3.loc[(m, k)].auroc_std)} & "
                                          f"{pm(e3.loc[(m, k)].fpr95_mean, e3.loc[(m, k)].fpr95_std)}" for k in [1, 3]) + r"\\")
write("tab_e3.tex", r"""\begin{table}[!htbp]
\caption{E3: detection of held-out glitch classes (mean$\pm$std over 10 random choices of the held-out classes; score-level
simulation in which the held-out score columns are removed and the remaining scores renormalized). AUROC: area under the ROC
curve for separating held-out from known glitches (0.5 = chance); FPR@95: false-positive rate on known glitches at 95\% recall
of held-out glitches. Coverage of the known classes stays at 0.90 and no held-out glitch receives an empty set.}\label{tab:e3}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}\toprule
 & \multicolumn{2}{c}{1 class held out} & \multicolumn{2}{c}{3 classes held out}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Novelty score & AUROC & FPR@95 & AUROC & FPR@95\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------- E4 ----------
e4 = pd.read_csv(os.path.join(R3, "E4_label_source_summary.csv"))
e4 = e4[e4.score == "lac"].set_index(["calibrated_on", "evaluated_on", "stratum"])
LB = {"final_label": "Final label", "vol_major": "Volunteer majority", "volunteer_majority": "Volunteer majority"}
lines = []
for cal_ in e4.index.get_level_values(0).unique():
    for ev_ in e4.index.get_level_values(1).unique():
        if (cal_, ev_, "all") not in e4.index: continue
        r = lambda st: e4.loc[(cal_, ev_, st)]
        lines.append(f"{LB.get(cal_, nice(cal_))} & {LB.get(ev_, nice(ev_))} & {r('all').coverage:.3f} & {r('contested').coverage:.3f} & "
                     f"{r('unanimous').coverage:.3f} & {r('all').worst_class_cov:.3f} & {r('all').set_size:.2f}" + r"\\")
write("tab_e4.tex", r"""\begin{table}[!htbp]
\caption{E4: sensitivity of Mondrian CP (LAC, $\alpha=0.1$) to the label used for calibration and for evaluation, on glitches with at
least three volunteer votes (mean of 5 random splits). Coverage is given for all glitches and for two strata: contested (majority
share below 0.7) and unanimous (all volunteers agree). Worst-class coverage and set size refer to all glitches.}\label{tab:e4}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccc}\toprule
 & & \multicolumn{3}{c}{Coverage} & & \\\cmidrule(lr){3-5}
Calibrated on & Evaluated on & All & Contested & Unanimous & Worst-class & Set size\\\midrule
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
write("tab_e5.tex", r"""\begin{table}[!htbp]
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
    e6all = pd.read_csv(p6)
    e6 = e6all[e6all.reference == "true"].set_index("method")
    vm = e6all[e6all.reference == "vol_major"].set_index("method")
    groups = [("Single sources of evidence", [("classifier_inset", "Classifier top label in the set"),
                                             ("retrieval_inset", "Retrieval tool, restricted to the set"),
                                             ("retrieval_unrestricted", "Retrieval tool, all 22 classes")]),
              ("Combinations of evidence", [("fusion_inset", r"\textbf{Classifier + retrieval fusion} (proposed)"),
                                           ("agent_tools", r"LLM agent over three tools$^\dagger$"),
                                           ("agent_tools_noscores", r"LLM agent over tools, no classifier scores$^\dagger$")])]
    bA = e6.acc_all.max(); bV = vm.acc_all.max()
    lines = []
    for gi, (glab, ms) in enumerate(groups):
        lines.append(r"\multicolumn{6}{l}{\emph{" + glab + r"}}\\")
        for m, lab in ms:
            r = e6.loc[m]; v = vm.loc[m].acc_all
            acc = f"{r.acc_all:.3f}"; acc = r"\textbf{" + acc + "}" if abs(r.acc_all - bA) < 1e-9 else acc
            vv = f"{v:.3f}"; vv = r"\textbf{" + vv + "}" if abs(v - bV) < 1e-9 else vv
            esc = "--" if r.escalation_rate == 0 else f"{r.escalation_rate:.3f}"
            dec = "--" if r.escalation_rate == 0 else f"{r.acc_decided:.3f}"
            lines.append(f"{lab} & {int(r.n)} & {acc} & {vv} & {esc} & {dec}" + r"\\")
        if gi == 0: lines.append(r"\midrule")
    write("tab_e6.tex", r"""\begin{table}[!htbp]
\caption{E6: decision modules for ambiguous glitches ($2\le|C(x)|\le4$; class-stratified sample of the deployment split).
Accuracy is measured against the final label and against the independent volunteer majority; escalated glitches count as not
correct. Escalated: fraction sent to a human by the agent; Acc.\ (decided): accuracy on the glitches the agent did not escalate.
Best accuracy in bold. $^\dagger$Run on the first 300 test glitches; on the same 300 glitches the classifier reaches 0.700 and
the fusion 0.727.}\label{tab:e6}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}\toprule
 & & \multicolumn{2}{c}{Accuracy} & & \\\cmidrule(lr){3-4}
Decision module & $n$ & Final label & Volunteers & Escalated & Acc.\ (decided)\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    pol = pd.read_csv(os.path.join(R4, "E6_workload_policies.csv"))
    PN = [("singleton auto, rest human", "Singletons automatic, every other set to a human"),
          ("classifier_inset", "+ classifier top label for ambiguous sets"),
          ("retrieval_inset", "+ retrieval (in set) for ambiguous sets"),
          ("retrieval_unrestricted", "+ retrieval (all classes) for ambiguous sets"),
          ("fusion_inset", "+ classifier--retrieval fusion for ambiguous sets"),
          ("agent_tools_noscores", "+ LLM agent over tools, no scores"),
          ("agent_tools on", "+ LLM agent over tools"),
          ("verifier", r"\textbf{+ verifier: automatic only if fusion and classifier agree}")]
    def pname(p):
        for k, v in PN:
            if k in p: return v
        return nice(p)
    lines = []
    for i, r in pol.iterrows():
        e = f"{100 * r.error:.1f}"; e = r"\textbf{" + e + "}" if "verifier" in r.policy else e
        lines.append(pname(r.policy) + f" & {100 * r.human_frac:.1f} & {e}" + r"\\")
        if i == 0: lines.append(r"\midrule")
    write("tab_e6_policy.tex", r"""\begin{table}[!htbp]
\caption{E6: end-to-end triage policies on all deployment glitches with images. Singleton sets are always labelled automatically
and sets with more than four labels always go to a human; the policies differ only in how ambiguous sets are handled. Human:
percentage of all glitches sent to an expert; error: percentage of all glitches that are labelled automatically and wrongly
(w.r.t.\ the final label).}\label{tab:e6pol}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcc}\toprule
Policy & Human (\%) & Error (\%)\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
for f_ in ["E0_reproduction.png"]:
    src = os.path.join(R3, "figs", f_)
    if os.path.exists(src):
        import shutil; shutil.copy(src, os.path.join(FIG, "fig_e0_reproduction.png")); print("copied", f_)

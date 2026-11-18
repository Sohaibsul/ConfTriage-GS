"""Build LaTeX tables and figures for the paper from the Kaggle result folders.
Usage (from the 'new project' folder):  python paper/make_tables.py
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = os.path.join(ROOT, "results_v3_7B")      # E0-E4 + E5 (7B)
R2 = os.path.join(ROOT, "results_v2_3B")      # E5 (3B)
R4 = os.path.join(ROOT, "results")            # latest run (E6)
TAB = os.path.join(ROOT, "paper", "tables"); FIG = os.path.join(ROOT, "paper", "figs")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

def nice(c): return c.replace("_", " ")
def pm(m, s, d=3): return f"{m:.{d}f}$\\pm${s:.{d}f}"
def write(name, s):
    open(os.path.join(TAB, name), "w", encoding="utf-8").write(s); print("wrote", name)
    # REVTeX: every table in one column, placed exactly where it is input ([H]); wide tables are scaled down to the column
    rt = (s.replace("\\begin{table}[!htbp]", "\\begin{table}[H]").replace("\\footnotesize", "\\scriptsize")
          .replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}")
          .replace("\\end{tabular}", "\\end{tabular}\\end{adjustbox}"))
    open(os.path.join(TAB, "rt_" + name), "w", encoding="utf-8").write(rt)   # REVTeX two-column version

# ---------- data per run ----------
d = pd.read_csv(os.path.join(R3, "data_summary.csv")).set_index("final_label")
d = d[["O1", "O2", "O3a", "O3b", "unknown"]]
rows = [f"{nice(c)} & " + " & ".join(f"{int(v):,}" for v in d.loc[c]) + r" \\" for c in d.index if c != "TOTAL"]
write("tab_data.tex", r"""\begin{table}[!htbp]
\caption{Glitches per class and observing run after joining the volunteer-classification release with the
machine-learning catalogue. ``Unmatched'' glitches have no run information and are used only in the experiments that do not need it.}\label{tab:data}
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
\caption{Score functions and conditioning strategies at $\alpha=0.1$ (mean over 10 random 50/50 calibration/test splits;
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

# ---------- E2 (revision: v5 = E2 with delay, gamma sweep and CIs; v6 = delay-aware Mondrian-ACI) ----------
R5 = os.path.join(ROOT, "results_v5"); R6 = os.path.join(ROOT, "results_v6"); R7 = os.path.join(ROOT, "results_v7")
e2 = pd.concat([pd.read_csv(os.path.join(R5, "E2_shift.csv")), pd.read_csv(os.path.join(R6, "E2b_delay_aware.csv"))],
               ignore_index=True)
scen = [(s, t) for (s, t), _ in e2.groupby(["source", "target"], sort=False)]
def E(m, s, t): return e2[(e2.source == s) & (e2.target == t) & (e2.method == m)].iloc[0]
keep = [("split_marginal", "Split CP, marginal", ""), ("split_mondrian", "Split CP, Mondrian", ""),
        ("weighted_mondrian", "Weighted CP, Mondrian", ""), ("aci_g0.005", "ACI, $\\gamma=0.005$", ""),
        ("rolling_mondrian", "Rolling Mondrian$^{\\ast}$", ""),
        ("mondrian_aci_g0.02", "MACI, $\\gamma=0.02$, no delay", "b"),
        ("mondrian_aci_g0.02_d2000", "MACI, $\\gamma=0.02^{\\ast}$", ""),
        ("da_mondrian_aci_g0.02_d2000", "DA-MACI, $\\gamma=0.02^{\\ast}$", ""),
        ("da_mondrian_aci_g0.05_d2000", "DA-MACI, $\\gamma=0.05^{\\ast}$", "b"),
        ("oracle_mondrian", "\\emph{Oracle Mondrian}", "")]
blocks = [("coverage", "(a) Marginal coverage", "{:.3f}"), ("worst_class_cov", "(b) Worst-class coverage (95\\% bootstrap interval)", "{:.3f}"),
          ("covgap", "(c) Class-coverage gap CovGap (percentage points)", "{:.2f}"), ("set_size", "(d) Average set size", "{:.2f}")]
lines = []
for bi, (col, lab, fmt) in enumerate(blocks):
    lines.append(r"\multicolumn{5}{l}{\emph{" + lab + r"}}\\")
    for m, ml, st in keep:
        vals = []
        for s, t in scen:
            r = E(m, s, t); txt = fmt.format(r[col])
            if st == "b": txt = r"\textbf{" + txt + "}"
            if col == "worst_class_cov": txt += r"\,{\tiny(" + f"{r.worst_lo:.2f}--{r.worst_hi:.2f}" + ")}"
            vals.append(txt)
        if m == "oracle_mondrian": lines.append(r"\cmidrule(l){1-5}")
        lines.append((r"\textbf{" + ml + "}" if st == "b" else ml) + " & " + " & ".join(vals) + r"\\")
    if bi < len(blocks) - 1: lines.append(r"\midrule")
hdr = " & ".join(f"{s}$\\rightarrow${t}" for s, t in scen)
write("tab_e2.tex", r"""\begin{table}[!htbp]
\caption{Conformal prediction calibrated on earlier observing runs and deployed on a later run, with glitches processed in GPS
order ($\alpha=0.1$, LAC score). Each column is one source$\rightarrow$target transition. $^{\ast}$Labels are revealed with a delay of
2000 glitches. MACI: Mondrian-ACI; DA: delay-aware step size (Sec.~\ref{sec:shiftmethods}). Worst-class intervals are 95\% bootstrap intervals of the
minimum class-wise coverage (classes with at least 30 target glitches). The oracle is calibrated on a random half of the target run and
evaluated on the other half; it is not available in practice. The step sizes $\gamma\in\{0.005,0.02\}$ were fixed before the
experiments; $\gamma=0.05$ for the delay-aware variant was selected after inspecting the sweep of Appendix~\ref{app:gamma} and is
therefore optimistic.}\label{tab:e2}
\centering\scriptsize\setlength{\tabcolsep}{1.8pt}
\begin{tabular}{lcccc}\toprule
Method & """ + hdr + r"""\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
# appendix table: informativeness of the main methods
INF = [("split_mondrian", "Split CP, Mondrian"), ("rolling_mondrian", "Rolling recalibration, Mondrian"),
       ("mondrian_aci_g0.02", "Mondrian-ACI ($\\gamma=0.02$), no delay"), ("da_mondrian_aci_g0.05_d2000", "DA Mondrian-ACI ($\\gamma=0.05$), delay"),
       ("oracle_mondrian", "Oracle Mondrian")]
lines = []
for col, lab in [("singleton", "(a) Singleton rate"), ("macro_f1", "(b) Set F1")]:
    lines.append(r"\multicolumn{5}{l}{\emph{" + lab + r"}}\\")
    for m, ml in INF: lines.append(ml + " & " + " & ".join(f"{E(m, s, t)[col]:.3f}" for s, t in scen) + r"\\")
    if col == "singleton": lines.append(r"\midrule")
write("tab_e2_info.tex", r"""\begin{table}[!htbp]
\caption{Informativeness of the sets in the run-shift experiments: singleton rate and set F1~\cite{malz2025} for the main methods.}\label{tab:e2info}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}\toprule
Method & """ + hdr + r"""\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
# figure: worst-class coverage per scenario with bootstrap intervals, and set size
meth = [("split_mondrian", "Split Mondrian"), ("aci_g0.005", "ACI"), ("weighted_mondrian", "Weighted Mondrian"),
        ("rolling_mondrian", "Rolling Mondrian (delay)"), ("mondrian_aci_g0.02", "Mondrian-ACI (no delay)"),
        ("da_mondrian_aci_g0.05_d2000", "DA Mondrian-ACI (delay)"), ("oracle_mondrian", "Oracle")]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.7)); x = np.arange(len(scen)); wdt = 0.12
for k, (m, lab) in enumerate(meth):
    rr = [E(m, s, t) for s, t in scen]; off = (k - (len(meth) - 1) / 2) * wdt
    wc = np.array([r.worst_class_cov for r in rr]); lo = np.array([r.worst_lo for r in rr]); hi = np.array([r.worst_hi for r in rr])
    ax[0].bar(x + off, wc, wdt, label=lab, yerr=[np.clip(wc - lo, 0, None), np.clip(hi - wc, 0, None)], capsize=1.5,
              error_kw=dict(lw=0.6))
    ax[1].bar(x + off, [r.set_size for r in rr], wdt, label=lab)
for a_, yl in zip(ax, ["worst-class coverage", "average set size"]):
    a_.set_xticks(x); a_.set_xticklabels([f"{s}→{t}" for s, t in scen], fontsize=8); a_.set_ylabel(yl)
ax[0].axhline(0.9, c="k", ls=":", lw=1)
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=4, fontsize=8, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.86)); plt.savefig(os.path.join(FIG, "fig_e2_worstclass.png"), dpi=250); plt.close()
print("wrote fig_e2_worstclass.png")
# figure: silent failure over time, O1+O2 -> O3b
rec = pd.read_csv(os.path.join(R5, "E2_records_O1O2_O3b.csv.gz"))
rda = pd.read_csv(os.path.join(R6, "E2b_records_O1O2_O3b.csv.gz"))
assert (rec.y.values == rda.y.values).all()
rec["cov_da"] = rda["cov_da_g0.05"].values
days = (rec.gps - rec.gps.iloc[0]) / 86400
fig, ax = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
for col, lab, c_ in [("cov_split_mondrian", "Split CP, Mondrian", "tab:red"), ("cov_aci_g0.005", "ACI, marginal", "tab:orange"),
                     ("cov_da", "DA Mondrian-ACI (delay)", "tab:blue")]:
    ax[0].plot(days, rec[col].rolling(5000, min_periods=1000).mean(), color=c_, lw=1.4, label=lab)
ax[0].axhline(0.9, c="k", ls=":", lw=1); ax[0].set_title("(a) marginal coverage, rolling 5000 glitches", fontsize=9)
ax[0].set_xlabel("days since start of O3b"); ax[0].set_ylabel("coverage"); ax[0].legend(fontsize=8, frameon=False, loc="lower left")
cls_cov = rec.groupby("y").agg(n=("y", "size"), c=("cov_split_mondrian", "mean"))
worst = cls_cov[cls_cov.n >= 150].sort_values("c").index[:4]
wk = (days // 7).astype(int)
for j, cname in enumerate(worst):
    sub = rec[rec.y == cname].assign(wk=wk[rec.y == cname]); col_ = plt.cm.tab10(j)
    g_ = sub.groupby("wk").agg(n=("y", "size"), s=("cov_split_mondrian", "mean"), d=("cov_da", "mean")); g_ = g_[g_.n >= 5]
    ax[1].plot(g_.index * 7 + 3.5, g_.s, color=col_, ls="--", lw=1.1, marker=".", ms=3)
    ax[1].plot(g_.index * 7 + 3.5, g_.d, color=col_, lw=1.8, marker="o", ms=3, label=cname.replace("_", " ") + f" (n={len(sub)})")
ax[1].axhline(0.9, c="k", ls=":", lw=1)
ax[1].set_title("(b) weekly coverage of the four worst-covered classes", fontsize=9); ax[1].set_xlabel("days since start of O3b")
ax[1].legend(fontsize=7, frameon=True, loc="center right", title="dashed: split CP; solid: DA Mondrian-ACI", title_fontsize=7)
for a_ in ax: a_.grid(alpha=0.25); a_.set_ylim(0, 1.02)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_e2_time.png"), dpi=250); plt.close(); print("wrote fig_e2_time.png", list(worst))
# appendix figure: gamma sweep, worst-class coverage vs set size
fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for a_, (s, t) in zip(axs, scen):
    for pat, lab, mk in [("mondrian_aci_g{}", "no delay", "o"), ("mondrian_aci_g{}_d2000", "delay, plain step", "s"),
                         ("da_mondrian_aci_g{}_d2000", "delay, DA step", "^")]:
        gs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        pts = [(g, E(pat.format(g), s, t)) for g in gs if ((e2.method == pat.format(g)) & (e2.source == s) & (e2.target == t)).any()]
        a_.plot([p.set_size for _, p in pts], [p.worst_class_cov for _, p in pts], marker=mk, ms=4, lw=1, label=lab)
        for g, p in pts: a_.annotate(f"{g:g}", (p.set_size, p.worst_class_cov), fontsize=6, xytext=(2, 2), textcoords="offset points")
    for m, mk, lab in [("rolling_mondrian", "x", "rolling (delay)"), ("oracle_mondrian", "*", "oracle")]:
        p = E(m, s, t); a_.plot(p.set_size, p.worst_class_cov, mk, color="k", ms=7, label=lab)
    a_.axhline(0.9, c="k", ls=":", lw=1); a_.set_title(f"{s}→{t}", fontsize=9); a_.set_xlabel("average set size"); a_.grid(alpha=0.25)
axs[0].set_ylabel("worst-class coverage"); axs[0].legend(fontsize=7, frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_gamma_sweep.png"), dpi=250); plt.close(); print("wrote fig_gamma_sweep.png")

# ---------- E9: calibration agent ----------
b9 = pd.read_csv(os.path.join(R6, "E9_budget.csv")); a9 = pd.read_csv(os.path.join(R6, "E9_alerts.csv"))
lines = []
for bud in [0.02, 0.05, 0.1, 0.25, 1.0]:
    for pol_ in (["random", "agent"] if bud < 1 else ["all"]):
        cells = []
        for s, t in scen:
            r = b9[(b9.source == s) & (b9.target == t) & (b9.budget == bud) & (b9.policy == pol_)].iloc[0]
            o = b9[(b9.source == s) & (b9.target == t) & (b9.budget == bud) & (b9.policy == ("random" if pol_ == "agent" else "agent"))]
            txt = f"{r.worst_class_cov:.2f} / {r.set_size:.1f}"
            if pol_ == "agent" and len(o) and r.worst_class_cov > o.iloc[0].worst_class_cov + 1e-9: txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        lab = {"random": "random", "agent": "agent", "all": "all labels"}[pol_]
        lines.append((f"{int(bud * 100)}\\%" if pol_ != "agent" else "") + f" & {lab} & " + " & ".join(cells) + r"\\")
    if bud < 1: lines.append(r"\addlinespace[2pt]")
write("tab_e9.tex", r"""\begin{table}[!htbp]
\caption{Calibration agent. Worst-class coverage / average set size of delay-aware Mondrian-ACI ($\gamma=0.02$, delay 2000)
when only a fraction of the target glitches (the label budget) is labelled, with the glitches to label chosen at random or by the
agent. Bold: the agent is better than random labelling at the same budget.}\label{tab:e9}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccc}\toprule
Budget & Labelling & """ + hdr + r"""\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
al = []
for (s, t), g in a9.groupby(["source", "target"], sort=False):
    u = g.undercovered.astype(bool); al_ = g.alerted.astype(bool)
    al.append(f"{s}$\\rightarrow${t} & {int(u.sum())} & {int((u & al_).sum())} & {int((~u & al_).sum())} & "
              f"{g[u & al_].alert_days.median():.0f} & {g.run_days.iloc[0]:.0f}" + r"\\")
write("tab_e9_alerts.tex", r"""\begin{table}[!htbp]
\caption{Coverage alerts of the calibration agent on top of static split Mondrian CP (the deployment that fails silently), with 5\%
of the glitches labelled at random and a label delay of 2000 glitches. An alert is raised for a class when the miss rate among its last
50 labelled glitches exceeds $\alpha+0.1$; a class is under-covered if its coverage over the whole target run is below 0.8.}\label{tab:e9alerts}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}\toprule
Transition & Under-covered & Detected & False alerts & Median delay (days) & Run length (days)\\\midrule
""" + "\n".join(al) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------- E3 ----------
e3 = pd.read_csv(os.path.join(R3, "E3_openset_summary.csv")).set_index(["method", "k_out"])
NOV = [("MSP", "Maximum softmax probability (MSP)"), ("entropy", "Predictive entropy"),
       ("set_size_or_empty", "Conformal set size"), ("conformal_pmax", "Conformal $p$-value, $1-\\max_y p_y(x)$")]
lines = []
for m, lab in NOV:
    lines.append(lab + " & " + " & ".join(f"{pm(e3.loc[(m, k)].auroc_mean, e3.loc[(m, k)].auroc_std)} & "
                                          f"{pm(e3.loc[(m, k)].fpr95_mean, e3.loc[(m, k)].fpr95_std)}" for k in [1, 3]) + r"\\")
write("tab_e3.tex", r"""\begin{table}[!htbp]
\caption{Detection of held-out glitch classes (mean$\pm$std over 10 random choices of the held-out classes; score-level
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
\caption{Sensitivity of Mondrian CP (LAC, $\alpha=0.1$) to the label used for calibration and for evaluation, on glitches with at
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
\caption{Accuracy of a general-purpose vision--language model (Qwen2.5-VL) on ambiguous glitches
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
\caption{Decision modules for ambiguous glitches ($2\le|C(x)|\le4$; class-stratified sample of the deployment split).
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
          ("classifier_inset", "+ classifier top label"),
          ("retrieval_inset", "+ retrieval in the set"),
          ("retrieval_unrestricted", "+ retrieval (all classes) for ambiguous sets"),
          ("fusion_inset", "+ classifier--retrieval fusion"),
          ("agent_tools_noscores", "+ LLM agent over tools, no scores"),
          ("agent_tools on", "+ LLM agent over tools"),
          ("verifier", r"\textbf{+ verifier (fusion agrees with classifier)}")]
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
\caption{End-to-end triage policies on all deployment glitches with images. Singleton sets are always labelled automatically
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
# ---------- E8: end-to-end triage, natural samples (A: in-run, from v5; B: under shift, from v7) ----------
pA_ = pd.read_csv(os.path.join(R5, "E8_policies.csv")); pA_ = pA_[pA_.setting == "A"]
pB_ = pd.read_csv(os.path.join(R7, "E8_policies.csv")) if os.path.exists(os.path.join(R7, "E8_policies.csv")) else None
PN8 = [("singletons auto", "Singletons automatic, rest to a human"),
       ("classifier_inset", "+ classifier top label"),
       ("retrieval_inset", "+ retrieval in the set"),
       ("fusion_inset", "+ classifier--retrieval fusion"),
       ("verifier", r"\textbf{+ verifier (fusion agrees with classifier)}")]
lines = []
for i, (k_, lab) in enumerate(PN8):
    ra = pA_[pA_.policy.str.contains(k_)].iloc[0]
    rb = pB_[pB_.policy.str.contains(k_)].iloc[0] if pB_ is not None else None
    f = lambda r: (f"{100 * r.human_frac:.1f} & {100 * r.error:.2f}") if r is not None else "-- & --"
    lines.append(f"{lab} & {f(ra)} & {f(rb)}" + r"\\")
    if i == 0: lines.append(r"\midrule")
write("tab_e8.tex", r"""\begin{table}[!htbp]
\caption{End-to-end triage policies. Human: percentage of all deployment glitches (with images) sent to an expert; error:
percentage of all glitches labelled automatically and wrongly (w.r.t.\ the final label). Error rates of the decision modules are
estimated on a random sample of 800 ambiguous glitches that follows the natural class distribution. In-run: Mondrian split CP on the
50/50 deployment split of Secs.~\ref{sec:e5} and~\ref{sec:e6}. Under shift: delay-aware Mondrian-ACI ($\gamma=0.02$, delay 2000) calibrated on O1+O2 and deployed on
O3b in GPS order, with a retrieval memory built from O1+O2 only.}\label{tab:e8}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}\toprule
 & \multicolumn{2}{c}{In-run} & \multicolumn{2}{c}{Under shift (O1+O2$\rightarrow$O3b)}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Policy & Human (\%) & Error (\%) & Human (\%) & Error (\%)\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------- E0 figure: class-wise coverage as grey lines, min-max band, marginal coverage in bold ----------
e0 = pd.read_csv(os.path.join(R3, "E0_malz_reproduction.csv"))
lvl = 1 - e0["alpha"].values
cc = e0[[c for c in e0.columns if c.startswith("cov_")]].values
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax[0].fill_between(lvl, np.nanmin(cc, 1), np.nanmax(cc, 1), color="0.85", lw=0, label="range over 22 classes")
for j in range(cc.shape[1]): ax[0].plot(lvl, cc[:, j], color="0.55", lw=0.6, alpha=0.7)
ax[0].plot(lvl, e0["coverage"], color="k", lw=2.2, label="marginal coverage")
ax[0].plot([0, 1], [0, 1], ls=":", color="tab:red", lw=1.2, label="nominal $1-\\alpha$")
ax[0].set_xlabel(r"nominal coverage $1-\alpha$"); ax[0].set_ylabel("empirical coverage"); ax[0].legend(fontsize=8, frameon=False)
ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1.02)
ax[1].plot(lvl, e0["set_size"], color="k", lw=2.2); ax[1].set_xlabel(r"nominal coverage $1-\alpha$"); ax[1].set_ylabel("average set size")
ax[1].set_xlim(0, 1)
for a_ in ax: a_.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_e0_reproduction.png"), dpi=250); plt.close(); print("wrote fig_e0_reproduction.png")

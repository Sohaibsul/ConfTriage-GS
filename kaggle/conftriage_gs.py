# %% [markdown]
# # ConfTriage-GS
# **Shift-aware, open-set conformal triage of Gravity Spy glitches with a set-constrained vision–language agent**
#
# Extends Malz, Ashton & Colombo, *Phys. Rev. D* **111**, 084078 (2025).
#
# | Exp | Question |
# |---|---|
# | E0 | Reproduce Malz 2025 (Mondrian CP, baseline score, 1500/class) |
# | E1 | Score functions (LAC/APS/RAPS) × conditioning (marginal/Mondrian/clustered), full imbalanced data |
# | E2 | Coverage under observing-run shift (O1+O2 → O3a/O3b): split vs weighted CP vs ACI vs rolling recalibration |
# | E3 | Conformal novelty detection, leave-k-classes-out |
# | E4 | Label-source sensitivity: final label vs independent volunteer majority |
# | E5 | Agentic triage: set-constrained Qwen2.5-VL adjudication + workload analysis |
#
# Data: Zenodo 5911227 (retired labels + volunteer votes) joined with Zenodo 5649212 (GPS time, detector, metadata, image URLs).

# %%
import os, sys, json, time, math, re, io, random, subprocess, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

CFG = dict(
    ALPHA=0.1,
    N_SEEDS=10,                 # random cal/test splits for E0/E1/E3
    RUN_E0=True, RUN_E1=True, RUN_E2=True, RUN_E3=True, RUN_E4=True, RUN_E5=False, RUN_E6=True,
    E6_BANK_PER_CLASS=120,      # labelled memory-bank glitches per class (retrieval tool)
    E6_N_TEST=600,              # ambiguous deployment glitches for E6
    E6_N_LLM=300,               # of which sent to the LLM agent
    VLM_MODEL="Qwen/Qwen2.5-VL-7B-Instruct",   # loaded in 4-bit; falls back to 3B fp16 if that fails
    VLM_FALLBACK="Qwen/Qwen2.5-VL-3B-Instruct",
    N_VLM=300,                  # ambiguous glitches sent to the VLM per condition
    N_REF=2,                    # reference examples per candidate class
    VLM_MAX_PIXELS=448 * 448,
    E3_REPEATS=10,
    SYNTH=os.environ.get("CT_SYNTH") == "1",   # local smoke test only (no downloads)
)
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("out_local")
FIG = os.path.join(OUT, "figs"); os.makedirs(FIG, exist_ok=True)
T0 = time.time()
def log(*a):
    print(f"[{(time.time()-T0)/60:6.1f} min]", *a, flush=True)
def save_csv(df, name):
    df.to_csv(os.path.join(OUT, name), index=False); log("saved", name, df.shape)
print(json.dumps(CFG, indent=1))

# %% [markdown]
# ## 1. Data

# %%
CLASSES = ["1080Lines", "1400Ripples", "Air_Compressor", "Blip", "Chirp", "Extremely_Loud", "Helix", "Koi_Fish",
           "Light_Modulation", "Low_Frequency_Burst", "Low_Frequency_Lines", "No_Glitch", "None_of_the_Above",
           "Paired_Doves", "Power_Line", "Repeating_Blips", "Scattered_Light", "Scratchy", "Tomte", "Violin_Mode",
           "Wandering_Line", "Whistle"]
K = len(CLASSES); CIDX = {c: i for i, c in enumerate(CLASSES)}
RUN_FILES = ["H1_O1", "L1_O1", "H1_O2", "L1_O2", "H1_O3a", "L1_O3a", "H1_O3b", "L1_O3b"]
META_COLS = ["event_time", "ifo", "duration", "peak_frequency", "central_freq", "bandwidth", "amplitude", "snr",
             "q_value", "gravityspy_id", "url1", "url2", "url3", "url4"]

DATA = "/tmp/gsdata"; os.makedirs(DATA, exist_ok=True)
def wget(url, dst):
    dst = os.path.join(DATA, dst)
    if not os.path.exists(dst):
        subprocess.run(["wget", "-q", "--tries=5", "-O", dst, url], check=True)

def load_real():
    wget("https://zenodo.org/api/records/5911227/files/retired_fulldata_min2_max50_ret0p9.hdf5/content", "retired.hdf5")
    ret = pd.read_hdf(f"{DATA}/retired.hdf5", key="image_db")
    log("retired", ret.shape)
    # volunteer vote fractions from `tracks` (tracks[0] = ML scores, tracks[1:] = one-hot volunteer votes)
    tr = ret["tracks"].values
    ml0 = np.array([np.asarray(t[0], float) for t in tr])
    agree = (ml0.argmax(1) == ret["ml_label"].map(CIDX).values).mean()
    log(f"tracks[0] argmax == ml_label for {agree:.4f} of rows (checks class order of tracks)")
    votes = np.zeros((len(ret), K), np.float32)
    for i, t in enumerate(tr):
        if len(t) > 1:
            votes[i] = np.asarray(t[1:], np.float32).sum(0)
    ret = ret.drop(columns=["tracks"])
    metas = []
    for f in RUN_FILES:
        wget(f"https://zenodo.org/api/records/5649212/files/{f}.csv/content", f"{f}.csv")
        m = pd.read_csv(f"{DATA}/{f}.csv", usecols=META_COLS)
        m["run"] = f.split("_")[1]
        metas.append(m); log("meta", f, m.shape)
    meta = pd.concat(metas, ignore_index=True).drop_duplicates("gravityspy_id")
    df = ret.merge(meta, on="gravityspy_id", how="left")
    log("joined", df.shape, "with run info:", df["run"].notna().mean().round(4))
    return df, votes

def load_synth(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    pri = np.array([327, 93, 67, 56, 54, 34, 34, 24, 21, 19, 18, 18, 11, 10, 6, 4, 3, 2, 2, 2, 2, 1], float)
    order = ["Scattered_Light", "Low_Frequency_Burst", "Blip", "Low_Frequency_Lines", "1080Lines", "Koi_Fish", "Tomte",
             "Violin_Mode", "No_Glitch", "1400Ripples", "Extremely_Loud", "Whistle", "Power_Line", "Repeating_Blips",
             "Scratchy", "Chirp", "None_of_the_Above", "Air_Compressor", "Helix", "Paired_Doves", "Light_Modulation",
             "Wandering_Line"]
    p = np.zeros(K); p[[CIDX[c] for c in order]] = pri; p /= p.sum()
    y = rng.choice(K, n, p=p)
    run = rng.choice(["O1", "O2", "O3a", "O3b"], n, p=[.1, .2, .35, .35])
    shift = np.where(np.isin(run, ["O3a", "O3b"]), 1.0, 0.0)
    logits = rng.normal(0, 1, (n, K)); logits[np.arange(n), y] += 4.0 - 1.5 * shift * rng.random(n)
    P = np.exp(logits); P /= P.sum(1, keepdims=True)
    df = pd.DataFrame(P, columns=CLASSES)
    df["final_label"] = [CLASSES[i] for i in y]; df["ml_label"] = [CLASSES[i] for i in P.argmax(1)]
    df["ml_confidence"] = P.max(1); df["final_score"] = 0.9; df["Nclassifications"] = rng.integers(2, 20, n)
    df["gravityspy_id"] = [f"g{i}" for i in range(n)]
    df["run"] = run; df["ifo"] = rng.choice(["H1", "L1"], n)
    base = {"O1": 1.127e9, "O2": 1.17e9, "O3a": 1.24e9, "O3b": 1.26e9}
    df["event_time"] = [base[r] + rng.random() * 1e7 for r in run]
    for c in ["duration", "peak_frequency", "central_freq", "bandwidth", "amplitude", "snr", "q_value"]:
        df[c] = rng.lognormal(0, 1, n) * (1 + shift)
    for u in ["url1", "url2", "url3", "url4"]: df[u] = None
    votes = np.zeros((n, K), np.float32)
    for i in range(n):
        nv = rng.integers(2, 8); lab = np.where(rng.random(nv) < 0.8, y[i], rng.integers(0, K, nv))
        np.add.at(votes[i], lab, 1)
    return df, votes

df, VOTES = load_synth() if CFG["SYNTH"] else load_real()
_P = df[CLASSES].values.astype(np.float64)
_ok = np.isfinite(_P).all(1) & (_P.sum(1) > 0) & df["final_label"].isin(CLASSES).values
log(f"dropping {(~_ok).sum()} rows with missing/invalid classifier scores ({(~_ok).mean():.4f})")
log("rows with run info among dropped:", df.loc[~_ok, "run"].notna().mean() if (~_ok).any() else 0)
df = df.loc[_ok].reset_index(drop=True); VOTES = VOTES[_ok]
log("ml_label == final_label:", (df["ml_label"] == df["final_label"]).mean().round(4))
P_ALL = df[CLASSES].values.astype(np.float64)
P_ALL = P_ALL / P_ALL.sum(1, keepdims=True)
Y_ALL = df["final_label"].map(CIDX).values
assert not np.isnan(Y_ALL.astype(float)).any()
nv = VOTES.sum(1)
Y_VOL = np.where(nv > 0, VOTES.argmax(1), -1)                     # volunteer-majority label (independent of ML)
VOL_AGREE = np.where(nv > 0, VOTES.max(1) / np.maximum(nv, 1), np.nan)
log("N =", len(df), " K =", K, " top-1 acc vs final_label:", (P_ALL.argmax(1) == Y_ALL).mean().round(4),
    " vs volunteer majority:", (P_ALL.argmax(1)[Y_VOL >= 0] == Y_VOL[Y_VOL >= 0]).mean().round(4))

summ = (df.assign(run=df["run"].fillna("unknown")).groupby(["run", "final_label"]).size().unstack(0).fillna(0).astype(int))
summ.loc["TOTAL"] = summ.sum()
summ.reset_index().to_csv(os.path.join(OUT, "data_summary.csv"), index=False)
print(summ.to_string())

# %% [markdown]
# ## 2. Conformal prediction library

# %%
def score_matrix(P, kind="lac", u=None, lam=0.01, kreg=2):
    """Nonconformity score s(x, y) for every class. Returns (n, K)."""
    if kind == "lac":
        return 1.0 - P
    n, k = P.shape
    order = np.argsort(-P, axis=1)
    Ps = np.take_along_axis(P, order, 1)
    cum = np.cumsum(Ps, 1)
    if u is None: u = np.random.default_rng(0).random(n)
    S_sorted = cum - u[:, None] * Ps                              # randomized APS
    if kind == "raps":
        rank = np.arange(1, k + 1)[None, :]
        S_sorted = S_sorted + lam * np.maximum(rank - kreg, 0)
    S = np.empty_like(S_sorted)
    np.put_along_axis(S, order, S_sorted, 1)
    return S

def conf_quantile(s, alpha):
    n = len(s)
    if n == 0: return np.inf
    lvl = math.ceil((n + 1) * (1 - alpha)) / n
    if lvl > 1: return np.inf
    return np.quantile(s, lvl, method="higher")

def calibrate(S_cal, y_cal, alpha, cond="marginal", rng=None, n_min=50, M=5):
    """Returns a threshold vector q (K,) so that C(x) = {y : S[x, y] <= q[y]}."""
    true = S_cal[np.arange(len(y_cal)), y_cal]
    if cond == "marginal":
        return np.full(S_cal.shape[1], conf_quantile(true, alpha))
    if cond == "mondrian":
        return np.array([conf_quantile(true[y_cal == c], alpha) for c in range(S_cal.shape[1])])
    if cond == "clustered":                                         # Ding et al. (NeurIPS 2023)
        from sklearn.cluster import KMeans
        rng = rng or np.random.default_rng(0)
        idx = rng.permutation(len(y_cal)); half = int(0.3 * len(idx))
        A, B = idx[:half], idx[half:]
        qmarg = conf_quantile(true[B], alpha)
        levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1 - alpha]
        emb, good = [], []
        for c in range(S_cal.shape[1]):
            sc = true[A][y_cal[A] == c]
            if len(sc) >= n_min:
                emb.append(np.quantile(sc, levels)); good.append(c)
        q = np.full(S_cal.shape[1], qmarg)
        if len(good) >= 2:
            km = KMeans(n_clusters=min(M, len(good)), n_init=10, random_state=0).fit(np.array(emb))
            for cl in np.unique(km.labels_):
                members = [good[i] for i in np.where(km.labels_ == cl)[0]]
                mask = np.isin(y_cal[B], members)
                qc = conf_quantile(true[B][mask], alpha)
                for c in members: q[c] = qc
        return q
    raise ValueError(cond)

def predict_sets(S, q):
    return S <= q[None, :] if np.ndim(q) == 1 else S <= q

def metrics(sets, y, alpha, min_class=30):
    n, k = sets.shape
    cov = sets[np.arange(n), y]
    size = sets.sum(1)
    out = dict(coverage=cov.mean(), set_size=size.mean(), singleton=(size == 1).mean(), empty=(size == 0).mean(),
               singleton_acc=(cov[size == 1].mean() if (size == 1).any() else np.nan))
    ccov = []
    for c in range(k):
        m = y == c
        if m.sum() >= min_class: ccov.append(cov[m].mean())
    ccov = np.array(ccov)
    out.update(worst_class_cov=ccov.min(), covgap=np.mean(np.abs(ccov - (1 - alpha))) * 100,
               frac_classes_undercovered=np.mean(ccov < 1 - alpha - 0.05))
    f1s = []                                                        # Malz-style set F1 per class
    for c in range(k):
        tp = (sets[:, c] & (y == c)).sum(); fp = (sets[:, c] & (y != c)).sum(); fn = ((~sets[:, c]) & (y == c)).sum()
        if tp + fn > 0: f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    out["macro_f1"] = np.mean(f1s)
    return out

def per_class_cov(sets, y):
    cov = sets[np.arange(len(y)), y]
    return {CLASSES[c]: cov[y == c].mean() for c in range(K) if (y == c).sum() > 0}

# %% [markdown]
# ## E0. Reproduction of Malz et al. (2025)

# %%
if CFG["RUN_E0"]:
    sub = np.concatenate([np.where(Y_ALL == c)[0][:1500] for c in range(K)])   # first 1500 per class, as in their code
    from sklearn.model_selection import train_test_split
    cal, te = train_test_split(sub, test_size=0.5, random_state=123)
    S = score_matrix(P_ALL, "lac")
    alphas = np.linspace(0.01, 0.99, 20)
    rows = []
    for a in alphas:
        q = calibrate(S[cal], Y_ALL[cal], a, "mondrian")
        sets = predict_sets(S[te], q)
        m = metrics(sets, Y_ALL[te], a); m["alpha"] = a
        m.update({f"cov_{k}": v for k, v in per_class_cov(sets, Y_ALL[te]).items()})
        rows.append(m)
    e0 = pd.DataFrame(rows); save_csv(e0, "E0_malz_reproduction.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c in CLASSES: ax[0].plot(1 - alphas, e0[f"cov_{c}"], lw=0.8)
    ax[0].plot(1 - alphas, e0["coverage"], "k--", lw=2, label="marginal"); ax[0].plot([0, 1], [0, 1], ":", c="grey")
    ax[0].set_xlabel(r"$1-\alpha$"); ax[0].set_ylabel("coverage"); ax[0].legend()
    ax[1].plot(1 - alphas, e0["set_size"], "k--"); ax[1].set_xlabel(r"$1-\alpha$"); ax[1].set_ylabel("average set size")
    plt.tight_layout(); plt.savefig(f"{FIG}/E0_reproduction.png", dpi=200); plt.close()
    log("E0 at alpha~0.1:", e0.iloc[np.argmin(np.abs(alphas - 0.1))][["coverage", "set_size", "singleton", "macro_f1"]].to_dict())

# %% [markdown]
# ## E1. Score functions × conditioning (full imbalanced data and balanced subset)

# %%
def balanced_subset(rng, per=1500):
    return np.concatenate([rng.permutation(np.where(Y_ALL == c)[0])[:per] for c in range(K)])

if CFG["RUN_E1"]:
    rows = []
    for regime in ["balanced1500", "full"]:
        for seed in range(CFG["N_SEEDS"]):
            rng = np.random.default_rng(seed)
            idx = balanced_subset(rng) if regime == "balanced1500" else rng.permutation(len(Y_ALL))
            idx = rng.permutation(idx); h = len(idx) // 2
            cal, te = idx[:h], idx[h:]
            u = rng.random(len(Y_ALL))
            for kind in ["lac", "aps", "raps"]:
                S = score_matrix(P_ALL, kind, u=u)
                for cond in ["marginal", "mondrian", "clustered"]:
                    q = calibrate(S[cal], Y_ALL[cal], CFG["ALPHA"], cond, rng=rng)
                    m = metrics(predict_sets(S[te], q), Y_ALL[te], CFG["ALPHA"])
                    m.update(regime=regime, seed=seed, score=kind, cond=cond); rows.append(m)
        log("E1 done", regime)
    e1 = pd.DataFrame(rows); save_csv(e1, "E1_all_runs.csv")
    cols = ["coverage", "worst_class_cov", "covgap", "set_size", "singleton", "singleton_acc", "macro_f1"]
    agg = e1.groupby(["regime", "score", "cond"])[cols].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]; agg = agg.reset_index()
    save_csv(agg, "E1_summary.csv"); print(agg.to_string())

# %% [markdown]
# ## E2. Coverage under observing-run shift

# %%
def weighted_thresholds(cal_scores, cal_w, test_w, alpha):
    """Tibshirani et al. (2019) weighted split-CP quantile for each test point (marginal)."""
    o = np.argsort(cal_scores); s = cal_scores[o]; w = cal_w[o]
    cw = np.cumsum(w); W = cw[-1]
    target = (1 - alpha) * (W + test_w)
    pos = np.searchsorted(cw, target, side="left")
    q = np.where(pos < len(s), s[np.minimum(pos, len(s) - 1)], np.inf)
    return q

def domain_weights(Xc, Xt, rng, n_fit=60000):
    from sklearn.ensemble import HistGradientBoostingClassifier
    ic = rng.choice(len(Xc), min(n_fit, len(Xc)), replace=False)
    it = rng.choice(len(Xt), min(n_fit, len(Xt)), replace=False)
    X = np.vstack([Xc[ic], Xt[it]]); d = np.r_[np.zeros(len(ic)), np.ones(len(it))]
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X, d)
    prior = len(ic) / len(it)
    def w(X_):
        p = np.clip(clf.predict_proba(X_)[:, 1], 1e-4, 1 - 1e-4)
        return np.clip(p / (1 - p) * prior, 0.05, 20.0)
    return w(Xc), w(Xt), clf

def features(idx):
    num = df.loc[idx, ["duration", "peak_frequency", "central_freq", "bandwidth", "amplitude", "snr", "q_value"]].values.astype(float)
    num = np.log(np.abs(num) + 1e-30)
    ifo = (df.loc[idx, "ifo"].values == "H1").astype(float)[:, None]
    return np.nan_to_num(np.hstack([np.log(P_ALL[idx] + 1e-12), num, ifo]))

if CFG["RUN_E2"]:
    run = df["run"].values; t = df["event_time"].values
    have = pd.notna(run) & pd.notna(t)
    a = CFG["ALPHA"]; rng = np.random.default_rng(0)
    S = score_matrix(P_ALL, "lac")
    scenarios = [("O1+O2", ["O1", "O2"], "O3a"), ("O1+O2", ["O1", "O2"], "O3b"), ("O3a", ["O3a"], "O3b"),
                 ("O1", ["O1"], "O2")]
    rows, curves = [], {}
    for src_name, src_runs, tgt in scenarios:
        src = np.where(have & np.isin(run, src_runs))[0]
        tgt_idx = np.where(have & (run == tgt))[0]
        if len(src) < 1000 or len(tgt_idx) < 1000:
            log("skip", src_name, tgt); continue
        tgt_idx = tgt_idx[np.argsort(t[tgt_idx])]                    # deployment order = GPS time
        ys, yt = Y_ALL[src], Y_ALL[tgt_idx]
        Ss, St = S[src], S[tgt_idx]
        res = {}
        # (1) split CP calibrated on the source run(s)
        for cond in ["marginal", "mondrian"]:
            q = calibrate(Ss, ys, a, cond); res[f"split_{cond}"] = predict_sets(St, q)
        # (2) oracle: calibrated on a random half of the target run, evaluated on the other half (reference only)
        perm = rng.permutation(len(tgt_idx)); oc, ot = perm[: len(perm) // 2], perm[len(perm) // 2:]
        oracle = {}
        for cond in ["marginal", "mondrian"]:
            q = calibrate(St[oc], yt[oc], a, cond); oracle[f"oracle_{cond}"] = (predict_sets(St[ot], q), yt[ot])
        # (3) weighted CP with a domain classifier (labels of target never used)
        wc, wt, _ = domain_weights(features(src), features(tgt_idx), rng)
        true_s = Ss[np.arange(len(ys)), ys]
        qw = weighted_thresholds(true_s, wc, wt, a)
        res["weighted_marginal"] = St <= qw[:, None]
        Qm = np.empty_like(St)
        for c in range(K):
            m = ys == c
            Qm[:, c] = weighted_thresholds(true_s[m], wc[m], wt, a) if m.sum() > 0 else np.inf
        res["weighted_mondrian"] = St <= Qm
        # (4) ACI (Gibbs & Candes 2021), labels revealed after each glitch, marginal LAC
        for gamma in [0.001, 0.005, 0.02]:
            sorted_true = np.sort(true_s); n = len(sorted_true)
            at = a; sets = np.zeros_like(St, bool)
            for i in range(len(yt)):
                if at <= 0: q = np.inf
                elif at >= 1: q = -np.inf
                else:
                    lvl = math.ceil((n + 1) * (1 - at)) / n
                    q = np.inf if lvl > 1 else sorted_true[min(int(math.ceil(lvl * n)) - 1, n - 1)]
                sets[i] = St[i] <= q
                err = 0.0 if sets[i, yt[i]] else 1.0
                at = at + gamma * (a - err)
            res[f"aci_g{gamma}"] = sets
        # (5) rolling recalibration on recently labelled target glitches (window W, label delay D)
        Wn, D, B = 5000, 2000, 1000
        sets = np.zeros_like(St, bool); true_t = St[np.arange(len(yt)), yt]
        for b0 in range(0, len(yt), B):
            hi = max(0, b0 - D); lo = max(0, hi - Wn)
            calsc = true_t[lo:hi] if hi - lo >= 500 else true_s
            sets[b0:b0 + B] = St[b0:b0 + B] <= conf_quantile(calsc, a)
        res["rolling_marginal"] = sets
        # (6) NEW: class-conditional (Mondrian) ACI — one alpha per class, updated only when that class is revealed
        nmax = max(int((ys == c).sum()) for c in range(K))
        pad = np.full((K, max(nmax, 1)), np.inf); ny = np.zeros(K, int)
        for c in range(K):
            v = np.sort(true_s[ys == c]); ny[c] = len(v); pad[c, :len(v)] = v
        for gamma in [0.005, 0.02]:
            ay = np.full(K, a); sets = np.zeros_like(St, bool)
            for i in range(len(yt)):
                lvl_idx = np.ceil((ny + 1) * (1 - ay)).astype(int)            # 1-based rank of the quantile
                q = np.where((ay <= 0) | (lvl_idx > ny), np.inf,
                             pad[np.arange(K), np.clip(lvl_idx - 1, 0, pad.shape[1] - 1)])
                q = np.where(ay >= 1, -np.inf, q)
                sets[i] = St[i] <= q
                c = yt[i]; err = 0.0 if sets[i, c] else 1.0
                ay[c] += gamma * (a - err)
            res[f"mondrian_aci_g{gamma}"] = sets
        # (7) NEW: class-conditional rolling recalibration (per-class window of recent labelled target glitches)
        Wc, minc = 1000, 50
        sets = np.zeros_like(St, bool)
        q_src = np.array([conf_quantile(true_s[ys == c], a) for c in range(K)])
        for b0 in range(0, len(yt), B):
            hi = max(0, b0 - D); q = q_src.copy()
            if hi > 0:
                yy, ss = yt[:hi], true_t[:hi]
                for c in range(K):
                    sc_c = ss[yy == c][-Wc:]
                    if len(sc_c) >= minc: q[c] = conf_quantile(sc_c, a)
            sets[b0:b0 + B] = St[b0:b0 + B] <= q[None, :]
        res["rolling_mondrian"] = sets
        for name, sets in res.items():
            m = metrics(sets, yt, a); m.update(source=src_name, target=tgt, method=name, n_cal=len(src), n_test=len(yt))
            rows.append(m)
        for name, (sets, yy) in oracle.items():
            m = metrics(sets, yy, a); m.update(source=src_name, target=tgt, method=name, n_cal=len(oc), n_test=len(yy))
            rows.append(m)
        # rolling coverage curves in deployment order
        chunk = max(len(yt) // 60, 500)
        curves[(src_name, tgt)] = {name: [sets[i:i + chunk][np.arange(len(sets[i:i + chunk])), yt[i:i + chunk]].mean()
                                          for i in range(0, len(yt) - chunk + 1, chunk)]
                                   for name, sets in res.items()}
        log("E2 done", src_name, "->", tgt)
    e2 = pd.DataFrame(rows); save_csv(e2, "E2_shift.csv")
    print(e2[["source", "target", "method", "coverage", "worst_class_cov", "covgap", "set_size", "singleton"]].round(4).to_string())
    for (s_, t_), cv in curves.items():
        plt.figure(figsize=(8, 4))
        for name in ["split_marginal", "split_mondrian", "aci_g0.005", "mondrian_aci_g0.005", "rolling_mondrian"]:
            if name in cv: plt.plot(cv[name], label=name)
        plt.axhline(1 - a, c="k", ls=":"); plt.xlabel("deployment chunk (GPS order)"); plt.ylabel("coverage")
        plt.title(f"calibrate {s_} → deploy {t_}"); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(f"{FIG}/E2_coverage_{s_.replace('+','')}_{t_}.png", dpi=200); plt.close()
    pd.DataFrame([dict(source=s_, target=t_, method=n_, chunk=i, coverage=v) for (s_, t_), cv in curves.items()
                  for n_, vals in cv.items() for i, v in enumerate(vals)]).to_csv(f"{OUT}/E2_curves.csv", index=False)

# %% [markdown]
# ## E3. Detecting unseen glitch classes (leave-k-classes-out)
# Score-level simulation: held-out class columns are removed and the remaining scores renormalized.
# The archived classifier saw all classes, so this is an approximation (stated in the paper).

# %%
def auroc_fpr95(score_known, score_novel):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(score_known)), np.ones(len(score_novel))]; s = np.r_[score_known, score_novel]
    au = roc_auc_score(y, s)
    thr = np.quantile(score_novel, 0.05)                           # 95% of novel flagged
    return au, (score_known >= thr).mean()

if CFG["RUN_E3"]:
    a = CFG["ALPHA"]
    candidates = [c for c in range(K) if (Y_ALL == c).sum() >= 1500 and CLASSES[c] not in ("None_of_the_Above", "No_Glitch")]
    rows = []
    for k_out in [1, 3]:
        for r in range(CFG["E3_REPEATS"]):
            rng = np.random.default_rng(100 + r)
            held = rng.choice(candidates, k_out, replace=False); known = np.array([c for c in range(K) if c not in held])
            remap = -np.ones(K, int); remap[known] = np.arange(len(known))
            idx = np.concatenate([rng.permutation(np.where(Y_ALL == c)[0])[:5000] for c in range(K)])
            kn = idx[np.isin(Y_ALL[idx], known)]; nov = idx[np.isin(Y_ALL[idx], held)]
            kn = rng.permutation(kn); cal, te = kn[: len(kn) // 2], kn[len(kn) // 2:]
            Pk = P_ALL[:, known]; Pk = Pk / Pk.sum(1, keepdims=True)
            yc, yt = remap[Y_ALL[cal]], remap[Y_ALL[te]]
            S = score_matrix(Pk, "lac")
            true_c = S[cal][np.arange(len(cal)), yc]
            def pmax(ix):                                           # Mondrian conformal p-values, max over classes
                pv = np.zeros((len(ix), len(known)))
                for j in range(len(known)):
                    sc = np.sort(true_c[yc == j]); n = len(sc)
                    ge = n - np.searchsorted(sc, S[ix, j], side="left")
                    pv[:, j] = (1 + ge) / (n + 1)
                return pv.max(1)
            sc = {
                "MSP": (1 - Pk[te].max(1), 1 - Pk[nov].max(1)),
                "entropy": (-(Pk[te] * np.log(Pk[te] + 1e-12)).sum(1), -(Pk[nov] * np.log(Pk[nov] + 1e-12)).sum(1)),
                "conformal_pmax": (1 - pmax(te), 1 - pmax(nov)),
            }
            q = calibrate(S[cal], yc, a, "mondrian")
            sz_t, sz_n = predict_sets(S[te], q).sum(1), predict_sets(S[nov], q).sum(1)
            sc["set_size_or_empty"] = (np.where(sz_t == 0, 99, sz_t), np.where(sz_n == 0, 99, sz_n))
            for name, (sk, sn) in sc.items():
                au, fpr = auroc_fpr95(sk, sn)
                rows.append(dict(k_out=k_out, repeat=r, held=";".join(CLASSES[h] for h in held), method=name,
                                 auroc=au, fpr95=fpr, known_cov=predict_sets(S[te], q)[np.arange(len(te)), yt].mean(),
                                 novel_empty_rate=(sz_n == 0).mean(), novel_avg_set=sz_n.mean()))
        log("E3 done k =", k_out)
    e3 = pd.DataFrame(rows); save_csv(e3, "E3_openset_all.csv")
    e3s = e3.groupby(["k_out", "method"])[["auroc", "fpr95", "known_cov", "novel_empty_rate"]].agg(["mean", "std"]).round(4)
    e3s.columns = ["_".join(c) for c in e3s.columns]; save_csv(e3s.reset_index(), "E3_openset_summary.csv"); print(e3s)

# %% [markdown]
# ## E4. Label-source sensitivity (final label vs independent volunteer majority)

# %%
if CFG["RUN_E4"]:
    a = CFG["ALPHA"]; rows = []
    okv = np.where((Y_VOL >= 0) & (nv >= 3))[0]
    for seed in range(5):
        rng = np.random.default_rng(seed); idx = rng.permutation(okv); h = len(idx) // 2
        cal, te = idx[:h], idx[h:]
        for kind in ["lac", "aps"]:
            S = score_matrix(P_ALL, kind, u=rng.random(len(Y_ALL)))
            for cal_lab_name, YL in [("final_label", Y_ALL), ("volunteer_majority", Y_VOL)]:
                q = calibrate(S[cal], YL[cal], a, "mondrian")
                sets = predict_sets(S[te], q)
                for eval_name, YE in [("final_label", Y_ALL), ("volunteer_majority", Y_VOL)]:
                    for stratum, msk in [("all", np.ones(len(te), bool)), ("unanimous", VOL_AGREE[te] == 1.0),
                                         ("contested", VOL_AGREE[te] < 0.7)]:
                        if msk.sum() < 100: continue
                        m = metrics(sets[msk], YE[te][msk], a, min_class=20)
                        m.update(seed=seed, score=kind, calibrated_on=cal_lab_name, evaluated_on=eval_name, stratum=stratum,
                                 n=int(msk.sum())); rows.append(m)
    e4 = pd.DataFrame(rows); save_csv(e4, "E4_label_source_all.csv")
    e4s = e4.groupby(["score", "calibrated_on", "evaluated_on", "stratum"])[["coverage", "worst_class_cov", "set_size", "n"]].mean().round(4)
    save_csv(e4s.reset_index(), "E4_label_source_summary.csv"); print(e4s)

# %% [markdown]
# ## E5. ConfTriage: set-gated vision–language agent
# Policy: |C|=1 → auto-label; 2≤|C|≤4 → VLM chooses **inside C** (or escalates); otherwise → human / new-class queue.

# %%
DESC = {
    "1080Lines": "a narrow horizontal line near 1080 Hz, often repeating in short segments",
    "1400Ripples": "short ripples or a smudge of power centred near 1400 Hz, lasting tenths of a second",
    "Air_Compressor": "a narrow line near 50 Hz from an air compressor, lasting about a second",
    "Blip": "a short (~0.1 s), symmetric teardrop-shaped burst spanning roughly 30-500 Hz",
    "Chirp": "a curve rising in frequency over time, like a compact-binary chirp",
    "Extremely_Loud": "a very high-energy event that saturates a large time-frequency area",
    "Helix": "a periodic helical / braided pattern at low-mid frequencies",
    "Koi_Fish": "like a large blip with a broad head and a thin tail, spanning a wide frequency range",
    "Light_Modulation": "repeating narrow spikes/stripes at low frequency caused by light modulation",
    "Low_Frequency_Burst": "a hump of excess power below ~50 Hz shaped like a fin or bump, lasting about a second",
    "Low_Frequency_Lines": "horizontal lines at low frequency (< 50 Hz) persisting across the window",
    "No_Glitch": "no visible glitch, only background noise",
    "None_of_the_Above": "a glitch that does not match any known class",
    "Paired_Doves": "repeating pairs of chirp-like curves that move up and down in frequency",
    "Power_Line": "a narrow horizontal line at 60 Hz (or harmonics), short in time",
    "Repeating_Blips": "several blips repeating at regular intervals (~0.25-0.5 s apart)",
    "Scattered_Light": "one or more stacked arches (archways) below ~100 Hz from scattered light",
    "Scratchy": "a band of wiggly, scratchy power at mid frequencies (~100-400 Hz)",
    "Tomte": "a short, triangular burst at low-mid frequency, narrower and lower than a blip",
    "Violin_Mode": "a short line near 500 Hz (violin-mode resonance of the suspension)",
    "Wandering_Line": "a thin line whose frequency drifts slowly over time",
    "Whistle": "a V- or W-shaped trace at high frequency (hundreds of Hz to kHz)",
}

def fetch_img(url, cache="/tmp/imgcache"):
    import requests
    from PIL import Image
    os.makedirs(cache, exist_ok=True)
    fn = os.path.join(cache, re.sub(r"[^A-Za-z0-9]", "", url)[-40:] + ".png")
    if not os.path.exists(fn):
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30); r.raise_for_status(); open(fn, "wb").write(r.content); break
            except Exception:
                time.sleep(2)
        else:
            return None
    try:
        return Image.open(fn).convert("RGB")
    except Exception:
        return None

def grid4(i, side=448):
    from PIL import Image
    ims = [fetch_img(df.at[i, u]) for u in ["url1", "url2", "url3", "url4"] if isinstance(df.at[i, u], str)]
    ims = [im for im in ims if im is not None]
    if len(ims) < 4: return None
    h = side // 2; canvas = Image.new("RGB", (side, side), "white")
    for k_, im in enumerate(ims[:4]):
        canvas.paste(im.resize((h, h)), ((k_ % 2) * h, (k_ // 2) * h))
    return canvas

if CFG["RUN_E5"] and not CFG["SYNTH"] and df["url1"].notna().any():
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLM
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.49", "accelerate"], check=True)
        from transformers import Qwen2_5_VLForConditionalGeneration as VLM
    a = CFG["ALPHA"]; rng = np.random.default_rng(7)
    # deployment split = E1 seed-0 full split, LAC + Mondrian
    idx = np.random.default_rng(0).permutation(len(Y_ALL)); h = len(idx) // 2; cal, te = idx[:h], idx[h:]
    te = te[df.loc[te, "url1"].notna().values]
    S = score_matrix(P_ALL, "lac"); q = calibrate(S[cal], Y_ALL[cal], a, "mondrian")
    sets_te = predict_sets(S[te], q); sz = sets_te.sum(1)
    strata = dict(singleton=(sz == 1).mean(), ambiguous=((sz >= 2) & (sz <= 4)).mean(), escalate=((sz == 0) | (sz > 4)).mean())
    strata.update(singleton_err=1 - sets_te[sz == 1][np.arange((sz == 1).sum()), Y_ALL[te][sz == 1]].mean(),
                  top1_err_all=1 - (P_ALL[te].argmax(1) == Y_ALL[te]).mean())
    log("strata", strata)
    amb = np.where((sz >= 2) & (sz <= 4))[0]
    # stratify the VLM sample by true class so rare classes are represented
    pick = []
    by = pd.Series(amb).groupby(Y_ALL[te][amb])
    per = max(1, CFG["N_VLM"] // max(by.ngroups, 1))
    for _, g in by: pick += list(rng.permutation(g.values)[:per])
    pick = rng.permutation(pick)[: CFG["N_VLM"]]
    # reference examples: confident, unanimous, singleton calibration glitches
    refs = {}
    Sc_sets = predict_sets(S[cal], q).sum(1)
    for c in range(K):
        cand = cal[(Y_ALL[cal] == c) & (Sc_sets == 1) & (np.nan_to_num(VOL_AGREE[cal]) == 1.0) & df.loc[cal, "url1"].notna().values]
        if len(cand) == 0: cand = cal[(Y_ALL[cal] == c) & df.loc[cal, "url1"].notna().values]
        got = []
        for i in rng.permutation(cand)[:10]:
            g = grid4(i, 336)
            if g is not None: got.append(g)
            if len(got) == CFG["N_REF"]: break
        refs[c] = got
    log("references ready", {CLASSES[c]: len(v) for c, v in refs.items()})

    dtype = torch.float16
    VLM_USED = CFG["VLM_MODEL"]
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"], check=True)
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = VLM.from_pretrained(VLM_USED, quantization_config=bnb, device_map="auto")
    except Exception as e:
        log("4-bit load failed:", repr(e)[:300], "-> fallback", CFG["VLM_FALLBACK"])
        VLM_USED = CFG["VLM_FALLBACK"]
        model = VLM.from_pretrained(VLM_USED, torch_dtype=dtype, device_map="auto")
    log("VLM loaded:", VLM_USED, "GPUs:", torch.cuda.device_count())
    proc = AutoProcessor.from_pretrained(VLM_USED, min_pixels=128 * 28 * 28, max_pixels=CFG["VLM_MAX_PIXELS"])

    def ask(images, text, max_new_tokens=120):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        msgs = [{"role": "user", "content": content}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=[prompt], images=images or None, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def parse(ans, allowed):
        lab, conf = None, "low"
        m = re.search(r'"label"\s*:\s*"([^"]+)"', ans)
        cand = m.group(1) if m else ans
        norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        for c in sorted(allowed, key=len, reverse=True):
            if norm(c) and norm(c) in norm(cand): lab = c; break
        mc = re.search(r'"confidence"\s*:\s*"(high|low)"', ans, re.I)
        if mc: conf = mc.group(1).lower()
        return lab, conf

    HEAD = ("You are an expert in LIGO detector characterization. The first image shows one glitch as four "
            "Omega-scan spectrograms (time on x, frequency on y, colour = normalized energy), arranged 2x2 for time "
            "windows 0.5 s, 1 s, 2 s and 4 s. ")
    def prompt_constrained(cands, with_refs, with_scores=True):
        s = HEAD
        if with_refs:
            s += "The following images are reference examples, in this order: " + ", ".join(
                f"{len(refs[c])} of {CLASSES[c]}" for c in cands) + ". "
        if with_scores:
            s += "A calibrated classifier says the true class is one of: " + "; ".join(
                f"{CLASSES[c]} ({DESC[CLASSES[c]]}, classifier score {p:.2f})" for c, p in cands_scores) + ". "
        else:
            s += "A calibrated classifier says the true class is one of (in no particular order): " + "; ".join(
                f"{CLASSES[c]} ({DESC[CLASSES[c]]})" for c in rng.permutation(cands)) + ". "
        s += ('Choose exactly one of these classes. Answer only with JSON: {"label": "<class>", "confidence": "high" or '
              '"low", "reason": "<one sentence>"}')
        return s
    def prompt_free():
        s = HEAD + "Possible classes: " + "; ".join(f"{c} ({DESC[c]})" for c in CLASSES) + ". "
        s += ('Choose exactly one class. Answer only with JSON: {"label": "<class>", "confidence": "high" or "low", '
              '"reason": "<one sentence>"}')
        return s

    # sanity check of fp16 output; fall back to fp32 if decoding is garbage
    test_i = te[pick[0]]; g = grid4(test_i)
    probe = ask([g], prompt_free())
    log("VLM probe answer:", probe[:200])
    CONDS = ["constrained_noscores", "constrained_scores", "constrained_refs_noscores", "unconstrained"]

    recs = []
    for n_, j in enumerate(pick):
        i = te[j]; g = grid4(i)
        if g is None: continue
        cands = list(np.where(sets_te[j])[0]); cands_scores = [(c, P_ALL[i, c]) for c in cands]
        y = Y_ALL[i]; rec = dict(gravityspy_id=df.at[i, "gravityspy_id"], true=CLASSES[y],
                                 vol_major=CLASSES[Y_VOL[i]] if Y_VOL[i] >= 0 else None,
                                 vol_agree=VOL_AGREE[i], set="|".join(CLASSES[c] for c in cands),
                                 set_size=len(cands), true_in_set=bool(sets_te[j, y]),
                                 classifier_top1=CLASSES[P_ALL[i].argmax()])
        try:
            ref_imgs = [im for c in cands for im in refs[c]]
            allowed_c = [CLASSES[c] for c in cands]
            for cond, imgs, txt, allowed in [
                ("constrained_noscores", [g], prompt_constrained(cands, False, False), allowed_c),
                ("constrained_scores", [g], prompt_constrained(cands, False, True), allowed_c),
                ("constrained_refs_noscores", [g] + ref_imgs, prompt_constrained(cands, True, False), allowed_c),
                ("unconstrained", [g], prompt_free(), CLASSES),
            ]:
                ans = ask(imgs, txt); lab, conf = parse(ans, allowed)
                rec[f"{cond}_label"] = lab; rec[f"{cond}_conf"] = conf; rec[f"{cond}_raw"] = ans[:300]
        except Exception as e:
            log("VLM error", repr(e)[:200]); torch.cuda.empty_cache(); continue
        recs.append(rec)
        if n_ % 25 == 0:
            log(f"E5 {n_}/{len(pick)}"); pd.DataFrame(recs).to_csv(f"{OUT}/E5_vlm_records.csv", index=False)
    e5 = pd.DataFrame(recs); save_csv(e5, "E5_vlm_records.csv")

    summ_rows = []
    for ref_name in ["true", "vol_major"]:
        sub = e5[e5[ref_name].notna()]
        r = dict(reference=ref_name, n=len(sub), classifier_top1_acc=(sub["classifier_top1"] == sub[ref_name]).mean(),
                 true_in_set=(sub["true_in_set"]).mean())
        r["vlm"] = VLM_USED
        for cond in CONDS:
            r[f"{cond}_acc"] = (sub[f"{cond}_label"] == sub[ref_name]).mean()
            r[f"{cond}_agree_cls"] = (sub[f"{cond}_label"] == sub["classifier_top1"]).mean()
            r[f"{cond}_invalid"] = sub[f"{cond}_label"].isna().mean()
            hi = sub[f"{cond}_conf"] == "high"
            r[f"{cond}_high_frac"] = hi.mean()
            r[f"{cond}_acc_high"] = (sub.loc[hi, f"{cond}_label"] == sub.loc[hi, ref_name]).mean() if hi.any() else np.nan
        summ_rows.append(r)
    e5s = pd.DataFrame(summ_rows); save_csv(e5s, "E5_vlm_summary.csv"); print(e5s.T)

    # end-to-end workload / error trade-off (strata proportions from the full deployment split)
    sub = e5
    acc_cls = (sub["classifier_top1"] == sub["true"]).mean()
    pS, pA, pE, eS = strata["singleton"], strata["ambiguous"], strata["escalate"], strata["singleton_err"]
    pol = [
        dict(policy="classifier top-1 only", human_frac=0.0, error=strata["top1_err_all"]),
        dict(policy="CP: singleton auto, rest human", human_frac=pA + pE, error=pS * eS),
        dict(policy="CP + classifier top-1 on ambiguous", human_frac=pE, error=pS * eS + pA * (1 - acc_cls)),
    ]
    for cond in [c for c in CONDS if c != "unconstrained"]:
        lab = sub[f"{cond}_label"]; ok = lab == sub["true"]
        hi = sub[f"{cond}_conf"] == "high"
        agree = lab == sub["classifier_top1"]
        pol += [
            dict(policy=f"CP + VLM decides ({cond})", human_frac=pE, error=pS * eS + pA * (1 - ok.mean())),
            dict(policy=f"CP + VLM decides, high conf only ({cond})", human_frac=pE + pA * (1 - hi.mean()),
                 error=pS * eS + pA * (hi & ~ok).mean()),
            dict(policy=f"CP + VLM verifier: agree->auto, disagree->human ({cond})", human_frac=pE + pA * (1 - agree.mean()),
                 error=pS * eS + pA * (agree & (sub["classifier_top1"] != sub["true"])).mean()),
        ]
    pol = pd.DataFrame(pol)
    save_csv(pol, "E5_workload_policies.csv"); print(pol)
    json.dump({k: float(v) for k, v in strata.items()}, open(f"{OUT}/E5_strata.json", "w"), indent=1)
else:
    log("E5 skipped (SYNTH mode, disabled, or no image URLs)")

# %% [markdown]
# ## E6. Tool-using triage agent inside the conformal set
# Tools: (1) classifier scores, (2) **image retrieval** — DINOv2 nearest labelled glitches from a memory bank,
# (3) **physics metadata** — peak frequency / duration / SNR / bandwidth vs. per-class typical ranges.
# An LLM (Qwen2.5-VL-7B, text-only) reads the tool outputs and decides inside the conformal set or escalates.

# %%
def fetch_path(url, cache="/tmp/imgcache"):
    import requests
    os.makedirs(cache, exist_ok=True)
    fn = os.path.join(cache, re.sub(r"[^A-Za-z0-9]", "", url)[-40:] + ".png")
    if os.path.exists(fn): return fn
    for _ in range(3):
        try:
            r = requests.get(url, timeout=30); r.raise_for_status(); open(fn, "wb").write(r.content); return fn
        except Exception:
            time.sleep(1)
    return None

def load_llm():
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLM
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.49", "accelerate"], check=True)
        from transformers import Qwen2_5_VLForConditionalGeneration as VLM
    name = CFG["VLM_MODEL"]
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"], check=True)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        m = VLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
    except Exception as e:
        log("4-bit load failed:", repr(e)[:200]); name = CFG["VLM_FALLBACK"]
        m = VLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
    p = AutoProcessor.from_pretrained(name)
    return m, p, name

if CFG["RUN_E6"] and not CFG["SYNTH"] and df["url1"].notna().any():
    import torch, concurrent.futures as cf
    from PIL import Image
    a = CFG["ALPHA"]; rng = np.random.default_rng(11)
    VIEWS = ["url1", "url2", "url3", "url4"]
    idx = np.random.default_rng(0).permutation(len(Y_ALL)); h = len(idx) // 2; cal, te = idx[:h], idx[h:]
    S = score_matrix(P_ALL, "lac"); q = calibrate(S[cal], Y_ALL[cal], a, "mondrian")
    has_url = df["url1"].notna().values
    te_u, cal_u = te[has_url[te]], cal[has_url[cal]]
    sets_te = predict_sets(S[te_u], q); sz = sets_te.sum(1)
    strata6 = dict(singleton=(sz == 1).mean(), ambiguous=((sz >= 2) & (sz <= 4)).mean(),
                   escalate=((sz == 0) | (sz > 4)).mean())
    s1 = sz == 1
    strata6["singleton_err"] = 1 - sets_te[s1][np.arange(s1.sum()), Y_ALL[te_u][s1]].mean()
    log("E6 strata (glitches with images)", strata6)

    amb = np.where((sz >= 2) & (sz <= 4))[0]
    by = pd.Series(amb).groupby(Y_ALL[te_u][amb]); per = max(1, CFG["E6_N_TEST"] // by.ngroups)
    pick = []
    for _, g in by: pick += list(rng.permutation(g.values)[:per])
    pick = np.array(rng.permutation(pick)[: CFG["E6_N_TEST"]])
    test_ids = te_u[pick]; test_sets = sets_te[pick]

    sets_cal = predict_sets(S[cal_u], q); szc = sets_cal.sum(1)
    ambc = np.where((szc >= 2) & (szc <= 4))[0]
    vpick = rng.permutation(ambc)[:300]
    val_ids, val_sets = cal_u[vpick], sets_cal[vpick]

    vset = set(val_ids.tolist()); bank = []
    for c in range(K):
        cand = [i for i in rng.permutation(cal_u[Y_ALL[cal_u] == c])[: CFG["E6_BANK_PER_CLASS"] * 2] if i not in vset]
        bank += cand[: CFG["E6_BANK_PER_CLASS"]]
    bank = np.array(bank)

    need = np.unique(np.r_[bank, val_ids, test_ids])
    urls = [df.at[i, u] for i in need for u in VIEWS]
    log("E6 downloading", len(urls), "images")
    with cf.ThreadPoolExecutor(32) as ex: got = list(ex.map(fetch_path, urls))
    log("E6 downloaded", sum(g is not None for g in got), "/", len(urls))

    # ---- tool 2: DINOv2 image embeddings (4 views, each L2-normalised, concatenated) ----
    from transformers import AutoImageProcessor, AutoModel
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dmod = AutoModel.from_pretrained("facebook/dinov2-base", torch_dtype=torch.float16).to("cuda").eval()
    def embed(ids):
        out = np.zeros((len(ids), 4 * 768), np.float32); ok = np.zeros(len(ids), bool)
        for b in range(0, len(ids), 16):
            ims, rows = [], []
            for r, i in enumerate(ids[b:b + 16]):
                paths = [fetch_path(df.at[i, u]) for u in VIEWS]
                if all(p is not None for p in paths):
                    try:
                        ims += [Image.open(p).convert("RGB") for p in paths]; rows.append(b + r)
                    except Exception:
                        pass
            if not rows: continue
            px = dproc(images=ims, return_tensors="pt")["pixel_values"].to("cuda").half()
            with torch.no_grad():
                f = dmod(pixel_values=px).pooler_output.float().cpu().numpy()
            f = f / np.linalg.norm(f, axis=1, keepdims=True)
            out[rows] = f.reshape(len(rows), 4 * 768) / 2.0; ok[rows] = True
        return out, ok
    Eb, okb = embed(bank); bank, Eb = bank[okb], Eb[okb]; yb = Y_ALL[bank]
    Ev, okv = embed(val_ids); val_ids, val_sets, Ev = val_ids[okv], val_sets[okv], Ev[okv]
    Et, okt = embed(test_ids); test_ids, test_sets, Et = test_ids[okt], test_sets[okt], Et[okt]
    log("E6 embedded: bank", len(bank), "val", len(val_ids), "test", len(test_ids))
    del dmod; torch.cuda.empty_cache()

    def knn(E, restrict=None, k=10, tau=0.05):
        sims = E @ Eb.T
        V = np.zeros((len(E), K)); neigh = []
        for r in range(len(E)):
            s = sims[r].copy()
            if restrict is not None: s[~restrict[r][yb]] = -np.inf
            top = np.argsort(-s)[:k]; top = top[np.isfinite(s[top])]
            np.add.at(V[r], yb[top], np.exp(s[top] / tau))
            neigh.append([(CLASSES[yb[t]], float(s[t])) for t in top])
        V = V / np.maximum(V.sum(1, keepdims=True), 1e-300)
        return V, neigh

    def in_set_argmax(M, sets):
        M = np.where(sets, M, -np.inf); return M.argmax(1)

    # fusion weight chosen on the validation glitches (calibration side, never on test)
    Vv, _ = knn(Ev, val_sets); Pv = P_ALL[val_ids]; yv = Y_ALL[val_ids]
    grid = np.linspace(0, 1, 11); vacc = []
    for w in grid:
        F = w * np.log(Pv + 1e-12) + (1 - w) * np.log(Vv + 1e-6)
        vacc.append((in_set_argmax(F, val_sets) == yv).mean())
    w_star = float(grid[int(np.argmax(vacc))])
    log("E6 fusion weight on validation:", dict(zip(np.round(grid, 1), np.round(vacc, 3))), "-> w* =", w_star)

    Vt, neigh_t = knn(Et, test_sets); Vt_all, _ = knn(Et, None)
    Pt = P_ALL[test_ids]; yt = Y_ALL[test_ids]
    pred = {
        "classifier_inset": in_set_argmax(Pt, test_sets),
        "retrieval_inset": in_set_argmax(Vt, test_sets),
        "retrieval_unrestricted": Vt_all.argmax(1),
        "fusion_inset": in_set_argmax(w_star * np.log(Pt + 1e-12) + (1 - w_star) * np.log(Vt + 1e-6), test_sets),
    }

    # ---- tool 3: physics metadata, per-class typical ranges from the calibration side ----
    MCOLS = [("peak_frequency", "Hz"), ("duration", "s"), ("snr", ""), ("bandwidth", "Hz")]
    calm = df.loc[cal_u]
    stats = {c: {col: np.nanpercentile(calm.loc[Y_ALL[cal_u] == c, col].values.astype(float), [25, 50, 75])
                 for col, _ in MCOLS} for c in range(K) if (Y_ALL[cal_u] == c).sum() > 10}
    def meta_text(i, cands):
        s = "This glitch: " + ", ".join(f"{col}={float(df.at[i, col]):.3g}{u}" for col, u in MCOLS) + ". Typical values (median [25%-75%]): "
        parts = []
        for c in cands:
            if c in stats:
                parts.append(CLASSES[c] + ": " + ", ".join(f"{col} {stats[c][col][1]:.3g} [{stats[c][col][0]:.3g}-{stats[c][col][2]:.3g}]"
                                                          for col, _ in MCOLS))
        return s + "; ".join(parts) + ". "

    # ---- LLM agent over tool outputs ----
    model, proc, LLM_USED = load_llm(); log("E6 LLM:", LLM_USED)
    def ask_text(text, max_new_tokens=150):
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=[prompt], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    def parse6(ans, allowed):
        m = re.search(r'"label"\s*:\s*"([^"]+)"', ans); cand = m.group(1) if m else ans
        norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        if "escalate" in norm(cand): return "ESCALATE"
        for c in sorted(allowed, key=len, reverse=True):
            if norm(c) in norm(cand): return c
        return None
    def agent_prompt(r, i, cands, use_scores):
        s = ("You are a triage agent for LIGO detector glitches. A calibrated conformal classifier guarantees (90% coverage) "
             "that the true class is one of: " + ", ".join(f"{CLASSES[c]} ({DESC[CLASSES[c]]})" for c in cands) + ".\n"
             "Tool outputs:\n")
        if use_scores:
            s += "[classifier] scores: " + ", ".join(f"{CLASSES[c]}={P_ALL[i, c]:.2f}" for c in cands) + "\n"
        s += ("[image-retrieval] 10 most visually similar labelled glitches (label, cosine similarity): "
              + ", ".join(f"{l} {v:.2f}" for l, v in neigh_t[r]) + "; vote share: "
              + ", ".join(f"{CLASSES[c]}={Vt[r, c]:.2f}" for c in cands) + "\n")
        s += "[physics] " + meta_text(i, cands) + "\n"
        s += ("Weigh the evidence. If the tools clearly agree, choose that class. If they conflict strongly, answer ESCALATE "
              "so a human expert reviews it. Answer only with JSON: "
              '{"label": "<one class from the list, or ESCALATE>", "confidence": "high" or "low", "reason": "<one sentence>"}')
        return s

    recs = []
    n_llm = min(CFG["E6_N_LLM"], len(test_ids))
    for r in range(len(test_ids)):
        i = test_ids[r]; cands = list(np.where(test_sets[r])[0])
        rec = dict(gravityspy_id=df.at[i, "gravityspy_id"], true=CLASSES[yt[r]],
                   vol_major=CLASSES[Y_VOL[i]] if Y_VOL[i] >= 0 else None, set="|".join(CLASSES[c] for c in cands),
                   set_size=len(cands), true_in_set=bool(test_sets[r, yt[r]]))
        for k_, v in pred.items(): rec[k_] = CLASSES[v[r]]
        if r < n_llm:
            allowed = [CLASSES[c] for c in cands]
            for cond, us in [("agent_tools", True), ("agent_tools_noscores", False)]:
                try:
                    ans = ask_text(agent_prompt(r, i, cands, us))
                except Exception as e:
                    ans = f"ERROR {e!r}"; torch.cuda.empty_cache()
                rec[cond] = parse6(ans, allowed); rec[cond + "_raw"] = ans[:300]
        recs.append(rec)
        if r % 50 == 0:
            log(f"E6 {r}/{len(test_ids)}"); pd.DataFrame(recs).to_csv(f"{OUT}/E6_records.csv", index=False)
    e6 = pd.DataFrame(recs); save_csv(e6, "E6_records.csv")

    rows6 = []
    methods = list(pred) + ["agent_tools", "agent_tools_noscores"]
    for ref in ["true", "vol_major"]:
        for mth in methods:
            sub = e6[e6[ref].notna() & e6[mth].notna()] if mth.startswith("agent") else e6[e6[ref].notna()]
            if len(sub) == 0: continue
            esc = sub[mth] == "ESCALATE"; dec = sub[~esc]
            rows6.append(dict(reference=ref, method=mth, n=len(sub), escalation_rate=esc.mean(),
                              acc_all=(sub[mth] == sub[ref]).mean(),
                              acc_decided=(dec[mth] == dec[ref]).mean() if len(dec) else np.nan,
                              wrong_rate=((~esc) & (sub[mth] != sub[ref])).mean()))
    e6s = pd.DataFrame(rows6); save_csv(e6s, "E6_summary.csv"); print(e6s.round(4).to_string())

    pS, pA, pE, eS = strata6["singleton"], strata6["ambiguous"], strata6["escalate"], strata6["singleton_err"]
    pol = [dict(policy="CP: singleton auto, rest human", human_frac=pA + pE, error=pS * eS)]
    for _, rr in e6s[e6s.reference == "true"].iterrows():
        pol.append(dict(policy=f"CP + {rr['method']} on ambiguous", human_frac=pE + pA * rr["escalation_rate"],
                        error=pS * eS + pA * rr["wrong_rate"]))
    agree = e6["fusion_inset"] == e6["classifier_inset"]
    pol.append(dict(policy="CP + verifier: fusion agrees with classifier -> auto, else human",
                    human_frac=pE + pA * (1 - agree.mean()),
                    error=pS * eS + pA * (agree & (e6["classifier_inset"] != e6["true"])).mean()))
    pol = pd.DataFrame(pol); save_csv(pol, "E6_workload_policies.csv"); print(pol.round(4).to_string())
    json.dump(dict({k: float(v) for k, v in strata6.items()}, w_star=w_star, val_acc=[float(x) for x in vacc],
                   llm=LLM_USED, bank_size=int(len(bank))), open(f"{OUT}/E6_meta.json", "w"), indent=1)
else:
    log("E6 skipped")

# %% [markdown]
# ## Done

# %%
log("ALL DONE. Outputs:", sorted(os.listdir(OUT)))

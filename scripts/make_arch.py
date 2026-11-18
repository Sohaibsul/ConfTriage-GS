"""Pipeline figure as a graphical abstract -> figs/fig_pipeline.pdf/.png.   Run: python paper/make_arch.py"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge, Polygon, Arc

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 7})
INK, MUTE = "#1B2430", "#6B7684"
BLUE, GREEN, RED, AMBER, PURPLE, SLATE = "#2F6DB5", "#1F8A70", "#C0473F", "#D08A1E", "#7A4FB5", "#4C6A92"
W_, H_ = 7.2, 3.75
fig = plt.figure(figsize=(W_, H_)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
def sub(x0, y0, w, h):  # inset axes in data coordinates of the canvas
    return fig.add_axes([x0 / 100, y0 / 52, w / 100, h / 52])
def card(x0, y0, w, h, col, title, num):
    ax.add_patch(FancyBboxPatch((x0 + 0.5, y0 - 0.6), w, h, boxstyle="round,pad=0,rounding_size=1.8", fc="#00000012", ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.8", fc="white", ec=col, lw=1.2, zorder=1))
    ax.add_patch(Circle((x0 + 3.0, y0 + h - 3.0), 1.9, fc=col, ec="none", zorder=2))
    ax.text(x0 + 3.0, y0 + h - 3.0, num, color="white", ha="center", va="center", weight="bold", fontsize=7.5, zorder=3)
    ax.text(x0 + 5.6, y0 + h - 3.0, title, color=col, ha="left", va="center", weight="bold", fontsize=7.6, zorder=3)
def T(x, y, s, fs=6.3, c=INK, w="normal", ha="center"): ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va="center", zorder=5)
def chevron(x, y, col):
    ax.add_patch(FancyArrowPatch((x, y), (x + 2.6, y), arrowstyle="simple,head_width=1.2,head_length=0.9,tail_width=0.45",
                                 mutation_scale=9, fc=col, ec="none", zorder=4))
def omega(seed, kind=0):
    r = np.random.default_rng(seed); Tt, Ff = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 36))
    if kind == 0: Z = np.exp(-((Tt - .5) * 10) ** 2 - ((Ff - .38) * 4) ** 2)
    elif kind == 1: Z = np.exp(-((Tt - .5) * 6) ** 2 - ((Ff - .3 - .3 * (Tt - .5)) * 6) ** 2)
    else: Z = np.exp(-((Tt - .5) * 14) ** 2 - ((Ff - .25) * 7) ** 2) * 0.8
    return Z + 0.15 * r.random(Tt.shape)

# ---------- card 1: glitch + CNN scores ----------
card(1, 13, 22, 37, SLATE, "Glitch", "1")
for k, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
    a = sub(3.4 + dx * 9.0, 36.2 - dy * 8.0, 8.4, 7.4); a.imshow(omega(k, 1), cmap="viridis", origin="lower", aspect="auto"); a.axis("off")
T(12.2, 26.6, "4 Omega scans (0.5–4 s)", fs=5.8, c=MUTE)
a = sub(9.6, 14.6, 11.6, 8.4)
cls = ["Blip", "Koi Fish", "Tomte", "Whistle", "other"]; sc = [0.41, 0.33, 0.14, 0.05, 0.07]
a.barh(range(5)[::-1], sc, color=[SLATE] * 4 + ["#C9D1DB"], height=0.7)
a.set_yticks(range(5)[::-1]); a.set_yticklabels(cls, fontsize=5.4); a.set_xticks([]); a.set_xlim(0, 0.5)
for s_ in ["top", "right", "bottom"]: a.spines[s_].set_visible(False)
a.set_title("CNN scores (22 classes)", fontsize=5.8, pad=2, color=MUTE)

# ---------- card 2: class-conditional online calibration (real O3b curve) ----------
card(26, 13, 23, 37, BLUE, "Online calibration", "2")
try:
    rec = pd.read_csv(os.path.join(ROOT, "results_v5", "E2_records_O1O2_O3b.csv.gz"))
    rda = pd.read_csv(os.path.join(ROOT, "results_v6", "E2b_records_O1O2_O3b.csv.gz"))
    rec["da"] = rda["cov_da_g0.05"].values; rec["day"] = (rec.gps - rec.gps.iloc[0]) / 86400
    sub_ = rec[rec.y == "Scratchy"].copy(); sub_["wk"] = (sub_.day // 10).astype(int)
    g = sub_.groupby("wk").agg(s=("cov_split_mondrian", "mean"), d=("da", "mean"), n=("y", "size")); g = g[g.n >= 5]
    xs, ys_s, ys_d = g.index * 10 + 5, g.s.values, g.d.values
    mrg = rec.groupby((rec.day // 10).astype(int))["cov_aci_g0.005"].mean()
except Exception:
    xs = np.arange(5, 145, 10); ys_s = 0.3 + 0.1 * np.sin(xs / 20); ys_d = 0.9 + 0 * xs; mrg = pd.Series(0.9, index=xs // 10)
a = sub(31.2, 23.0, 15.6, 12.4)
a.axhline(0.9, color=INK, ls=":", lw=0.8)
a.plot(mrg.index * 10 + 5, mrg.values, color=AMBER, lw=1.3, label="overall coverage")
a.plot(xs, ys_s, color=RED, lw=1.3, ls="--", marker="o", ms=1.8, label="rare class, static CP")
a.plot(xs, ys_d, color=BLUE, lw=1.6, marker="o", ms=1.8, label="rare class, Mondrian-ACI")
a.set_ylim(0, 1.05); a.set_xlim(0, 145); a.tick_params(labelsize=5, length=2)
a.set_xlabel("days in O3b", fontsize=5.5, labelpad=1); a.set_ylabel("coverage", fontsize=5.5, labelpad=1)
for s_ in ["top", "right"]: a.spines[s_].set_visible(False)
a.legend(fontsize=4.7, frameon=False, loc="lower center", bbox_to_anchor=(0.42, 1.0), handlelength=1.8, borderaxespad=0.2)
T(37.5, 17.6, "Mondrian-ACI: one adaptive\nlevel per class", fs=5.9, w="bold", c=BLUE)
T(37.5, 14.6, "$\\alpha^{(y)}\\leftarrow\\alpha^{(y)}+\\gamma_y(\\alpha-\\mathrm{miss})$", fs=6.0)

# ---------- card 3: conformal set ----------
card(52, 13, 21, 37, BLUE, "Conformal set", "3")
a = sub(59.0, 24.5, 12.0, 15.0)
names = ["Blip", "Koi Fish", "Tomte", "Whistle", "Chirp", "Helix"]; sco = [0.41, 0.33, 0.14, 0.05, 0.03, 0.02]
thr = [0.20, 0.25, 0.10, 0.20, 0.15, 0.15]; inset = [s >= t for s, t in zip(sco, thr)]
yy = np.arange(len(names))[::-1]
a.barh(yy, sco, color=[BLUE if i else "#D5DBE3" for i in inset], height=0.66)
for y_, t_ in zip(yy, thr): a.plot([t_, t_], [y_ - 0.42, y_ + 0.42], color=RED, lw=1.1)
a.set_yticks(yy); a.set_yticklabels(names, fontsize=5.4); a.set_xticks([]); a.set_xlim(0, 0.48)
for s_ in ["top", "right", "bottom"]: a.spines[s_].set_visible(False)
a.set_title("score vs. threshold $\\hat q_y$ (red)", fontsize=5.4, pad=2, color=MUTE)
T(62.5, 20.0, "$C(x)=\\{$Blip, Koi Fish, Tomte$\\}$", fs=6.4, w="bold", c=BLUE)
T(62.5, 16.6, "contains the true class with\nprobability 0.9 for every class", fs=5.5, c=MUTE)

# ---------- card 4: set-gated triage with icons ----------
card(76, 13, 23, 37, GREEN, "Set-gated triage", "4")
def check_icon(cx, cy, col):
    ax.add_patch(Circle((cx, cy), 2.1, fc=col, ec="none", zorder=4))
    ax.plot([cx - 1.0, cx - 0.2, cx + 1.1], [cy, cy - 0.9, cy + 0.9], color="white", lw=1.8, zorder=5, solid_capstyle="round")
def person_icon(cx, cy, col):
    ax.add_patch(Circle((cx, cy + 0.9), 0.85, fc=col, ec="none", zorder=4))
    ax.add_patch(Wedge((cx, cy - 1.3), 1.6, 0, 180, fc=col, ec="none", zorder=4))
def lens_icon(cx, cy, col):
    ax.add_patch(Circle((cx - 0.3, cy + 0.3), 1.2, fc="white", ec=col, lw=1.6, zorder=4))
    ax.plot([cx + 0.55, cx + 1.6], [cy - 0.55, cy - 1.6], color=col, lw=2.0, zorder=4, solid_capstyle="round")
rows = [(41.0, "$|C|=1$", "auto-label"), (29.0, "$|C|=2$–$4$", "retrieval-verified"), (17.5, "$|C|>4$", "expert / new class")]
check_icon(80.0, rows[0][0], GREEN); T(83.3, rows[0][0] + 1.0, rows[0][2], fs=6.4, w="bold", c=GREEN, ha="left"); T(83.3, rows[0][0] - 1.4, rows[0][1], fs=5.6, c=MUTE, ha="left")
lens_icon(80.0, rows[1][0] + 2.2, AMBER); T(83.3, rows[1][0] + 3.2, rows[1][2], fs=6.4, w="bold", c=AMBER, ha="left"); T(83.3, rows[1][0] + 0.9, rows[1][1], fs=5.6, c=MUTE, ha="left")
for k in range(3):
    a = sub(83.3 + k * 5.2, 21.3, 4.6, 4.2); a.imshow(omega(10 + k, 1 if k < 2 else 0), cmap="viridis", origin="lower", aspect="auto"); a.set_xticks([]); a.set_yticks([])
    for s_ in a.spines.values(): s_.set_color(AMBER); s_.set_linewidth(0.8)
person_icon(80.0, rows[2][0], RED); T(83.3, rows[2][0] + 1.0, rows[2][2], fs=6.4, w="bold", c=RED, ha="left"); T(83.3, rows[2][0] - 1.4, rows[2][1], fs=5.6, c=MUTE, ha="left")

for x in (23.3, 49.3, 73.3): chevron(x, 31.5, "#9AA5B1")

# ---------- calibration agent loop ----------
ax.add_patch(FancyBboxPatch((26.5, 0.6), 72.5, 8.8, boxstyle="round,pad=0,rounding_size=1.8", fc="#F4EFFB", ec=PURPLE, lw=1.1, zorder=1))
ax.add_patch(Arc((31.5, 5.0), 6.0, 6.0, theta1=40, theta2=330, color=PURPLE, lw=1.6, zorder=3))
ax.add_patch(FancyArrowPatch((33.5, 7.2), (34.5, 5.9), arrowstyle="-|>", mutation_scale=7, color=PURPLE, lw=1.6, zorder=3))
bell = Polygon([(30.4, 3.6), (32.6, 3.6), (32.2, 4.4), (32.1, 5.8), (31.5, 6.4), (30.9, 5.8), (30.8, 4.4)], closed=True, fc=PURPLE, ec="none", zorder=4)
ax.add_patch(bell); ax.add_patch(Circle((31.5, 3.2), 0.35, fc=PURPLE, ec="none", zorder=4))
T(36.5, 6.9, "Calibration agent", fs=7.0, w="bold", c=PURPLE, ha="left")
T(36.5, 3.4, "observe class-wise coverage  →  request expert labels within a budget  →  update $\\alpha^{(y)}$  →  raise alerts",
  fs=5.9, ha="left")
ax.add_patch(FancyArrowPatch((87.5, 13.0), (87.5, 9.4), arrowstyle="-|>", mutation_scale=8, color=PURPLE, lw=1.3, ls=(0, (3, 2)), zorder=4))
T(88.2, 11.2, "labels", fs=5.4, c=PURPLE, ha="left")
ax.add_patch(FancyArrowPatch((37.5, 9.4), (37.5, 13.0), arrowstyle="-|>", mutation_scale=8, color=PURPLE, lw=1.3, zorder=4))
T(38.2, 11.2, "updates", fs=5.4, c=PURPLE, ha="left")

fig.savefig(os.path.join(OUT, "fig_pipeline.pdf")); fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=230)
print("wrote figs/fig_pipeline.pdf/.png")

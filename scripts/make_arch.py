"""Pipeline (architecture) figure -> figs/fig_pipeline.pdf/.png.   Run: python paper/make_arch.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), "paper", "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 7})
INK = "#1B2430"; MUTE = "#5B6675"
COL = dict(data="#4C6A92", cal="#2F6DB5", tri="#1F8A70", tool="#D08A1E", ok="#2E8B57", hum="#C0473F", agent="#7A4FB5")

fig = plt.figure(figsize=(7.4, 3.9)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 53); ax.axis("off")

def card(x0, y0, w, h, color, title=None, fill=None, lw=1.1, dashed=False, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill or "white",
                                ec=color, lw=lw, ls=(0, (3, 2)) if dashed else "-", zorder=z))
    if title:
        ax.add_patch(FancyBboxPatch((x0, y0 + h - 3.2), w, 3.2, boxstyle=f"round,pad=0,rounding_size={r}", fc=color, ec=color, lw=lw, zorder=z + 0.1))
        ax.add_patch(plt.Rectangle((x0, y0 + h - 3.2), w, 1.6, fc=color, ec=color, lw=0, zorder=z + 0.2))
        ax.text(x0 + w / 2, y0 + h - 1.6, title, color="white", ha="center", va="center", weight="bold", fontsize=7.2, zorder=z + 1)
def txt(x, y, s, fs=6.6, c=INK, w="normal", ha="center", z=5, **kw): ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va="center", zorder=z, **kw)
def arr(p, q, c=INK, ls="-", lw=1.0, cs="arc3", ms=8, z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw, ls=ls, connectionstyle=cs, shrinkA=0, shrinkB=0, zorder=z))
def chip(x, y, s, color, w=None):
    w = w or (len(s) * 0.62 + 1.4)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 1.05), w, 2.1, boxstyle="round,pad=0,rounding_size=1.0", fc=color, ec="none", alpha=0.18, zorder=5))
    txt(x, y, s, fs=6.0, c=color, w="bold", z=6)

# ---------- stage 1: input glitch with a stylized Omega scan ----------
card(1, 9, 17, 41, COL["data"], "1  Glitch stream", fill="#F4F6F9")
rng = np.random.default_rng(3); tt = np.linspace(0, 1, 60); ff = np.linspace(0, 1, 40); T, F = np.meshgrid(tt, ff)
for k, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
    sc = 1.0 + 0.5 * k
    Z = np.exp(-((T - 0.5) * 9 / sc) ** 2 - ((F - 0.35 - 0.25 * (T - 0.5)) * 5) ** 2) + 0.12 * rng.random(T.shape)
    x0, y0 = 2.6 + dx * 7.3, 32.6 - dy * 7.6
    ax.imshow(Z, extent=(x0, x0 + 6.8, y0, y0 + 7.0), origin="lower", cmap="viridis", aspect="auto", zorder=3)
txt(9.5, 24.0, "4 Omega scans (0.5–4 s)", fs=6.0, c=MUTE)
txt(9.5, 19.6, "CNN scores $\\hat\\pi(x_t)$", fs=6.6, w="bold")
txt(9.5, 16.8, "22 classes", fs=6.0, c=MUTE)
txt(9.5, 13.2, "GPS order", fs=6.0, c=MUTE); txt(9.5, 11.0, "O1 → O2 → O3a → O3b", fs=6.0, c=MUTE)

# ---------- stage 2: class-conditional online calibration ----------
card(21, 9, 25, 41, COL["cal"], "2  Online calibration", fill="#F2F6FC")
card(23, 37.5, 21, 8.5, COL["cal"], r=0.8, z=2)
txt(33.5, 43.6, "Mondrian split CP", fs=6.8, w="bold"); txt(33.5, 40.0, "one threshold $\\hat q_y$ per class\n(calibrated on earlier runs)", fs=6.0, c=MUTE)
card(23, 23.5, 21, 11, COL["cal"], r=0.8, lw=1.8, fill="#E3EDFA", z=2)
txt(33.5, 32.2, "Mondrian-ACI  (ours)", fs=6.9, w="bold", c=COL["cal"])
txt(33.5, 28.7, "$\\alpha^{(y)}\\leftarrow\\alpha^{(y)}+\\gamma_y(\\alpha-\\mathrm{miss})$", fs=6.6)
txt(33.5, 25.5, "per class · delay-aware $\\gamma_y$", fs=6.0, c=MUTE)
card(23, 11, 21, 10, COL["cal"], r=0.8, z=2)
txt(33.5, 18.3, "Prediction set $C(x_t)$", fs=6.8, w="bold")
chip(27.4, 14.2, "Blip", COL["cal"]); chip(33.5, 14.2, "Koi Fish", COL["cal"]); chip(39.6, 14.2, "Tomte", COL["cal"])
arr((33.5, 37.5), (33.5, 34.5), c=COL["cal"]); arr((33.5, 23.5), (33.5, 21.0), c=COL["cal"])
arr((18, 16), (23, 16), c=INK, lw=1.2)

# ---------- stage 3: set-gated triage ----------
card(49, 9, 50, 41, COL["tri"], "3  Set-gated triage", fill="#F1F8F5")
ax.add_patch(Polygon([(51.5, 29.5), (56, 34), (60.5, 29.5), (56, 25)], closed=True, fc="white", ec=COL["tri"], lw=1.3, zorder=3))
txt(56, 30.2, "$|C|$", fs=7.2, w="bold"); txt(56, 27.6, "router", fs=5.6, c=MUTE)
ax.plot([44, 47.6, 47.6], [16, 16, 29.5], color=INK, lw=1.2, zorder=4, solid_capstyle="butt")
arr((47.6, 29.5), (51.5, 29.5), c=INK, lw=1.2)
# outcome lanes
card(51, 39.5, 10, 6.5, COL["ok"], fill="#E6F4EC", r=0.8, z=2); txt(56, 43.6, "auto-label", fs=6.6, w="bold", c=COL["ok"]); txt(56, 41.2, "$|C|=1$", fs=6.0, c=MUTE)
card(51, 11, 10, 9.5, COL["hum"], fill="#FBE9E7", r=0.8, z=2); txt(56, 17.6, "expert /", fs=6.6, w="bold", c=COL["hum"]); txt(56, 15.2, "new class", fs=6.6, w="bold", c=COL["hum"]); txt(56, 12.7, "$|C|>4$", fs=6.0, c=MUTE)
arr((56, 34), (56, 39.5), c=COL["ok"], lw=1.2); arr((56, 25), (56, 20.5), c=COL["hum"], lw=1.2)
arr((60.5, 29.5), (64, 29.5), c=COL["tri"], lw=1.2); txt(62.2, 31.3, "2–4", fs=6.0, c=COL["tri"], w="bold")
# decision module
card(64, 11, 33.5, 35.5, COL["tri"], fill="white", dashed=True, r=1.0, z=1.5)
txt(80.75, 44.2, "Retrieval-verified decision module", fs=6.8, w="bold", c=COL["tri"])
card(65.5, 33.5, 14.5, 8, COL["tool"], fill="#FFF6E6", r=0.8, z=2); txt(72.75, 39.3, "CNN in set", fs=6.6, w="bold"); txt(72.75, 36.2, "$\\arg\\max_{y\\in C}\\hat\\pi_y$", fs=6.2, c=MUTE)
card(81.5, 33.5, 14.5, 8, COL["tool"], fill="#FFF6E6", r=0.8, z=2); txt(88.75, 39.3, "DINOv2 retrieval", fs=6.6, w="bold"); txt(88.75, 36.2, "k-NN, labelled memory", fs=5.8, c=MUTE)
card(65.5, 23.5, 30.5, 7.5, COL["tri"], fill="#EAF6F1", r=0.8, z=2)
txt(80.75, 28.7, "fusion  $w\\log\\hat\\pi_y+(1-w)\\log r_y$", fs=6.6, w="bold"); txt(80.75, 25.6, "restricted to the conformal set", fs=6.0, c=MUTE)
arr((72.75, 33.5), (72.75, 31.0), c=COL["tool"]); arr((88.75, 33.5), (88.75, 31.0), c=COL["tool"])
ax.add_patch(Polygon([(75.5, 17.5), (80.75, 21.2), (86, 17.5), (80.75, 13.8)], closed=True, fc="white", ec=COL["tri"], lw=1.2, zorder=3))
txt(80.75, 17.5, "agree?", fs=6.4, w="bold"); arr((80.75, 23.5), (80.75, 21.2), c=COL["tri"])
card(65.5, 12.2, 8.5, 4.6, COL["ok"], fill="#E6F4EC", r=0.8, z=2); txt(69.75, 14.5, "yes: label", fs=6.0, w="bold", c=COL["ok"])
card(87.5, 12.2, 8.5, 4.6, COL["hum"], fill="#FBE9E7", r=0.8, z=2); txt(91.75, 14.5, "no: expert", fs=6.0, w="bold", c=COL["hum"])
arr((75.5, 17.5), (74.0, 15.6), c=COL["ok"]); arr((86, 17.5), (87.5, 15.6), c=COL["hum"])

# ---------- calibration agent loop ----------
card(21, 1.0, 76.5, 6.2, COL["agent"], fill="#F3EEFA", r=1.0, z=1.5)
txt(23, 4.1, "Calibration agent:", fs=6.8, w="bold", c=COL["agent"], ha="left")
txt(40.0, 4.1, "observe sets  →  request labels (budget)  →  update $\\alpha^{(y)}$  →  alert on under-covered classes",
    fs=6.3, ha="left", c=INK)
ax.add_patch(FancyArrowPatch((56, 11), (40, 7.2), arrowstyle="-|>", mutation_scale=8, color=COL["agent"], lw=1.1, ls=(0, (3, 2)),
                             connectionstyle="arc3,rad=-0.25", zorder=4))
ax.add_patch(FancyArrowPatch((25, 7.2), (23.0, 26), arrowstyle="-|>", mutation_scale=8, color=COL["agent"], lw=1.1, ls=(0, (3, 2)),
                             connectionstyle="arc3,rad=-0.35", zorder=4))
txt(62.5, 8.3, "expert / volunteer labels", fs=5.8, c=COL["agent"], ha="left")

fig.savefig(os.path.join(OUT, "fig_pipeline.pdf")); fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=230)
print("wrote figs/fig_pipeline.pdf/.png")

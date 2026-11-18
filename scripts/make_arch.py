"""Pipeline (architecture) figure -> figs/fig_pipeline.pdf/.png.   Run: python paper/make_arch.py
Layout rule: only horizontal/vertical connectors, no connector crosses a box."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), "paper", "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 7})
INK, MUTE = "#1B2430", "#5B6675"
C = dict(data="#4C6A92", cal="#2F6DB5", tri="#1F8A70", tool="#C9861A", ok="#2E8B57", hum="#C0473F", agent="#7A4FB5")

fig = plt.figure(figsize=(7.4, 3.7)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

def box(x0, y0, w, h, ec, fc="white", lw=1.1, r=1.0, dashed=False, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=(0, (3, 2)) if dashed else "-", zorder=z))
def panel(x0, y0, w, h, col, title, fc):
    box(x0, y0, w, h, col, fc=fc, lw=1.0, r=1.4, z=0.5)
    ax.add_patch(FancyBboxPatch((x0, y0 + h - 3.4), w, 3.4, boxstyle="round,pad=0,rounding_size=1.4", fc=col, ec=col, lw=1.0, zorder=0.6))
    ax.add_patch(plt.Rectangle((x0, y0 + h - 3.4), w, 1.7, fc=col, ec=col, lw=0, zorder=0.7))
    t(x0 + w / 2, y0 + h - 1.7, title, fs=7.4, c="white", w="bold")
def t(x, y, s, fs=6.6, c=INK, w="normal", ha="center", z=6): ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va="center", zorder=z)
def line(xs, ys, c=INK, lw=1.1, ls="-"): ax.plot(xs, ys, color=c, lw=lw, ls=ls, zorder=3, solid_capstyle="butt")
def arrow(p, q, c=INK, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=4))
def pill(cx, cy, w, h, col, fc, l1, l2=None):
    box(cx - w / 2, cy - h / 2, w, h, col, fc=fc, r=h / 2.2, lw=1.2)
    if l2: t(cx, cy + 1.0, l1, w="bold", c=col); t(cx, cy - 1.3, l2, fs=5.9, c=MUTE)
    else: t(cx, cy, l1, w="bold", c=col)
def chip(cx, cy, s, col):
    w = len(s) * 0.55 + 1.4
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - 1.0), w, 2.0, boxstyle="round,pad=0,rounding_size=0.9", fc=col, ec="none", alpha=0.17, zorder=5))
    t(cx, cy, s, fs=5.9, c=col, w="bold")

# ---------------- panels ----------------
panel(1, 12, 16.5, 36, C["data"], "1  Glitch stream", "#F4F6F9")
panel(20, 12, 20.5, 36, C["cal"], "2  Online calibration", "#F1F5FC")
panel(43, 12, 56, 36, C["tri"], "3  Set-gated triage", "#F0F8F4")

# ---------------- 1: input ----------------
rng = np.random.default_rng(3); T, F = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 40))
for k, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
    Z = np.exp(-((T - 0.5) * 9 / (1 + 0.5 * k)) ** 2 - ((F - 0.35 - 0.25 * (T - 0.5)) * 5) ** 2) + 0.12 * rng.random(T.shape)
    x0, y0 = 2.7 + dx * 6.8, 34.2 - dy * 6.9
    ax.imshow(Z, extent=(x0, x0 + 6.3, y0, y0 + 6.3), origin="lower", cmap="viridis", aspect="auto", zorder=3)
t(9.25, 25.6, "Omega scans, 0.5–4 s", fs=5.9, c=MUTE)
t(9.25, 21.8, "CNN scores $\\hat\\pi(x_t)$", w="bold"); t(9.25, 19.4, "22 classes", fs=5.9, c=MUTE)
t(9.25, 15.6, "GPS order: O1 → O3b", fs=5.9, c=MUTE)

# ---------------- 2: calibration ----------------
# order: split CP (top) -> conformal set (middle) <- Mondrian-ACI (bottom, fed by the agent band below)
box(22, 37.5, 16.5, 6.5, C["cal"])
t(30.25, 42.0, "Mondrian split CP", w="bold"); t(30.25, 39.4, "$\\hat q_y$ per class, earlier runs", fs=5.9, c=MUTE)
box(22, 26.0, 16.5, 8.5, C["cal"])
t(30.25, 31.9, "Conformal set $C(x_t)$", w="bold")
chip(25.0, 28.6, "Blip", C["cal"]); chip(30.4, 28.6, "Koi Fish", C["cal"]); chip(35.9, 28.6, "Tomte", C["cal"])
box(22, 13.5, 16.5, 9.0, C["cal"], fc="#E2ECFA", lw=1.8)
t(30.25, 19.9, "Mondrian-ACI (ours)", w="bold", c=C["cal"])
t(30.25, 17.3, "$\\alpha^{(y)}\\!\\leftarrow\\alpha^{(y)}+\\gamma_y(\\alpha-\\mathrm{miss})$", fs=6.2)
t(30.25, 15.0, "per class, delay-aware", fs=5.8, c=MUTE)
arrow((30.25, 37.5), (30.25, 34.5), c=C["cal"]); arrow((30.25, 22.5), (30.25, 26.0), c=C["cal"])
arrow((17.5, 30.2), (22, 30.2), lw=1.3)

# ---------------- 3: routing by set size (vertical bus, horizontal branches) ----------------
BX = 45.5; YT, YM, YB = 39.5, 28.5, 17.0
line([38.5, BX], [30.2, 30.2], lw=1.3); line([BX, BX], [YB, YT], lw=1.3)
t(BX, YT + 3.6, "$|C|$", fs=6.0, c=MUTE, w="bold")
for y, lab in [(YT, "1"), (YM, "2–4"), (YB, ">4")]:
    arrow((BX, y), (49.5, y), lw=1.3); t(47.5, y + 1.4, lab, fs=5.6, c=MUTE)
pill(55.5, YT, 12, 5.2, C["ok"], "#E6F4EC", "auto-label")
pill(55.5, YB, 12, 5.2, C["hum"], "#FBE9E7", "expert", "or new class")

# decision module (middle lane)
box(49.5, 21.8, 48, 13.6, C["tri"], fc="white", dashed=True, r=1.2, z=1)
t(73.5, 33.7, "Retrieval-verified decision module", fs=6.4, w="bold", c=C["tri"])
box(51.0, 27.6, 12.5, 4.4, C["tool"], fc="#FFF5E3"); t(57.25, 29.8, "CNN in set", w="bold")
box(51.0, 22.8, 12.5, 4.4, C["tool"], fc="#FFF5E3"); t(57.25, 25.0, "DINOv2 k-NN", w="bold")
box(66.0, 23.6, 12.5, 7.6, C["tri"], fc="#E8F5EF")
t(72.25, 29.0, "fusion", w="bold"); t(72.25, 26.0, "$w\\log\\hat\\pi_y+(1\\!-\\!w)\\log r_y$", fs=5.2)
line([63.5, 65.0, 65.0], [29.8, 29.8, 27.4], c=C["tool"]); line([63.5, 65.0], [25.0, 25.0], c=C["tool"])
line([65.0, 65.0], [25.0, 27.4], c=C["tool"]); arrow((65.0, 27.4), (66.0, 27.4), c=C["tool"])
ax.add_patch(Polygon([(80.0, 27.4), (83.2, 30.6), (86.4, 27.4), (83.2, 24.2)], closed=True, fc="white", ec=C["tri"], lw=1.2, zorder=3))
t(83.2, 27.4, "agree?", fs=5.8, w="bold")
arrow((78.0, 27.4), (80.0, 27.4), c=C["tri"])
pill(92.3, 31.0, 8.6, 3.6, C["ok"], "#E6F4EC", "label")
pill(92.3, 23.8, 8.6, 3.6, C["hum"], "#FBE9E7", "expert")
line([86.4, 87.6, 87.6], [27.4, 27.4, 31.0], c=C["ok"]); arrow((87.6, 31.0), (88.0, 31.0), c=C["ok"])
line([87.6, 87.6], [27.4, 23.8], c=C["hum"]); arrow((87.6, 23.8), (88.0, 23.8), c=C["hum"])
t(86.6, 29.6, "yes", fs=5.2, c=C["ok"], ha="right"); t(86.6, 25.2, "no", fs=5.2, c=C["hum"], ha="right")

# ---------------- calibration agent band + straight feedback connectors ----------------
box(20, 1.5, 79, 7.5, C["agent"], fc="#F3EEFA", r=1.2, z=1)
t(22.0, 6.6, "Calibration agent", fs=6.8, w="bold", c=C["agent"], ha="left")
t(22.0, 3.6, "observe sets  →  request labels within a budget  →  update $\\alpha^{(y)}$  →  alert on under-covered classes",
  fs=6.0, ha="left")
# expert labels go down into the agent band
line([55.5, 55.5], [YB - 2.6, 9.0], c=C["agent"], ls=(0, (3, 2))); arrow((55.5, 10.2), (55.5, 9.0), c=C["agent"])
t(56.3, 10.9, "labels", fs=5.6, c=C["agent"], ha="left")
# agent updates Mondrian-ACI (vertical line in the left margin of panel 2)
arrow((30.25, 9.0), (30.25, 13.5), c=C["agent"], lw=1.3); t(31.0, 11.0, "updates", fs=5.6, c=C["agent"], ha="left")

fig.savefig(os.path.join(OUT, "fig_pipeline.pdf")); fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=230)
print("wrote figs/fig_pipeline.pdf/.png")

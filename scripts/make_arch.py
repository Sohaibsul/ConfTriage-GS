"""Pipeline figure (flat flowchart, left-to-right) -> figs/fig_pipeline.pdf/.png.   Run: python paper/make_arch.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge, Polygon, Arc

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), "paper", "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
INK, SUB, EDGE, ARW = "#1F2933", "#66717E", "#A7B4C4", "#55606E"
BLUE, GREEN, AMBER, RED, PURPLE = "#2F6DB5", "#22906F", "#C9861A", "#C0473F", "#7A4FB5"

fig = plt.figure(figsize=(7.2, 2.95)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 41); ax.axis("off")
def rbox(x0, y0, w, h, ec=EDGE, fc="white", lw=1.0, r=1.2, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
def T(x, y, s, fs=6.2, c=INK, w="normal", ha="center"): ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va="center", zorder=6)
def arrow(p, q, c=ARW, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=4))
def seg(xs, ys, c=ARW, lw=1.2): ax.plot(xs, ys, color=c, lw=lw, zorder=3, solid_capstyle="butt")
def stage(x0, w, title, sub, icon, fc="white", ec=EDGE, tc=INK):
    rbox(x0, 19.5, w, 14.0, ec=ec, fc=fc, lw=1.1)
    icon(x0 + w / 2, 29.3); T(x0 + w / 2, 24.6, title, fs=6.6, w="bold", c=tc); T(x0 + w / 2, 21.9, sub, fs=5.5, c=SUB)

# ---------------- icons ----------------
def ic_spec(cx, cy):
    Tt, Ff = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 30))
    Z = np.exp(-((Tt - .5) * 7) ** 2 - ((Ff - .35 - .3 * (Tt - .5)) * 6) ** 2) + 0.1 * np.random.default_rng(1).random(Tt.shape)
    ax.imshow(Z, extent=(cx - 3.2, cx + 3.2, cy - 2.6, cy + 2.6), cmap="viridis", origin="lower", aspect="auto", zorder=3)
def ic_net(cx, cy):
    cols = [(-2.6, [-1.6, 0, 1.6]), (0, [-2.2, -0.75, 0.75, 2.2]), (2.6, [-0.8, 0.8])]
    for (x1, ys1), (x2, ys2) in zip(cols[:-1], cols[1:]):
        for a in ys1:
            for b in ys2: ax.plot([cx + x1, cx + x2], [cy + a, cy + b], color="#C5CEDA", lw=0.5, zorder=3)
    for x, ys in cols:
        for y in ys: ax.add_patch(Circle((cx + x, cy + y), 0.5, fc=SUB, ec="none", zorder=4))
def ic_bars(cx, cy):
    hs = [3.6, 2.6, 4.2, 1.8, 3.0]
    for i, h in enumerate(hs):
        ax.add_patch(plt.Rectangle((cx - 3.0 + i * 1.25, cy - 2.4), 0.85, h, fc=BLUE, ec="none", alpha=0.75, zorder=3))
    ax.plot([cx - 3.4, cx + 3.4], [cy + 0.3, cy + 0.3], color=RED, lw=1.0, ls="--", zorder=4)
def ic_set(cx, cy):
    for i, (dx, dy) in enumerate([(-1.9, 1.1), (1.6, 1.1), (-0.2, -1.4)]):
        rbox(cx + dx - 1.6, cy + dy - 0.8, 3.2, 1.6, ec="none", fc=BLUE if i < 3 else "#DDE3EA", r=0.8, z=3)
    ax.add_patch(FancyBboxPatch((cx - 4.0, cy - 2.7), 8.0, 5.4, boxstyle="round,pad=0,rounding_size=2.4", fc="none", ec=BLUE, lw=1.0, ls=(0, (2, 1.5)), zorder=3))
def ic_check(cx, cy, c):
    ax.add_patch(Circle((cx, cy), 1.5, fc=c, ec="none", zorder=4))
    ax.plot([cx - 0.7, cx - 0.15, cx + 0.8], [cy, cy - 0.6, cy + 0.65], color="white", lw=1.4, zorder=5, solid_capstyle="round")
def ic_lens(cx, cy, c):
    ax.add_patch(Circle((cx - 0.25, cy + 0.25), 1.0, fc="white", ec=c, lw=1.4, zorder=4))
    ax.plot([cx + 0.45, cx + 1.3], [cy - 0.45, cy - 1.3], color=c, lw=1.8, zorder=4, solid_capstyle="round")
def ic_person(cx, cy, c):
    ax.add_patch(Circle((cx, cy + 0.65), 0.65, fc=c, ec="none", zorder=4)); ax.add_patch(Wedge((cx, cy - 1.0), 1.2, 0, 180, fc=c, ec="none", zorder=4))

# ---------------- group backgrounds ----------------
rbox(29.2, 16.0, 31.6, 24.4, ec="none", fc="#EEF3FA", r=1.6, z=0.5); T(45.0, 38.6, "Online conformal calibration", fs=6.4, w="bold", c=BLUE)
rbox(62.0, 14.2, 37.4, 26.2, ec="none", fc="#EDF6F2", r=1.6, z=0.5); T(80.7, 38.6, "Set-gated triage", fs=6.4, w="bold", c=GREEN)

# ---------------- main row ----------------
stage(1.0, 12.0, "Glitch", "Omega scans", ic_spec)
stage(15.5, 12.0, "CNN", "class scores $\\hat\\pi(x)$", ic_net)
stage(31.0, 13.0, "Mondrian-ACI", "per-class thresholds,\nupdated online", ic_bars, fc="#E1EBF8", ec=BLUE, tc=BLUE)
stage(46.5, 12.5, "Conformal set", "$C(x)$, 90% per class", ic_set)
for x0, x1 in [(13.0, 15.5), (27.5, 31.0), (44.0, 46.5)]: arrow((x0, 26.5), (x1, 26.5))
# router
ax.add_patch(Polygon([(61.0, 26.5), (63.6, 29.1), (66.2, 26.5), (63.6, 23.9)], closed=True, fc="white", ec=GREEN, lw=1.2, zorder=3))
T(63.6, 26.5, "$|C|$", fs=6.4, w="bold")
arrow((59.0, 26.5), (61.0, 26.5))
# fan-out to outcomes (orthogonal)
BX = 68.0; seg([66.2, BX], [26.5, 26.5]); seg([BX, BX], [19.0, 34.0])
outs = [(34.0, "1", "Automatic label", "singleton set", GREEN, ic_check),
        (26.5, "2–4", "Retrieval-verified decision", "CNN ⊕ DINOv2; label only if both agree", AMBER, ic_lens),
        (19.0, ">4", "Expert / new-class queue", "large set: human inspection", RED, ic_person)]
for y, lab, title, sub, col, ic in outs:
    arrow((BX, y), (70.5, y), c=ARW); T(69.25, y + 1.3, lab, fs=5.4, c=SUB)
    rbox(70.5, y - 3.0, 28.2, 6.0, ec=col, fc="white", lw=1.1, r=1.0)
    ic(73.2, y, col); T(75.6, y + 1.15, title, fs=6.3, w="bold", c=col, ha="left"); T(75.6, y - 1.35, sub, fs=5.3, c=SUB, ha="left")

# ---------------- calibration agent (feedback) ----------------
rbox(29.2, 1.2, 70.2, 7.6, ec=PURPLE, fc="#F5F0FB", lw=1.0, r=1.4, z=1)
ax.add_patch(Arc((33.2, 5.0), 4.6, 4.6, theta1=50, theta2=320, color=PURPLE, lw=1.4, zorder=3))
ax.add_patch(FancyArrowPatch((34.6, 6.8), (35.4, 5.7), arrowstyle="-|>", mutation_scale=6, color=PURPLE, lw=1.4, zorder=3))
T(37.0, 6.4, "Calibration agent", fs=6.6, w="bold", c=PURPLE, ha="left")
T(37.0, 3.4, "monitors class-wise coverage  ·  requests expert labels within a budget  ·  raises alerts", fs=5.6, c=INK, ha="left")
arrow((84.6, 16.0), (84.6, 8.8), c=PURPLE, ls=(0, (3, 2)))       # expert labels -> agent
T(85.3, 11.2, "labels", fs=5.2, c=PURPLE, ha="left")
arrow((37.5, 8.8), (37.5, 19.5), c=PURPLE)                       # agent -> Mondrian-ACI
T(38.2, 12.0, "update $\\alpha^{(y)}$", fs=5.2, c=PURPLE, ha="left")

fig.savefig(os.path.join(OUT, "fig_pipeline.pdf")); fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=240)
print("wrote figs/fig_pipeline.pdf/.png")

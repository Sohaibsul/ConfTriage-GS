"""Pipeline figure (detailed flat flowchart, left-to-right) -> figs/fig_pipeline.pdf/.png.   Run: python paper/make_arch.py"""
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

fig = plt.figure(figsize=(7.2, 4.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
def rbox(x0, y0, w, h, ec=EDGE, fc="white", lw=1.0, r=1.2, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
def T(x, y, s, fs=6.0, c=INK, w="normal", ha="center"): ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va="center", zorder=6, linespacing=1.15)
def arrow(p, q, c=ARW, lw=1.15, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7.5, color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=4))
def seg(xs, ys, c=ARW, lw=1.15): ax.plot(xs, ys, color=c, lw=lw, zorder=3, solid_capstyle="butt")

# ---------------- icons ----------------
def ic_spec(cx, cy):
    Tt, Ff = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 30))
    Z = np.exp(-((Tt - .5) * 7) ** 2 - ((Ff - .35 - .3 * (Tt - .5)) * 6) ** 2) + 0.1 * np.random.default_rng(1).random(Tt.shape)
    ax.imshow(Z, extent=(cx - 3.2, cx + 3.2, cy - 2.4, cy + 2.4), cmap="viridis", origin="lower", aspect="auto", zorder=3)
def ic_net(cx, cy):
    cols = [(-2.6, [-1.5, 0, 1.5]), (0, [-2.1, -0.7, 0.7, 2.1]), (2.6, [-0.75, 0.75])]
    for (x1, a1), (x2, a2) in zip(cols[:-1], cols[1:]):
        for a in a1:
            for b in a2: ax.plot([cx + x1, cx + x2], [cy + a, cy + b], color="#C5CEDA", lw=0.5, zorder=3)
    for x, ys in cols:
        for y in ys: ax.add_patch(Circle((cx + x, cy + y), 0.45, fc=SUB, ec="none", zorder=4))
def ic_bars(cx, cy):
    for i, h in enumerate([3.4, 2.4, 4.0, 1.7, 2.8]):
        ax.add_patch(plt.Rectangle((cx - 3.0 + i * 1.25, cy - 2.2), 0.85, h, fc=BLUE, ec="none", alpha=0.75, zorder=3))
    ax.plot([cx - 3.4, cx + 3.4], [cy + 0.3, cy + 0.3], color=RED, lw=1.0, ls="--", zorder=4)
def ic_check(cx, cy, c):
    ax.add_patch(Circle((cx, cy), 1.45, fc=c, ec="none", zorder=4))
    ax.plot([cx - 0.65, cx - 0.12, cx + 0.75], [cy, cy - 0.55, cy + 0.6], color="white", lw=1.3, zorder=5, solid_capstyle="round")
def ic_lens(cx, cy, c):
    ax.add_patch(Circle((cx - 0.25, cy + 0.25), 0.95, fc="white", ec=c, lw=1.3, zorder=4))
    ax.plot([cx + 0.42, cx + 1.25], [cy - 0.42, cy - 1.25], color=c, lw=1.7, zorder=4, solid_capstyle="round")
def ic_person(cx, cy, c):
    ax.add_patch(Circle((cx, cy + 0.6), 0.6, fc=c, ec="none", zorder=4)); ax.add_patch(Wedge((cx, cy - 0.95), 1.15, 0, 180, fc=c, ec="none", zorder=4))
def chip(cx, cy, s, c=BLUE):
    w = len(s) * 0.52 + 1.2
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - 0.85), w, 1.7, boxstyle="round,pad=0,rounding_size=0.8", fc=c, ec="none", alpha=0.16, zorder=3))
    T(cx, cy, s, fs=5.1, c=c, w="bold")

MY = 34.0                                    # centre line of the main row
def stage(x0, w, title, lines, icon, fc="white", ec=EDGE, tc=INK):
    rbox(x0, 26.0, w, 16.0, ec=ec, fc=fc, lw=1.1)
    icon(x0 + w / 2, 37.6); T(x0 + w / 2, 33.0, title, fs=6.5, w="bold", c=tc); T(x0 + w / 2, 29.2, lines, fs=5.1, c=SUB)

# ---------------- group backgrounds ----------------
rbox(28.6, 23.0, 33.4, 31.0, ec="none", fc="#EEF3FA", r=1.6, z=0.5); T(45.3, 52.6, "Online conformal calibration", fs=6.3, w="bold", c=BLUE)
rbox(64.6, 13.2, 35.0, 40.8, ec="none", fc="#EDF6F2", r=1.6, z=0.5); T(82.1, 52.6, "Set-gated triage", fs=6.3, w="bold", c=GREEN)

# ---------------- main row ----------------
stage(1.0, 12.4, "Glitch $x_t$", "4 Omega scans\n(0.5–4 s), GPS order", ic_spec)
stage(15.4, 11.6, "CNN", "Gravity Spy, frozen\n22 class scores $\\hat\\pi$", ic_net)
stage(30.4, 14.2, "Mondrian-ACI", "$\\alpha^{(y)}\\!\\leftarrow\\!\\alpha^{(y)}\\!+\\!\\gamma_y(\\alpha\\!-\\!\\mathrm{miss})$\none level per class,\ndelay-aware $\\gamma_y$", ic_bars,
      fc="#E1EBF8", ec=BLUE, tc=BLUE)
rbox(46.6, 26.0, 13.6, 16.0, ec=EDGE, lw=1.1)
T(53.4, 40.3, "Conformal set", fs=6.5, w="bold")
chip(50.2, 37.3, "Blip"); chip(55.9, 37.3, "Koi Fish"); chip(53.4, 35.0, "Tomte")
T(53.4, 30.4, "$C(x_t)=\\{y:s(x_t,y)\\leq\\hat q_y\\}$\ncovers each class\nwith prob. 0.9", fs=5.0, c=SUB)
for x0, x1 in [(13.4, 15.4), (27.0, 30.4), (44.6, 46.6)]: arrow((x0, MY), (x1, MY))
# source-run calibration feeding Mondrian-ACI
rbox(30.4, 45.4, 14.2, 5.0, ec=BLUE, fc="white", lw=0.9, r=0.9)
T(37.5, 47.9, "source runs (O1+O2)\nsplit CP: $\\hat q_y$ per class", fs=5.0, c=BLUE)
arrow((37.5, 45.4), (37.5, 42.0), c=BLUE)
# router
ax.add_patch(Polygon([(61.6, MY), (63.8, MY + 2.2), (66.0, MY), (63.8, MY - 2.2)], closed=True, fc="white", ec=GREEN, lw=1.2, zorder=3))
T(63.8, MY, "$|C|$", fs=6.0, w="bold"); arrow((60.2, MY), (61.6, MY))

# ---------------- outcomes ----------------
BX = 67.6; seg([66.0, BX], [MY, MY]); seg([BX, BX], [17.8, 48.6])
def out_row(y, lab):
    arrow((BX, y), (70.0, y)); T(68.8, y + 1.2, lab, fs=5.2, c=SUB)
out_row(48.6, "1"); out_row(MY, "2–4"); out_row(17.8, ">4")
rbox(70.0, 45.6, 29.0, 6.0, ec=GREEN, lw=1.1, r=1.0)
ic_check(72.8, 48.6, GREEN); T(75.2, 49.7, "Automatic label", fs=6.2, w="bold", c=GREEN, ha="left"); T(75.2, 47.3, "singleton set (15.7% of glitches)", fs=5.0, c=SUB, ha="left")
rbox(70.0, 14.8, 29.0, 6.0, ec=RED, lw=1.1, r=1.0)
ic_person(72.8, 17.8, RED); T(75.2, 18.9, "Expert / new-class queue", fs=6.2, w="bold", c=RED, ha="left"); T(75.2, 16.5, "set with more than four classes", fs=5.0, c=SUB, ha="left")
# decision module (detailed)
rbox(70.0, 23.0, 29.0, 20.6, ec=AMBER, lw=1.1, r=1.0)
ic_lens(72.6, 41.0, AMBER); T(74.8, 41.0, "Retrieval-verified decision", fs=6.2, w="bold", c=AMBER, ha="left")
rbox(71.2, 33.4, 12.4, 5.2, ec=EDGE, fc="#FFF8EC", r=0.8); T(77.4, 36.0, "CNN top label\ninside $C(x_t)$", fs=5.1)
rbox(85.4, 33.4, 12.6, 5.2, ec=EDGE, fc="#FFF8EC", r=0.8); T(91.7, 36.0, "DINOv2 10-NN\nmemory 120/class", fs=5.1)
rbox(71.2, 25.0, 17.2, 5.6, ec=AMBER, fc="#FFF3DD", r=0.8); T(79.8, 27.8, "fusion $w\\log\\hat\\pi_y+(1\\!-\\!w)\\log r_y$\n$w=0.2$, restricted to $C(x_t)$", fs=5.0)
seg([77.4, 77.4], [33.4, 32.0], c=AMBER); seg([91.7, 91.7], [33.4, 32.0], c=AMBER); seg([77.4, 91.7], [32.0, 32.0], c=AMBER)
arrow((84.0, 32.0), (84.0, 30.6), c=AMBER)
ax.add_patch(Polygon([(89.6, 27.8), (92.4, 30.3), (95.2, 27.8), (92.4, 25.3)], closed=True, fc="white", ec=AMBER, lw=1.1, zorder=3))
T(92.4, 27.8, "agree?", fs=5.0, w="bold"); arrow((88.4, 27.8), (89.6, 27.8), c=AMBER)
T(92.4, 31.6, "yes → label", fs=4.8, c=GREEN, w="bold"); T(92.4, 24.0, "no → expert", fs=4.8, c=RED, w="bold")

# ---------------- calibration agent: observe -> decide -> act ----------------
rbox(28.6, 1.0, 71.0, 10.6, ec=PURPLE, fc="#F5F0FB", lw=1.0, r=1.4, z=1)
ax.add_patch(Arc((32.4, 7.6), 3.4, 3.4, theta1=50, theta2=320, color=PURPLE, lw=1.4, zorder=3))
ax.add_patch(FancyArrowPatch((33.45, 8.95), (34.0, 8.15), arrowstyle="-|>", mutation_scale=6, color=PURPLE, lw=1.4, zorder=3))
T(32.4, 3.9, "Calibration\nagent", fs=5.2, w="bold", c=PURPLE)
steps = [(36.6, "Observe", "sets, scores and\nrevealed labels"), (57.4, "Decide", "request a label? priority to label-\nstarved or under-covered classes"),
         (79.6, "Act", "update $\\alpha^{(y)}$; alert when a class's\nmiss rate exceeds $\\alpha+0.1$")]
for i, (x0, ttl, txt) in enumerate(steps):
    rbox(x0, 2.6, 18.6 if i else 17.0, 7.4, ec=PURPLE, fc="white", lw=0.9, r=0.9)
    w_ = 18.6 if i else 17.0
    T(x0 + w_ / 2, 8.2, ttl, fs=5.9, w="bold", c=PURPLE); T(x0 + w_ / 2, 4.9, txt, fs=4.9, c=INK)
arrow((53.6, 6.3), (57.4, 6.3), c=PURPLE); arrow((76.0, 6.3), (79.6, 6.3), c=PURPLE)
arrow((84.5, 14.8), (84.5, 11.6), c=PURPLE, ls=(0, (3, 2))); T(85.1, 13.2, "expert labels", fs=4.9, c=PURPLE, ha="left")
arrow((37.5, 11.6), (37.5, 26.0), c=PURPLE); T(38.1, 17.5, "updated\nlevels $\\alpha^{(y)}$", fs=4.9, c=PURPLE, ha="left")

fig.savefig(os.path.join(OUT, "fig_pipeline.pdf")); fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=240)
print("wrote figs/fig_pipeline.pdf/.png")

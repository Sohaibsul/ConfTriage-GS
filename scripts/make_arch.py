"""Architecture diagram of the full pipeline -> figs/fig_architecture.pdf (+ .png preview).
Run:  python scripts/make_arch.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "paper", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 6.6})

# palette (stage colours)
INK = "#1F2933"
C = {
    "data":  ("#F3F5F8", "#6B7C93"),
    "cal":   ("#EDF3FB", "#2F6DB5"),
    "tri":   ("#EEF7F2", "#2E8B62"),
    "tool":  ("#FFF7E8", "#C3841A"),
    "auto":  ("#E3F4E8", "#2E8B57"),
    "human": ("#FBE7E5", "#B5473F"),
    "prop":  ("#D9E8FA", "#1D4F91"),
}

fig = plt.figure(figsize=(7.4, 4.3))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")


def panel(x0, y0, x1, y1, key, title, num):
    fc, ec = C[key]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec=ec, lw=0.9, zorder=0))
    ax.add_patch(FancyBboxPatch((x0, y1 - 4.2), x1 - x0, 4.2, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=ec, ec=ec, lw=0.9, zorder=0.5))
    ax.add_patch(plt.Rectangle((x0, y1 - 4.2), x1 - x0, 2.0, fc=ec, ec=ec, lw=0, zorder=0.6))
    ax.text(x0 + 1.4, y1 - 2.1, f"{num}  {title}", color="white", weight="bold", va="center", ha="left",
            fontsize=7.0, zorder=1)


def box(x0, y0, x1, y1, key, text, bold_first=True, lw=0.9, dashed=False, fs=6.6):
    fc, ec = C[key]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc if key not in ("data", "cal", "tri") else "white", ec=ec, lw=lw,
                                ls=(0, (3, 2)) if dashed else "-", zorder=2))
    lines = text.split("\n")
    n = len(lines)
    h = 2.25  # line height in data units
    ymid = (y0 + y1) / 2
    for i, ln in enumerate(lines):
        y = ymid + (n - 1) / 2 * h - i * h
        ax.text((x0 + x1) / 2, y, ln, ha="center", va="center", color=INK, zorder=3, fontsize=fs,
                weight="bold" if (bold_first and i == 0) else "normal")


def arrow(p, q, color=INK, ls="-", lw=0.9, cs="arc3", label=None, lpos=0.5, loff=(0, 1.2), la="center"):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, ls=ls,
                        connectionstyle=cs, shrinkA=0, shrinkB=0, zorder=4)
    ax.add_patch(a)
    if label:
        x = p[0] + (q[0] - p[0]) * lpos + loff[0]
        y = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(x, y, label, ha=la, va="center", fontsize=6.0, color=color, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))


def poly(pts, color=INK, ls="-", lw=0.9):
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, zorder=4, solid_capstyle="butt")
    arrow(pts[-2], pts[-1], color=color, ls=ls, lw=lw)


# ---------------- panels ----------------
panel(1, 6, 20.5, 56, "data", "Data", "1")
panel(23, 6, 47, 56, "cal", "Online calibration", "2")
panel(49.5, 6, 99, 56, "tri", "Set-gated triage", "3")

# ---------------- 1. data ----------------
box(2.5, 39, 19, 49, "data", "Volunteer release\nZenodo 5911227\n809k glitches, 22 classes\nscores, labels, votes")
box(2.5, 25, 19, 35, "data", "ML catalogue\nZenodo 5649212\nGPS time, detector,\nOmicron, 4 Omega scans")
box(2.5, 9, 19, 20.5, "data", "Glitch $x_t$\nin GPS order\nscores $\\hat\\pi(x_t)$,\nmetadata $m_t$, images")
arrow((10.75, 39), (10.75, 35))
arrow((10.75, 25), (10.75, 20.5), label="join on ID", loff=(0, 0))

# ---------------- 2. calibration ----------------
box(24.5, 39, 45.5, 49, "cal", "Mondrian split CP\ncalibrate on source runs\n(e.g. O1+O2): one\nthreshold $\\hat q_y$ per class")
box(24.5, 24.5, 45.5, 35, "prop", "Mondrian-ACI (proposed)\nper-class level $\\alpha_t^{(y)}$\n$\\alpha^{(y_t)}_{t+1}=\\alpha^{(y_t)}_t+\\gamma(\\alpha-\\mathrm{err}_t)$\nonly class $y_t$ is updated", lw=1.4)
box(24.5, 9, 45.5, 20.5, "cal", "Prediction set\n$C(x_t)=\\{y:\\ s(x_t,y)\\leq\\hat q_y(\\alpha_t^{(y)})\\}$\nscore $s=1-\\hat\\pi_y$ (LAC)\nclass-wise coverage $\\approx 1-\\alpha$")
arrow((35, 39), (35, 35))
arrow((35, 24.5), (35, 20.5))
arrow((19, 14.75), (24.5, 14.75))

# ---------------- 3. triage ----------------
# router (diamond)
rx, ry, rw, rh = 56.5, 31.5, 5.6, 7.0
ax.add_patch(Polygon([(rx - rw, ry), (rx, ry + rh), (rx + rw, ry), (rx, ry - rh)], closed=True,
                     fc="white", ec=C["tri"][1], lw=1.1, zorder=2))
ax.text(rx, ry + 0.9, "$|C(x_t)|$", ha="center", va="center", fontsize=7, weight="bold", color=INK, zorder=3)
ax.text(rx, ry - 1.6, "router", ha="center", va="center", fontsize=5.8, color="#52606D", zorder=3)
poly([(45.5, 14.75), (48.2, 14.75), (48.2, ry), (rx - rw, ry)])

box(51, 43.5, 62, 50.5, "auto", "Automatic\nlabel")
box(51, 9, 62, 18.5, "human", "Human expert\n/ new-class\nqueue")
arrow((rx, ry + rh), (rx, 43.5), label="$=1$", loff=(2.6, 0), la="center")
arrow((rx, ry - rh), (rx, 18.5), label="$>4$", loff=(2.6, 0), la="center")

# decision sub-panel
dx0, dx1, dy0, dy1 = 65, 98, 8, 50.8
ax.add_patch(FancyBboxPatch((dx0, dy0), dx1 - dx0, dy1 - dy0, boxstyle="round,pad=0,rounding_size=1",
                            fc="white", ec=C["tri"][1], lw=0.9, ls=(0, (4, 2)), zorder=1))
ax.text((dx0 + dx1) / 2, dy1 - 1.9, "Retrieval-verified decision module", ha="center",
        va="center", fontsize=6.6, weight="bold", color=C["tri"][1], zorder=3)
arrow((rx + rw, ry), (dx0, ry), label="$2$–$4$", lpos=0.5, loff=(0, 1.4))

# tools
box(66.5, 38.5, 75.5, 47, "tool", "Classifier\nscores\nrestricted\nto $C(x_t)$", fs=6.2)
box(77, 38.5, 87.5, 47, "tool", "Retrieval tool\nDINOv2 $k$-NN\nmemory bank\n120 / class", fs=6.2)
box(89, 38.5, 96.8, 47, "tool", "Physics\ntool\n$f_{\\rm peak}$, dur.,\nSNR, bw.", fs=6.2)

# fusion
box(66.5, 28.5, 87.5, 36, "tri", "Set-restricted fusion\n$\\hat y=\\arg\\max_{y\\in C}\\ \\log\\hat\\pi_y+w^\\star\\log r_y$\n$w^\\star=0.2$ chosen on validation", fs=6.2)
arrow((71, 38.5), (71, 36))
arrow((82.25, 38.5), (82.25, 36))

# LLM agent (evaluated, not adopted)
box(89, 23, 96.8, 35, "data", "LLM agent\nover all\ntools\n(tested,\nnot used)", dashed=True, fs=5.8)
arrow((92.9, 38.5), (92.9, 35), ls=(0, (3, 2)), color="#6B7C93")

# verifier
box(70, 18, 84, 24.5, "tri", "Verifier\n$\\hat y_{\\rm fusion}=\\hat y_{\\rm clf}$ ?", fs=6.2)
arrow((77, 28.5), (77, 24.5))

# outcomes
box(66.5, 9.5, 76.5, 14.5, "auto", "agree: auto label", bold_first=False, fs=6.0)
box(78, 9.5, 89, 14.5, "human", "disagree: human", bold_first=False, fs=6.0)
arrow((74, 18), (71.5, 14.5))
arrow((80, 18), (83.5, 14.5))

# result badge
ax.text(93.4, 16.5, "error\n7.5%$\\rightarrow$5.2%", ha="center", va="center", fontsize=5.8,
        color=C["tri"][1], weight="bold", zorder=3)

# ---------------- feedback loop: labels -> Mondrian-ACI ----------------
FB = "#B5473F"
poly([(56.5, 9), (56.5, 2.6), (22, 2.6), (22, 29.75), (24.5, 29.75)], color=FB, ls=(0, (4, 2)))
ax.text(39, 2.6, "revealed labels $y_t$ (volunteers / experts) update $\\alpha^{(y_t)}$",
        ha="center", va="center", fontsize=6.0, color=FB, zorder=5, bbox=dict(fc="white", ec="none", pad=1.5))

fig.savefig(os.path.join(OUT, "fig_architecture.pdf"))
fig.savefig(os.path.join(OUT, "fig_architecture.png"), dpi=220)
print("wrote figs/fig_architecture.pdf/.png")

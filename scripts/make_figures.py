"""Figures for the CHASE manuscript (all numbers from the reported experiments)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 2, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------- Fig. 1 schematic
COCO = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
base = np.array([[0, 7.2], [-.25, 7.4], [.25, 7.4], [-.5, 7.25], [.5, 7.25], [-1, 6.3], [1, 6.3],
                 [-1.4, 5.0], [1.4, 5.0], [-1.6, 3.9], [1.6, 3.9], [-.6, 3.8], [.6, 3.8],
                 [-.7, 2.0], [.7, 2.0], [-.75, .3], [.75, .3]])
rng = np.random.default_rng(3)
persons = [base * 0.9 + [5.5, 2.2] + rng.normal(0, .05, base.shape),
           base * 0.85 + [9.5, 2.6] + rng.normal(0, .05, base.shape)]
valid = np.vstack(persons)
pad = np.zeros((3, 2))                                         # zero-padded slots at the image corner


def panel(ax, title):
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlim(-0.8, 12.2); ax.set_ylim(-1.9, 12.6); ax.set_aspect("equal")
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for P in persons:
        for a, b in COCO:
            ax.plot(*P[[a, b]].T, color=GRAY, lw=1, zorder=1)
        ax.scatter(*P.T, s=6, color=INK2, zorder=2, linewidths=0)


def hull(ax, pts, **kw):
    h = ConvexHull(pts)
    ax.add_patch(Polygon(pts[h.vertices], closed=True, fill=False, **kw))


fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.55))
# (a) as designed
panel(axs[0], "(a) CHASE as designed")
hull(axs[0], np.vstack([valid, pad]), ls="--", lw=1, ec=GRAY)
axs[0].scatter(*pad.T, s=14, marker="s", color="white", ec=INK2, lw=.8, zorder=3)
axs[0].scatter([7.4], [5.3], s=70, marker="*", color=BLUE, ec="white", lw=.6, zorder=4)
axs[0].annotate("adaptive origin in hull interior", (7.4, 5.3), (7.0, 11.2), fontsize=7.5, color=INK,
                ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
axs[0].annotate("zero padding", (0, 0), (0.6, -0.6), fontsize=7.5, color=INK2, va="top")
# (b) observed
panel(axs[1], "(b) Observed collapse")
hull(axs[1], np.vstack([valid, pad]), ls="--", lw=1, ec=GRAY)
axs[1].scatter(*pad.T, s=14, marker="s", color="white", ec=INK2, lw=.8, zorder=3)
kp = persons[0][4]
axs[1].scatter(*kp, s=70, marker="*", color=BLUE, ec="white", lw=.6, zorder=4)
axs[1].annotate("CAD: one fixed keypoint ($N_{\\mathrm{eff}}=1$)", kp, (3.6, 10.9), fontsize=7.5, color=INK,
                ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
cen = np.vstack([valid, pad]).mean(0)
axs[1].scatter(*cen, s=60, marker="D", color=ORANGE, ec="white", lw=.6, zorder=4)
axs[1].annotate("NTU: centroid incl. padding\n($N_{\\mathrm{eff}}=U$)", cen, (8.6, -0.5), fontsize=7.5,
                color=INK, ha="center", va="top", arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
# (c) CHASE+
panel(axs[2], "(c) CHASE+")
hull(axs[2], valid, ls="-", lw=1, ec=AQUA)
axs[2].scatter(*pad.T, s=14, marker="s", color="white", ec=GRAY, lw=.6, zorder=3, alpha=.6)
axs[2].annotate("padding masked ($\\alpha_u=0$)", (0, 0), (0.6, -0.6), fontsize=7.5, color=INK2, va="top")
sub = persons[0][[5, 6, 11, 12]].mean(0)
axs[2].scatter(*sub, s=70, marker="*", color=AQUA, ec="white", lw=.6, zorder=4)
axs[2].annotate("origin inside the valid hull", sub, (6.0, 11.2),
                fontsize=7.5, color=INK, ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
fig.tight_layout(w_pad=0.6)
save(fig, "fig1_schematic")

# ----------------------------------------------------------------- Fig. 2 effective support
rows = [  # label, U, p05, median, p95, colour
    ("NTU-11 · CHASE (released)", 3200, 3200, 3200, 3200, ORANGE),
    ("CAD · CHASE (released)", 1020, 1, 1, 1, BLUE),
    ("CAD · CHASE seed 1", 1020, 1, 1, 1, BLUE),
    ("CAD · CHASE seed 2", 1020, 1, 1, 1, BLUE),
    ("CAD · CHASE seed 3", 1020, 1, 1, 1, BLUE),
    ("CAD · CHASE+ (5 seeds)", 1020, 1, 1, 1, BLUE),
    ("CAD · CHASE+B seed 1", 1020, 46.4, 76.3, 76.3, AQUA),
    ("CAD · CHASE+B seed 2", 1020, 45.0, 87.0, 87.0, AQUA),
    ("CAD · CHASE+B seed 3", 1020, 55.6, 76.3, 76.3, AQUA),
    ("CAD · CHASE+B seed 4", 1020, 56.5, 81.2, 81.2, AQUA),
    ("CAD · CHASE+B seed 5", 1020, 58.5, 87.0, 87.0, AQUA),
]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
y = np.arange(len(rows))[::-1]
for yi, (lab, U, lo, med, hi, col) in zip(y, rows):
    ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
    ax.scatter([med], [yi], s=34, color=col, ec="white", lw=1, zorder=3)
    ax.plot([U, U], [yi - .22, yi + .22], color=INK2, lw=1)
ax.axvline(8, color=GRAY, lw=1, ls=":")
ax.text(8 * 1.08, len(rows) - 0.55, "hard floor $\\kappa=8$", fontsize=7.5, color=INK2, va="center")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xscale("log"); ax.set_xlim(0.7, 5000)
ax.set_xlabel("Effective support $N_{\\mathrm{eff}}$ (log scale; dot = median, bar = 5th–95th percentile, tick = $U$)")
ax.grid(axis="y", visible=False)
save(fig, "fig2_neff")

# ----------------------------------------------------------------- Fig. 3 translation robustness
sig = [0, 0.5, 1, 2]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
ax = axs[0]
for s in ([0, 6.83, 8.67, 15.11], [0, 27.6, 35.61, 46.52], [0, 6.83, 8.67, 15.77]):
    ax.plot(sig, s, color=BLUE, lw=1, alpha=.45)
ax.plot(sig, [0, 5.65, 10.12, 15.90], color=BLUE, marker="o", ms=4, label="CHASE, real joints moved")
ax.plot(sig, [0, 0, 1.18, 1.71], color=GRAY, ls="--", marker="o", ms=3.5, label="CHASE, padding moved too")
ax.plot([0, 1, 2], [0, 0, 0], color=AQUA, marker="s", ms=4, label="CHASE+ / CHASE+B (all seeds)")
ax.text(2.04, 46.5, "seed 2", fontsize=7.5, color=BLUE, va="center")
ax.set_title("(a) CAD", loc="left"); ax.set_xlabel("Translation $\\sigma$ (RMS radii)")
ax.set_ylabel("Prediction flips (%)"); ax.set_xticks(sig)
ax.legend(loc="upper left", handlelength=1.6)
ax = axs[1]
ax.plot(sig, [0, 0.49, 1.47, 3.93], color=BLUE, marker="o", ms=4, label="CHASE, real joints moved")
ax.plot(sig, [0, 0, 0, 0], color=GRAY, ls="--", marker="o", ms=3.5, label="CHASE, padding moved too")
ax.set_title("(b) NTU Mutual 11", loc="left"); ax.set_xlabel("Translation $\\sigma$ (RMS radii)")
ax.set_xticks(sig); ax.legend(loc="upper left", handlelength=1.6)
fig.tight_layout(w_pad=1.0)
save(fig, "fig3_translation")

# ----------------------------------------------------------------- Fig. 4 permutation decomposition
groups = ["Full permutation", "Origin only\n(CHASE component)", "Slots only\n(backbone component)"]
cad = [42.62, 38.24, 32.63]; ntu = [1.01, 0.00, 1.01]
fig, ax = plt.subplots(figsize=(5.2, 2.3))
x = np.arange(3); w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, cad, w, color=BLUE, label="CAD (released CHASE)")
b2 = ax.bar(x + w / 2 + 0.01, ntu, w, color=ORANGE, label="NTU Mutual 11 (released CHASE)")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.8, f"{r.get_height():.1f}",
                ha="center", va="bottom", fontsize=7.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("Prediction flips (%)")
ax.set_ylim(0, 50); ax.grid(axis="x", visible=False); ax.legend(loc="upper right")
save(fig, "fig4_permutation")

# ----------------------------------------------------------------- Fig. 5 per-seed accuracy
arms = ["Vanilla", "CHASE", "Fixed origin", "CHASE+", "CHASE+B"]
acc = {"Vanilla": [58.43, 67.06, 66.54], "CHASE": [68.37, 63.79, 81.96, 63.79, 55.82],
       "Fixed origin": [73.33, 72.81, 78.30, 74.77, 76.34], "CHASE+": [73.73, 71.63, 81.05, 80.13, 85.23],
       "CHASE+B": [69.80, 73.33, 80.78, 85.75, 84.58]}
fig, ax = plt.subplots(figsize=(5.2, 2.7))
for i, a in enumerate(arms):
    v = np.array(acc[a]); xs = i + np.linspace(-0.14, 0.14, len(v))
    ax.scatter(xs, v, s=22, color=BLUE, alpha=.55, ec="white", lw=.6, zorder=3)
    m, sd = v.mean(), v.std(ddof=1)
    ax.plot([i - .26, i + .26], [m, m], color=INK, lw=2, solid_capstyle="round", zorder=4)
    ax.plot([i, i], [m - sd, m + sd], color=INK, lw=1, zorder=4)
    ax.text(i + .3, m, f"{m:.1f}", fontsize=7.5, color=INK, va="center")
ax.axhline(89.61, color=GRAY, ls=":", lw=1)
ax.text(4.45, 89.61, "CHASE reported (89.6)", fontsize=7.5, color=INK2, va="bottom", ha="right")
ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms)
ax.set_ylabel("Best test accuracy (%)"); ax.set_ylim(50, 93); ax.grid(axis="x", visible=False)
save(fig, "fig5_accuracy")
print("done")

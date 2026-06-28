"""
SnapUGC Dataset Visualization Script
Generates publication-quality figures for README and thesis.
Style: clean academic, white background — matching the original SnapUGC paper.

Figures:
  1. dataset_samples.png      – 2×4 grid of sample video frames (paper style)
  2. ecr_distribution.png     – ECR histogram + KDE (train vs test) + boxplot
  3. dataset_overview.png     – Quality-band bar chart, CDF, percentiles, stats table
  4. ecr_quality_bands.png    – Per-band distribution detail
"""

import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "official_5k_split"
VIDEO_DIR = ROOT / "data" / "official_balanced_5000_videos"
ASSETS    = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train_4000.csv"
TEST_CSV  = DATA_DIR / "test_1000.csv"
ALL_CSV   = DATA_DIR / "split_all_5000.csv"

# ── Academic colour palette (matches paper aesthetic) ──────────────────────
C_TRAIN   = "#3A86FF"   # steel blue  – train
C_TEST    = "#FF6B6B"   # coral red   – test
C_LOW     = "#E07B39"   # amber       – low quality band
C_MED     = "#F4A935"   # gold        – medium quality band
C_HIGH    = "#2D9E6B"   # green       – high quality band
C_ACCENT  = "#6C63FF"   # purple      – accent / "All"
C_GRAY    = "#888888"

# ── Global matplotlib style (clean white, paper-ready) ────────────────────
plt.rcParams.update({
    # Background
    "figure.facecolor":       "white",
    "axes.facecolor":         "white",
    # Spine / border
    "axes.edgecolor":         "#CCCCCC",
    "axes.linewidth":         0.8,
    # Grid
    "axes.grid":              True,
    "grid.color":             "#E8E8E8",
    "grid.linestyle":         "-",
    "grid.linewidth":         0.6,
    "grid.alpha":             1.0,
    "axes.axisbelow":         True,
    # Ticks
    "xtick.color":            "#333333",
    "ytick.color":            "#333333",
    "xtick.labelsize":        9,
    "ytick.labelsize":        9,
    "xtick.direction":        "out",
    "ytick.direction":        "out",
    "xtick.major.size":       3.5,
    "ytick.major.size":       3.5,
    # Labels / text
    "axes.labelcolor":        "#222222",
    "axes.titlecolor":        "#111111",
    "text.color":             "#222222",
    "font.family":            "DejaVu Sans",
    "font.size":              10,
    "axes.titlesize":         11,
    "axes.labelsize":         10,
    # Legend
    "legend.framealpha":      0.9,
    "legend.edgecolor":       "#CCCCCC",
    "legend.fontsize":        9,
    # Lines
    "lines.linewidth":        1.8,
    "patch.linewidth":        0.8,
})


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    all_df   = pd.read_csv(ALL_CSV)
    return train_df, test_df, all_df


def ecr_band(ecr):
    if ecr < 0.33:  return "Low"
    if ecr < 0.67:  return "Medium"
    return "High"


def extract_thumbnail(video_path: Path, size=(320, 180)):
    if not HAS_CV2 or not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1  –  SAMPLE GRID  (12 images, 2×6, no title/caption/border)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sample_grid(all_df, n_cols=7, n_rows=2, seed=42):
    """
    2×6 grid of video thumbnails.
    Pure images only — no title, no caption, no border, no colour coding.
    White gap between cells, matching the paper's Fig. 1 style.
    """
    print("  [1/4] Generating sample grid …")

    n = n_cols * n_rows          # 12
    rng = random.Random(seed)

    # Pick 12 random samples (simple random, no ECR stratification needed)
    sampled = all_df.sample(n, random_state=seed).reset_index(drop=True)
    rows = [sampled.iloc[i] for i in range(n)]

    thumb_w, thumb_h = 280, 240
    dpi    = 150
    gap_px = 20         # white gap between cells in pixels

    fig_w_px = n_cols * thumb_w + (n_cols - 1) * gap_px
    fig_h_px = n_rows * thumb_h + (n_rows - 1) * gap_px

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w_px / dpi, fig_h_px / dpi),
        facecolor="white",
        gridspec_kw={
            "wspace": gap_px / thumb_w,
            "hspace": gap_px / thumb_h,
        },
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for ax, row in zip(axes.flat, rows):
        vid_path = VIDEO_DIR / Path(row["video_path"]).name
        thumb = extract_thumbnail(vid_path, size=(thumb_w, thumb_h))

        if thumb is None:
            thumb = np.full((thumb_h, thumb_w, 3), [100, 110, 130], dtype=np.uint8)

        ax.imshow(thumb, aspect="auto", interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)   # no border at all

    out = ASSETS / "dataset_samples.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0)
    plt.close(fig)
    print(f"     ✓  Saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2  –  ECR DISTRIBUTION  (grouped bar histogram, academic style)
# ══════════════════════════════════════════════════════════════════════════════

def plot_ecr_distribution(train_df, test_df):
    """
    Single clean bar-chart: evenly-spaced bins across [0, 1].
    Train and Test bars shown side-by-side with quality-band shading.
    """
    print("  [2/4] Generating ECR distribution plot …")

    n_bins   = 20                              # 20 equal bins → each bin = 0.05
    bins     = np.linspace(0, 1, n_bins + 1)
    bw       = bins[1] - bins[0]              # bin width = 0.05
    centers  = (bins[:-1] + bins[1:]) / 2

    train_ecr = train_df["ECR"].dropna().values
    test_ecr  = test_df["ECR"].dropna().values

    # Count (not density) so y-axis is interpretable as number of videos
    train_counts, _ = np.histogram(train_ecr, bins=bins)
    test_counts,  _ = np.histogram(test_ecr,  bins=bins)

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.12)

    half_bw = bw * 0.44
    ax.bar(centers - half_bw / 2, train_counts, width=half_bw * 0.96,
           color=C_TRAIN, alpha=0.82, zorder=3)
    ax.bar(centers + half_bw / 2, test_counts,  width=half_bw * 0.96,
           color=C_TEST,  alpha=0.82, zorder=3)

    # Quality-band background shading
    band_defs = [
        (0.00, 0.33, C_LOW,  "Low\nQuality"),
        (0.33, 0.67, C_MED,  "Medium\nQuality"),
        (0.67, 1.00, C_HIGH, "High\nQuality"),
    ]
    for x0, x1, col, _ in band_defs:
        ax.axvspan(x0, x1, alpha=0.07, color=col, zorder=0)

    # Dashed dividers at quality boundaries
    for xv, col in [(0.33, C_LOW), (0.67, C_HIGH)]:
        ax.axvline(xv, color=col, linewidth=1.1, linestyle="--", alpha=0.6, zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

    # Mean-ECR dotted vertical lines
    for ecr_vals, col in [(train_ecr, C_TRAIN), (test_ecr, C_TEST)]:
        ax.axvline(ecr_vals.mean(), color=col, linewidth=1.4,
                   linestyle=":", alpha=0.85, zorder=4)

    ax.set_xlabel("ECR Score")
    ax.set_ylabel("Number of Videos")
    # Fewer x-ticks — one label every 0.1 to avoid crowding
    ax.set_xticks(np.arange(0.0, 1.05, 0.10))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="x", labelsize=9, rotation=0)
    ax.set_title(
        "ECR Score Distribution — SnapUGC-5K  (Train vs. Test)",
        fontsize=12, fontweight="bold", pad=8,
    )

    legend_handles = [
        mpatches.Patch(color=C_TRAIN, alpha=0.82,
                       label=f"Train  (n=4,000,  μ={train_ecr.mean():.3f})"),
        mpatches.Patch(color=C_TEST,  alpha=0.82,
                       label=f"Test   (n=1,000,  μ={test_ecr.mean():.3f})"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.9, loc="upper left")

    out = ASSETS / "ecr_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"     ✓  Saved → {out}")
    return out





# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3  –  DATASET OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def plot_dataset_overview(train_df, test_df, all_df):
    print("  [3/4] Generating dataset overview …")

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        wspace=0.38, hspace=0.50,
        left=0.08, right=0.97,
        top=0.91, bottom=0.10,
    )

    fig.suptitle(
        "SnapUGC-5K Dataset Overview",
        fontsize=14, fontweight="bold", y=0.97,
    )

    # ── (a) Pie: split sizes ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sizes  = [len(train_df), len(test_df)]
    labels = [f"Train  ({len(train_df):,})", f"Test  ({len(test_df):,})"]
    wedges, _, auts = ax1.pie(
        sizes, labels=labels,
        autopct="%1.1f%%",
        colors=[C_TRAIN, C_TEST],
        startangle=90,
        wedgeprops=dict(linewidth=1.2, edgecolor="white"),
        textprops=dict(fontsize=9),
        pctdistance=0.68,
    )
    for at in auts:
        at.set_fontweight("bold")
        at.set_fontsize(10)
    ax1.set_title("(a)  Split Proportion", pad=8)

    # ── (b) Grouped bar: quality bands ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    band_labels = ["Low\n(<0.33)", "Medium\n(0.33–0.67)", "High\n(>0.67)"]
    train_b = [
        (train_df["ECR"] <  0.33).sum(),
        ((train_df["ECR"] >= 0.33) & (train_df["ECR"] < 0.67)).sum(),
        (train_df["ECR"] >= 0.67).sum(),
    ]
    test_b = [
        (test_df["ECR"] <  0.33).sum(),
        ((test_df["ECR"] >= 0.33) & (test_df["ECR"] < 0.67)).sum(),
        (test_df["ECR"] >= 0.67).sum(),
    ]
    x = np.arange(3)
    w = 0.38
    b1 = ax2.bar(x - w/2, train_b, w, color=C_TRAIN, alpha=0.80,
                 label="Train", zorder=3)
    b2 = ax2.bar(x + w/2, test_b,  w, color=C_TEST,  alpha=0.80,
                 label="Test",  zorder=3)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 8,
                     f"{h:,}", ha="center", va="bottom",
                     fontsize=8, color="#333333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_labels, fontsize=9)
    ax2.set_ylabel("Count")
    ax2.set_title("(b)  Quality Band Distribution", pad=8)
    ax2.legend(fontsize=9)

    # ── (c) CDF ───────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for df, lbl, col in [(train_df, "Train", C_TRAIN),
                         (test_df,  "Test",  C_TEST)]:
        s   = np.sort(df["ECR"].dropna().values)
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax3.plot(s, cdf, color=col, linewidth=2.0, label=lbl)
    ax3.axvline(0.33, color=C_LOW,  linewidth=1.0, linestyle="--", alpha=0.7)
    ax3.axvline(0.67, color=C_HIGH, linewidth=1.0, linestyle="--", alpha=0.7)
    ax3.set_xlabel("ECR Score")
    ax3.set_ylabel("Cumulative Probability")
    ax3.set_title("(c)  Cumulative Distribution (CDF)", pad=8)
    ax3.legend(fontsize=9)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # ── (d) ECR percentile bar chart ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    pcts = [10, 25, 50, 75, 90]
    tp   = np.percentile(train_df["ECR"].dropna(), pcts)
    ep   = np.percentile(test_df["ECR"].dropna(), pcts)
    xp   = np.arange(len(pcts))
    ax4.bar(xp - 0.2, tp, 0.38, color=C_TRAIN, alpha=0.80, label="Train", zorder=3)
    ax4.bar(xp + 0.2, ep, 0.38, color=C_TEST,  alpha=0.80, label="Test",  zorder=3)
    for xi, tv, ev in zip(xp, tp, ep):
        ax4.text(xi - 0.2, tv + 0.008, f"{tv:.3f}", ha="center", va="bottom",
                 fontsize=7.5, color="#333333")
        ax4.text(xi + 0.2, ev + 0.008, f"{ev:.3f}", ha="center", va="bottom",
                 fontsize=7.5, color="#333333")
    ax4.set_xticks(xp)
    ax4.set_xticklabels([f"P{p}" for p in pcts], fontsize=9)
    ax4.set_ylabel("ECR Score")
    ax4.set_ylim(0, 1.05)
    ax4.set_title("(d)  ECR Percentiles", pad=8)
    ax4.legend(fontsize=9)

    # ── (e) Summary stats table ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis("off")

    def row_stats(df, name):
        ecr  = df["ECR"].dropna()
        n    = len(df)
        low  = (ecr < 0.33).sum()
        med  = ((ecr >= 0.33) & (ecr < 0.67)).sum()
        high = (ecr >= 0.67).sum()
        return [
            name,
            f"{n:,}",
            f"{ecr.mean():.4f}",
            f"{ecr.std():.4f}",
            f"{ecr.min():.4f}",
            f"{ecr.max():.4f}",
            f"{ecr.median():.4f}",
            f"{low:,}  ({low/n*100:.1f}%)",
            f"{med:,}  ({med/n*100:.1f}%)",
            f"{high:,}  ({high/n*100:.1f}%)",
        ]

    col_hdrs = [
        "Split", "Count",
        "Mean", "Std", "Min", "Max", "Median",
        "Low\n(<0.33)", "Med\n(0.33–0.67)", "High\n(>0.67)",
    ]
    tbl_data = [
        row_stats(train_df, "Train"),
        row_stats(test_df,  "Test"),
        row_stats(all_df,   "All"),
    ]

    tbl = ax5.table(
        cellText=tbl_data,
        colLabels=col_hdrs,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.1)

    split_colors = [C_TRAIN, C_TEST, C_ACCENT]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(fontweight="bold", color="#111111", fontsize=8.5)
        else:
            cell.set_facecolor("white" if r % 2 == 1 else "#FAFAFA")
            cell.set_text_props(color="#333333")
            if c == 0:
                cell.set_text_props(color=split_colors[r - 1],
                                    fontweight="bold")

    ax5.set_title("(e)  Summary Statistics", pad=10)

    out = ASSETS / "dataset_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"     ✓  Saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4  –  ECR QUALITY BAND DETAIL
# ══════════════════════════════════════════════════════════════════════════════

def plot_ecr_quality_bands(train_df, test_df):
    print("  [4/4] Generating ECR quality band detail …")

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.5),
        facecolor="white",
        sharey=False,
    )
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97,
                        top=0.88, bottom=0.15)
    fig.suptitle(
        "ECR Distribution by Quality Band — Train vs. Test",
        fontsize=12, fontweight="bold", y=0.97,
    )

    configs = [
        ("(a)  Low Quality  (ECR < 0.33)",   (0.00, 0.33), C_LOW),
        ("(b)  Medium Quality  (0.33–0.67)", (0.33, 0.67), C_MED),
        ("(c)  High Quality  (ECR > 0.67)",  (0.67, 1.00), C_HIGH),
    ]

    for ax, (title, (lo, hi), bc) in zip(axes, configs):
        bins = np.linspace(lo, hi, 21)
        bw   = bins[1] - bins[0]

        for df, lbl, col in [
            (train_df, "Train", C_TRAIN),
            (test_df,  "Test",  C_TEST),
        ]:
            sub = df[(df["ECR"] >= lo) & (df["ECR"] < hi)]["ECR"].dropna().values
            if len(sub) == 0:
                continue
            cts, edges = np.histogram(sub, bins=bins, density=True)
            cen = (edges[:-1] + edges[1:]) / 2
            ax.bar(cen, cts, width=bw * 0.85,
                   color=col, alpha=0.28, zorder=2, linewidth=0)

            if HAS_SCIPY and len(sub) >= 5:
                kde = gaussian_kde(sub, bw_method=0.14)
                xk  = np.linspace(lo, hi, 300)
                yk  = kde(xk)
                ax.plot(xk, yk, color=col, linewidth=2.0,
                        label=f"{lbl}  (n={len(sub):,}, μ={np.mean(sub):.3f})",
                        zorder=4)
            else:
                ax.plot(cen, cts, color=col, linewidth=2.0,
                        label=f"{lbl}  (n={len(sub):,})", zorder=4)

        ax.set_title(title, pad=7, fontsize=10)
        ax.set_xlabel("ECR Score")
        ax.set_ylabel("Density")
        ax.set_xlim(lo, hi)
        ax.legend(fontsize=8.5)

        # Colour the top spine to indicate band
        ax.spines["top"].set_edgecolor(bc)
        ax.spines["top"].set_linewidth(2.5)
        ax.spines["right"].set_visible(False)

    out = ASSETS / "ecr_quality_bands.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"     ✓  Saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SnapUGC Dataset Visualization  (academic style)")
    print("=" * 60)

    train_df, test_df, all_df = load_data()
    print(f"  Loaded: train={len(train_df)}, test={len(test_df)}, all={len(all_df)}")
    print()

    outputs = [
        plot_sample_grid(all_df),
        plot_ecr_distribution(train_df, test_df),
        plot_dataset_overview(train_df, test_df, all_df),
        plot_ecr_quality_bands(train_df, test_df),
    ]

    print()
    print("=" * 60)
    print("  Done. Figures saved to:  assets/")
    for p in outputs:
        print(f"    • {p.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
10_pca_scatter.py

Publication-quality PCA scatter plot of learned drug embeddings.

Points are coloured by cluster (k=15). Named pharmacological drugs are
labelled with leader lines. Three annotation boxes highlight the key
findings: kinase cluster (C5), triptan co-cluster (C9), and SE-only
cluster (C10).

Outputs:
    results/pca_scatter.png   (300 dpi, suitable for thesis)
    results/pca_scatter.svg   (vector, suitable for LaTeX)

Requirements:
    pip install matplotlib adjustText

Usage:
    python 10_pca_scatter.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       9,
    "axes.linewidth":  0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

RES = Path("results")
RES.mkdir(exist_ok=True)
RAW = Path("data/raw")

# ── cluster colour palette (15 clusters) ──────────────────────────────────────
# C5  (kinase)   → teal/green,  highlighted
# C9  (triptans) → coral/red,   highlighted
# C10 (SE-only)  → amber,       highlighted
# rest           → muted blues/purples/greys

CLUSTER_COLORS = {
    0:  "#9B97D4",   # muted purple
    1:  "#7EB8E8",   # muted blue
    2:  "#88CBB5",   # muted teal
    3:  "#B0ABCD",   # lavender
    4:  "#6DAEDB",   # medium blue
    5:  "#1D9E75",   # HIGHLIGHT — kinase (teal-green)
    6:  "#94C4DE",   # pale blue
    7:  "#AAAAAA",   # grey
    8:  "#C4BCDC",   # pale purple
    9:  "#D85A30",   # HIGHLIGHT — triptans (coral)
    10: "#E8A020",   # HIGHLIGHT — SE-only (amber)
    11: "#8BBBD4",   # steel blue
    12: "#CCCCCC",   # light grey (tiny cluster)
    13: "#B8A090",   # warm grey — catch-all
    14: "#A0B8A0",   # sage
}

CLUSTER_LABELS = {
    0:  "C0",   1:  "C1",   2:  "C2",   3:  "C3",   4:  "C4",
    5:  "C5 — kinase inhibitors",
    6:  "C6",   7:  "C7",   8:  "C8",
    9:  "C9 — triptans",
    10: "C10 — SE-only (no targets)",
    11: "C11",  12: "C12",  13: "C13 — catch-all",  14: "C14",
}

# ── drugs to label on the plot ─────────────────────────────────────────────────
# Format: CID -> (display_name, label_offset_x, label_offset_y)
# Offsets are in data coordinates; tune to avoid overlaps.
LABELED_DRUGS = {
    # ── kinase inhibitors (C5) ─────────────────────────────────────────────
    "CID000002987":  ("Imatinib",         0.20, -0.18),
    "CID000005493":  ("Erlotinib",         0.20, -0.18),
    "CID000005329102":("Sorafenib",        0.20,  0.18),
    "CID000216239":  ("Lapatinib",         0.20,  0.18),
    # ── triptans (C9) ─────────────────────────────────────────────────────
    "CID000077992":  ("Sumatriptan",      -0.25,  0.22),
    "CID000004440":  ("Rizatriptan",       0.20,  0.18),
    "CID000077993":  ("Zolmitriptan",      0.20, -0.18),
    # ── SE-only cluster (C10) ─────────────────────────────────────────────
    "CID000005593":  ("Tolterodine",      -0.28, -0.20),
    # ── cholinergic / serotonergic drugs (scattered) ──────────────────────
    "CID000001065":  ("Atropine",          0.20,  0.18),
    "CID000003494":  ("Scopolamine",      -0.28,  0.18),
    "CID000003661":  ("Clozapine",         0.20,  0.18),
    "CID000003042":  ("Olanzapine",        0.20,  0.18),
    "CID000124087":  ("Clomipramine",      0.20, -0.20),
    "CID000004601":  ("Imipramine",        0.20,  0.18),
    "CID000002370":  ("Benztropine",      -0.28, -0.20),
    "CID000002725":  ("Chlorpromazine",   -0.28, -0.20),
}

# ── annotation boxes for key findings ────────────────────────────────────────
ANNOTATIONS = [
    {
        "text":  "Kinase cluster (C5)\nn=66 drugs\np(kinase activity)=10⁻²³²",
        "xy":    (0.55, -1.5),    # arrow tip (data coords)
        "xytext":(2.2,  -2.7),    # text box position
        "color": "#1D9E75",
    },
    {
        "text":  "Triptan co-cluster (C9)\nSumatriptan + Rizatriptan\nshared HTR1A/1B/1D targets",
        "xy":    (-1.8,  3.5),
        "xytext":(-0.5,  4.5),
        "color": "#D85A30",
    },
    {
        "text":  "SE-only cluster (C10)\nn=19 drugs, 16 with no\nprotein targets",
        "xy":    (-3.8, -1.0),
        "xytext":(-2.2, -2.8),
        "color": "#E8A020",
    },
]


# ── data loaders ───────────────────────────────────────────────────────────────

def load_data():
    # Cluster assignments and node indices
    drug_cluster = {}
    node_idx     = {}
    with open(RES / "drug_clusters.csv") as f:
        for row in csv.DictReader(f):
            drug_cluster[row["drug_id"]] = int(row["cluster_id"])
            node_idx[row["drug_id"]]     = int(row["node_idx"])

    # PCA embeddings
    pca = np.load(RES / "drug_embeddings_pca2d.npy")

    # Has protein targets flag
    has_targets = {}
    cb_path = RES / "channel_balance.csv"
    if cb_path.exists():
        with open(cb_path) as f:
            for row in csv.DictReader(f):
                has_targets[row["drug_id"]] = row["has_protein_targets"] == "True"

    return drug_cluster, node_idx, pca, has_targets


# ── main plotting function ─────────────────────────────────────────────────────

def make_scatter(drug_cluster, node_idx, pca, has_targets):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAF8")

    # ── 1. scatter all drugs by cluster ───────────────────────────────────
    # Separate highlighted clusters so they render on top
    highlight_clusters = {5, 9, 10}
    background_ids = []
    foreground_ids = []

    for drug_id, c in drug_cluster.items():
        idx = node_idx.get(drug_id)
        if idx is None or idx >= len(pca):
            continue
        if c in highlight_clusters:
            foreground_ids.append(drug_id)
        else:
            background_ids.append(drug_id)

    # Background clusters — smaller, more transparent
    for drug_id in background_ids:
        idx   = node_idx[drug_id]
        c     = drug_cluster[drug_id]
        color = CLUSTER_COLORS.get(c, "#BBBBBB")
        ax.scatter(pca[idx, 0], pca[idx, 1],
                   c=color, s=18, alpha=0.55,
                   edgecolors="none", zorder=2,
                   rasterized=True)

    # Foreground clusters — larger, fully opaque
    cluster_marker = {5: "o", 9: "^", 10: "s"}
    cluster_size   = {5: 38,  9: 45,  10: 42}
    for drug_id in foreground_ids:
        idx    = node_idx[drug_id]
        c      = drug_cluster[drug_id]
        color  = CLUSTER_COLORS[c]
        marker = cluster_marker.get(c, "o")
        size   = cluster_size.get(c, 38)
        ax.scatter(pca[idx, 0], pca[idx, 1],
                   c=color, s=size, alpha=0.92,
                   marker=marker,
                   edgecolors="white", linewidths=0.5,
                   zorder=4, rasterized=True)

    # ── 2. labelled drugs ─────────────────────────────────────────────────
    for drug_id, (name, dx, dy) in LABELED_DRUGS.items():
        idx = node_idx.get(drug_id)
        if idx is None:
            continue
        c      = drug_cluster.get(drug_id, -1)
        color  = CLUSTER_COLORS.get(c, "#555555")
        x, y   = pca[idx, 0], pca[idx, 1]
        xt, yt = x + dx, y + dy

        # Draw dot highlight
        ax.scatter(x, y, c=color, s=55, zorder=6,
                   edgecolors="white", linewidths=0.8)

        # Leader line
        ax.annotate(
            "",
            xy=(x, y), xytext=(xt, yt),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.7,
                connectionstyle="arc3,rad=0.0",
            ),
            zorder=5,
        )

        # Text label with white stroke for readability
        ax.text(
            xt, yt, name,
            fontsize=7.5, color=color,
            ha="center", va="center",
            fontweight="medium",
            path_effects=[
                pe.withStroke(linewidth=2.2, foreground="white")
            ],
            zorder=7,
        )

    # ── 3. finding annotation boxes ──────────────────────────────────────
    for ann in ANNOTATIONS:
        ax.annotate(
            ann["text"],
            xy=ann["xy"],
            xytext=ann["xytext"],
            fontsize=7.5,
            color=ann["color"],
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=ann["color"],
                linewidth=1.0,
                alpha=0.92,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=ann["color"],
                lw=0.9,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=8,
        )

    # ── 4. axes and labels ────────────────────────────────────────────────
    ax.set_xlabel("PC1", fontsize=10, labelpad=6)
    ax.set_ylabel("PC2", fontsize=10, labelpad=6)
    ax.set_title(
        "Drug embedding space — PCA projection of learned HGT representations\n"
        "Coloured by k=15 cluster assignment",
        fontsize=10, fontweight="bold", pad=10,
    )
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.18, linewidth=0.4, color="#CCCCCC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)

    # ── 5. legend ─────────────────────────────────────────────────────────
    # Two sections: highlighted clusters + "other clusters" single entry
    legend_handles = []

    # Highlighted clusters
    for c, marker, label_suffix in [
        (5,  "o", "— kinase inhibitors"),
        (9,  "^", "— triptans"),
        (10, "s", "— SE-only (no protein targets)"),
    ]:
        h = Line2D([0], [0],
                   marker=marker, color="w",
                   markerfacecolor=CLUSTER_COLORS[c],
                   markeredgecolor="white",
                   markeredgewidth=0.5,
                   markersize=7,
                   label=f"C{c} {label_suffix}")
        legend_handles.append(h)

    # Spacer
    legend_handles.append(
        Line2D([0], [0], color="none", label="")
    )

    # Other clusters as a single representative entry
    legend_handles.append(
        Line2D([0], [0],
               marker="o", color="w",
               markerfacecolor="#9B97D4",
               markeredgecolor="none",
               markersize=6,
               alpha=0.7,
               label="C0–C4, C6–C8, C11–C14 (GPCR / other)")
    )

    # Has-targets indicator
    legend_handles.append(
        Line2D([0], [0], color="none", label="")
    )
    legend_handles.append(
        Line2D([0], [0],
               marker="o", color="w",
               markerfacecolor="#888888",
               markeredgecolor="white",
               markeredgewidth=0.8,
               markersize=7,
               label="Named drugs (labelled)")
    )

    ax.legend(
        handles=legend_handles,
        fontsize=7.5,
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        handletextpad=0.5,
        labelspacing=0.4,
    )

    # ── 6. drug count annotation ──────────────────────────────────────────
    ax.text(
        0.01, 0.01,
        f"n = {len(drug_cluster)} drugs  ·  k = 15 clusters",
        transform=ax.transAxes,
        fontsize=7.5, color="#999999",
        va="bottom", ha="left",
    )

    plt.tight_layout()
    return fig


# ── save ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    drug_cluster, node_idx, pca, has_targets = load_data()

    print(f"  {len(drug_cluster)} drugs  |  {len(set(drug_cluster.values()))} clusters")
    print(f"  PCA shape: {pca.shape}")

    print("Generating scatter plot...")
    fig = make_scatter(drug_cluster, node_idx, pca, has_targets)

    png_path = RES / "pca_scatter.png"
    svg_path = RES / "pca_scatter.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    print(f"  Saved PNG → {png_path}")
    print(f"  Saved SVG → {svg_path}")
    print("Done.")


if __name__ == "__main__":
    main()

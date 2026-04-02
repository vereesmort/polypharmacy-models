"""
09_cholinergic_serotonergic_fingerprint.py

Three-panel figure illustrating the finding:
  "20 known cholinergic/serotonergic drugs share a CHRM/HTR target fingerprint
   yet disperse across 9 clusters — the model distinguished pharmacological
   sub-families rather than grouping the entire class."

Panel A  —  Drug × gene target matrix
Panel B  —  PCA scatter (opacity = SE-channel fraction)
Panel C  —  Cluster distribution + mean SE-fraction overlay

Outputs (all in results/):
    fingerprint_combined.png      (requires matplotlib >= 3.5)

Usage:
    python 09_cholinergic_serotonergic_fingerprint.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW = Path("data/raw")
RES = Path("results")
RES.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Drug roster                                                                  #
# --------------------------------------------------------------------------- #
DRUGS = {
    "CID000001065": ("Atropine",         "anticholinergic"),
    "CID000004819": ("Pirenzepine",      "anticholinergic"),
    "CID000002955": ("Mianserin",        "antidepressant"),
    "CID000003494": ("Scopolamine",      "anticholinergic"),
    "CID000003661": ("Clozapine",        "antipsychotic"),
    "CID000002370": ("Benztropine",      "anticholinergic"),
    "CID000004034": ("Trihexyphenidyl",  "anticholinergic"),
    "CID000072054": ("Darifenacin",      "anticholinergic"),
    "CID000005572": ("Procyclidine",     "anticholinergic"),
    "CID000005184": ("Biperiden",        "anticholinergic"),
    "CID000003746": ("Oxybutynin",       "anticholinergic"),
    "CID000003042": ("Olanzapine",       "antipsychotic"),
    "CID000124087": ("Clomipramine",     "TCA antidepressant"),
    "CID000002725": ("Chlorpromazine",   "antipsychotic"),
    "CID000005593": ("Tolterodine",      "anticholinergic"),
    "CID000004601": ("Imipramine",       "TCA antidepressant"),
    "CID000004634": ("Doxepin",          "TCA antidepressant"),
    "CID000077992": ("Sumatriptan",      "triptan"),
    "CID000004440": ("Rizatriptan",      "triptan"),
    "CID000077993": ("Zolmitriptan",     "triptan"),
}

GENE_IDS = [
    "1128","1129","1131","1132","1133",  # CHRM1-5
    "3350","3351","3352",                 # HTR1A/1B/1D
    "3356","3357","3358","3362","6532",   # HTR2A/2B/2C, HTR6, SERT
]
GENE_NAMES = [
    "CHRM1","CHRM2","CHRM3","CHRM4","CHRM5",
    "HTR1A","HTR1B","HTR1D",
    "HTR2A","HTR2B","HTR2C","HTR6","SERT",
]
GENE_FAMILIES = [
    ("CHRM (muscarinic)",  0,  4, "#534AB7"),
    ("HTR1 (5-HT1)",       5,  7, "#D85A30"),
    ("HTR2 / HTR6 / SERT", 8, 12, "#1D9E75"),
]

CLASS_COLOR = {
    "anticholinergic":    "#534AB7",
    "antipsychotic":      "#0F6E56",
    "TCA antidepressant": "#BA7517",
    "antidepressant":     "#BA7517",
    "triptan":            "#D85A30",
}
CLASS_MARKER = {
    "anticholinergic":    "s",
    "antipsychotic":      "D",
    "TCA antidepressant": "^",
    "antidepressant":     "^",
    "triptan":            "o",
}
CLUSTER_LABEL = {
    1:"C1", 2:"C2", 3:"C3", 5:"C5",
    6:"C6", 9:"C9 (triptans)", 10:"C10", 11:"C11", 14:"C14",
}


# --------------------------------------------------------------------------- #
# Data loaders                                                                 #
# --------------------------------------------------------------------------- #

def load_targets():
    dp = defaultdict(set)
    with open(RAW / "bio-decagon-targets.csv") as f:
        for row in csv.DictReader(f):
            dp[row["STITCH"]].add(row["Gene"])
    return dp


def load_clusters_pca():
    cluster  = {}
    node_idx = {}
    with open(RES / "drug_clusters.csv") as f:
        for row in csv.DictReader(f):
            cluster[row["drug_id"]]  = int(row["cluster_id"])
            node_idx[row["drug_id"]] = int(row["node_idx"])
    pca = np.load(RES / "drug_embeddings_pca2d.npy")
    return cluster, node_idx, pca


def load_channel_balance():
    cb   = {}
    path = RES / "channel_balance.csv"
    if not path.exists():
        return cb
    with open(path) as f:
        for row in csv.DictReader(f):
            cb[row["drug_id"]] = float(row["se_fraction"])
    return cb


def build_records(dp, cluster, node_idx, pca, channel_balance):
    records = []
    for did, (name, cls) in DRUGS.items():
        targets  = dp.get(did, set())
        fam_hits = [gid in targets for gid in GENE_IDS]
        idx      = node_idx.get(did)
        x, y     = (float(pca[idx, 0]), float(pca[idx, 1])) if idx is not None else (0.0, 0.0)
        records.append({
            "id":       did,
            "name":     name,
            "cls":      cls,
            "cluster":  cluster.get(did, -1),
            "x":        x,
            "y":        y,
            "fam_hits": fam_hits,
            "n_fam":    sum(fam_hits),
            "se_frac":  channel_balance.get(did, 1.0),
        })
    return records


# --------------------------------------------------------------------------- #
# Panel A — target matrix                                                      #
# --------------------------------------------------------------------------- #

def panel_a(records, ax):
    import matplotlib.patches as mpatches

    n_drugs = len(records)
    n_genes = len(GENE_NAMES)

    for i, rec in enumerate(records):
        for j in range(n_genes):
            fam_color = next(c for (_, s, e, c) in GENE_FAMILIES if s <= j <= e)
            fc  = fam_color if rec["fam_hits"][j] else "none"
            ec  = "#D0CEC8"
            lw  = 0.4
            rect = mpatches.FancyBboxPatch(
                (j - 0.42, i - 0.38), 0.84, 0.76,
                boxstyle="round,pad=0.05",
                facecolor=fc, edgecolor=ec, linewidth=lw, alpha=0.85,
            )
            ax.add_patch(rect)

    ax.set_xlim(-0.6, n_genes - 0.4)
    ax.set_ylim(-0.6, n_drugs - 0.4)
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(GENE_NAMES, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(n_drugs))
    ax.set_yticklabels(
        [f"{r['name']}  " for r in records],
        fontsize=8,
        color=[CLASS_COLOR.get(r["cls"], "#888") for r in records],
    )
    ax.tick_params(length=0)
    ax.set_title("A — CHRM / HTR target fingerprint", fontsize=9,
                 fontweight="bold", pad=30)

    # Family brackets
    for label, start, end, color in GENE_FAMILIES:
        mid = (start + end) / 2
        ax.annotate("", xy=(end + 0.4, n_drugs + 0.3),
                    xytext=(start - 0.4, n_drugs + 0.3),
                    xycoords="data",
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.5),
                    annotation_clip=False)
        ax.text(mid, n_drugs + 0.65, label, ha="center", va="bottom",
                fontsize=7, color=color, fontweight="bold",
                transform=ax.transData, clip_on=False)

    # Family dividers
    for _, _, end, _ in GENE_FAMILIES[:-1]:
        ax.axvline(end + 0.5, color="#E0DED6", lw=0.8, zorder=0)

    handles = [mpatches.Patch(color=c, label=k)
               for k, c in CLASS_COLOR.items() if k != "antidepressant"]
    ax.legend(handles=handles, fontsize=7, loc="lower right",
              bbox_to_anchor=(1.02, -0.36), ncol=2, frameon=False)

    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")


# --------------------------------------------------------------------------- #
# Panel B — PCA scatter                                                        #
# --------------------------------------------------------------------------- #

def panel_b(records, ax):
    import matplotlib.patheffects as pe
    import matplotlib.lines as mlines

    for rec in records:
        col    = CLASS_COLOR.get(rec["cls"], "#888")
        marker = CLASS_MARKER.get(rec["cls"], "o")
        # Higher SE fraction → more opaque (model leans on SE channel)
        alpha  = 0.3 + 0.7 * rec["se_frac"]
        ax.scatter(rec["x"], rec["y"],
                   c=col, marker=marker, s=55,
                   alpha=alpha, edgecolors="white", linewidths=0.8, zorder=3)

    # Selective labels
    labeled = {"Sumatriptan", "Rizatriptan", "Tolterodine",
               "Clomipramine", "Benztropine", "Imipramine", "Clozapine"}
    for rec in records:
        if rec["name"] not in labeled:
            continue
        col   = CLASS_COLOR.get(rec["cls"], "#888")
        off_x = 0.18 if rec["x"] > 0 else -0.18
        ha    = "left" if rec["x"] > 0 else "right"
        ax.text(rec["x"] + off_x, rec["y"], rec["name"],
                fontsize=6.5, color=col, ha=ha, va="center",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    # Cluster centroids
    centroids = defaultdict(lambda: {"xs": [], "ys": []})
    for r in records:
        centroids[r["cluster"]]["xs"].append(r["x"])
        centroids[r["cluster"]]["ys"].append(r["y"])
    for c, v in centroids.items():
        mx  = float(np.mean(v["xs"]))
        my  = float(np.mean(v["ys"]))
        lbl = CLUSTER_LABEL.get(c, f"C{c}")
        ax.text(mx, my + 0.25, lbl, fontsize=6.5, color="#B4B2A9",
                ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.set_title("B — Embedding space (PCA)\nOpacity ∝ SE-channel fraction",
                 fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.18, linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [mlines.Line2D([], [], color=CLASS_COLOR[cls], marker=mk,
                             linestyle="None", markersize=6, label=cls)
               for cls, mk in CLASS_MARKER.items() if cls != "antidepressant"]
    ax.legend(handles=handles, fontsize=6.5, loc="lower left", frameon=False)


# --------------------------------------------------------------------------- #
# Panel C — cluster distribution + SE fraction                                #
# --------------------------------------------------------------------------- #

def panel_c(records, ax):
    cluster_drugs = defaultdict(list)
    for r in records:
        cluster_drugs[r["cluster"]].append(r)

    sorted_clusters = sorted(cluster_drugs.items(), key=lambda x: -len(x[1]))
    labels   = [CLUSTER_LABEL.get(c, f"C{c}") for c, _ in sorted_clusters]
    se_fracs = [float(np.mean([r["se_frac"] for r in v])) for _, v in sorted_clusters]

    x     = np.arange(len(labels))
    bar_w = 0.55

    class_order = ["anticholinergic", "antipsychotic",
                   "TCA antidepressant", "triptan", "antidepressant"]
    bottoms = np.zeros(len(labels))
    for cls in class_order:
        heights = np.array([sum(1 for r in v if r["cls"] == cls)
                            for _, v in sorted_clusters], dtype=float)
        if heights.sum() == 0:
            continue
        ax.bar(x, heights, bar_w, bottom=bottoms,
               color=CLASS_COLOR.get(cls, "#888"), alpha=0.85,
               label=cls if cls != "antidepressant" else None)
        bottoms += heights

    # SE-fraction overlay
    ax2 = ax.twinx()
    ax2.plot(x, se_fracs, "o--", color="#E24B4A", linewidth=1.2,
             markersize=4, zorder=5)
    for xi, sf in zip(x, se_fracs):
        ax2.text(xi, sf + 0.04, f"{sf:.0%}", ha="center", fontsize=6.5,
                 color="#E24B4A")
    ax2.set_ylim(0, 1.25)
    ax2.set_ylabel("mean SE-channel fraction", fontsize=7.5, color="#E24B4A")
    ax2.tick_params(axis="y", labelcolor="#E24B4A", labelsize=7)
    ax2.spines[["top", "left"]].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("number of drugs", fontsize=8)
    ax.set_ylim(0, max(len(v) for _, v in sorted_clusters) + 2)
    ax.tick_params(labelsize=7)
    ax.set_title("C — Cluster distribution\n(9 clusters, mean SE-fraction overlay)",
                 fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("Loading data...")
    dp              = load_targets()
    cluster, node_idx, pca = load_clusters_pca()
    channel_balance = load_channel_balance()

    print("Building drug records...")
    records = build_records(dp, cluster, node_idx, pca, channel_balance)

    # Summary table
    print()
    print(f"{'Drug':<20} {'Class':<20} {'C':>3}  {'Fam':>4}  "
          f"{'SE%':>6}  Family targets")
    print("─" * 85)
    for r in records:
        genes = [GENE_NAMES[i] for i, h in enumerate(r["fam_hits"]) if h]
        print(f"{r['name']:<20} {r['cls']:<20} {r['cluster']:>3}  "
              f"{r['n_fam']:>4}  {r['se_frac']*100:>5.1f}%  {','.join(genes)}")

    n_clusters = len({r["cluster"] for r in records})
    print(f"\nDispersed across {n_clusters} clusters.")
    print(f"Mean SE-channel fraction: {np.mean([r['se_frac'] for r in records]):.1%}")

    print("\nGenerating figure...")
    fig = plt.figure(figsize=(18, 7.5))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)

    panel_a(records, fig.add_subplot(gs[0]))
    panel_b(records, fig.add_subplot(gs[1]))
    panel_c(records, fig.add_subplot(gs[2]))

    fig.text(
        0.5, -0.04,
        "Finding: 20 cholinergic/serotonergic drugs share a CHRM/HTR target fingerprint "
        "but spread across 9 clusters. The model distinguishes pharmacological sub-families "
        "(triptans → C9, anticholinergics → multiple GPCR clusters, TCAs → C6/C11) rather than "
        "grouping the entire class. SE-channel fraction >70% for most drugs confirms that "
        "combination side-effect co-occurrence is the primary embedding signal.",
        ha="center", va="top", fontsize=8, color="#5F5E5A",
    )

    out = RES / "fingerprint_combined.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

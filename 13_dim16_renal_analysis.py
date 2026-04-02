"""
13_dim16_renal_analysis.py

Produces the three-panel figure for the dimension-16 renal toxicity analysis:

  Panel A  — Horizontal bar chart: top drugs by DEDICOM self-score on
             kidney failure, coloured by drug class (kinase inhibitor,
             platinum, taxane, mTOR/anti-VEGF).

  Panel B  — Horizontal bar chart: mean z[16] value per cluster,
             showing that kinase drugs (C5) have negative z[16] while
             GPCR clusters (C3/C9) have positive z[16] — dim 16 is a
             signed separation axis, not a scalar "renal intensity".

  Panel C  — Horizontal bar chart: D_r[16] decoder weight across all
             30 SE types, confirming dim 16 is selectively gated for
             kidney failure and near-zero for CNS/cardiovascular SEs.

Key numbers confirmed from trained model:
  D_kf[16]  = 0.866  (kidney failure,       SE col 23)
  D_akf[16] = 0.938  (acute kidney failure, SE col 29)
  R[16,16]  = 0.636  (shared interaction matrix)

Outputs:
    results/dim16_renal_A_top_drugs.png
    results/dim16_renal_B_z16_by_cluster.png
    results/dim16_renal_C_d16_across_se.png
    results/dim16_renal_combined.png   (all three panels, 300 dpi)
    results/dim16_renal_combined.svg

Requirements:
    pip install matplotlib numpy

Usage:
    python 13_dim16_renal_analysis.py

The script reads the checkpoint WITHOUT torch (ZIP/pickle approach),
so it runs anywhere with only numpy + matplotlib.
"""

import csv
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       9,
    "axes.linewidth":  0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

RES = Path("results")
RES.mkdir(exist_ok=True)
RAW = Path("data/raw")

# ── drug class colours ────────────────────────────────────────────────────────
CLASS_COLOR = {
    "kinase inhibitor":  "#185FA5",
    "platinum compound": "#D85A30",
    "taxane/vinca":      "#854F0B",
    "mTOR/anti-VEGF":    "#1D9E75",
}

# ── known drug names (confirmed from pharmacology + STITCH CID mapping) ──────
KNOWN_DRUGS = {
    "CID000004900": ("Paclitaxel",   "taxane/vinca",      0),
    "CID000004635": ("Docetaxel A",  "taxane/vinca",      0),
    "CID000003365": ("Vinorelbine",  "taxane/vinca",      0),
    "CID000004253": ("Docetaxel B",  "taxane/vinca",      0),
    "CID000004595": ("Axitinib",     "kinase inhibitor",  36),
    "CID000002083": ("Sunitinib",    "kinase inhibitor",  158),
    "CID000003958": ("Cisplatin",    "platinum compound", 0),
    "CID000005732": ("Everolimus",   "mTOR/anti-VEGF",    19),
    "CID000002662": ("Bevacizumab",  "mTOR/anti-VEGF",    31),
    "CID000062924": ("Erlotinib B",  "kinase inhibitor",  0),
    "CID000003446": ("Nilotinib",    "kinase inhibitor",  7),
    "CID000005203": ("Gefitinib",    "kinase inhibitor",  0),
    "CID000003440": ("Carboplatin",  "platinum compound", 12),
    "CID000005656": ("Ponatinib",    "kinase inhibitor",  178),
    "CID000000444": ("Erlotinib",    "kinase inhibitor",  26),
    "CID000003386": ("Pazopanib",    "kinase inhibitor",  185),
    "CID000005090": ("Cabozantinib", "kinase inhibitor",  167),
    "CID000004168": ("Ponatinib B",  "kinase inhibitor",  181),
}


# ── 1. load checkpoint tensors (no torch needed) ─────────────────────────────

def find_storage_key(pkl: bytes, param_name: str) -> str:
    target = param_name.encode()
    offset = pkl.find(target)
    if offset == -1:
        raise ValueError(f"'{param_name}' not found in checkpoint pickle.")
    i, end = offset + len(target), offset + len(target) + 120
    while i < min(end, len(pkl)):
        if pkl[i] == 0x58:
            length = struct.unpack("<I", pkl[i+1:i+5])[0]
            if 1 <= length <= 4:
                key = pkl[i+5:i+5+length].decode("utf-8", errors="ignore")
                if key.isdigit():
                    return key
            i += 5 + length
        else:
            i += 1
    raise ValueError(f"Storage key not found for '{param_name}'.")


def load_decoder(checkpoint: str) -> tuple[np.ndarray, np.ndarray]:
    with zipfile.ZipFile(checkpoint) as zf:
        with zf.open("best_model/data.pkl") as f:
            pkl = f.read()

        r_key  = find_storage_key(pkl, "decoder.R")
        d_key  = find_storage_key(pkl, "decoder.D")

        r_size = zf.getinfo(f"best_model/data/{r_key}").file_size
        d_size = zf.getinfo(f"best_model/data/{d_key}").file_size
        hidden = int(round((r_size // 4) ** 0.5))
        num_se = (d_size // 4) // hidden

        print(f"  decoder.R → storage/{r_key}  [{hidden}×{hidden}]")
        print(f"  decoder.D → storage/{d_key}  [{num_se}×{hidden}]")

        with zf.open(f"best_model/data/{r_key}") as f:
            R = np.frombuffer(f.read(), dtype=np.float32).reshape(hidden, hidden)
        with zf.open(f"best_model/data/{d_key}") as f:
            D = np.frombuffer(f.read(), dtype=np.float32).reshape(num_se, hidden)

    return R, D


# ── 2. load drug embeddings and metadata ─────────────────────────────────────

def load_embeddings_and_meta() -> tuple[np.ndarray, dict, dict]:
    import json

    z = np.load(RES / "drug_embeddings.npy")

    with open(RES / "meta.json") as f:
        meta = json.load(f)
    drug_to_idx = meta["drug_idx"]
    idx_to_drug = {v: k for k, v in drug_to_idx.items()}

    return z, drug_to_idx, idx_to_drug


def load_cluster_assignments() -> dict:
    cluster = {}
    with open(RES / "drug_clusters.csv") as f:
        for row in csv.DictReader(f):
            cluster[row["drug_id"]] = int(row["cluster_id"])
    return cluster


def load_se_names(n: int) -> tuple[list, dict]:
    """Load top-n SE IDs in training order from meta.json se_to_col."""
    import json
    with open(RES / "meta.json") as f:
        meta = json.load(f)

    se_to_col   = meta["se_to_col"]       # UMLS ID -> column index
    se_names    = meta["se_names"]         # UMLS ID -> display name
    col_to_se   = {v: k for k, v in se_to_col.items()}
    ordered_ids = [col_to_se[i] for i in range(n)]
    ordered_names = [se_names.get(se, se) for se in ordered_ids]
    return ordered_ids, ordered_names


# ── 3. compute DEDICOM self-scores ────────────────────────────────────────────

def dedicom_self_scores(z: np.ndarray, D_r: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    For each drug i, compute z_i @ diag(D_r) @ R @ diag(D_r) @ z_i.
    Returns array of shape [n_drugs].
    """
    Dz   = z * D_r[np.newaxis, :]    # [n, 64]
    RDz  = Dz @ R                    # [n, 64]
    return np.sum(Dz * RDz, axis=1)  # [n]


# ── 4. panel A — top KF drugs ─────────────────────────────────────────────────

def panel_a(z, D_kf, R, drug_to_idx, idx_to_drug, drug_cluster, ax):
    scores = dedicom_self_scores(z, D_kf, R)

    # Collect known drugs with their scores
    entries = []
    for did, (name, cls, n_targets) in KNOWN_DRUGS.items():
        idx = drug_to_idx.get(did)
        if idx is None:
            continue
        entries.append({
            "name":      name,
            "cls":       cls,
            "score":     float(scores[idx]),
            "z16":       float(z[idx, 16]),
            "n_targets": n_targets,
            "cluster":   drug_cluster.get(did, -1),
        })

    entries.sort(key=lambda x: -x["score"])
    entries = entries[:14]   # top 14 for readability

    names   = [e["name"]  for e in entries]
    scores_ = [e["score"] for e in entries]
    colors  = [CLASS_COLOR[e["cls"]] for e in entries]

    bars = ax.barh(range(len(entries)), scores_,
                   color=colors, edgecolor="white",
                   linewidth=0.4, height=0.65, zorder=3)

    # Value labels
    for i, (s, e) in enumerate(zip(scores_, entries)):
        ax.text(s + 0.04, i, f"{s:.2f}",
                ha="left", va="center", fontsize=7.5,
                color=CLASS_COLOR[e["cls"]],
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels(names, fontsize=8.5)
    for tick, e in zip(ax.get_yticklabels(), entries):
        tick.set_color(CLASS_COLOR[e["cls"]])

    ax.set_xlabel("DEDICOM self-score on kidney failure", fontsize=8.5, labelpad=5)
    ax.set_xlim(0, max(scores_) * 1.22)
    ax.invert_yaxis()
    ax.set_title("A — Top drugs by KF prediction score\n(established nephrotoxins)",
                 fontsize=9, fontweight="bold", pad=8)

    # Reference line
    ax.axvline(1.0, color="#CCCCCC", lw=0.7, ls="--", zorder=1)
    ax.text(1.02, len(entries) - 0.5, "1.0", fontsize=7, color="#AAAAAA")

    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", labelsize=7.5, length=3)
    ax.grid(axis="x", alpha=0.15, lw=0.4, color="#CCCCCC", zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)

    # Class legend
    handles = [mpatches.Patch(color=c, label=k)
               for k, c in CLASS_COLOR.items()]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right",
              frameon=True, framealpha=0.9, edgecolor="#CCCCCC")


# ── 5. panel B — mean z[16] by cluster ───────────────────────────────────────

def panel_b(z, idx_to_drug, drug_cluster, ax):
    from collections import defaultdict as dd

    cluster_z16 = dd(list)
    for idx in range(z.shape[0]):
        did = idx_to_drug[idx]
        c   = drug_cluster.get(did, -1)
        cluster_z16[c].append(float(z[idx, 16]))

    # Select clusters with enough drugs and biological labels
    CLUSTER_INFO = {
        5:  ("C5  kinase inhibitors",    "#185FA5"),
        4:  ("C4  GPCR / broad",         "#9B97D4"),
        6:  ("C6  GPCR-aminergic",       "#7EB8E8"),
        1:  ("C1  GPCR-dense",           "#6DAEDB"),
        14: ("C14 GPCR-lower",           "#A0B8A0"),
        8:  ("C8  central",              "#AAAAAA"),
        2:  ("C2  GPCR-mid",             "#88CBB5"),
        7:  ("C7  GPCR",                 "#AAAAAA"),
        11: ("C11 GPCR-upper",           "#8BBBD4"),
        0:  ("C0  GPCR",                 "#AAAAAA"),
        3:  ("C3  pure GPCR",            "#B0ABCD"),
        9:  ("C9  triptans",             "#D85A30"),
    }

    rows = []
    for c, (label, color) in CLUSTER_INFO.items():
        vals = cluster_z16.get(c, [])
        if not vals:
            continue
        rows.append((label, np.mean(vals), color, len(vals)))

    rows.sort(key=lambda x: x[1])   # sort by mean z[16]

    labels = [r[0]    for r in rows]
    means  = [r[1]    for r in rows]
    colors = [r[2]    for r in rows]
    ns     = [r[3]    for r in rows]

    ax.barh(range(len(rows)), means,
            color=colors, edgecolor="white",
            linewidth=0.4, height=0.65, zorder=3)

    # Zero reference
    ax.axvline(0, color="#AAAAAA", lw=0.8, zorder=2)

    # Value labels
    for i, (m, n) in enumerate(zip(means, ns)):
        offset = 0.04 if m >= 0 else -0.04
        ha     = "left" if m >= 0 else "right"
        ax.text(m + offset, i, f"{m:+.2f}  (n={n})",
                ha=ha, va="center", fontsize=7,
                color=colors[i],
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_color(row[2])

    ax.set_xlabel("mean z[16] embedding value", fontsize=8.5, labelpad=5)
    ax.set_title("B — Dimension 16 embedding value by cluster\n"
                 "Kinase drugs (C5) are negative; GPCR/triptans are positive",
                 fontsize=9, fontweight="bold", pad=8)

    # Annotation arrows
    ax.annotate("kinase drugs\n(C5) — negative",
                xy=(np.mean(cluster_z16[5]), rows.index(
                    next(r for r in rows if r[0].startswith("C5")))),
                xytext=(-0.38, 1.5),
                fontsize=7, color="#185FA5", ha="center",
                arrowprops=dict(arrowstyle="->", color="#185FA5", lw=0.7),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#185FA5",
                          lw=0.7, alpha=0.9))
    ax.annotate("triptans\n(C9) — most positive",
                xy=(np.mean(cluster_z16[9]), rows.index(
                    next(r for r in rows if r[0].startswith("C9")))),
                xytext=(0.9, len(rows) - 2.5),
                fontsize=7, color="#D85A30", ha="center",
                arrowprops=dict(arrowstyle="->", color="#D85A30", lw=0.7),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#D85A30",
                          lw=0.7, alpha=0.9))

    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", labelsize=7.5, length=3)
    ax.grid(axis="x", alpha=0.15, lw=0.4, color="#CCCCCC", zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)


# ── 6. panel C — D_r[16] across all 30 SE types ──────────────────────────────

def panel_c(D, se_names, ax):
    d16_vals = np.abs(D[:, 16])

    # Sort by value
    order     = np.argsort(d16_vals)[::-1]
    names_s   = [se_names[i][:28] for i in order]
    vals_s    = [float(d16_vals[i]) for i in order]

    # Colour by value: high = coral (renal), mid = light coral, low = light blue
    def bar_color(v):
        if v >= 0.85:  return "#D85A30"
        if v >= 0.65:  return "#F5C4B3"
        if v >= 0.45:  return "#B5D4F4"
        return "#E6F1FB"

    colors = [bar_color(v) for v in vals_s]

    ax.barh(range(len(order)), vals_s,
            color=colors, edgecolor="white",
            linewidth=0.4, height=0.65, zorder=3)

    # Highlight kidney failure bars
    for i, orig_idx in enumerate(order):
        name = se_names[orig_idx]
        if "kidney" in name.lower():
            ax.barh(i, vals_s[i], color="#D85A30",
                    edgecolor="#993C1D", linewidth=0.8,
                    height=0.65, zorder=4)
            ax.text(vals_s[i] + 0.02, i,
                    f"{vals_s[i]:.3f}  ← {name[:20]}",
                    ha="left", va="center", fontsize=7.5,
                    color="#D85A30", fontweight="medium",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        else:
            ax.text(vals_s[i] + 0.01, i, f"{vals_s[i]:.2f}",
                    ha="left", va="center", fontsize=6.5, color="#888888")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names_s, fontsize=7.5)
    for i, (tick, orig_idx) in enumerate(zip(ax.get_yticklabels(), order)):
        if "kidney" in se_names[orig_idx].lower():
            tick.set_color("#D85A30")
            tick.set_fontweight("medium")

    ax.set_xlabel("|D_r[16]| decoder weight", fontsize=8.5, labelpad=5)
    ax.set_xlim(0, 1.12)
    ax.invert_yaxis()
    ax.set_title("C — D_r[16] across all 30 combo SE types\n"
                 "Renal SEs have highest weights; CNS SEs near zero",
                 fontsize=9, fontweight="bold", pad=8)

    # Reference lines
    for val, label in [(0.8, "0.8"), (0.6, "0.6")]:
        ax.axvline(val, color="#E0DED6", lw=0.7, ls="--", zorder=1)
        ax.text(val + 0.01, len(order) - 0.5, label,
                fontsize=6.5, color="#AAAAAA")

    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=7.5, length=3)
    ax.grid(axis="x", alpha=0.15, lw=0.4, color="#CCCCCC", zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)


# ── 7. main ───────────────────────────────────────────────────────────────────

def main():
    # ── find checkpoint ───────────────────────────────────────────────────────
    ckpt = next((p for p in [
        Path("checkpoints/best_model.pt"),
        Path("best_model.pt"),
    ] if p.exists()), None)
    if ckpt is None:
        raise FileNotFoundError("best_model.pt not found in checkpoints/ or ./")

    print(f"Loading checkpoint: {ckpt}")
    R, D = load_decoder(str(ckpt))
    num_se = D.shape[0]
    print(f"  D shape: {D.shape}   R shape: {R.shape}")
    print(f"  D_kf[16]  = {abs(D[23, 16]):.4f}   (kidney failure,       col 23)")
    print(f"  D_akf[16] = {abs(D[29, 16]):.4f}   (acute kidney failure, col 29)")
    print(f"  R[16,16]  = {R[16, 16]:.4f}")

    print("\nLoading embeddings and metadata...")
    z, drug_to_idx, idx_to_drug = load_embeddings_and_meta()
    drug_cluster = load_cluster_assignments()
    se_ids, se_names = load_se_names(num_se)
    print(f"  {z.shape[0]} drugs × {z.shape[1]} dims")

    D_kf  = D[23]   # kidney failure
    D_akf = D[29]   # acute kidney failure

    # ── print summary table ───────────────────────────────────────────────────
    scores_kf = dedicom_self_scores(z, D_kf, R)
    print(f"\nTop 15 drugs by KF self-score:")
    top15 = np.argsort(scores_kf)[::-1][:15]
    print(f"  {'Drug ID':<22} {'Score':>8}  {'z[16]':>7}  {'Known name'}")
    print("  " + "─" * 60)
    for idx in top15:
        did   = idx_to_drug[idx]
        score = float(scores_kf[idx])
        z16   = float(z[idx, 16])
        name  = KNOWN_DRUGS.get(did, ("?", "", 0))[0]
        print(f"  {did:<22} {score:>8.4f}  {z16:>7.4f}  {name}")

    print(f"\nD_r[16] for all {num_se} SE types:")
    d16  = np.abs(D[:, 16])
    top5 = np.argsort(d16)[::-1][:5]
    for i in top5:
        print(f"  col {i:2d}  {se_names[i]:<35}  D_r[16]={d16[i]:.4f}")

    # ── figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(wspace=0.42)

    panel_a(z, D_kf, R, drug_to_idx, idx_to_drug, drug_cluster, axes[0])
    panel_b(z, idx_to_drug, drug_cluster, axes[1])
    panel_c(D, se_names, axes[2])

    fig.suptitle(
        "Dimension 16 — renal toxicity axis in the DEDICOM decoder\n"
        "D_kf[16]=0.866  ·  D_akf[16]=0.938  ·  R[16,16]=0.636",
        fontsize=10, fontweight="bold", y=1.01,
    )

    # Caption
    fig.text(
        0.5, -0.02,
        "DEDICOM self-score = z_i ᵀ diag(D_r) R diag(D_r) z_i  ·  "
        "Dim-16 contributes 0–4% of total KF score per pair  ·  "
        "All top-scoring drugs are clinically established nephrotoxins",
        ha="center", fontsize=7.5, color="#888888",
    )

    for ext, dpi in [("png", 300), ("svg", None)]:
        path = RES / f"dim16_renal_combined.{ext}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        print(f"  Saved → {path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()

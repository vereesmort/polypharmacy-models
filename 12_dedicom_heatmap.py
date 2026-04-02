"""
12_dedicom_heatmap.py

Publication-quality heatmap of the DEDICOM decoder's D_r weights.

Rows  = 30 most frequent combo side-effect types
Cols  = 15 most discriminative embedding dimensions
       (ranked by variance of |D_r[dim]| across the 30 SE types)
Cells = |D_r[dim]| — the importance of embedding dimension `dim`
        for predicting side-effect type `r`

Design choices:
  - Each column is independently min-max normalised so within-column
    contrast is maximised regardless of absolute scale differences.
  - Rows are sorted by their primary dimension (argmax of |D_r|),
    grouping pharmacologically similar SE types together.
  - Row labels are coloured by pharmacological group.
  - Column headers show the raw dimension index.
  - Three columns are outlined (d15, d16, d55) — the clearest
    single-dimension clusters discussed in the thesis.
  - A colour bar legend is shown on the right.
  - Disease-class annotations are added as a strip on the left.

Outputs:
    results/dedicom_heatmap.png   (300 dpi)
    results/dedicom_heatmap.svg   (vector)

Requirements:
    pip install matplotlib numpy

Usage:
    python 12_dedicom_heatmap.py
"""

import csv
import struct
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       9,
    "axes.linewidth":  0.5,
})

RES = Path("results")
RES.mkdir(exist_ok=True)
RAW = Path("data/raw")

# ── checkpoint path ────────────────────────────────────────────────────────────
CHECKPOINT = Path("checkpoints/best_model.pt")
NUM_SE     = 30
HIDDEN_DIM = 64
TOP_N_DIMS = 15   # number of most discriminative dims to show

# ── pharmacological group colours for row labels ───────────────────────────────
DIM_GROUPS = {
    15: ("CNS / cardiovascular",       "#534AB7"),
    16: ("Renal toxicity",             "#D85A30"),
    55: ("Haematological",             "#1D9E75"),
    41: ("Circulatory / GI",           "#BA7517"),
    10: ("Respiratory",                "#185FA5"),
     5: ("Metabolic",                  "#993556"),
    57: ("Systemic / constitutional",  "#5F5E5A"),
    32: ("Fever / thermoregulation",   "#D85A30"),
    30: ("Pain",                       "#888780"),
    17: ("Pleuritic / pain",           "#888780"),
    23: ("Respiratory (dyspnoea)",     "#185FA5"),
    28: ("CNS / confusion",            "#534AB7"),
    45: ("GI",                         "#BA7517"),
    54: ("Abdominal",                  "#BA7517"),
    43: ("Oedema",                     "#888780"),
     4: ("CNS (headache)",             "#534AB7"),
}
DEFAULT_GROUP_COLOR = "#888780"

# Columns to draw an outline around (key thesis findings)
HIGHLIGHT_COLS = {16: "#D85A30", 15: "#534AB7", 55: "#1D9E75"}

# ── disease-class strip colours ────────────────────────────────────────────────
DISEASE_COLORS = {
    "unannotated":               "#E8E6E0",
    "hematopoietic system disease":  "#9FE1CB",
    "urinary system disease":    "#F5C4B3",
    "cardiovascular system disease": "#AFA9EC",
    "respiratory system disease":    "#85B7EB",
    "acquired metabolic disease":    "#FAC775",
}


# ── data extraction ────────────────────────────────────────────────────────────

def extract_D(checkpoint: Path, num_se: int, hidden_dim: int) -> np.ndarray:
    """Extract decoder D matrix from PyTorch checkpoint without torch."""
    with zipfile.ZipFile(checkpoint) as zf:
        with zf.open("best_model/data.pkl") as f:
            pkl = f.read()

        def find_storage_key(name: str) -> str:
            offset = pkl.find(name.encode())
            if offset == -1:
                raise ValueError(f"Key '{name}' not found in checkpoint.")
            i = offset + len(name)
            while i < offset + 120:
                if pkl[i] == 0x58:
                    length = struct.unpack("<I", pkl[i + 1:i + 5])[0]
                    if 1 <= length <= 4:
                        key = pkl[i + 5:i + 5 + length].decode("utf-8", errors="ignore")
                        if key.isdigit():
                            return key
                    i += 5 + length
                else:
                    i += 1
            raise ValueError(f"Storage key for '{name}' not found.")

        d_key = find_storage_key("decoder.D")
        with zf.open(f"best_model/data/{d_key}") as f:
            D = np.frombuffer(f.read(), dtype=np.float32).reshape(num_se, hidden_dim)

    return D


def load_se_metadata(num_se: int):
    """Return (se_ids, se_names, disease_categories) for the top-N SE types."""
    se_counts   = Counter()
    se_names    = {}
    combo_files = sorted(RAW.glob("bio-decagon-combo*.csv"))
    if not combo_files:
        raise FileNotFoundError(
            f"No combo CSV files found in {RAW}. "
            "Expected files matching bio-decagon-combo*.csv"
        )
    for path in combo_files:
        with open(path) as f:
            for row in csv.DictReader(f):
                se_counts[row["Polypharmacy Side Effect"]] += 1
                se_names[row["Polypharmacy Side Effect"]] = row["Side Effect Name"]

    top_ids  = [se for se, _ in se_counts.most_common(num_se)]
    top_names = [se_names[se] for se in top_ids]

    se_category = {}
    cat_path = RAW / "bio-decagon-effectcategories.csv"
    if cat_path.exists():
        with open(cat_path) as f:
            for row in csv.DictReader(f):
                se_category[row["Side Effect"]] = row["Disease Class"]

    categories = [se_category.get(se, "unannotated") for se in top_ids]
    return top_ids, top_names, categories


# ── sorting ────────────────────────────────────────────────────────────────────

def sort_rows_by_primary_dim(D_abs: np.ndarray, se_names, categories):
    """
    Sort SE rows so that SEs sharing the same primary dimension are grouped.
    Within each primary-dim group, sort by descending weight on that dimension.
    """
    primary_dims = np.argmax(D_abs, axis=1)

    # Build group order: sort groups by the maximum weight in that group
    group_max = {}
    for i, pdim in enumerate(primary_dims):
        w = D_abs[i, pdim]
        if pdim not in group_max or w > group_max[pdim]:
            group_max[pdim] = w
    group_order = sorted(group_max.keys(), key=lambda d: -group_max[d])

    order = []
    for pdim in group_order:
        members = [(i, D_abs[i, pdim]) for i in range(len(se_names))
                   if primary_dims[i] == pdim]
        members.sort(key=lambda x: -x[1])
        order.extend(m[0] for m in members)

    return (
        order,
        [se_names[i]  for i in order],
        [categories[i] for i in order],
        [int(primary_dims[i]) for i in order],
    )


# ── plotting ───────────────────────────────────────────────────────────────────

def make_heatmap(D: np.ndarray, se_names, categories, top_dims):
    D_abs = np.abs(D)

    # Sort rows by primary dimension for visual grouping
    row_order, sorted_names, sorted_cats, primary_dims = \
        sort_rows_by_primary_dim(D_abs, se_names, categories)

    # Reorder D
    D_sorted = D_abs[row_order, :][:, top_dims]   # shape [30, 15]

    # Column-wise min-max normalisation for contrast
    col_min = D_sorted.min(axis=0, keepdims=True)
    col_max = D_sorted.max(axis=0, keepdims=True)
    D_norm  = (D_sorted - col_min) / (col_max - col_min + 1e-8)

    nRows, nCols = D_norm.shape

    # ── figure layout ──
    # Main heatmap + left disease strip + right colour bar
    fig = plt.figure(figsize=(11, 9.5))
    fig.patch.set_facecolor("white")

    # GridSpec: [disease strip | heatmap | gap | colorbar]
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        1, 4,
        figure=fig,
        width_ratios=[0.018, 1, 0.02, 0.045],
        wspace=0.03,
        left=0.28, right=0.96,
        top=0.92,  bottom=0.08,
    )

    ax_strip = fig.add_subplot(gs[0])   # disease category strip
    ax_main  = fig.add_subplot(gs[1])   # heatmap
    ax_cbar  = fig.add_subplot(gs[3])   # colour bar

    # ── custom blue colour map ──
    cmap = LinearSegmentedColormap.from_list(
        "blue_heatmap",
        ["#EFF5FC", "#C8DEF5", "#85B7EB", "#378ADD", "#185FA5", "#0C447C", "#042C53"],
    )

    # ── draw heatmap cells ──
    img = ax_main.imshow(
        D_norm,
        aspect="auto",
        cmap=cmap,
        vmin=0, vmax=1,
        interpolation="nearest",
    )

    # ── cell value annotations ──
    for i in range(nRows):
        for j in range(nCols):
            val      = D_sorted[i, j]
            norm_val = D_norm[i, j]
            txt_col  = "white" if norm_val > 0.60 else "#0C447C"
            ax_main.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=6.2, color=txt_col,
                fontweight="medium" if norm_val > 0.75 else "normal",
            )

    # ── column outlines for key findings ──
    for col_dim, outline_color in HIGHLIGHT_COLS.items():
        if col_dim in top_dims:
            j = top_dims.index(col_dim)
            rect = mpatches.FancyBboxPatch(
                (j - 0.5, -0.5), 1.0, nRows,
                boxstyle="square,pad=0",
                linewidth=1.8,
                edgecolor=outline_color,
                facecolor="none",
                zorder=5,
                clip_on=False,
            )
            ax_main.add_patch(rect)

            # Dim label above the column, coloured
            ax_main.text(
                j, -1.1, f"d{col_dim}",
                ha="center", va="bottom",
                fontsize=8, color=outline_color,
                fontweight="bold",
            )

    # ── column (x) axis: dimension labels ──
    ax_main.set_xticks(range(nCols))
    ax_main.set_xticklabels(
        [f"d{d}" for d in top_dims],
        fontsize=7.5, rotation=0,
    )
    ax_main.xaxis.set_tick_params(length=0)
    ax_main.xaxis.tick_top()
    ax_main.xaxis.set_label_position("top")
    ax_main.set_xlabel(
        "Embedding dimension  (top-15 by variance of |D_r| across 30 SE types)",
        fontsize=8.5, labelpad=8,
    )

    # ── row (y) axis: SE name labels, coloured by pharmacological group ──
    ax_main.set_yticks(range(nRows))
    ax_main.set_yticklabels([], fontsize=0)  # drawn manually for colour control

    for i, (name, pdim) in enumerate(zip(sorted_names, primary_dims)):
        _, col = DIM_GROUPS.get(pdim, ("other", DEFAULT_GROUP_COLOR))
        ax_main.text(
            -0.55, i, name,
            ha="right", va="center",
            fontsize=7.8, color=col,
            transform=ax_main.get_yaxis_transform(),
        )

    ax_main.tick_params(axis="both", length=0)

    # Light grid
    for i in range(nRows - 1):
        ax_main.axhline(i + 0.5, color="white", linewidth=0.6, zorder=3)
    for j in range(nCols - 1):
        ax_main.axvline(j + 0.5, color="white", linewidth=0.6, zorder=3)

    ax_main.set_xlim(-0.5, nCols - 0.5)
    ax_main.set_ylim(nRows - 0.5, -0.5)
    ax_main.spines[:].set_visible(False)

    # ── disease category strip ──
    ax_strip.set_xlim(0, 1)
    ax_strip.set_ylim(nRows - 0.5, -0.5)
    for i, cat in enumerate(sorted_cats):
        fc = DISEASE_COLORS.get(cat, "#E8E6E0")
        ax_strip.add_patch(
            mpatches.Rectangle((0, i - 0.5), 1, 1, color=fc)
        )
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    ax_strip.spines[:].set_visible(False)
    ax_strip.set_xlabel("", fontsize=0)

    # Disease strip title (rotated)
    ax_strip.text(
        0.5, -0.8, "Disease\nclass",
        ha="center", va="bottom",
        fontsize=6.5, color="#888888",
        transform=ax_strip.transData,
        rotation=0,
    )

    # ── colour bar ──
    cbar = fig.colorbar(img, cax=ax_cbar)
    cbar.set_label(
        "|D_r[dim]|\n(column-normalised)",
        fontsize=7.5, labelpad=6,
    )
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["min", "0.25", "0.50", "0.75", "max"])
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)

    # ── figure title ──
    fig.text(
        0.62, 0.96,
        "DEDICOM decoder — |D_r[dim]| weight heatmap\n"
        "Each cell: importance of embedding dimension for predicting that SE type",
        ha="center", va="bottom",
        fontsize=10, fontweight="bold",
    )

    # ── pharmacological group legend (row label colours) ──
    shown_groups = {}
    for pdim in primary_dims:
        label, col = DIM_GROUPS.get(pdim, ("other", DEFAULT_GROUP_COLOR))
        shown_groups[label] = col

    handles_rows = [
        mpatches.Patch(color=col, label=label)
        for label, col in shown_groups.items()
    ]
    legend_rows = fig.legend(
        handles=handles_rows,
        title="Primary decoder dim group\n(row label colour)",
        title_fontsize=7.5,
        fontsize=7,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        ncol=1,
        handlelength=1.2,
        handletextpad=0.6,
        labelspacing=0.4,
    )

    # ── disease strip legend ──
    handles_disease = [
        mpatches.Patch(color=col, label=cat)
        for cat, col in DISEASE_COLORS.items()
    ]
    fig.legend(
        handles=handles_disease,
        title="Disease class (left strip)",
        title_fontsize=7.5,
        fontsize=7,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        ncol=1,
        handlelength=1.2,
        handletextpad=0.6,
        labelspacing=0.4,
    )

    # ── column outline legend ──
    handles_col = [
        mpatches.Patch(
            facecolor="white",
            edgecolor=col,
            linewidth=1.5,
            label=f"d{dim} — {DIM_GROUPS.get(dim, ('?',''))[0]}",
        )
        for dim, col in HIGHLIGHT_COLS.items()
    ]
    fig.legend(
        handles=handles_col,
        title="Key finding columns",
        title_fontsize=7.5,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.92),
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        ncol=1,
        handlelength=1.4,
        handletextpad=0.6,
    )

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Extracting decoder weights...")
    D = extract_D(CHECKPOINT, NUM_SE, HIDDEN_DIM)
    print(f"  D shape: {D.shape}")

    print("Loading SE metadata...")
    top30_ids, top30_names, categories = load_se_metadata(NUM_SE)

    # Top-15 discriminative dims
    D_abs       = np.abs(D)
    var_per_dim = D_abs.var(axis=0)
    top15_dims  = np.argsort(var_per_dim)[::-1][:TOP_N_DIMS].tolist()
    print(f"  Top-{TOP_N_DIMS} dims: {top15_dims}")

    # Summary: primary dim and group per SE
    primary_dims = np.argmax(D_abs, axis=1)
    print()
    print(f"{'SE':<38} {'Top dim':>7}  {'Weight':>7}  Group")
    print("─" * 75)
    for i, name in enumerate(top30_names):
        pdim  = int(primary_dims[i])
        w     = float(D_abs[i, pdim])
        group = DIM_GROUPS.get(pdim, ("other", ""))[0]
        print(f"{name:<38} {'d'+str(pdim):>7}  {w:>7.3f}  {group}")

    print()
    print("Generating heatmap...")
    fig = make_heatmap(D, top30_names, categories, top15_dims)

    png_path = RES / "dedicom_heatmap.png"
    svg_path = RES / "dedicom_heatmap.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved PNG → {png_path}")
    print(f"  Saved SVG → {svg_path}")
    print("Done.")


if __name__ == "__main__":
    main()

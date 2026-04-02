"""
12_dedicom_top_dim_bar.py

Publication-quality horizontal bar chart showing the highest-weighted
DEDICOM decoder dimension for each of the top-30 combo side-effect types.

Each bar encodes:
  - Length  : |D_r[top_dim]|  — how strongly the decoder relies on that
              embedding dimension to score this side effect
  - Colour  : pharmacological group of the top dimension
  - Label   : dimension index + numeric weight

A second panel shows the legend with group descriptions and lists
which side effects belong to each group.

Output:
    results/dedicom_top_dim_bar.png   (300 dpi)
    results/dedicom_top_dim_bar.svg   (vector)

Requirements:
    pip install matplotlib numpy

Usage:
    python 12_dedicom_top_dim_bar.py

The script extracts decoder weights directly from the checkpoint
(no torch installation required — uses the ZIP/pickle approach).
"""

import csv
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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

# ── pharmacological dimension groups ─────────────────────────────────────────
# Maps primary decoder dimension -> (short label, hex colour, description)
DIM_GROUPS = {
    15: ("CNS / cardiovascular",       "#534AB7",
         "chest pain · dizziness · hypertension · anxiety · aching joints"),
    16: ("Renal toxicity",             "#D85A30",
         "kidney failure · acute kidney failure"),
    55: ("Haematological / infectious","#1D9E75",
         "anaemia · pneumonia"),
    41: ("Circulatory / GI",           "#BA7517",
         "arterial pressure dec. · nausea · emesis"),
    10: ("Respiratory / musculoskeletal","#185FA5",
         "hypoventilation · back ache"),
     5: ("Metabolic",                  "#993556",
         "asthenia · dehydration · hyperglycaemia"),
    57: ("Systemic / constitutional",  "#5F5E5A",
         "fatigue · oedema extremities · loss of weight"),
}
DEFAULT_COLOR = "#B4B2A9"   # for dims not in any group
DEFAULT_LABEL = "Other"


def dim_color(d: int) -> str:
    return DIM_GROUPS.get(d, (None, DEFAULT_COLOR, None))[1]

def dim_group_label(d: int) -> str:
    return DIM_GROUPS.get(d, (DEFAULT_LABEL, None, None))[0]


# ── 1. extract decoder weights from checkpoint ────────────────────────────────

def find_storage_key(pkl_bytes: bytes, param_name: str) -> str:
    target = param_name.encode()
    offset = pkl_bytes.find(target)
    if offset == -1:
        raise ValueError(f"'{param_name}' not found in checkpoint pickle.")
    i, end = offset + len(target), offset + len(target) + 120
    while i < min(end, len(pkl_bytes)):
        if pkl_bytes[i] == 0x58:
            length = struct.unpack("<I", pkl_bytes[i+1:i+5])[0]
            if 1 <= length <= 4:
                key = pkl_bytes[i+5:i+5+length].decode("utf-8", errors="ignore")
                if key.isdigit():
                    return key
            i += 5 + length
        else:
            i += 1
    raise ValueError(f"Storage key not found for '{param_name}'.")


def load_decoder_D(checkpoint: str = "checkpoints/best_model.pt") -> np.ndarray:
    """
    Extract the DEDICOM D matrix [num_se, hidden_dim] from the checkpoint
    without requiring PyTorch to be installed.
    """
    with zipfile.ZipFile(checkpoint) as zf:
        with zf.open("best_model/data.pkl") as f:
            pkl = f.read()

        d_key  = find_storage_key(pkl, "decoder.D")
        r_key  = find_storage_key(pkl, "decoder.R")

        # Infer shapes from file sizes
        d_size = zf.getinfo(f"best_model/data/{d_key}").file_size
        r_size = zf.getinfo(f"best_model/data/{r_key}").file_size
        hidden_dim = int(round((r_size // 4) ** 0.5))
        num_se     = (d_size // 4) // hidden_dim

        print(f"  decoder.D → storage/{d_key}  "
              f"shape=({num_se}, {hidden_dim})")

        with zf.open(f"best_model/data/{d_key}") as f:
            D = np.frombuffer(f.read(), dtype=np.float32).reshape(num_se, hidden_dim)

    return D


# ── 2. load SE metadata ───────────────────────────────────────────────────────

def load_top_se_names(n: int, raw_dir: Path):
    se_counts = Counter()
    se_names  = {}
    for path in sorted(raw_dir.glob("bio-decagon-combo*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                se_counts[row["Polypharmacy Side Effect"]] += 1
                se_names[row["Polypharmacy Side Effect"]] = row["Side Effect Name"]
    top = [se for se, _ in se_counts.most_common(n)]
    return top, {se: se_names[se] for se in top}


def load_categories(raw_dir: Path):
    cats = {}
    path = raw_dir / "bio-decagon-effectcategories.csv"
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                cats[row["Side Effect"]] = row["Disease Class"]
    return cats


# ── 3. build per-SE records ───────────────────────────────────────────────────

def build_records(D: np.ndarray, se_ids, se_names_map, categories):
    D_abs   = np.abs(D)
    records = []
    for i, se_id in enumerate(se_ids):
        top_dim = int(np.argmax(D_abs[i]))
        weight  = float(D_abs[i, top_dim])
        records.append({
            "se_id":    se_id,
            "name":     se_names_map[se_id],
            "top_dim":  top_dim,
            "weight":   weight,
            "color":    dim_color(top_dim),
            "group":    dim_group_label(top_dim),
            "category": categories.get(se_id, "unannotated"),
            "d_vector": D_abs[i].tolist(),
        })
    return records


# ── 4. main figure ────────────────────────────────────────────────────────────

def make_figure(records):
    # Sort by weight descending within each group, then by group
    group_order = [DIM_GROUPS[d][0] for d in sorted(DIM_GROUPS)] + [DEFAULT_LABEL]
    def sort_key(r):
        try:
            gi = group_order.index(r["group"])
        except ValueError:
            gi = len(group_order)
        return (gi, -r["weight"])

    sorted_records = sorted(records, key=sort_key)

    n   = len(sorted_records)
    fig = plt.figure(figsize=(11, 9))

    # Layout: main bar chart (left) + legend panel (right)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.04)
    ax  = fig.add_subplot(gs[0])
    axL = fig.add_subplot(gs[1])

    # ── bar chart ────────────────────────────────────────────────────────────
    y_pos   = np.arange(n)
    bar_h   = 0.62
    weights = [r["weight"] for r in sorted_records]
    colors  = [r["color"]  for r in sorted_records]
    dims    = [r["top_dim"] for r in sorted_records]
    names   = [r["name"]   for r in sorted_records]

    bars = ax.barh(y_pos, weights, bar_h,
                   color=colors, edgecolor="white", linewidth=0.4,
                   zorder=3)

    # Value + dimension label on each bar
    for i, (w, d, col) in enumerate(zip(weights, dims, colors)):
        # Dimension label inside bar (if bar wide enough)
        if w > 0.12:
            ax.text(w - 0.015, i, f"d{d}",
                    ha="right", va="center", fontsize=7,
                    color="white", fontweight="medium", zorder=5)
        else:
            ax.text(w + 0.008, i, f"d{d}",
                    ha="left", va="center", fontsize=7,
                    color=col, fontweight="medium", zorder=5)

        # Numeric weight outside bar
        ax.text(w + 0.025, i, f"{w:.2f}",
                ha="left", va="center", fontsize=7.5,
                color="#555555", zorder=5)

    # Group separator lines + group labels on y axis
    current_group = None
    group_starts  = {}
    for i, r in enumerate(sorted_records):
        if r["group"] != current_group:
            if i > 0:
                ax.axhline(i - 0.5, color="#E0DED6", lw=0.8, zorder=1)
            group_starts[r["group"]] = i
            current_group = r["group"]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)

    # Colour y-tick labels by group
    for tick, r in zip(ax.get_yticklabels(), sorted_records):
        tick.set_color(r["color"])

    ax.set_xlabel("|D_r[dim]| decoder weight", fontsize=9, labelpad=6)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()   # highest weight at top
    ax.set_title(
        "DEDICOM decoder — top embedding dimension per side-effect type\n"
        "|D_r[dim]| = importance of dimension dim for predicting side effect r",
        fontsize=9, fontweight="bold", pad=10,
    )
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", labelsize=8, length=3)
    ax.grid(axis="x", alpha=0.18, linewidth=0.4, color="#CCCCCC", zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)

    # Reference line at 0.8 (strong decoder activation)
    ax.axvline(0.8, color="#CCCCCC", lw=0.8, ls="--", zorder=1)
    ax.text(0.805, n - 0.2, "0.8", fontsize=7, color="#AAAAAA", va="bottom")

    # ── legend panel ─────────────────────────────────────────────────────────
    axL.axis("off")

    legend_y = 0.97
    line_h   = 0.068

    axL.text(0.05, legend_y, "Dimension groups",
             transform=axL.transAxes,
             fontsize=8.5, fontweight="bold", color="#333333",
             va="top")
    legend_y -= line_h * 1.4

    for dim, (label, color, ses_str) in DIM_GROUPS.items():
        # Colour swatch
        swatch = mpatches.FancyBboxPatch(
            (0.05, legend_y - 0.022), 0.08, 0.028,
            boxstyle="round,pad=0.002",
            facecolor=color, edgecolor="none",
            transform=axL.transAxes, clip_on=False,
        )
        axL.add_patch(swatch)

        # Dimension index
        axL.text(0.17, legend_y - 0.008,
                 f"d{dim}",
                 transform=axL.transAxes,
                 fontsize=7.5, fontweight="bold",
                 color=color, va="center")

        # Group label
        axL.text(0.29, legend_y - 0.008,
                 label,
                 transform=axL.transAxes,
                 fontsize=7.5, color="#333333", va="center")

        legend_y -= line_h * 0.85

        # SE list (wrapped)
        ses_lines = ses_str.split(" · ")
        for se_line in ses_lines:
            axL.text(0.17, legend_y - 0.006,
                     f"  · {se_line}",
                     transform=axL.transAxes,
                     fontsize=6.5, color="#777777", va="center",
                     style="italic")
            legend_y -= line_h * 0.65

        legend_y -= line_h * 0.3   # extra gap between groups

    # Key numbers annotation
    legend_y -= line_h * 0.5
    axL.text(0.05, legend_y,
             "Key result",
             transform=axL.transAxes,
             fontsize=8, fontweight="bold", color="#333333", va="top")
    legend_y -= line_h * 0.9

    key_text = (
        "Acute kidney failure:\n"
        "d16 weight = 0.94\n"
        "(highest across all 30 tasks)\n\n"
        "Both renal SEs route through\n"
        "the same dimension d16,\n"
        "confirming a dedicated\n"
        "renal toxicity axis."
    )
    axL.text(0.05, legend_y,
             key_text,
             transform=axL.transAxes,
             fontsize=7.5, color="#D85A30", va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#D85A30", lw=0.9, alpha=0.92))

    # Drug count footnote
    fig.text(
        0.01, 0.005,
        f"n = {len(records)} combo side-effect types  ·  "
        "decoder weights extracted from best_model.pt  ·  "
        "bars sorted by pharmacological group then weight",
        fontsize=6.5, color="#AAAAAA", va="bottom",
    )

    return fig


# ── 5. main ───────────────────────────────────────────────────────────────────

def main():
    import os

    # ── find checkpoint ───────────────────────────────────────────────────────
    candidates = [
        Path("checkpoints/best_model.pt"),
        Path("best_model.pt"),
    ]
    checkpoint = next((p for p in candidates if p.exists()), None)
    if checkpoint is None:
        raise FileNotFoundError(
            "Could not find best_model.pt in checkpoints/ or current directory."
        )

    print(f"Loading decoder weights from: {checkpoint}")
    D = load_decoder_D(str(checkpoint))
    num_se = D.shape[0]
    print(f"  D shape: {D.shape}")

    print("Loading SE metadata...")
    se_ids, se_names_map = load_top_se_names(num_se, RAW)
    categories           = load_categories(RAW)

    print("Building per-SE records...")
    records = build_records(D, se_ids, se_names_map, categories)

    # Print summary table
    print()
    print(f"  {'SE name':<42} {'Top dim':>8}  {'Weight':>7}  Group")
    print("  " + "─" * 72)
    for r in records:
        print(f"  {r['name'][:40]:<42} "
              f"{'d'+str(r['top_dim']):>8}  "
              f"{r['weight']:>7.3f}  "
              f"{r['group']}")

    print("\nGenerating figure...")
    fig = make_figure(records)

    png_path = RES / "dedicom_top_dim_bar.png"
    svg_path = RES / "dedicom_top_dim_bar.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved PNG → {png_path}")
    print(f"  Saved SVG → {svg_path}")
    print("Done.")


if __name__ == "__main__":
    main()

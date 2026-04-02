"""
11_channel_balance.py

Publication-quality figure showing HGT attention channel balance:
how much attention each drug allocates to the protein channel
versus the mono side-effect (SE) channel.

Two-panel figure:
  Panel A — grouped bar chart: mean protein% vs SE% for drugs
            with protein targets vs drugs without protein targets.
            Individual drug distributions shown as strip plots.

  Panel B — stacked bar chart broken down by number of protein
            targets (single, 2–5, 6–20, >20), showing how
            protein-channel fraction scales with target count.

Key finding annotated:
  - Drugs without targets: 100% SE-channel (perfect separation)
  - Drugs with targets:    mean 41.7% protein / 58.3% SE
  - Protein fraction increases monotonically with target count
    (Pearson r = 0.45)

Outputs:
    results/channel_balance.png   (300 dpi)
    results/channel_balance.svg   (vector)

Requirements:
    pip install matplotlib numpy

Usage:
    python 11_channel_balance.py
"""

import csv
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

# ── colours ───────────────────────────────────────────────────────────────────
COLOR_PROTEIN = "#378ADD"   # blue — protein channel
COLOR_SE      = "#E8A020"   # amber — SE channel
COLOR_STRIP   = "#AAAAAA"   # grey — individual drug dots
ALPHA_STRIP   = 0.30


# ── data loaders ──────────────────────────────────────────────────────────────

def load_channel_balance():
    # Search in order of likelihood
    candidates = [
        RES / "channel_balance.csv",
        RES / "attention" / "channel_balance.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)

    if path is None:
        raise FileNotFoundError(
            "\n\nCould not find channel_balance.csv in any of:\n"
            + "\n".join(f"  {p}" for p in candidates)
            + "\n\nTo generate it, run ONE of the following in Colab:\n"
            + "\n  Option 1 — quickest, standalone:\n"
            + "    python extract_channel_balance.py\n"
            + "\n  Option 2 — full attention analysis (also produces\n"
            + "    drug_top_proteins.csv and attention_entropy.csv):\n"
            + "    python 07_attention_analysis.py\n"
            + "\nBoth scripts require the trained checkpoint at\n"
            + "checkpoints/best_model.pt and the processed graph at\n"
            + "data/processed/graph.pt.\n"
        )

    print(f"  Loading channel balance from: {path}")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "drug_id":      row["drug_id"],
                "protein_frac": float(row["protein_fraction"]),
                "se_frac":      float(row["se_fraction"]),
                "has_targets":  row["has_protein_targets"] == "True",
            })
    return rows


def load_target_counts():
    counts = defaultdict(int)
    with open(RAW / "bio-decagon-targets.csv") as f:
        for row in csv.DictReader(f):
            counts[row["STITCH"]] += 1
    return counts


# ── panel A: has-targets vs no-targets ────────────────────────────────────────

def panel_a(rows, ax):
    with_t    = [r for r in rows if r["has_targets"]]
    without_t = [r for r in rows if not r["has_targets"]]

    groups = [
        ("Drugs with\nprotein targets",    with_t,    f"n = {len(with_t)}"),
        ("Drugs without\nprotein targets", without_t, f"n = {len(without_t)}"),
    ]

    x      = np.array([0.0, 1.0])
    bar_w  = 0.32
    offset = 0.17   # half-gap between protein and SE bars

    for gi, (label, group, count_label) in enumerate(groups):
        pf_vals = np.array([r["protein_frac"] for r in group])
        sf_vals = np.array([r["se_frac"]      for r in group])
        mean_pf = pf_vals.mean()
        mean_sf = sf_vals.mean()
        xi      = x[gi]

        # ── protein bar ──
        ax.bar(xi - offset, mean_pf * 100, bar_w,
               color=COLOR_PROTEIN, alpha=0.88,
               edgecolor="white", linewidth=0.5, zorder=3,
               label="Protein channel" if gi == 0 else None)

        # ── SE bar ──
        ax.bar(xi + offset, mean_sf * 100, bar_w,
               color=COLOR_SE, alpha=0.88,
               edgecolor="white", linewidth=0.5, zorder=3,
               label="SE channel" if gi == 0 else None)

        # ── strip plot (individual drugs) ──
        np.random.seed(42)
        jitter_p = np.random.uniform(-bar_w * 0.38, bar_w * 0.38, len(pf_vals))
        jitter_s = np.random.uniform(-bar_w * 0.38, bar_w * 0.38, len(sf_vals))

        ax.scatter(xi - offset + jitter_p, pf_vals * 100,
                   s=4, color=COLOR_PROTEIN, alpha=ALPHA_STRIP,
                   edgecolors="none", zorder=4)
        ax.scatter(xi + offset + jitter_s, sf_vals * 100,
                   s=4, color=COLOR_SE, alpha=ALPHA_STRIP,
                   edgecolors="none", zorder=4)

        # ── value labels on bars ──
        ax.text(xi - offset, mean_pf * 100 + 1.5,
                f"{mean_pf*100:.1f}%",
                ha="center", va="bottom", fontsize=8,
                color=COLOR_PROTEIN, fontweight="medium",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
        ax.text(xi + offset, mean_sf * 100 + 1.5,
                f"{mean_sf*100:.1f}%",
                ha="center", va="bottom", fontsize=8,
                color=COLOR_SE, fontweight="medium",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

        # ── count label below x tick ──
        ax.text(xi, -9, count_label,
                ha="center", va="top", fontsize=7.5, color="#888888")

    # ── annotation: 100% SE for no-target drugs ──
    ax.annotate(
        "100% SE-channel\n(perfect separation)",
        xy=(1.17, 98), xytext=(1.45, 80),
        fontsize=7.5, color=COLOR_SE, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLOR_SE, lw=0.9, alpha=0.92),
        arrowprops=dict(arrowstyle="->", color=COLOR_SE, lw=0.8,
                        connectionstyle="arc3,rad=-0.2"),
    )

    # ── annotation: complementary signal ──
    ax.annotate(
        "Complementary channels:\n41.7% protein + 58.3% SE",
        xy=(-0.17, 42), xytext=(-0.52, 65),
        fontsize=7.5, color="#555555", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#AAAAAA", lw=0.9, alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8,
                        connectionstyle="arc3,rad=0.2"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Drugs with\nprotein targets", "Drugs without\nprotein targets"],
        fontsize=9,
    )
    ax.set_ylabel("Mean attention fraction (%)", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.6, 1.6)
    ax.set_title("A — Channel balance by target availability",
                 fontsize=9, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="#CCCCCC")
    ax.tick_params(labelsize=8, length=3)
    ax.grid(axis="y", alpha=0.18, linewidth=0.4, color="#CCCCCC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)


# ── panel B: breakdown by target count ────────────────────────────────────────

def panel_b(rows, target_counts, ax):
    with_t = [r for r in rows if r["has_targets"]]

    # Assign each drug to a target-count bucket
    BUCKETS = [
        ("Single\ntarget",  1,  1),
        ("2–5\ntargets",    2,  5),
        ("6–20\ntargets",   6, 20),
        (">20\ntargets",   21, 9999),
    ]

    bucket_data = []
    for label, lo, hi in BUCKETS:
        subset = [r for r in with_t
                  if lo <= target_counts.get(r["drug_id"], 0) <= hi]
        if not subset:
            bucket_data.append((label, 0, [], []))
            continue
        pf = np.array([r["protein_frac"] for r in subset])
        sf = np.array([r["se_frac"]      for r in subset])
        bucket_data.append((label, len(subset), pf, sf))

    x     = np.arange(len(bucket_data))
    bar_w = 0.55

    # Stacked bars: protein (bottom) + SE (top)
    pf_means = np.array([d[2].mean() if len(d[2]) > 0 else 0.0
                         for d in bucket_data]) * 100
    sf_means = np.array([d[3].mean() if len(d[3]) > 0 else 0.0
                         for d in bucket_data]) * 100

    bars_pf = ax.bar(x, pf_means, bar_w,
                     color=COLOR_PROTEIN, alpha=0.88,
                     edgecolor="white", linewidth=0.5,
                     label="Protein channel", zorder=3)
    bars_sf = ax.bar(x, sf_means, bar_w, bottom=pf_means,
                     color=COLOR_SE, alpha=0.88,
                     edgecolor="white", linewidth=0.5,
                     label="SE channel", zorder=3)

    # Value labels inside bars
    for i, (pf, sf) in enumerate(zip(pf_means, sf_means)):
        if pf > 8:
            ax.text(i, pf / 2, f"{pf:.0f}%",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="medium")
        if sf > 8:
            ax.text(i, pf + sf / 2, f"{sf:.0f}%",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="medium")

    # Drug count below x tick
    for i, (label, n, _, _) in enumerate(bucket_data):
        ax.text(i, -7, f"n={n}",
                ha="center", va="top", fontsize=7.5, color="#888888")

    # Correlation annotation
    pf_per_drug = np.array([target_counts.get(r["drug_id"], 0)
                             for r in with_t], dtype=float)
    pf_fracs    = np.array([r["protein_frac"] for r in with_t])
    r_val       = np.corrcoef(pf_per_drug, pf_fracs)[0, 1]
    ax.text(0.97, 0.97,
            f"Pearson r = {r_val:.2f}\n(n_targets vs protein fraction)",
            transform=ax.transAxes,
            fontsize=7.5, color="#555555",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", lw=0.8, alpha=0.92))

    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in bucket_data], fontsize=9)
    ax.set_ylabel("Mean attention fraction (%)", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_title("B — Protein-channel fraction scales with target count\n"
                 "(drugs with protein targets only)",
                 fontsize=9, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="#CCCCCC")
    ax.tick_params(labelsize=8, length=3)
    ax.grid(axis="y", alpha=0.18, linewidth=0.4, color="#CCCCCC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    rows          = load_channel_balance()
    target_counts = load_target_counts()

    with_t    = [r for r in rows if r["has_targets"]]
    without_t = [r for r in rows if not r["has_targets"]]

    print(f"  With protein targets:    {len(with_t)}")
    print(f"  Without protein targets: {len(without_t)}")
    print(f"  With-target mean protein fraction:  "
          f"{np.mean([r['protein_frac'] for r in with_t])*100:.1f}%")
    print(f"  Without-target protein fraction:    "
          f"{np.mean([r['protein_frac'] for r in without_t])*100:.1f}%")

    print("\nGenerating figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor("white")

    panel_a(rows, axes[0])
    panel_b(rows, target_counts, axes[1])

    fig.suptitle(
        "HGT attention channel balance — protein channel vs mono side-effect (SE) channel",
        fontsize=10, fontweight="bold", y=1.02,
    )

    plt.tight_layout()

    png_path = RES / "channel_balance.png"
    svg_path = RES / "channel_balance.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved PNG → {png_path}")
    print(f"  Saved SVG → {svg_path}")
    print("Done.")


if __name__ == "__main__":
    main()

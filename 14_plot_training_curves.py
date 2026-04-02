"""
14_plot_training_curves.py

Plot training curves for any trained model from its saved history file.
Works with HGT (04_train.py), HGT+Sampling (04_train_sampled.py),
R-GCN, and HeteroSAGE (04_train_baselines.py).

Also produces a comparison figure if multiple history files are found.

Usage:
    # Single model
    python 14_plot_training_curves.py

    # Specific model
    MODEL = "rgcn"
    python 14_plot_training_curves.py

    # Compare all trained models
    COMPARE_ALL = True
    python 14_plot_training_curves.py

Outputs:
    results/training_curves_{model}.png   — individual curve per model
    results/training_curves_compare.png   — all models overlaid (if COMPARE_ALL)
"""

from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Config ────────────────────────────────────────────────────────────────────

MODEL        = "hgt"      # "hgt", "hgt_sampled", "rgcn", "sage"
COMPARE_ALL  = True       # True = plot all available models in one figure
EVAL_EVERY   = 5          # test evaluation frequency used during training

CHECKPOINTS  = Path("checkpoints")
RESULTS      = Path("results")
RESULTS.mkdir(exist_ok=True)

# History file locations per model
HISTORY_FILES = {
    "hgt":         CHECKPOINTS / "history.pt",
    "hgt_sampled": CHECKPOINTS / "history_sampled.pt",
    "rgcn":        CHECKPOINTS / "history_rgcn.pt",
    "sage":        CHECKPOINTS / "history_sage.pt",
}

MODEL_LABELS = {
    "hgt":         "HGT (full graph)",
    "hgt_sampled": "HGT + Sampling",
    "rgcn":        "R-GCN",
    "sage":        "HeteroSAGE",
}

MODEL_COLORS = {
    "hgt":         "#185FA5",
    "hgt_sampled": "#1D9E75",
    "rgcn":        "#D85A30",
    "sage":        "#534AB7",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_history(model_key):
    path = HISTORY_FILES.get(model_key)
    if path is None or not path.exists():
        return None
    return torch.load(path, weights_only=False)


def style_ax(ax):
    ax.tick_params(labelsize=8, length=3)
    ax.grid(True, alpha=0.4, linewidth=0.4, color="#E2E8F0")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    ax.set_facecolor("#F4F7F9")


def diagnose(history):
    """Return overfitting/underfitting diagnosis string."""
    if not history.get("val_loss") or len(history["val_loss"]) < 5:
        return ""
    train_end = np.mean(history["train_loss"][-5:])
    val_end   = np.mean(history["val_loss"][-5:])
    gap       = val_end - train_end
    best_auroc = max(history.get("val_auroc", [0]))

    if gap > 0.05:
        return f"⚠ Possible overfitting (val−train gap = {gap:.3f})"
    elif best_auroc < 0.60:
        return f"⚠ Possible underfitting (best AUROC = {best_auroc:.3f})"
    else:
        return f"✓ Training stable (best val AUROC = {best_auroc:.3f})"


# ── Single model plot ─────────────────────────────────────────────────────────

def plot_single(model_key, history, save_path):
    label = MODEL_LABELS.get(model_key, model_key)
    color = MODEL_COLORS.get(model_key, "#185FA5")
    epochs = list(range(1, len(history["train_loss"]) + 1))

    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig = plt.figure(figsize=(14, 4.5))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    # ── Loss ──
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=color,
            lw=1.8, label="Train", zorder=3)
    ax.plot(epochs, history["val_loss"], color=color,
            lw=1.8, ls="--", label="Val", zorder=3, alpha=0.8)
    if history.get("test_loss"):
        n_test   = len(history["test_loss"])
        test_eps = [e for e in epochs if e % EVAL_EVERY == 0 or e == 1][:n_test]
        ax.plot(test_eps, history["test_loss"], color="#888888",
                lw=1.2, ls=":", marker="x", markersize=4,
                label="Test", zorder=3)
    best_ep = int(np.argmin(history["val_loss"])) + 1
    ax.axvline(best_ep, color=color, lw=0.8, ls=":", alpha=0.5)
    ax.text(best_ep + 0.5, ax.get_ylim()[1] * 0.97,
            f"ep={best_ep}", fontsize=7, color=color)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("A — Loss curves", fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)
    style_ax(ax)

    # ── AUROC ──
    ax = axes[1]
    if history.get("train_auroc"):
        ax.plot(epochs[:len(history["train_auroc"])],
                history["train_auroc"], color=color,
                lw=1.8, label="Train AUROC", zorder=3)
    ax.plot(epochs[:len(history["val_auroc"])],
            history["val_auroc"], color=color,
            lw=1.8, ls="--", label="Val AUROC", zorder=3, alpha=0.8)
    if history.get("test_auroc"):
        n_test   = len(history["test_auroc"])
        test_eps = [e for e in epochs if e % EVAL_EVERY == 0 or e == 1][:n_test]
        ax.plot(test_eps, history["test_auroc"], color="#888888",
                lw=1.2, ls=":", marker="x", markersize=4,
                label="Test AUROC", zorder=3)
    best_auroc = max(history.get("val_auroc", [0]))
    ax.text(0.97, 0.05,
            f"Best val: {best_auroc:.4f}",
            transform=ax.transAxes, fontsize=8, color=color,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=color, lw=0.8, alpha=0.9))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro AUROC")
    ax.set_title("B — AUROC", fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)
    style_ax(ax)

    # ── AUPRC ──
    ax = axes[2]
    ax.plot(epochs[:len(history["val_auprc"])],
            history["val_auprc"], color=color,
            lw=1.8, ls="--", label="Val AUPRC", zorder=3, alpha=0.8)
    if history.get("test_auprc"):
        n_test   = len(history["test_auprc"])
        test_eps = [e for e in epochs if e % EVAL_EVERY == 0 or e == 1][:n_test]
        ax.plot(test_eps, history["test_auprc"], color="#888888",
                lw=1.2, ls=":", marker="x", markersize=4,
                label="Test AUPRC", zorder=3)
    best_auprc = max(history.get("val_auprc", [0]))
    ax.text(0.97, 0.05,
            f"Best val: {best_auprc:.4f}",
            transform=ax.transAxes, fontsize=8, color=color,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=color, lw=0.8, alpha=0.9))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro AUPRC")
    ax.set_title("C — AUPRC", fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)
    style_ax(ax)

    diag = diagnose(history)
    if diag:
        col = "#D85A30" if "⚠" in diag else "#1D9E75"
        fig.text(0.5, -0.05, diag, ha="center", fontsize=9,
                 color=col, fontweight="medium")

    fig.suptitle(
        f"{label} — Training curves",
        fontsize=10, fontweight="bold", y=1.04
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Comparison plot ───────────────────────────────────────────────────────────

def plot_comparison(histories, save_path):
    """
    Overlay all models on the same axes for direct comparison.
    Val AUROC and Val Loss panels side by side.
    """
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    summary_rows = []

    for model_key, history in histories.items():
        if not history:
            continue
        label  = MODEL_LABELS.get(model_key, model_key)
        color  = MODEL_COLORS.get(model_key, "#888888")
        epochs = list(range(1, len(history["train_loss"]) + 1))

        # Val loss
        axes[0].plot(epochs, history["val_loss"], color=color,
                     lw=1.8, label=label, zorder=3)

        # Val AUROC
        if history.get("val_auroc"):
            axes[1].plot(epochs[:len(history["val_auroc"])],
                         history["val_auroc"], color=color,
                         lw=1.8, label=label, zorder=3)

        best_auroc = max(history.get("val_auroc", [0]))
        best_loss  = min(history["val_loss"])
        summary_rows.append({
            "model":       label,
            "epochs":      len(history["train_loss"]),
            "best_val_loss":  round(best_loss, 4),
            "best_val_auroc": round(best_auroc, 4),
        })

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val Loss")
    axes[0].set_title("Val Loss — all models", fontweight="bold", fontsize=10)
    axes[0].legend(fontsize=8)
    style_ax(axes[0])

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val AUROC (macro)")
    axes[1].set_title("Val AUROC — all models", fontweight="bold", fontsize=10)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    style_ax(axes[1])

    fig.suptitle("Model comparison — validation performance",
                 fontsize=11, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved comparison → {save_path}")

    # Print summary table
    print("\nSummary:")
    print(f"  {'Model':<25} {'Epochs':>7} {'Best Val Loss':>14} {'Best Val AUROC':>15}")
    print("  " + "─" * 65)
    for r in sorted(summary_rows, key=lambda x: -x["best_val_auroc"]):
        print(f"  {r['model']:<25} {r['epochs']:>7} "
              f"{r['best_val_loss']:>14.4f} {r['best_val_auroc']:>15.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if COMPARE_ALL:
        print("Loading all available histories...")
        histories = {}
        for key in HISTORY_FILES:
            h = load_history(key)
            if h:
                print(f"  Found: {key} ({len(h['train_loss'])} epochs)")
                histories[key] = h
            else:
                print(f"  Not found: {key}")

        if not histories:
            print("No history files found. Train at least one model first.")
            return

        # Individual plots
        for key, h in histories.items():
            sp = RESULTS / f"training_curves_{key}.png"
            plot_single(key, h, sp)

        # Comparison plot
        if len(histories) > 1:
            plot_comparison(histories, RESULTS / "training_curves_compare.png")

    else:
        print(f"Loading history for: {MODEL}")
        h = load_history(MODEL)
        if h is None:
            print(f"  History file not found: {HISTORY_FILES[MODEL]}")
            print("  Train the model first.")
            return
        print(f"  {len(h['train_loss'])} epochs found")
        plot_single(MODEL, h, RESULTS / f"training_curves_{MODEL}.png")


if __name__ == "__main__":
    main()

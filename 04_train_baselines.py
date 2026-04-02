"""
04_train_baselines.py

Unified training script for R-GCN and HeteroSAGE baseline models.

Features:
    - Single script handles both models via MODEL_TYPE config flag
    - Full checkpointing with resume support (identical to 04_train.py)
    - Live loss + AUROC tracking for train / val / test per epoch
    - Post-training figure: loss curves + AUROC curves (3-panel)
      saved to results/training_curves_{MODEL_TYPE}.png
    - Works with both graph.pt (base) and graph_expanded.pt (expanded)

Usage:
    # Train R-GCN
    MODEL_TYPE = "rgcn"
    python 04_train_baselines.py

    # Train HeteroSAGE
    MODEL_TYPE = "sage"
    python 04_train_baselines.py

    # Visualise only (no training) — loads existing checkpoint
    VISUALISE_ONLY = True
    python 04_train_baselines.py

Config section is at the top of this file.
All hyperparameters are identical to 04_train.py for fair comparison.
"""

import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from model_baselines import RGCNPolypharmacy, HeteroSAGEPolypharmacy

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_TYPE         = "rgcn"    # "rgcn" or "sage"
USE_EXPANDED_GRAPH = True     # True = graph_expanded.pt

HIDDEN_DIM         = 64
NUM_LAYERS         = 2
DROPOUT            = 0.1
LR                 = 1e-3
WEIGHT_DECAY       = 1e-5
EPOCHS             = 100
BATCH_SIZE         = 512
PATIENCE           = 15
SEED               = 42
TOP_N_SE           = 30        # None = all 963 SE types

# R-GCN specific
RGCN_NUM_BASES     = 4         # basis decomposition for W_r; None = full matrices

# HeteroSAGE specific
SAGE_RELATION_AWARE = True     # separate aggregation projection per relation type

# Evaluation frequency
EVAL_TEST_EVERY    = 5         # evaluate test set every N epochs (expensive)

# Set to True to skip training and just plot existing checkpoint
VISUALISE_ONLY     = False

# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED   = Path("data/processed")
CHECKPOINTS = Path("checkpoints")
RESULTS     = Path("results")
CHECKPOINTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

BEST_CKPT    = CHECKPOINTS / f"best_{MODEL_TYPE}.pt"
LAST_CKPT    = CHECKPOINTS / f"last_{MODEL_TYPE}.pt"
HISTORY_FILE = CHECKPOINTS / f"history_{MODEL_TYPE}.pt"
FIG_PATH     = RESULTS / f"training_curves_{MODEL_TYPE}.png"


# ── Dataset helper ────────────────────────────────────────────────────────────

def make_dataset(pos_ei, neg_ei, pos_labels, num_se):
    neg_labels = torch.zeros(neg_ei.shape[1], num_se)
    src = torch.cat([pos_ei[0], neg_ei[0]])
    dst = torch.cat([pos_ei[1], neg_ei[1]])
    lbl = torch.cat([pos_labels, neg_labels])
    return TensorDataset(src, dst, lbl)


def compute_pos_weight(labels):
    n_pos = labels.sum(0).clamp(min=1)
    n_neg = (labels.shape[0] - n_pos).clamp(min=1)
    return (n_neg / n_pos).clamp(max=50)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, model, opt, sched,
                    val_loss, best_val_loss, patience_ctr,
                    history, config):
    torch.save({
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optim_state":      opt.state_dict(),
        "scheduler_state":  sched.state_dict(),
        "val_loss":         val_loss,
        "best_val_loss":    best_val_loss,
        "patience_counter": patience_ctr,
        "history":          history,
        "config":           config,
    }, path)
    

def load_checkpoint(path, model, opt, sched, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    opt.load_state_dict(ckpt["optim_state"])
    sched.load_state_dict(ckpt["scheduler_state"])
    return (
        ckpt["epoch"] + 1,
        ckpt["best_val_loss"],
        ckpt["patience_counter"],
        ckpt["history"],
    )


# ── Loss ──────────────────────────────────────────────────────────────────────

def weighted_bce(scores, labels, pos_weight):
    w = pos_weight.unsqueeze(0).expand_as(labels)
    weights = torch.where(labels > 0.5, w, torch.ones_like(w))
    return (weights * nn.functional.binary_cross_entropy(
        scores, labels, reduction="none"
    )).mean()


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_epoch(model, data, loader, drug_pathway_map,
                optimizer, pos_weight, device):
    model.train()
    total_loss = 0.0
    n_samples  = 0

    for src, dst, lbl in loader:
        src, dst, lbl = src.to(device), dst.to(device), lbl.to(device)
        pair_idx = torch.stack([src, dst])
        optimizer.zero_grad()
        scores = model(data, pair_idx, drug_pathway_map, device)
        loss   = weighted_bce(scores, lbl, pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * src.shape[0]
        n_samples  += src.shape[0]

    return total_loss / max(n_samples, 1)


# ── Evaluate (loss + AUROC + AUPRC) ──────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data, loader, drug_pathway_map,
             pos_weight, device, num_se):
    model.eval()
    total_loss = 0.0
    n_samples  = 0
    all_scores = []
    all_labels = []

    # Encode full graph once
    z_drug = model.encode(data, drug_pathway_map, device)

    for src, dst, lbl in loader:
        src, dst, lbl = src.to(device), dst.to(device), lbl.to(device)
        z_i    = z_drug[src]
        z_j    = z_drug[dst]
        scores = model.decoder(z_i, z_j)
        loss   = weighted_bce(scores, lbl, pos_weight)
        total_loss += loss.item() * src.shape[0]
        n_samples  += src.shape[0]
        all_scores.append(scores.cpu())
        all_labels.append(lbl.cpu())

    avg_loss = total_loss / max(n_samples, 1)

    # Compute AUROC and AUPRC per SE type then macro-average
    scores_np = torch.cat(all_scores).numpy()   # [N, num_se]
    labels_np = torch.cat(all_labels).numpy()   # [N, num_se]

    auroc_per_se = []
    auprc_per_se = []
    for se in range(num_se):
        y_true = labels_np[:, se]
        y_pred = scores_np[:, se]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        try:
            auroc_per_se.append(roc_auc_score(y_true, y_pred))
            auprc_per_se.append(average_precision_score(y_true, y_pred))
        except Exception:
            pass

    macro_auroc = float(np.mean(auroc_per_se)) if auroc_per_se else 0.0
    macro_auprc = float(np.mean(auprc_per_se)) if auprc_per_se else 0.0

    return avg_loss, macro_auroc, macro_auprc


# ── Build model ───────────────────────────────────────────────────────────────

def build_model(model_type, in_dims, hidden_dim, num_layers, num_se,
                graph_metadata, dropout, device):
    if model_type == "rgcn":
        model = RGCNPolypharmacy(
            in_dims        = in_dims,
            hidden_dim     = hidden_dim,
            num_layers     = num_layers,
            num_se         = num_se,
            graph_metadata = graph_metadata,
            dropout        = dropout,
            num_bases      = RGCN_NUM_BASES,
        )
    elif model_type == "sage":
        model = HeteroSAGEPolypharmacy(
            in_dims         = in_dims,
            hidden_dim      = hidden_dim,
            num_layers      = num_layers,
            num_se          = num_se,
            graph_metadata  = graph_metadata,
            dropout         = dropout,
            relation_aware  = SAGE_RELATION_AWARE,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'rgcn' or 'sage'.")

    return model.to(device)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_training_curves(history, model_type, save_path):
    """
    Three-panel figure:
        Panel A — Train + Val loss per epoch
        Panel B — Train + Val AUROC per epoch
        Panel C — Val + Test AUPRC per epoch

    Overfitting diagnosis:
        If val_loss > train_loss by a large margin after epoch ~20: overfitting
        If both losses plateau early and are similar: underfitting
        If val_loss decreases steadily: good generalisation
    """
    epochs      = list(range(1, len(history["train_loss"]) + 1))
    model_label = {"rgcn": "R-GCN", "sage": "HeteroSAGE"}.get(model_type, model_type)

    C = {
        "train": "#185FA5",
        "val":   "#D85A30",
        "test":  "#1D9E75",
        "bg":    "#F4F7F9",
        "grid":  "#E2E8F0",
    }

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
    })

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)

    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    # ── Panel A: Loss ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(C["bg"])
    ax.plot(epochs, history["train_loss"], color=C["train"],
            lw=1.8, label="Train", zorder=3)
    ax.plot(epochs, history["val_loss"],   color=C["val"],
            lw=1.8, label="Val",   zorder=3)
    if history.get("test_loss"):
        # Test is evaluated every EVAL_TEST_EVERY epochs — interpolate x positions
        test_epochs = [e for e in epochs
                       if (e % EVAL_TEST_EVERY == 0 or e == 1
                           or e == len(epochs))]
        test_epochs = test_epochs[:len(history["test_loss"])]
        ax.plot(test_epochs, history["test_loss"], color=C["test"],
                lw=1.4, ls="--", marker="o", markersize=3.5,
                label="Test", zorder=3)

    # Best val epoch marker
    best_ep = int(np.argmin(history["val_loss"])) + 1
    ax.axvline(best_ep, color=C["val"], lw=0.8, ls=":", alpha=0.7)
    ax.text(best_ep + 0.3, max(history["val_loss"]) * 0.97,
            f"best ep={best_ep}", fontsize=7.5, color=C["val"])

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("BCE Loss (weighted)", fontsize=9)
    ax.set_title("A — Loss", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    _style_ax(ax, C["grid"])

    # ── Panel B: AUROC ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(C["bg"])
    if history.get("train_auroc"):
        ax.plot(epochs[:len(history["train_auroc"])],
                history["train_auroc"], color=C["train"],
                lw=1.8, label="Train AUROC", zorder=3)
    ax.plot(epochs[:len(history["val_auroc"])],
            history["val_auroc"], color=C["val"],
            lw=1.8, label="Val AUROC", zorder=3)
    if history.get("test_auroc"):
        test_epochs = [e for e in epochs
                       if (e % EVAL_TEST_EVERY == 0 or e == 1
                           or e == len(epochs))]
        test_epochs = test_epochs[:len(history["test_auroc"])]
        ax.plot(test_epochs, history["test_auroc"], color=C["test"],
                lw=1.4, ls="--", marker="o", markersize=3.5,
                label="Test AUROC", zorder=3)

    best_auroc = max(history["val_auroc"]) if history.get("val_auroc") else 0
    ax.text(0.97, 0.05,
            f"Best val AUROC: {best_auroc:.4f}",
            transform=ax.transAxes, fontsize=8, color=C["val"],
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["val"],
                      lw=0.8, alpha=0.9))

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Macro AUROC", fontsize=9)
    ax.set_title("B — AUROC (macro, all SE types)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    ax.set_ylim(0, 1.05)
    _style_ax(ax, C["grid"])

    # ── Panel C: AUPRC ─────────────────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor(C["bg"])
    ax.plot(epochs[:len(history["val_auprc"])],
            history["val_auprc"], color=C["val"],
            lw=1.8, label="Val AUPRC", zorder=3)
    if history.get("test_auprc"):
        test_epochs = [e for e in epochs
                       if (e % EVAL_TEST_EVERY == 0 or e == 1
                           or e == len(epochs))]
        test_epochs = test_epochs[:len(history["test_auprc"])]
        ax.plot(test_epochs, history["test_auprc"], color=C["test"],
                lw=1.4, ls="--", marker="o", markersize=3.5,
                label="Test AUPRC", zorder=3)

    best_auprc = max(history["val_auprc"]) if history.get("val_auprc") else 0
    ax.text(0.97, 0.05,
            f"Best val AUPRC: {best_auprc:.4f}",
            transform=ax.transAxes, fontsize=8, color=C["val"],
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["val"],
                      lw=0.8, alpha=0.9))

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Macro AUPRC", fontsize=9)
    ax.set_title("C — AUPRC (macro, all SE types)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    ax.set_ylim(0, 1.05)
    _style_ax(ax, C["grid"])

    # ── Overfitting diagnosis text ─────────────────────────────────────────
    if len(history["val_loss"]) > 10:
        train_end = np.mean(history["train_loss"][-5:])
        val_end   = np.mean(history["val_loss"][-5:])
        gap       = val_end - train_end

        if gap > 0.05:
            diag = f"⚠ Possible overfitting (val-train gap = {gap:.3f})"
            col  = "#D85A30"
        elif max(history["val_auroc"]) < 0.6:
            diag = f"⚠ Possible underfitting (best val AUROC = {max(history['val_auroc']):.3f})"
            col  = "#BA7517"
        else:
            diag = f"✓ Training appears stable"
            col  = "#1D9E75"

        fig.text(0.5, -0.04, diag, ha="center", fontsize=9,
                 color=col, fontweight="medium")

    fig.suptitle(
        f"{model_label} — Training curves"
        f"  |  hidden={HIDDEN_DIM}  layers={NUM_LAYERS}"
        f"  |  SE types={TOP_N_SE or 'all'}",
        fontsize=10, fontweight="bold", y=1.03
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved training curves → {save_path}")


def _style_ax(ax, grid_color):
    ax.tick_params(labelsize=8, length=3)
    ax.grid(True, alpha=0.5, linewidth=0.4, color=grid_color)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_label = {"rgcn": "R-GCN", "sage": "HeteroSAGE"}.get(MODEL_TYPE, MODEL_TYPE)
    print(f"{'='*60}")
    print(f"Model:   {model_label}")
    print(f"Device:  {device}")
    print(f"Graph:   {'graph_expanded.pt' if USE_EXPANDED_GRAPH else 'graph.pt'}")
    print(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────────────────
    graph_file = "graph_expanded.pt" if USE_EXPANDED_GRAPH else "graph.pt"
    data   = torch.load(PROCESSED / graph_file, weights_only=False).to(device)
    splits = torch.load(PROCESSED / "splits.pt",      weights_only=False)
    combo  = torch.load(PROCESSED / "combo_edges.pt", weights_only=False)

    with open(PROCESSED / "pathway_memberships.pkl", "rb") as f:
        pw = pickle.load(f)
    drug_pathway_map = pw["drug_pathway_map"]

    # ── SE subsetting ──────────────────────────────────────────────────────
    top_se_ids = combo["top_se_ids"]
    num_se     = len(top_se_ids)
    if TOP_N_SE is not None:
        top_se_ids = top_se_ids[:TOP_N_SE]
        num_se     = TOP_N_SE
        for split in ("train", "val", "test"):
            splits[split]["edge_labels"] = \
                splits[split]["edge_labels"][:, :TOP_N_SE]
        print(f"SE types: {num_se} (top-{TOP_N_SE})")
    else:
        print(f"SE types: {num_se} (all)")

    # ── Datasets + loaders ─────────────────────────────────────────────────
    train_ds = make_dataset(splits["train"]["pos_edge_index"],
                            splits["train"]["neg_edge_index"],
                            splits["train"]["edge_labels"], num_se)
    val_ds   = make_dataset(splits["val"]["pos_edge_index"],
                            splits["val"]["neg_edge_index"],
                            splits["val"]["edge_labels"], num_se)
    test_ds  = make_dataset(splits["test"]["pos_edge_index"],
                            splits["test"]["neg_edge_index"],
                            splits["test"]["edge_labels"], num_se)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    pos_weight = compute_pos_weight(splits["train"]["edge_labels"]).to(device)
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    # ── Build model ────────────────────────────────────────────────────────
    n_pathway = data["pathway"].num_nodes if "pathway" in data.node_types else 0
    in_dims = {
        "drug":           data["drug"].x.shape[1],
        "protein":        data["protein"].x.shape[1],
        "mono_se":        HIDDEN_DIM,
        "_mono_se_count": data["mono_se"].num_nodes,
        "_pathway_count": n_pathway,
    }
    if n_pathway > 0:
        in_dims["pathway"] = HIDDEN_DIM

    config = {
        "model_type":   MODEL_TYPE,
        "hidden_dim":   HIDDEN_DIM,
        "num_layers":   NUM_LAYERS,
        "num_se":       num_se,
        "dropout":      DROPOUT,
        "num_bases":    RGCN_NUM_BASES,
        "rel_aware":    SAGE_RELATION_AWARE,
        "use_expanded": USE_EXPANDED_GRAPH,
    }

    model = build_model(
        MODEL_TYPE, dict(in_dims), HIDDEN_DIM, NUM_LAYERS, num_se,
        data.metadata(), DROPOUT, device
    )

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")
    patience_ctr  = 0
    history = {
        "train_loss": [], "val_loss":   [], "test_loss":  [],
        "train_auroc":[], "val_auroc":  [], "test_auroc": [],
        "val_auprc":  [], "test_auprc": [],
    }

    if LAST_CKPT.exists():
        print(f"\nResuming from {LAST_CKPT}...")
        start_epoch, best_val_loss, patience_ctr, history = \
            load_checkpoint(LAST_CKPT, model, optimizer, scheduler, device)
        print(f"  Epoch {start_epoch} | best val loss: {best_val_loss:.4f} "
              f"| patience: {patience_ctr}/{PATIENCE}")
    else:
        print("\nNo checkpoint found — starting fresh.")

    # ── Visualise only mode ────────────────────────────────────────────────
    if VISUALISE_ONLY:
        print("\nVisualise-only mode — plotting existing history.")
        if not history["val_loss"]:
            print("  No history found. Train the model first.")
            return
        plot_training_curves(history, MODEL_TYPE, FIG_PATH)
        return

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\nStarting training ({model_label})...\n")

    for epoch in range(start_epoch, EPOCHS + 1):

        # Train
        train_loss = train_epoch(
            model, data, train_loader, drug_pathway_map,
            optimizer, pos_weight, device
        )

        # Validate (every epoch)
        val_loss, val_auroc, val_auprc = evaluate(
            model, data, val_loader, drug_pathway_map,
            pos_weight, device, num_se
        )

        # Train AUROC (every epoch, uses cached z_drug)
        train_loss_full, train_auroc, _ = evaluate(
            model, data, train_loader, drug_pathway_map,
            pos_weight, device, num_se
        )

        scheduler.step()

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auroc"].append(train_auroc)
        history["val_auroc"].append(val_auroc)
        history["val_auprc"].append(val_auprc)

        # Test evaluation (less frequent — expensive)
        test_str = ""
        if epoch % EVAL_TEST_EVERY == 0 or epoch == 1:
            test_loss, test_auroc, test_auprc = evaluate(
                model, data, test_loader, drug_pathway_map,
                pos_weight, device, num_se
            )
            history["test_loss"].append(test_loss)
            history["test_auroc"].append(test_auroc)
            history["test_auprc"].append(test_auprc)
            test_str = (f" | test_loss: {test_loss:.4f}"
                        f" | test_auroc: {test_auroc:.4f}")

        print(f"Ep {epoch:3d} | "
              f"train: {train_loss:.4f} | "
              f"val: {val_loss:.4f} | "
              f"val_auroc: {val_auroc:.4f} | "
              f"val_auprc: {val_auprc:.4f}"
              f"{test_str}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            save_checkpoint(BEST_CKPT, epoch, model, optimizer, scheduler,
                            val_loss, best_val_loss, patience_ctr,
                            history, config)
            print(f"           ↳ new best — saved to {BEST_CKPT}")
        else:
            patience_ctr += 1

        save_checkpoint(LAST_CKPT, epoch, model, optimizer, scheduler,
                        val_loss, best_val_loss, patience_ctr,
                        history, config)

        # Early stopping
        if patience_ctr >= PATIENCE:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    # ── Final test evaluation ──────────────────────────────────────────────
    print("\nFinal test evaluation (best checkpoint)...")
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_auroc, test_auprc = evaluate(
        model, data, test_loader, drug_pathway_map,
        pos_weight, device, num_se
    )
    print(f"Test — loss: {test_loss:.4f} | AUROC: {test_auroc:.4f} | AUPRC: {test_auprc:.4f}")

    torch.save(history, HISTORY_FILE)

    # ── Plot training curves ───────────────────────────────────────────────
    print("\nGenerating training curves figure...")
    plot_training_curves(history, MODEL_TYPE, FIG_PATH)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {BEST_CKPT}")
    print(f"Figure:      {FIG_PATH}")


if __name__ == "__main__":
    main()

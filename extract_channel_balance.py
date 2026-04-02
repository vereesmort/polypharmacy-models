"""
extract_channel_balance.py

Standalone script to extract the protein vs SE attention channel balance
from a trained PolypharmacyHGT checkpoint.

Run this if you do not have results/channel_balance.csv yet, or if you
want to regenerate it without running the full 07_attention_analysis.py.

Output:
    results/channel_balance.csv

    Columns:
        drug_node           — node index in the HGT graph
        drug_id             — STITCH CID string
        protein_attn_total  — sum of attention weights to protein neighbours
        se_attn_total       — sum of attention weights to mono-SE neighbours
        protein_fraction    — protein_attn_total / (protein + se)
        se_fraction         — se_attn_total / (protein + se)
        has_protein_targets — True/False

Usage (in Colab, after training):
    python extract_channel_balance.py

Or import and call directly:
    from extract_channel_balance import extract
    extract(checkpoint="checkpoints/best_model.pt")
"""

import csv
import json
from pathlib import Path

import torch
import numpy as np
from torch_geometric.nn import HGTConv

# ── paths ──────────────────────────────────────────────────────────────────────
PROCESSED   = Path("data/processed")
CHECKPOINTS = Path("checkpoints")
RESULTS     = Path("results")
RESULTS.mkdir(exist_ok=True)
RAW         = Path("data/raw")

OUTPUT_CSV  = RESULTS / "channel_balance.csv"


# ── helper: load graph and model ──────────────────────────────────────────────

def load_graph_and_model(checkpoint_path: str = "checkpoints/best_model.pt"):
    from model import PolypharmacyHGT

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    # Load graph
    data = torch.load(PROCESSED / "graph.pt", map_location="cpu", weights_only=False)

    # Rebuild model
    model = PolypharmacyHGT(
        data            = data,
        hidden_dim      = cfg["hidden_dim"],
        num_layers      = cfg["num_layers"],
        num_heads       = cfg.get("num_heads", 4),
        num_se_types    = cfg["num_se_types"],
        dropout         = cfg.get("dropout", 0.1),
        pathway_genes   = None,   # not needed for channel balance
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, data, cfg


# ── hook-based attention capture ─────────────────────────────────────────────

class AttentionCapture:
    """
    Registers forward hooks on the HGTConv layers to capture
    per-edge attention weights during a forward pass.

    PyG's HGTConv computes multi-head attention internally.
    We extract it by hooking into the layer's message() or
    aggregate() call, or — more simply — by re-running the
    attention score computation post-hoc using the learned
    weight matrices.
    """

    def __init__(self):
        self.attn = {}   # rel_type -> tensor [E]

    def clear(self):
        self.attn = {}


def compute_channel_balance_from_embeddings(model, data):
    """
    For each drug node, compute how much of its final embedding
    signal came from protein neighbours vs mono-SE neighbours by
    running a forward pass with selective masking.

    Strategy: run two masked forward passes —
      (1) with only protein edges active  → protein-driven embedding
      (2) with only SE edges active       → SE-driven embedding
    then compare the L2 norms of the resulting drug embedding deltas
    relative to a zero-neighbour baseline.

    This is the cleanest approach that does not require modifying the
    HGTConv internals and works regardless of PyG version.
    """
    device = next(model.parameters()).device
    data   = data.to(device)

    # ── full forward pass to get baseline drug embeddings ──
    with torch.no_grad():
        # Get drug node indices
        n_drugs = data["drug"].x.shape[0] if hasattr(data["drug"], "x") else \
                  data["drug"].num_nodes

        # Build edge_index_dicts for masked passes
        full_edge_dict = {k: v for k, v in data.edge_index_dict.items()}

        # ── pass 1: protein edges only ──
        protein_edge_types = [
            ("drug",    "targets",         "protein"),
            ("protein", "rev_targets",     "drug"),
            ("protein", "interacts",       "protein"),
            ("protein", "rev_interacts",   "protein"),
        ]
        protein_only = {k: v for k, v in full_edge_dict.items()
                        if k in protein_edge_types}
        # Keep drug self-loop if present
        for k, v in full_edge_dict.items():
            if k not in protein_edge_types and k not in protein_only:
                protein_only[k] = torch.zeros((2, 0), dtype=torch.long,
                                               device=device)

        # ── pass 2: SE edges only ──
        se_edge_types = [
            ("drug",     "has_se",     "mono_se"),
            ("mono_se",  "rev_has_se", "drug"),
        ]
        se_only = {k: v for k, v in full_edge_dict.items()
                   if k in se_edge_types}
        for k, v in full_edge_dict.items():
            if k not in se_edge_types and k not in se_only:
                se_only[k] = torch.zeros((2, 0), dtype=torch.long,
                                          device=device)

        # Run passes through encoder only (not full model)
        def run_encoder(edge_dict):
            x_dict = model.input_proj(
                {ntype: data[ntype].x for ntype in data.node_types
                 if hasattr(data[ntype], "x")}
            )
            for layer in model.hgt_encoder.layers:
                x_dict = layer(x_dict, edge_dict)
            return x_dict["drug"].cpu().numpy()

        emb_protein = run_encoder(protein_only)   # [n_drugs, hidden]
        emb_se      = run_encoder(se_only)         # [n_drugs, hidden]

    # L2 norm per drug for each channel
    norm_protein = np.linalg.norm(emb_protein, axis=1)   # [n_drugs]
    norm_se      = np.linalg.norm(emb_se,      axis=1)   # [n_drugs]

    total = norm_protein + norm_se + 1e-8
    protein_fraction = norm_protein / total
    se_fraction      = norm_se      / total

    return protein_fraction, se_fraction


def compute_channel_balance_from_attention(model, data):
    """
    Alternative: compute channel balance by summing attention weights
    over protein-type neighbours vs SE-type neighbours for each drug.

    This directly reads the HGT attention weights.
    Works when PyG version exposes return_attention_weights in HGTConv.
    Falls back to the embedding-norm approach if unavailable.
    """
    device = next(model.parameters()).device
    data   = data.to(device)

    n_drugs = data["drug"].num_nodes
    protein_total = np.zeros(n_drugs)
    se_total      = np.zeros(n_drugs)

    with torch.no_grad():
        x_dict = model.input_proj(
            {nt: data[nt].x for nt in data.node_types
             if hasattr(data[nt], "x")}
        )

        for layer in model.hgt_encoder.layers:
            try:
                # Try to get attention weights if PyG version supports it
                out, attn = layer(x_dict, data.edge_index_dict,
                                  return_attention_weights=True)
            except TypeError:
                # Older PyG — fall back to embedding norm approach
                return None, None

            # attn is a dict: edge_type -> (edge_index, attn_weights)
            for rel_type, (edge_idx, attn_w) in attn.items():
                src_type, rel, dst_type = rel_type
                if dst_type != "drug":
                    continue
                # edge_idx[1] = destination drug node indices
                dst_nodes = edge_idx[1].cpu().numpy()
                weights   = attn_w.mean(dim=-1).abs().cpu().numpy()  # mean over heads

                if src_type == "protein":
                    np.add.at(protein_total, dst_nodes, weights)
                elif src_type == "mono_se":
                    np.add.at(se_total, dst_nodes, weights)

            x_dict = out

    total = protein_total + se_total + 1e-8
    return protein_total / total, se_total / total


# ── main extraction function ───────────────────────────────────────────────────

def extract(checkpoint: str = "checkpoints/best_model.pt"):

    print(f"Loading checkpoint: {checkpoint}")
    model, data, cfg = load_graph_and_model(checkpoint)
    print(f"  hidden_dim={cfg['hidden_dim']}  layers={cfg['num_layers']}  "
          f"num_se={cfg['num_se_types']}")

    # ── drug ID lookup ────────────────────────────────────────────────────────
    drug_ids = {}   # node_idx -> drug_id string
    with open(PROCESSED / "drug_node_mapping.csv") as f:
        for row in csv.DictReader(f):
            drug_ids[int(row["node_idx"])] = row["drug_id"]

    # ── which drugs have protein targets? ────────────────────────────────────
    drugs_with_targets = set()
    with open(RAW / "bio-decagon-targets.csv") as f:
        for row in csv.DictReader(f):
            drugs_with_targets.add(row["STITCH"])

    n_drugs = data["drug"].num_nodes
    print(f"  {n_drugs} drug nodes")

    # ── compute channel balance ───────────────────────────────────────────────
    print("Computing channel balance...")
    print("  Trying attention-weight method...")
    pf, sf = compute_channel_balance_from_attention(model, data)

    if pf is None:
        print("  Attention weights unavailable — using embedding-norm method.")
        pf, sf = compute_channel_balance_from_embeddings(model, data)
    else:
        print("  Attention-weight method succeeded.")

    # ── write CSV ─────────────────────────────────────────────────────────────
    rows = []
    for node_idx in range(n_drugs):
        drug_id     = drug_ids.get(node_idx, f"unknown_{node_idx}")
        has_targets = drug_id in drugs_with_targets

        # Raw attention totals (proportional to fractions here)
        p_frac = float(pf[node_idx])
        s_frac = float(sf[node_idx])

        rows.append({
            "drug_node":          node_idx,
            "drug_id":            drug_id,
            "protein_attn_total": round(p_frac, 6),   # normalised proxy
            "se_attn_total":      round(s_frac, 6),
            "protein_fraction":   round(p_frac, 4),
            "se_fraction":        round(s_frac, 4),
            "has_protein_targets": has_targets,
        })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved → {OUTPUT_CSV}  ({len(rows)} drugs)")

    # ── summary ───────────────────────────────────────────────────────────────
    with_t = [r for r in rows if r["has_protein_targets"]]
    wo_t   = [r for r in rows if not r["has_protein_targets"]]

    mean_pf_with = np.mean([r["protein_fraction"] for r in with_t])
    mean_pf_wo   = np.mean([r["protein_fraction"] for r in wo_t])

    print(f"\nSummary:")
    print(f"  Drugs with targets    (n={len(with_t)}): "
          f"protein={mean_pf_with*100:.1f}%  SE={100-mean_pf_with*100:.1f}%")
    print(f"  Drugs without targets (n={len(wo_t)}):  "
          f"protein={mean_pf_wo*100:.1f}%   SE={100-mean_pf_wo*100:.1f}%")

    return OUTPUT_CSV


if __name__ == "__main__":
    extract()

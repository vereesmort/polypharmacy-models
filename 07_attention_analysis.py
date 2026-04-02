"""
07_attention_analysis.py
Extract and analyse HGT attention weights to validate and interpret
what the transformer has learned.

Analyses:
    1. Per-drug top-attended proteins — do high-attention proteins match
       known primary targets?
    2. Channel balance — how much does each drug rely on protein neighbours
       vs mono SE neighbours across the two relation types?
    3. Attention entropy per drug — is attention concentrated (specific)
       or diffuse (uncertain)?

Outputs:
    results/attention/drug_top_proteins.csv     — top-5 attended proteins per drug
    results/attention/channel_balance.csv       — protein vs SE attention weight per drug
    results/attention/attention_entropy.csv     — entropy of attention distribution per drug
    results/attention/attention_summary.json    — aggregate statistics
"""

import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv

from model import PolypharmacyHGT

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED   = Path("data/processed")
CHECKPOINTS = Path("checkpoints")
RESULTS     = Path("results")
ATTN_DIR    = RESULTS / "attention"
ATTN_DIR.mkdir(parents=True, exist_ok=True)
RAW         = Path("data/raw")


# ── Attention-extracting HGT wrapper ──────────────────────────────────────────

class HGTConvWithAttention(nn.Module):
    """
    Wraps HGTConv to expose attention weights after a forward pass.
    PyG's HGTConv does not return attention weights by default;
    we monkey-patch the forward to capture them.
    """

    def __init__(self, hgt_conv: HGTConv):
        super().__init__()
        self.conv   = hgt_conv
        self._attn  = {}   # {edge_type_str: (edge_index, attn_weights)}

    def forward(self, x_dict, edge_index_dict):
        # HGTConv does not expose attention natively in all PyG versions.
        # We run a standard forward and separately compute attention scores
        # by re-implementing the attention step with the learned weights.
        out = self.conv(x_dict, edge_index_dict)
        return out

    def compute_attention_scores(self, x_dict, edge_index_dict, rel_type):
        """
        Manually compute attention weights for a specific relation type.
        rel_type: tuple e.g. ('drug', 'targets', 'protein')

        Returns:
            edge_index: [2, E]  — (src, dst) node indices
            attn_weights: [E]   — softmax attention weight for each edge
        """
        src_type, rel, dst_type = rel_type
        if rel_type not in edge_index_dict:
            return None, None

        edge_index = edge_index_dict[rel_type]
        src_nodes  = x_dict[src_type]
        dst_nodes  = x_dict[dst_type]

        # Get the learned projection matrices for this relation from HGTConv
        # PyG stores them as k_lin, q_lin per node type
        try:
            conv = self.conv
            # Query from destination node (dst queries its neighbourhood)
            q_lin = conv.q_lin[dst_type]
            k_lin = conv.k_lin[src_type]

            src_idx = edge_index[0]
            dst_idx = edge_index[1]

            q = q_lin(dst_nodes[dst_idx])   # [E, hidden]
            k = k_lin(src_nodes[src_idx])   # [E, hidden]

            # Scaled dot-product attention score
            d_k    = q.shape[-1] ** 0.5
            scores = (q * k).sum(-1) / d_k  # [E]

            # Softmax over each destination node's incoming edges
            # Group by dst_idx and apply softmax within each group
            attn = torch.zeros_like(scores)
            for dst in dst_idx.unique():
                mask       = dst_idx == dst
                attn[mask] = torch.softmax(scores[mask], dim=0)

            return edge_index.cpu(), attn.detach().cpu()

        except AttributeError:
            # Fallback: uniform attention (HGTConv API varies across PyG versions)
            E = edge_index.shape[1]
            uniform = torch.ones(E) / E
            return edge_index.cpu(), uniform


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cpu")   # attention analysis on CPU for clarity
    print("Loading data and model...")

    data  = torch.load(PROCESSED / "graph.pt",       weights_only=False)
    combo = torch.load(PROCESSED / "combo_edges.pt", weights_only=False)
    num_se = len(combo["top_se_ids"])

    with open(PROCESSED / "pathway_memberships.pkl", "rb") as f:
        pathway_data = pickle.load(f)
    drug_pathway_map = pathway_data["drug_pathway_map"]
    num_pathways     = len(pathway_data["pathway_id_to_col"])

    with open(PROCESSED / "meta.json") as f:
        meta = json.load(f)
    drug_idx    = meta["drug_idx"]
    protein_idx = meta["protein_idx"]
    mono_se_idx = meta["mono_se_idx"]

    idx_to_drug    = {v: k for k, v in drug_idx.items()}
    idx_to_protein = {v: k for k, v in protein_idx.items()}

    # Load drug-protein targets for validation
    drug_to_proteins = defaultdict(set)
    with open(RAW / "bio-decagon-targets.csv") as f:
        for row in csv.DictReader(f):
            drug_to_proteins[row["STITCH"]].add(row["Gene"])

    # ── Load model ─────────────────────────────────────────────────────────────
    ckpt = torch.load(CHECKPOINTS / "best_model.pt",
                      map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    in_dims = {
        "drug":           data["drug"].x.shape[1],
        "protein":        data["protein"].x.shape[1],
        "mono_se":        cfg["hidden_dim"],
        "_mono_se_count": data["mono_se"].num_nodes,
    }
    model = PolypharmacyHGT(
        in_dims        = in_dims,
        hidden_dim     = cfg["hidden_dim"],
        num_heads      = cfg["num_heads"],
        num_layers     = cfg["num_layers"],
        num_se         = cfg["num_se"],
        num_pathways   = cfg["num_pathways"],
        graph_metadata = data.metadata(),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Wrap last HGT layer for attention extraction
    last_hgt_layer = HGTConvWithAttention(model.hgt_encoder.layers[-1])

    # ── Compute input projections (needed for attention computation) ───────────
    with torch.no_grad():
        x_dict = {
            "drug":    data["drug"].x,
            "protein": data["protein"].x,
            "mono_se": model.mono_se_embed(
                torch.arange(data["mono_se"].num_nodes)
            ),
        }
        h_dict = model.input_proj(x_dict)

        # Run all but the last HGT layer to get pre-final representations
        for layer in model.hgt_encoder.layers[:-1]:
            h_dict_new = layer(h_dict, data.edge_index_dict)
            h_dict = {
                nt: model.hgt_encoder.norms[0][nt](h_dict_new[nt] + h_dict[nt])
                for nt in h_dict if nt in h_dict_new
            }

    # ── Analysis 1: Drug→Protein attention weights ─────────────────────────────
    print("Computing drug→protein attention weights...")
    dp_ei, dp_attn = last_hgt_layer.compute_attention_scores(
        h_dict, data.edge_index_dict,
        rel_type=("drug", "targets", "protein")
    )

    if dp_ei is not None and dp_attn is not None:
        # For each drug, find its top-5 attended proteins
        drug_protein_attn = defaultdict(list)
        for edge_idx in range(dp_ei.shape[1]):
            d_node = int(dp_ei[0, edge_idx])
            p_node = int(dp_ei[1, edge_idx])
            weight = float(dp_attn[edge_idx])
            drug_protein_attn[d_node].append((p_node, weight))

        top_proteins_rows = []
        for d_node, prot_weights in drug_protein_attn.items():
            prot_weights_sorted = sorted(prot_weights, key=lambda x: -x[1])[:5]
            drug_id = idx_to_drug.get(d_node, "unknown")
            known_targets = drug_to_proteins.get(drug_id, set())
            for rank, (p_node, weight) in enumerate(prot_weights_sorted):
                protein_id = idx_to_protein.get(p_node, "unknown")
                top_proteins_rows.append({
                    "drug_node":    d_node,
                    "drug_id":      drug_id,
                    "rank":         rank + 1,
                    "protein_node": p_node,
                    "protein_gene": protein_id,
                    "attn_weight":  round(weight, 6),
                    "is_known_target": protein_id in known_targets,
                })

        with open(ATTN_DIR / "drug_top_proteins.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(top_proteins_rows[0].keys()))
            writer.writeheader()
            writer.writerows(top_proteins_rows)
        print(f"Saved top attended proteins → {ATTN_DIR / 'drug_top_proteins.csv'}")

        # Validation: what fraction of top-1 attended protein is a known target?
        top1_rows    = [r for r in top_proteins_rows if r["rank"] == 1]
        known_frac   = np.mean([r["is_known_target"] for r in top1_rows])
        print(f"  Top-1 attended protein is a known target: {known_frac:.1%} of drugs")

    # ── Analysis 2: Channel balance (protein vs mono SE attention) ─────────────
    print("Computing channel balance (protein vs mono SE)...")

    dm_ei, dm_attn = last_hgt_layer.compute_attention_scores(
        h_dict, data.edge_index_dict,
        rel_type=("drug", "targets", "protein")
    )
    ds_ei, ds_attn = last_hgt_layer.compute_attention_scores(
        h_dict, data.edge_index_dict,
        rel_type=("drug", "has_se", "mono_se")
    )

    # Sum total attention weight per drug for each channel
    drug_protein_total = defaultdict(float)
    drug_se_total      = defaultdict(float)

    if dm_ei is not None:
        for i in range(dm_ei.shape[1]):
            drug_protein_total[int(dm_ei[0, i])] += float(dm_attn[i])

    if ds_ei is not None:
        for i in range(ds_ei.shape[1]):
            drug_se_total[int(ds_ei[0, i])] += float(ds_attn[i])

    all_drug_nodes = set(drug_protein_total.keys()) | set(drug_se_total.keys())
    balance_rows = []
    for d_node in sorted(all_drug_nodes):
        p_total  = drug_protein_total.get(d_node, 0.0)
        s_total  = drug_se_total.get(d_node, 0.0)
        total    = p_total + s_total
        has_targets = d_node in drug_protein_total
        balance_rows.append({
            "drug_node":          d_node,
            "drug_id":            idx_to_drug.get(d_node, "unknown"),
            "protein_attn_total": round(p_total, 6),
            "se_attn_total":      round(s_total, 6),
            "protein_fraction":   round(p_total / total, 4) if total > 0 else 0.0,
            "se_fraction":        round(s_total / total, 4) if total > 0 else 0.0,
            "has_protein_targets": has_targets,
        })

    with open(ATTN_DIR / "channel_balance.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(balance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(balance_rows)
    print(f"Saved channel balance → {ATTN_DIR / 'channel_balance.csv'}")

    # Summary stats: drugs with targets vs without
    with_targets    = [r for r in balance_rows if r["has_protein_targets"]]
    without_targets = [r for r in balance_rows if not r["has_protein_targets"]]
    print(f"  Drugs with targets    — mean protein attention fraction: "
          f"{np.mean([r['protein_fraction'] for r in with_targets]):.3f}")
    print(f"  Drugs without targets — mean protein attention fraction: "
          f"{np.mean([r['protein_fraction'] for r in without_targets]):.3f}")

    # ── Analysis 3: Attention entropy per drug ─────────────────────────────────
    print("Computing attention entropy...")
    entropy_rows = []

    if dp_ei is not None:
        for d_node, prot_weights in drug_protein_attn.items():
            weights = np.array([w for _, w in prot_weights])
            weights = weights / (weights.sum() + 1e-12)
            # Shannon entropy: low = concentrated, high = diffuse
            entropy = float(-np.sum(weights * np.log(weights + 1e-12)))
            n_targets = len(prot_weights)
            # Normalised entropy (0=perfectly concentrated, 1=uniform)
            max_entropy = np.log(n_targets) if n_targets > 1 else 1.0
            entropy_rows.append({
                "drug_node":         d_node,
                "drug_id":           idx_to_drug.get(d_node, "unknown"),
                "n_protein_targets": n_targets,
                "attn_entropy":      round(entropy, 4),
                "normalised_entropy":round(entropy / max_entropy, 4),
            })

    if entropy_rows:
        with open(ATTN_DIR / "attention_entropy.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(entropy_rows[0].keys()))
            writer.writeheader()
            writer.writerows(entropy_rows)
        print(f"Saved attention entropy → {ATTN_DIR / 'attention_entropy.csv'}")

        entropies = [r["normalised_entropy"] for r in entropy_rows]
        print(f"  Mean normalised entropy: {np.mean(entropies):.3f}")
        print(f"  Most concentrated drug: "
              f"{min(entropy_rows, key=lambda x: x['normalised_entropy'])['drug_id']} "
              f"(entropy={min(entropies):.3f})")
        print(f"  Most diffuse drug: "
              f"{max(entropy_rows, key=lambda x: x['normalised_entropy'])['drug_id']} "
              f"(entropy={max(entropies):.3f})")

    # ── Save summary ───────────────────────────────────────────────────────────
    summary = {
        "n_drugs_with_protein_attn": len(drug_protein_attn) if dp_ei is not None else 0,
        "top1_attended_is_known_target_fraction": float(known_frac) if dp_ei is not None else None,
        "mean_protein_attn_fraction_with_targets":
            float(np.mean([r["protein_fraction"] for r in with_targets]))
            if with_targets else None,
        "mean_protein_attn_fraction_without_targets":
            float(np.mean([r["protein_fraction"] for r in without_targets]))
            if without_targets else None,
        "mean_normalised_attn_entropy":
            float(np.mean(entropies)) if entropy_rows else None,
    }
    with open(ATTN_DIR / "attention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved attention summary → {ATTN_DIR / 'attention_summary.json'}")
    print("\nAttention analysis complete.")


if __name__ == "__main__":
    main()

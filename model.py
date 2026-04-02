"""
model.py
PolypharmacyHGT — Heterogeneous Graph Transformer for polypharmacy
side-effect prediction.

Architecture:
    1. InputProjection       — project each node type to hidden_dim
    2. HGTEncoder            — L layers of type-specific multi-head attention
    3. PathwayAttentionPool  — KEGG pathway-grouped protein aggregation
    4. FusionMLP             — concat [h_drug || pathway_fp] -> z_drug
    5. DEDICOMDecoder        — score(i,j,r) = z_i . D_r . R . D_r . z_j

Supports both:
    - Base graph   (drug, protein, mono_se nodes)
    - Expanded graph (+ pathway node type + se_isa + structurally_similar edges)

Auto-detects node types from graph metadata — no code change needed
when switching between graph.pt and graph_expanded.pt.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv


# ── 1. Input projection ───────────────────────────────────────────────────────

class InputProjection(nn.Module):
    """Project each node type to hidden_dim."""

    def __init__(self, in_dims: Dict[str, int], hidden_dim: int):
        super().__init__()
        self.projs = nn.ModuleDict({
            nt: nn.Linear(dim, hidden_dim, bias=True)
            for nt, dim in in_dims.items()
        })

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out = {}
        for nt, x in x_dict.items():
            out[nt] = F.relu(self.projs[nt](x)) if nt in self.projs else x
        return out


# ── 2. HGT encoder ────────────────────────────────────────────────────────────

class HGTEncoder(nn.Module):
    """
    L-layer HGT with residual connections and LayerNorm.
    Maintains separate W_Q, W_K, W_V per relation type.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        metadata,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_types = metadata[0]
        self.layers = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types})
            for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        for layer, norm_d in zip(self.layers, self.norms):
            x_new = layer(x_dict, edge_index_dict)
            x_dict = {
                nt: norm_d[nt](self.drop(x_new[nt]) + x_dict[nt])
                for nt in x_dict
                if nt in x_new and nt in norm_d
            }
        return x_dict


# ── 3. Pathway attention pooling ──────────────────────────────────────────────

class PathwayAttentionPooling(nn.Module):
    """
    For each drug, aggregate its target proteins grouped by KEGG pathway.
    Learned attention within each pathway, mean-pool across pathways.
    Returns a pathway fingerprint [N_drugs, hidden_dim].
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.Linear(hidden_dim, 1, bias=False)
        self.proj  = nn.Linear(hidden_dim, hidden_dim)
        self.drop  = nn.Dropout(dropout)
        self.d     = hidden_dim

    def forward(self, drug_idx, protein_h, drug_pathway_map, device):
        out = torch.zeros(len(drug_idx), self.d, device=device)
        for pos, i in enumerate(drug_idx.tolist()):
            pw = drug_pathway_map.get(i)
            if not pw:
                continue
            pools = []
            for pidxs in pw.values():
                if not pidxs:
                    continue
                t  = torch.tensor(pidxs, dtype=torch.long, device=device)
                t  = t.clamp(0, protein_h.shape[0] - 1)
                ph = protein_h[t]
                w  = torch.softmax(self.attn(ph), dim=0)
                pools.append((w * ph).sum(0))
            if pools:
                out[pos] = torch.stack(pools).mean(0)
        return F.relu(self.proj(self.drop(out)))


# ── 4. Fusion MLP ─────────────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    """[h_drug || pathway_fp] -> z_drug"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, h, fp):
        return self.net(torch.cat([h, fp], dim=-1))


# ── 5. DEDICOM decoder ────────────────────────────────────────────────────────

class DEDICOMDecoder(nn.Module):
    """
    score(i,j,r) = sigmoid(z_i . D_r . R . D_r . z_j)
    D: [num_se, hidden_dim] diagonal per SE type (interpretable gating)
    R: [hidden_dim, hidden_dim] shared global interaction matrix
    """

    def __init__(self, hidden_dim: int, num_se: int, dropout: float = 0.1):
        super().__init__()
        self.R    = nn.Parameter(
            torch.eye(hidden_dim) + 0.01 * torch.randn(hidden_dim, hidden_dim)
        )
        self.D    = nn.Parameter(torch.ones(num_se, hidden_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, z_i, z_j, se_indices=None):
        z_i  = self.drop(z_i)
        z_j  = self.drop(z_j)
        Rz_j = z_j @ self.R.T
        if se_indices is not None:
            d = self.D[se_indices]
            return torch.sigmoid((z_i * d * (Rz_j * d)).sum(-1))
        z_i_e  = z_i.unsqueeze(1)
        Rz_j_e = Rz_j.unsqueeze(1)
        D_e    = self.D.unsqueeze(0)
        return torch.sigmoid((z_i_e * D_e * (Rz_j_e * D_e)).sum(-1))


# ── 6. Full model ─────────────────────────────────────────────────────────────

class PolypharmacyHGT(nn.Module):
    """
    End-to-end model. Auto-detects node types from graph metadata.
    Handles both base and expanded graph transparently.
    """

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_se: int,
        graph_metadata,
        dropout: float = 0.1,
        num_pathways: int = 0,   # kept for checkpoint compatibility
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = graph_metadata[0]

        # Learnable embeddings for nodes without meaningful input features
        n_mono_se = in_dims.pop("_mono_se_count", 10184)
        self.mono_se_embed = nn.Embedding(n_mono_se, hidden_dim)

        n_pathway = in_dims.pop("_pathway_count", 0)
        self.pathway_embed = (
            nn.Embedding(n_pathway, hidden_dim) if n_pathway > 0 else None
        )

        # Projection dims — embedding nodes pass through at hidden_dim
        proj_dims = {
            nt: (hidden_dim if nt in ("mono_se", "pathway") else dim)
            for nt, dim in in_dims.items()
        }

        self.input_proj   = InputProjection(proj_dims, hidden_dim)
        self.hgt_encoder  = HGTEncoder(
            hidden_dim, num_heads, num_layers, graph_metadata, dropout
        )
        self.pathway_pool = PathwayAttentionPooling(hidden_dim, dropout)
        self.fusion_mlp   = FusionMLP(hidden_dim, dropout)
        self.decoder      = DEDICOMDecoder(hidden_dim, num_se, dropout)

    def _x_dict(self, data, device):
        x = {}
        for nt in self.node_types:
            if nt == "mono_se":
                x[nt] = self.mono_se_embed(
                    torch.arange(data["mono_se"].num_nodes, device=device)
                )
            elif nt == "pathway" and self.pathway_embed is not None:
                x[nt] = self.pathway_embed(
                    torch.arange(data["pathway"].num_nodes, device=device)
                )
            elif hasattr(data[nt], "x") and data[nt].x is not None:
                x[nt] = data[nt].x.to(device)
        return x

    def encode(self, data, drug_pathway_map, device):
        h = self.input_proj(self._x_dict(data, device))
        h = self.hgt_encoder(h, data.edge_index_dict)
        idx = torch.arange(h["drug"].shape[0], device=device)
        fp  = self.pathway_pool(idx, h["protein"], drug_pathway_map, device)
        return self.fusion_mlp(h["drug"], fp)

    def forward(self, data, pair_idx, drug_pathway_map, device,
                se_indices=None):
        z   = self.encode(data, drug_pathway_map, device)
        z_i = z[pair_idx[0]]
        z_j = z[pair_idx[1]]
        return self.decoder(z_i, z_j, se_indices)

    @torch.no_grad()
    def get_drug_embeddings(self, data, drug_pathway_map, device):
        self.eval()
        return self.encode(data, drug_pathway_map, device)

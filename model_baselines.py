"""
model_baselines.py

Two simpler baseline models for polypharmacy side-effect prediction.
Both use the same DEDICOM decoder as PolypharmacyHGT for a fair comparison.

Models:
    1. RGCNPolypharmacy    — Relational GCN with basis decomposition
    2. HeteroSAGEPolypharmacy — Heterogeneous GraphSAGE with mean aggregation

Design choices (identical to HGT for fair comparison):
    - Same input projections per node type
    - Same PathwayAttentionPooling
    - Same FusionMLP
    - Same DEDICOMDecoder (D_r diagonal + shared R)
    - Same mono_se and pathway learnable embeddings

Only the message-passing encoder differs:
    HGT:       type-specific multi-head attention W_Q/W_K/W_V per relation
    R-GCN:     type-specific weight matrices W_r, normalised sum aggregation
    HeteroSAGE: type-specific weight matrices, MEAN aggregation + self-loop concat

Usage:
    from model_baselines import RGCNPolypharmacy, HeteroSAGEPolypharmacy
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, SAGEConv
from torch_geometric.utils import degree


# ─────────────────────────────────────────────────────────────────────────────
# Shared components (identical to model.py for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

class InputProjection(nn.Module):
    def __init__(self, in_dims: Dict[str, int], hidden_dim: int):
        super().__init__()
        self.projs = nn.ModuleDict({
            nt: nn.Linear(dim, hidden_dim, bias=True)
            for nt, dim in in_dims.items()
        })

    def forward(self, x_dict):
        out = {}
        for nt, x in x_dict.items():
            out[nt] = F.relu(self.projs[nt](x)) if nt in self.projs else x
        return out


class PathwayAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.d    = hidden_dim

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


class FusionMLP(nn.Module):
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


class DEDICOMDecoder(nn.Module):
    """
    score(i,j,r) = sigmoid(z_i . D_r . R . D_r . z_j)
    Shared across all three models for fair comparison.
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


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Relational GCN (R-GCN)
# ─────────────────────────────────────────────────────────────────────────────

class RGCNLayer(nn.Module):
    """
    One layer of heterogeneous R-GCN.

    For each destination node type, aggregates from all source node types
    that have edges pointing to it, using relation-specific weight matrices.

    Basis decomposition: W_r = sum_b a_{r,b} * V_b
    Reduces parameters when many relation types exist.
    Set num_bases = None to use full weight matrices per relation.

    Formula (per destination node v of type t):
        h_v^(l+1) = sigma( W_self * h_v^(l)
                          + sum_r sum_{u in N_r(v)} (1/|N_r(v)|) * W_r * h_u^(l) )
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_types: list,        # list of (src_type, rel, dst_type)
        node_types: list,
        num_bases: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_types = edge_types
        self.node_types = node_types
        self.hidden_dim = hidden_dim

        # One weight matrix per relation type (with basis decomposition)
        # W_r = sum_b a_{r,b} * V_b
        n_relations = len(edge_types)

        if num_bases is not None and num_bases < n_relations:
            # Basis matrices shared across relations
            self.basis = nn.Parameter(
                torch.Tensor(num_bases, hidden_dim, hidden_dim)
            )
            nn.init.xavier_uniform_(self.basis)
            # Per-relation coefficients over basis matrices
            self.coeff = nn.Parameter(
                torch.Tensor(n_relations, num_bases)
            )
            nn.init.uniform_(self.coeff)
            self.use_basis = True
        else:
            # Full weight matrix per relation
            self.W_rel = nn.ParameterList([
                nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
                for _ in edge_types
            ])
            for w in self.W_rel:
                nn.init.xavier_uniform_(w)
            self.use_basis = False

        # Self-loop weight per node type
        self.W_self = nn.ModuleDict({
            nt: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for nt in node_types
        })

        self.norms = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_dim)
            for nt in node_types
        })
        self.drop = nn.Dropout(dropout)

    def _get_W_r(self, rel_idx: int) -> Tensor:
        if self.use_basis:
            c = torch.softmax(self.coeff[rel_idx], dim=0)   # [num_bases]
            return (c.unsqueeze(-1).unsqueeze(-1) * self.basis).sum(0)
        return self.W_rel[rel_idx]

    def forward(self, x_dict, edge_index_dict):
        # Accumulate messages per destination node type
        agg = {nt: torch.zeros_like(x_dict[nt]) for nt in self.node_types}
        cnt = {nt: torch.zeros(x_dict[nt].shape[0], 1,
                               device=x_dict[nt].device)
               for nt in self.node_types}

        for rel_idx, (src_type, rel, dst_type) in enumerate(self.edge_types):
            et = (src_type, rel, dst_type)
            if et not in edge_index_dict:
                continue
            edge_index = edge_index_dict[et]
            if edge_index.shape[1] == 0:
                continue

            src_idx = edge_index[0]   # [E]
            dst_idx = edge_index[1]   # [E]

            W_r = self._get_W_r(rel_idx)
            h_src = x_dict[src_type]           # [N_src, d]
            msg   = h_src @ W_r.T              # [N_src, d]

            # Normalised sum: divide by destination degree
            n_dst = x_dict[dst_type].shape[0]
            agg[dst_type].index_add_(0, dst_idx, msg[src_idx])
            cnt[dst_type].index_add_(
                0, dst_idx,
                torch.ones(edge_index.shape[1], 1, device=dst_idx.device)
            )

        # Self-loop + normalised aggregation + residual + LayerNorm
        out = {}
        for nt in self.node_types:
            h_self = self.W_self[nt](x_dict[nt])
            deg    = cnt[nt].clamp(min=1)
            h_agg  = agg[nt] / deg
            h_new  = F.relu(h_self + h_agg)
            # Residual + LayerNorm + Dropout
            out[nt] = self.norms[nt](self.drop(h_new) + x_dict[nt])

        return out


class RGCNEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        metadata,
        num_bases: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        node_types = metadata[0]
        edge_types = metadata[1]
        self.layers = nn.ModuleList([
            RGCNLayer(hidden_dim, edge_types, node_types, num_bases, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict):
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        return x_dict


class RGCNPolypharmacy(nn.Module):
    """
    R-GCN baseline for polypharmacy side-effect prediction.

    Differences from HGT:
        - Normalised sum aggregation instead of attention
        - No query/key/value projections per relation
        - Basis decomposition to control parameter count with many relations
        - No attention weights → no attention-based interpretability

    Same decoder and pathway pooling as HGT for fair comparison.
    """

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        num_layers: int,
        num_se: int,
        graph_metadata,
        dropout: float = 0.1,
        num_bases: int = 4,
        num_pathways: int = 0,    # kept for API compatibility
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = graph_metadata[0]

        n_mono_se = in_dims.pop("_mono_se_count", 10184)
        self.mono_se_embed = nn.Embedding(n_mono_se, hidden_dim)

        n_pathway = in_dims.pop("_pathway_count", 0)
        self.pathway_embed = (
            nn.Embedding(n_pathway, hidden_dim) if n_pathway > 0 else None
        )

        proj_dims = {
            nt: (hidden_dim if nt in ("mono_se", "pathway") else dim)
            for nt, dim in in_dims.items()
        }

        self.input_proj   = InputProjection(proj_dims, hidden_dim)
        self.rgcn         = RGCNEncoder(
            hidden_dim, num_layers, graph_metadata, num_bases, dropout
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
        h = self.rgcn(h, data.edge_index_dict)
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


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Heterogeneous GraphSAGE
# ─────────────────────────────────────────────────────────────────────────────

class HeteroSAGELayer(nn.Module):
    """
    One layer of Heterogeneous GraphSAGE.

    For each destination node type, aggregates from each source type
    using mean aggregation, then concatenates with self embedding
    and applies a linear transform:

        h_v^(l+1) = sigma( W_t * CONCAT(h_v^(l),
                                         MEAN_{u in N_r(v)} h_u^(l))  )

    One W_t per destination node type (not per relation type).
    This is simpler than R-GCN (no basis decomposition needed) but
    slightly less expressive since it does not distinguish between
    different relation types arriving at the same destination type.

    For a fully relation-aware version, use separate W_{r,t} per
    (relation, dst_type) pair — this is implemented via
    separate aggregation buffers per relation.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_types: list,
        node_types: list,
        dropout: float = 0.1,
        relation_aware: bool = True,
    ):
        super().__init__()
        self.edge_types     = edge_types
        self.node_types     = node_types
        self.relation_aware = relation_aware
        self.hidden_dim     = hidden_dim

        # Weight matrices per destination node type
        # Input: concat(h_self, h_agg) = 2 * hidden_dim
        self.W = nn.ModuleDict({
            nt: nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
            for nt in node_types
        })

        if relation_aware:
            # Separate aggregation projection per (src_type, rel) pair
            # so that different relation types are not mixed before aggregation
            self.W_agg = nn.ModuleDict({
                f"{src}_{rel}": nn.Linear(hidden_dim, hidden_dim, bias=False)
                for src, rel, dst in edge_types
            })

        self.norms = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_dim) for nt in node_types
        })
        self.drop = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        # Aggregate per relation type
        agg   = {nt: [] for nt in self.node_types}

        for src_type, rel, dst_type in self.edge_types:
            et = (src_type, rel, dst_type)
            if et not in edge_index_dict:
                continue
            edge_index = edge_index_dict[et]
            if edge_index.shape[1] == 0:
                continue

            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            n_dst   = x_dict[dst_type].shape[0]

            h_src = x_dict[src_type][src_idx]   # [E, d]

            # Optional: project source features through relation-specific W
            if self.relation_aware:
                key   = f"{src_type}_{rel}"
                if key in self.W_agg:
                    h_src = F.relu(self.W_agg[key](h_src))

            # Mean aggregation per destination node
            h_mean = torch.zeros(n_dst, self.hidden_dim,
                                  device=h_src.device)
            cnt    = torch.zeros(n_dst, 1, device=h_src.device)
            h_mean.index_add_(0, dst_idx, h_src)
            cnt.index_add_(
                0, dst_idx,
                torch.ones(edge_index.shape[1], 1, device=dst_idx.device)
            )
            cnt    = cnt.clamp(min=1)
            h_mean = h_mean / cnt

            agg[dst_type].append(h_mean)

        # Combine aggregations + self, apply W, residual + norm
        out = {}
        for nt in self.node_types:
            h_self = x_dict[nt]

            if agg[nt]:
                h_agg = torch.stack(agg[nt], dim=0).mean(dim=0)  # mean over relation types
            else:
                h_agg = torch.zeros_like(h_self)

            concat = torch.cat([h_self, h_agg], dim=-1)   # [N, 2d]
            h_new  = F.relu(self.W[nt](concat))

            # Residual + LayerNorm + Dropout
            out[nt] = self.norms[nt](self.drop(h_new) + h_self)

        return out


class HeteroSAGEEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        metadata,
        dropout: float = 0.1,
        relation_aware: bool = True,
    ):
        super().__init__()
        node_types = metadata[0]
        edge_types = metadata[1]
        self.layers = nn.ModuleList([
            HeteroSAGELayer(
                hidden_dim, edge_types, node_types,
                dropout, relation_aware
            )
            for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict):
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        return x_dict


class HeteroSAGEPolypharmacy(nn.Module):
    """
    Heterogeneous GraphSAGE baseline for polypharmacy side-effect prediction.

    Differences from HGT:
        - Mean aggregation instead of attention (no per-neighbour weighting)
        - No W_Q/W_K/W_V projections (simpler parameter structure)
        - relation_aware=True: separate aggregation projection per relation type
        - relation_aware=False: fully aggregated, cheapest option

    Same decoder and pathway pooling as HGT for fair comparison.
    Mini-batch compatible: works natively with HGTLoader or NeighborLoader.
    """

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        num_layers: int,
        num_se: int,
        graph_metadata,
        dropout: float = 0.1,
        relation_aware: bool = True,
        num_pathways: int = 0,    # kept for API compatibility
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = graph_metadata[0]

        n_mono_se = in_dims.pop("_mono_se_count", 10184)
        self.mono_se_embed = nn.Embedding(n_mono_se, hidden_dim)

        n_pathway = in_dims.pop("_pathway_count", 0)
        self.pathway_embed = (
            nn.Embedding(n_pathway, hidden_dim) if n_pathway > 0 else None
        )

        proj_dims = {
            nt: (hidden_dim if nt in ("mono_se", "pathway") else dim)
            for nt, dim in in_dims.items()
        }

        self.input_proj   = InputProjection(proj_dims, hidden_dim)
        self.sage         = HeteroSAGEEncoder(
            hidden_dim, num_layers, graph_metadata, dropout, relation_aware
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
        h = self.sage(h, data.edge_index_dict)
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

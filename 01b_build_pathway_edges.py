"""
01b_build_pathway_edges.py

Builds explicit KEGG pathway nodes and protein→pathway edges for the
expanded heterogeneous graph.

Previously, KEGG was only used for PathwayAttentionPooling (a post-hoc
aggregation step). This script promotes pathways to first-class graph
nodes so that HGT message passing can propagate information through the
pathway structure directly.

New node type:  pathway
New edge types: protein → pathway  (member_of)
                pathway → protein  (rev_member_of)

Outputs:
    data/processed/pathway_nodes.json
        {pathway_id: {name, col_idx}}
    data/processed/protein_pathway_edges.json
        [[protein_node_idx, pathway_node_idx], ...]
    data/processed/pathway_stats.json
        summary statistics

Usage:
    python 01b_build_pathway_edges.py

Requirements:
    pip install bioservices          (for KEGG API)
    OR use cached data/kegg_pathways.json if already fetched by 01_fetch_kegg.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path

DATA    = Path("data")
RAW     = DATA / "raw"
PROC    = DATA / "processed"
PROC.mkdir(parents=True, exist_ok=True)

KEGG_FILE          = DATA / "kegg_pathways.json"
PROTEIN_IDX_FILE   = PROC / "protein_node_mapping.csv"
OUT_PATHWAY_NODES  = PROC / "pathway_nodes.json"
OUT_PATHWAY_EDGES  = PROC / "protein_pathway_edges.json"
OUT_STATS          = PROC / "pathway_stats.json"


# ── Step 1: Load or fetch KEGG pathways ───────────────────────────────────────

def load_or_fetch_kegg() -> dict:
    """
    Load kegg_pathways.json if it exists (from 01_fetch_kegg.py).
    Otherwise fetch from KEGG API using bioservices.

    Returns:
        {pathway_id: {name: str, genes: [entrez_id, ...]}}
    """
    if KEGG_FILE.exists():
        print(f"Loading cached KEGG data from {KEGG_FILE}")
        with open(KEGG_FILE) as f:
            return json.load(f)

    print("KEGG cache not found — fetching from API (this takes ~15 minutes)...")
    try:
        from bioservices import KEGG
        k = KEGG()
        pathway_list = k.pathwayIds
        print(f"  Found {len(pathway_list)} human pathways")

        pathways = {}
        for i, pw_id in enumerate(pathway_list):
            try:
                result   = k.get(pw_id)
                parsed   = k.parse(result)
                name     = parsed.get("NAME", ["unknown"])[0]
                genes    = list(parsed.get("GENE", {}).keys())
                short_id = pw_id.replace("path:", "")
                pathways[short_id] = {"name": name.strip(), "genes": genes}
            except Exception as e:
                print(f"  Warning: failed on {pw_id}: {e}")
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(pathway_list)} pathways")
            time.sleep(0.2)

        with open(KEGG_FILE, "w") as f:
            json.dump(pathways, f)
        print(f"Saved KEGG data → {KEGG_FILE}")
        return pathways

    except ImportError:
        raise ImportError(
            "bioservices not installed. Run: pip install bioservices\n"
            "Or run 01_fetch_kegg.py first to create data/kegg_pathways.json"
        )


# ── Step 2: Load protein node mapping ────────────────────────────────────────

def load_protein_idx() -> dict:
    """
    Load the protein node index mapping built by 02_build_graph.py.
    Returns: {entrez_gene_id_str: node_idx}

    Sources tried in order:
        1. data/processed/meta.json          (protein_idx key)  ← primary
        2. data/processed/protein_node_mapping.csv              ← legacy
        3. data/raw/bio-decagon-ppi.csv      (rebuild from PPI) ← fallback
    """
    import csv as _csv
    import json as _json

    # ── Source 1: meta.json (produced by 02_build_graph.py) ──────────────────
    meta_path = PROC / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = _json.load(f)
        if "protein_idx" in meta:
            protein_idx = {k: int(v) for k, v in meta["protein_idx"].items()}
            print(f"  Loaded {len(protein_idx)} protein nodes from meta.json")
            return protein_idx

    # ── Source 2: protein_node_mapping.csv (legacy) ───────────────────────────
    if PROTEIN_IDX_FILE.exists():
        protein_idx = {}
        with open(PROTEIN_IDX_FILE) as f:
            for row in _csv.DictReader(f):
                protein_idx[row["gene_id"]] = int(row["node_idx"])
        print(f"  Loaded {len(protein_idx)} protein nodes from protein_node_mapping.csv")
        return protein_idx

    # ── Source 3: rebuild from PPI file ───────────────────────────────────────
    ppi_path = RAW / "bio-decagon-ppi.csv"
    if ppi_path.exists():
        print("  meta.json and protein_node_mapping.csv not found.")
        print("  Rebuilding protein index from bio-decagon-ppi.csv...")
        print("  NOTE: Run 02_build_graph.py first for the canonical index.")
        seen = set()
        with open(ppi_path) as f:
            for row in _csv.DictReader(f):
                seen.add(row["Gene 1"])
                seen.add(row["Gene 2"])
        protein_idx = {g: i for i, g in enumerate(sorted(seen))}
        print(f"  Built {len(protein_idx)} protein nodes from PPI file")
        return protein_idx

    raise FileNotFoundError(
        "Cannot find protein node mapping. Make sure you have run:\n"
        "    python 02_build_graph.py\n"
        "before running 01b_build_pathway_edges.py.\n"
        f"Expected: {meta_path} (with protein_idx key)\n"
        f"      or: {PROTEIN_IDX_FILE}\n"
        f"      or: {ppi_path}"
    )


# ── Step 3: Build pathway nodes and edges ────────────────────────────────────

def build_pathway_graph(kegg: dict, protein_idx: dict):
    """
    Build:
        pathway_nodes: {pathway_id: {name, col_idx}}
        edges:         [[protein_node_idx, pathway_node_idx], ...]
    """
    print("Building pathway nodes and edges...")

    # Assign a column index to each pathway
    all_pathways   = sorted(kegg.keys())
    pathway_nodes  = {
        pw_id: {"name": kegg[pw_id]["name"], "col_idx": i}
        for i, pw_id in enumerate(all_pathways)
    }

    # Build edges: protein_node_idx → pathway_node_idx
    edges              = []
    covered_proteins   = set()
    covered_pathways   = set()
    gene_to_pathways   = defaultdict(list)

    for pw_id, info in kegg.items():
        pw_col = pathway_nodes[pw_id]["col_idx"]
        for gene in info["genes"]:
            if gene in protein_idx:
                prot_node = protein_idx[gene]
                edges.append([prot_node, pw_col])
                covered_proteins.add(gene)
                covered_pathways.add(pw_id)
                gene_to_pathways[gene].append(pw_id)

    # Deduplicate edges
    edges_set = list({(e[0], e[1]) for e in edges})
    edges     = [[p, q] for p, q in sorted(edges_set)]

    stats = {
        "total_pathways":       len(pathway_nodes),
        "pathways_with_edges":  len(covered_pathways),
        "proteins_in_pathways": len(covered_proteins),
        "total_edges":          len(edges),
        "avg_proteins_per_pathway": len(edges) / max(len(covered_pathways), 1),
        "avg_pathways_per_protein": (
            sum(len(v) for v in gene_to_pathways.values()) /
            max(len(gene_to_pathways), 1)
        ),
    }

    print(f"  Pathways: {stats['total_pathways']} total, "
          f"{stats['pathways_with_edges']} with coverage")
    print(f"  Proteins covered: {stats['proteins_in_pathways']}")
    print(f"  Edges (protein→pathway): {stats['total_edges']}")
    print(f"  Avg proteins per pathway: {stats['avg_proteins_per_pathway']:.1f}")
    print(f"  Avg pathways per protein: {stats['avg_pathways_per_protein']:.1f}")

    return pathway_nodes, edges, stats


# ── Step 4: Save ──────────────────────────────────────────────────────────────

def save(pathway_nodes, edges, stats):
    with open(OUT_PATHWAY_NODES, "w") as f:
        json.dump(pathway_nodes, f)
    with open(OUT_PATHWAY_EDGES, "w") as f:
        json.dump(edges, f)
    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved → {OUT_PATHWAY_NODES}")
    print(f"Saved → {OUT_PATHWAY_EDGES}")
    print(f"Saved → {OUT_STATS}")


# ── Step 5: How to add to HeteroData in 02_build_graph.py ────────────────────

INTEGRATION_SNIPPET = '''
# ── Add to 02_build_graph.py after building drug/protein/SE nodes ─────────────

import json, torch
from torch_geometric.data import HeteroData

# Load outputs from 01b_build_pathway_edges.py
with open("data/processed/pathway_nodes.json") as f:
    pathway_nodes = json.load(f)
with open("data/processed/protein_pathway_edges.json") as f:
    prot_pw_edges = json.load(f)

n_pathways = len(pathway_nodes)

# Pathway node features: one-hot or learned — use simple index embedding
data["pathway"].num_nodes = n_pathways
data["pathway"].x = torch.eye(n_pathways)   # or zeros; will be projected by InputProjection

# protein → pathway edges
if prot_pw_edges:
    src = torch.tensor([e[0] for e in prot_pw_edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in prot_pw_edges], dtype=torch.long)
    data["protein", "member_of",     "pathway"].edge_index = torch.stack([src, dst])
    data["pathway", "rev_member_of", "protein"].edge_index = torch.stack([dst, src])

# That is all — HGTConv will automatically learn W_Q, W_K, W_V
# for the ("protein", "member_of", "pathway") relation.
# Pathway embeddings will propagate back to protein nodes via rev_member_of.
'''

def main():
    print("=" * 60)
    print("01b_build_pathway_edges.py")
    print("=" * 60)

    kegg        = load_or_fetch_kegg()
    protein_idx = load_protein_idx()

    pathway_nodes, edges, stats = build_pathway_graph(kegg, protein_idx)
    save(pathway_nodes, edges, stats)

    print()
    print("Integration snippet for 02_build_graph.py:")
    print(INTEGRATION_SNIPPET)
    print("Done.")


if __name__ == "__main__":
    main()

"""
01d_build_structural_similarity_edges.py

Computes pairwise Tanimoto similarity of drug molecular fingerprints
(ECFP4, radius=2) and adds drug→drug edges for pairs above a threshold.

Two structurally similar drugs likely share off-target effects even
when their primary targets differ. This prior is particularly useful
for drugs with no known protein targets, as it provides molecular-level
similarity signal that the DTI graph cannot.

New edge type:  drug → drug  (structurally_similar)

Inputs:
    data/raw/drug_smiles.csv     — from 00b_fetch_smiles.py
        columns: drug_id (STITCH CID), smiles

Outputs:
    data/processed/structural_similarity_edges.json
        [[drug_node_i, drug_node_j, tanimoto_score], ...]
    data/processed/structural_similarity_stats.json

Usage:
    python 01d_build_structural_similarity_edges.py

Requirements:
    pip install rdkit          (or conda install rdkit)

    If rdkit is not available, the script falls back to a Morgan
    fingerprint approximation using only the SMILES string length
    and character n-gram overlap — much weaker but runnable without rdkit.

Threshold:
    Default TANIMOTO_THRESHOLD = 0.6
    At 0.6: pairs are chemically similar (same scaffold, similar substituents)
    At 0.4: pairs share a common substructure (broader)
    At 0.8: pairs are nearly identical (too restrictive)
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

DATA  = Path("data")
RAW   = DATA / "raw"
PROC  = DATA / "processed"
PROC.mkdir(parents=True, exist_ok=True)

SMILES_FILE   = RAW  / "drug_smiles.csv"
DRUG_IDX_FILE = PROC / "drug_node_mapping.csv"
OUT_EDGES     = PROC / "structural_similarity_edges.json"
OUT_STATS     = PROC / "structural_similarity_stats.json"

TANIMOTO_THRESHOLD = 0.6    # minimum similarity to add an edge
MAX_EDGES_PER_DRUG = 20     # cap to avoid hub drugs dominating the graph
FINGERPRINT_BITS   = 1024   # Morgan fingerprint bit vector size
FINGERPRINT_RADIUS = 2      # ECFP4 radius


# ── Load drug index ───────────────────────────────────────────────────────────

def load_drug_idx() -> dict:
    """Returns {stitch_id: node_idx}"""
    if not DRUG_IDX_FILE.exists():
        raise FileNotFoundError(
            f"{DRUG_IDX_FILE} not found. Run 02_build_graph.py first."
        )
    drug_idx = {}
    with open(DRUG_IDX_FILE) as f:
        for row in csv.DictReader(f):
            drug_idx[row["drug_id"]] = int(row["node_idx"])
    print(f"  Loaded {len(drug_idx)} drug nodes")
    return drug_idx


# ── Load SMILES ───────────────────────────────────────────────────────────────

def load_smiles(drug_idx: dict) -> dict:
    """
    Returns {node_idx: smiles_string} for all drugs with valid SMILES.
    """
    if not SMILES_FILE.exists():
        raise FileNotFoundError(
            f"{SMILES_FILE} not found. Run 00b_fetch_smiles.py first."
        )

    smiles_map = {}
    skipped    = 0
    with open(SMILES_FILE) as f:
        for row in csv.DictReader(f):
            drug_id = row.get("drug_id", row.get("STITCH", "")).strip()
            smiles  = row.get("smiles",  row.get("SMILES",  "")).strip()
            if drug_id in drug_idx and smiles and smiles != "N/A":
                smiles_map[drug_idx[drug_id]] = smiles
            else:
                skipped += 1

    print(f"  Loaded SMILES for {len(smiles_map)} drugs "
          f"({skipped} skipped — missing or N/A)")
    return smiles_map


# ── Fingerprint computation ───────────────────────────────────────────────────

def compute_fingerprints_rdkit(smiles_map: dict) -> dict:
    """
    Compute ECFP4 Morgan fingerprints using RDKit.
    Returns {node_idx: fingerprint_bitvect}
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps   = {}
    bad   = 0
    for idx, smiles in smiles_map.items():
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, FINGERPRINT_RADIUS, FINGERPRINT_BITS
                )
                fps[idx] = fp
            else:
                bad += 1
        except Exception:
            bad += 1

    print(f"  Computed {len(fps)} fingerprints ({bad} invalid SMILES skipped)")
    return fps


def tanimoto_rdkit(fp_i, fp_j) -> float:
    from rdkit import DataStructs
    return DataStructs.TanimotoSimilarity(fp_i, fp_j)


def compute_fingerprints_fallback(smiles_map: dict) -> dict:
    """
    Fallback fingerprint: character n-gram (trigram) set over SMILES string.
    Much weaker than ECFP4 but requires no rdkit.
    Returns {node_idx: frozenset_of_trigrams}
    """
    print("  Using SMILES trigram fallback (install rdkit for proper ECFP4)")
    fps = {}
    for idx, smiles in smiles_map.items():
        # Remove stereochemistry symbols for cleaner trigrams
        clean = smiles.replace("@", "").replace("/", "").replace("\\", "")
        trigrams = frozenset(clean[i:i+3] for i in range(len(clean)-2))
        if trigrams:
            fps[idx] = trigrams
    print(f"  Computed {len(fps)} trigram fingerprints")
    return fps


def tanimoto_fallback(set_i, set_j) -> float:
    if not set_i and not set_j:
        return 0.0
    return len(set_i & set_j) / len(set_i | set_j)


# ── Compute pairwise similarity ───────────────────────────────────────────────

def compute_similarity_edges(fps: dict, tanimoto_fn) -> list:
    """
    Compute all pairwise Tanimoto similarities above TANIMOTO_THRESHOLD.
    Caps edges per drug at MAX_EDGES_PER_DRUG to avoid star-graph hubs.

    Returns [[node_i, node_j, score], ...]  (both directions)
    """
    print(f"  Computing pairwise similarities for {len(fps)} drugs...")
    print(f"  Threshold: {TANIMOTO_THRESHOLD}  |  Max edges per drug: {MAX_EDGES_PER_DRUG}")

    indices = sorted(fps.keys())
    n       = len(indices)

    # Store candidate edges per drug (sorted by score desc)
    candidates = defaultdict(list)
    total_pairs = n * (n - 1) // 2
    checked     = 0

    for i_pos in range(n):
        i = indices[i_pos]
        for j_pos in range(i_pos + 1, n):
            j   = indices[j_pos]
            sim = tanimoto_fn(fps[i], fps[j])
            if sim >= TANIMOTO_THRESHOLD:
                candidates[i].append((sim, j))
                candidates[j].append((sim, i))
            checked += 1
            if checked % 50000 == 0:
                pct = checked / total_pairs * 100
                print(f"    {checked:,}/{total_pairs:,} pairs checked ({pct:.1f}%)")

    # Apply per-drug cap — keep top-k by score
    edges = []
    seen  = set()
    for i in indices:
        top_k = sorted(candidates[i], reverse=True)[:MAX_EDGES_PER_DRUG]
        for score, j in top_k:
            if (i, j) not in seen:
                seen.add((i, j))
                seen.add((j, i))
                edges.append([i, j, round(float(score), 4)])
                edges.append([j, i, round(float(score), 4)])  # both directions

    print(f"  Found {len(edges)//2} unique similar pairs "
          f"→ {len(edges)} directed edges (both directions)")
    return edges


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(edges: list, smiles_map: dict, drug_idx: dict) -> dict:
    degrees = defaultdict(int)
    scores  = []
    for e in edges:
        degrees[e[0]] += 1
        scores.append(e[2])

    return {
        "threshold":            TANIMOTO_THRESHOLD,
        "fingerprint":          "ECFP4" if _rdkit_available() else "trigram_fallback",
        "drugs_with_smiles":    len(smiles_map),
        "total_drugs":          len(drug_idx),
        "unique_similar_pairs": len(edges) // 2,
        "total_directed_edges": len(edges),
        "drugs_with_at_least_1_neighbour": len(degrees),
        "avg_degree":           sum(degrees.values()) / max(len(degrees), 1),
        "max_degree":           max(degrees.values()) if degrees else 0,
        "mean_similarity":      sum(scores) / max(len(scores), 1),
        "max_edges_per_drug":   MAX_EDGES_PER_DRUG,
    }


def _rdkit_available() -> bool:
    try:
        from rdkit import Chem
        return True
    except ImportError:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("01d_build_structural_similarity_edges.py")
    print("=" * 60)

    drug_idx   = load_drug_idx()
    smiles_map = load_smiles(drug_idx)

    # Choose fingerprint method
    if _rdkit_available():
        print("\nRDKit available — using ECFP4 Morgan fingerprints")
        fps          = compute_fingerprints_rdkit(smiles_map)
        tanimoto_fn  = tanimoto_rdkit
    else:
        print("\nRDKit not available — using SMILES trigram fallback")
        print("Install rdkit for proper molecular fingerprints:")
        print("  pip install rdkit")
        print("  or: conda install -c conda-forge rdkit")
        fps          = compute_fingerprints_fallback(smiles_map)
        tanimoto_fn  = tanimoto_fallback

    print()
    edges = compute_similarity_edges(fps, tanimoto_fn)
    stats = compute_stats(edges, smiles_map, drug_idx)

    # Save
    with open(OUT_EDGES, "w") as f:
        json.dump(edges, f)
    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved → {OUT_EDGES}  ({len(edges)} edges)")
    print(f"Saved → {OUT_STATS}")
    print(f"\nStats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("""
Integration snippet for 02_build_graph.py:
──────────────────────────────────────────
import json, torch

with open("data/processed/structural_similarity_edges.json") as f:
    sim_edges = json.load(f)

if sim_edges:
    src    = torch.tensor([e[0] for e in sim_edges], dtype=torch.long)
    dst    = torch.tensor([e[1] for e in sim_edges], dtype=torch.long)
    weight = torch.tensor([e[2] for e in sim_edges], dtype=torch.float)

    data["drug", "structurally_similar", "drug"].edge_index = torch.stack([src, dst])
    # Optional: store Tanimoto score as edge attribute
    data["drug", "structurally_similar", "drug"].edge_attr  = weight.unsqueeze(1)

# HGTConv will learn separate W_Q, W_K, W_V for this relation.
# The edge_attr can be incorporated via an edge feature MLP if desired,
# but HGT works well with topology alone.
""")
    print("Done.")


if __name__ == "__main__":
    main()

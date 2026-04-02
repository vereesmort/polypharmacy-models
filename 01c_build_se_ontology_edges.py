"""
01c_build_se_ontology_edges.py

Connects mono side-effect nodes to each other via the Human Phenotype
Ontology (HPO) is-a hierarchy.

Currently each mono SE node is isolated — two nodes representing
"kidney failure" and "acute kidney failure" are not connected even
though one is a subtype of the other. This script adds is-a edges
between SE nodes so that HGT message passing can propagate signal
through phenotypic similarity.

New edge type:  mono_se → mono_se  (se_isa)

Strategy:
1. Load all UMLS CUI identifiers from the DECAGON mono SE file
2. Map UMLS CUIs to HPO term IDs using a pre-built mapping file
   (downloaded from HPO annotations or via UMLS REST API)
3. Load the HPO ontology hierarchy and traverse is-a relationships
4. For each SE pair (i, j) where j is an ancestor of i in HPO,
   add a directed edge i → j (child → parent, more specific → more general)
5. Filter to maximum ancestor depth = 3 to avoid overly distant connections

Fallback strategy (no HPO mapping available):
- Use the MeSH disease class from bio-decagon-effectcategories.csv
- Connect SE nodes that share the same disease class
- This is coarser but requires no external download

Outputs:
    data/processed/se_ontology_edges.json
        [[se_node_i, se_node_j, relation], ...]
        relation: "hpo_isa" or "same_disease_class"
    data/processed/se_ontology_stats.json

Usage:
    python 01c_build_se_ontology_edges.py

    # For HPO-based edges (better), first download:
    # wget https://purl.obolibrary.org/obo/hp.obo -O data/hp.obo
    # HPO to UMLS mapping:
    # wget https://data.bioontology.org/ontologies/HP/download -O data/hp_umls.csv
    # (requires BioPortal API key — free registration)

    # For disease-class fallback (no download needed):
    # Script auto-detects and falls back to this mode
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

DATA  = Path("data")
RAW   = DATA / "raw"
PROC  = DATA / "processed"
PROC.mkdir(parents=True, exist_ok=True)

HPO_OBO_FILE     = DATA / "hp.obo"
HPO_UMLS_FILE    = DATA / "hp_umls_mapping.csv"
MONO_SE_FILE     = RAW  / "bio-decagon-mono.csv"
EFFECT_CAT_FILE  = RAW  / "bio-decagon-effectcategories.csv"
SE_IDX_FILE      = PROC / "se_node_mapping.csv"

OUT_EDGES        = PROC / "se_ontology_edges.json"
OUT_STATS        = PROC / "se_ontology_stats.json"

MAX_ANCESTOR_DEPTH = 3   # how many hops up the ontology to follow


# ── Load SE node mapping ──────────────────────────────────────────────────────

def load_se_idx() -> dict:
    """
    Returns {umls_cui: node_idx} for all mono SE nodes.
    Builds from mono SE file if mapping file doesn't exist.
    """
    se_idx = {}

    if SE_IDX_FILE.exists():
        with open(SE_IDX_FILE) as f:
            for row in csv.DictReader(f):
                se_idx[row["se_id"]] = int(row["node_idx"])
        print(f"  Loaded {len(se_idx)} SE nodes from mapping file")
        return se_idx

    # Build from raw mono SE file
    print("  Building SE index from bio-decagon-mono.csv...")
    seen = set()
    with open(MONO_SE_FILE) as f:
        for row in csv.DictReader(f):
            seen.add(row["Individual Side Effect"])
    se_idx = {cui: i for i, cui in enumerate(sorted(seen))}
    print(f"  Found {len(se_idx)} unique mono SE nodes")
    return se_idx


# ── Strategy A: HPO-based edges ───────────────────────────────────────────────

def load_hpo_graph(obo_path: Path) -> dict:
    """
    Parse hp.obo file and build {hpo_id: [parent_hpo_id, ...]} dict.
    Only reads is-a relationships.
    """
    print("  Parsing HPO OBO file...")
    parents  = defaultdict(list)
    current  = None

    with open(obo_path) as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current = None
            elif line.startswith("id: HP:"):
                current = line.split("id: ")[1].strip()
            elif line.startswith("is_a:") and current:
                parent = line.split("is_a:")[1].strip().split(" ")[0]
                if parent.startswith("HP:"):
                    parents[current].append(parent)
            elif line.startswith("is_obsolete: true"):
                current = None

    print(f"  Loaded {len(parents)} HPO terms with parents")
    return dict(parents)


def load_umls_to_hpo(mapping_path: Path) -> dict:
    """
    Load UMLS CUI → HPO ID mapping.
    Expected CSV columns: umls_cui, hpo_id
    """
    umls_to_hpo = {}
    with open(mapping_path) as f:
        for row in csv.DictReader(f):
            cui = row.get("umls_cui", row.get("CUI", "")).strip()
            hpo = row.get("hpo_id",  row.get("HPO",  "")).strip()
            if cui and hpo:
                umls_to_hpo[cui] = hpo
    print(f"  Loaded {len(umls_to_hpo)} UMLS→HPO mappings")
    return umls_to_hpo


def get_ancestors_up_to_depth(hpo_id: str, parents: dict, max_depth: int) -> list:
    """BFS ancestors up to max_depth hops."""
    visited = []
    frontier = [(hpo_id, 0)]
    seen = {hpo_id}
    while frontier:
        node, depth = frontier.pop(0)
        if depth > 0:
            visited.append((node, depth))
        if depth < max_depth:
            for parent in parents.get(node, []):
                if parent not in seen:
                    seen.add(parent)
                    frontier.append((parent, depth + 1))
    return visited


def build_hpo_edges(se_idx: dict, hpo_parents: dict,
                    umls_to_hpo: dict) -> list:
    """
    For each pair of SE nodes where one is an HPO ancestor of the other
    (within MAX_ANCESTOR_DEPTH), add a directed edge child → ancestor.
    """
    print("  Building HPO is-a edges...")

    # Map each SE node to its HPO ancestors
    se_to_ancestors = {}
    for cui, node_idx in se_idx.items():
        hpo_id = umls_to_hpo.get(cui)
        if hpo_id:
            ancestors = get_ancestors_up_to_depth(hpo_id, hpo_parents,
                                                   MAX_ANCESTOR_DEPTH)
            se_to_ancestors[cui] = (hpo_id, ancestors)

    # Build reverse map: hpo_id → list of (cui, node_idx)
    hpo_to_ses = defaultdict(list)
    for cui, (hpo_id, _) in se_to_ancestors.items():
        hpo_to_ses[hpo_id].append((cui, se_idx[cui]))

    # For each SE, connect to SEs that map to its ancestor HPO terms
    edges = []
    for cui, (hpo_id, ancestors) in se_to_ancestors.items():
        src_idx = se_idx[cui]
        for anc_hpo, depth in ancestors:
            for anc_cui, anc_idx in hpo_to_ses.get(anc_hpo, []):
                if anc_idx != src_idx:
                    edges.append([src_idx, anc_idx, "hpo_isa", depth])

    # Deduplicate
    seen_pairs = set()
    deduped = []
    for e in edges:
        key = (e[0], e[1])
        if key not in seen_pairs:
            seen_pairs.add(key)
            deduped.append(e)

    print(f"  Built {len(deduped)} HPO is-a edges")
    return deduped


# ── Strategy B: Disease-class fallback edges ──────────────────────────────────

def build_disease_class_edges(se_idx: dict) -> list:
    """
    Connect SE nodes that share the same disease class from
    bio-decagon-effectcategories.csv. This is a coarse fallback
    when HPO mapping is unavailable.

    Edges are undirected (add both directions) since disease class
    is a flat grouping, not a hierarchy.
    """
    print("  Building disease-class edges (fallback mode)...")

    if not EFFECT_CAT_FILE.exists():
        print("  bio-decagon-effectcategories.csv not found — skipping")
        return []

    # Load disease class per SE
    se_to_class = {}
    with open(EFFECT_CAT_FILE) as f:
        for row in csv.DictReader(f):
            cui = row["Side Effect"]
            cls = row["Disease Class"]
            se_to_class[cui] = cls

    # Group SE nodes by disease class
    class_to_ses = defaultdict(list)
    for cui, node_idx in se_idx.items():
        cls = se_to_class.get(cui)
        if cls:
            class_to_ses[cls].append((cui, node_idx))

    # Connect all pairs within same disease class
    edges = []
    for cls, members in class_to_ses.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                _, idx_i = members[i]
                _, idx_j = members[j]
                # Both directions (undirected)
                edges.append([idx_i, idx_j, "same_disease_class", 1])
                edges.append([idx_j, idx_i, "same_disease_class", 1])

    # Filter to classes with a reasonable number of members
    # Very large classes (e.g. "unannotated") would add too many edges
    class_sizes = {cls: len(m) for cls, m in class_to_ses.items()}
    max_class_size = 50  # cap to avoid O(n^2) explosion in large classes
    filtered = [e for e in edges
                if class_sizes.get(se_to_class.get(
                    next(cui for cui, idx in
                         sum(class_to_ses.values(), [])
                         if idx == e[0]), ""), 0) <= max_class_size]

    print(f"  Disease classes found: {len(class_to_ses)}")
    print(f"  Built {len(filtered)} disease-class edges "
          f"(max class size = {max_class_size})")
    return filtered


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("01c_build_se_ontology_edges.py")
    print("=" * 60)

    se_idx = load_se_idx()

    # Try HPO-based edges first
    if HPO_OBO_FILE.exists() and HPO_UMLS_FILE.exists():
        print("\nHPO files found — using HPO is-a edges (preferred)")
        hpo_parents  = load_hpo_graph(HPO_OBO_FILE)
        umls_to_hpo  = load_umls_to_hpo(HPO_UMLS_FILE)
        edges        = build_hpo_edges(se_idx, hpo_parents, umls_to_hpo)
        method       = "hpo_isa"
    else:
        print("\nHPO files not found — using disease-class fallback")
        print("  To use HPO edges (better), download:")
        print("  wget https://purl.obolibrary.org/obo/hp.obo -O data/hp.obo")
        print("  (UMLS→HPO mapping requires BioPortal API key)")
        print()
        edges  = build_disease_class_edges(se_idx)
        method = "same_disease_class"

    # Statistics
    stats = {
        "method":           method,
        "total_se_nodes":   len(se_idx),
        "total_edges":      len(edges),
        "se_nodes_covered": len({e[0] for e in edges} | {e[1] for e in edges}),
        "max_depth":        MAX_ANCESTOR_DEPTH if method == "hpo_isa" else 1,
    }

    # Save
    with open(OUT_EDGES, "w") as f:
        json.dump(edges, f)
    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved → {OUT_EDGES}  ({len(edges)} edges)")
    print(f"Saved → {OUT_STATS}")

    # Print integration snippet
    print("""
Integration snippet for 02_build_graph.py:
──────────────────────────────────────────
import json, torch

with open("data/processed/se_ontology_edges.json") as f:
    se_ont_edges = json.load(f)

if se_ont_edges:
    src = torch.tensor([e[0] for e in se_ont_edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in se_ont_edges], dtype=torch.long)
    data["mono_se", "se_isa", "mono_se"].edge_index = torch.stack([src, dst])
    # For undirected (disease class fallback):
    # the fallback already adds both directions, so no reverse needed
    # For directed HPO edges, optionally add reverse:
    # data["mono_se", "rev_se_isa", "mono_se"].edge_index = torch.stack([dst, src])
""")
    print("Done.")


if __name__ == "__main__":
    main()

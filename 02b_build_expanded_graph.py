"""
02b_build_expanded_graph.py

Extends the base heterogeneous graph with Priority 1 biological priors:

  1. Pathway nodes + protein→pathway edges  (KEGG)
  2. SE→SE ontology edges                   (phenotypic keyword groups,
                                             upgrades to HPO if hp.obo available)
  3. Drug→drug structural similarity edges  (RDKit ECFP4, or SMILES trigram fallback)

Loads the base graph from data/processed/graph.pt and returns an
expanded HeteroData object with three additional edge types:

  ("protein",  "member_of",          "pathway")
  ("pathway",  "rev_member_of",      "protein")
  ("mono_se",  "se_isa",             "mono_se")
  ("drug",     "structurally_similar","drug"  )

The expanded graph is saved to data/processed/graph_expanded.pt.
All downstream scripts (03_build_splits.py, 04_train.py) can then
load this file in place of graph.pt with no other changes required.

Usage:
    python 02b_build_expanded_graph.py

Requirements:
    - data/processed/graph.pt           (from 02_build_graph.py)
    - data/kegg_pathways.json           (from 01_fetch_kegg.py)
    - data/raw/bio-decagon-mono.csv     (DECAGON)
    - data/raw/drug_smiles.csv          (from 00b_fetch_smiles.py)
    - data/raw/drug_smiles.csv          (for structural similarity)
    Optional:
    - data/hp.obo                       (HPO ontology — for richer SE edges)
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

DATA = Path("data")
RAW  = DATA / "raw"
PROC = DATA / "processed"

IN_GRAPH   = PROC / "graph.pt"
KEGG_FILE  = DATA / "kegg_pathways.json"
MONO_FILE  = RAW  / "bio-decagon-mono.csv"
SMILES_FILE= RAW  / "drug_smiles.csv"
HPO_FILE   = DATA / "hp.obo"

OUT_GRAPH  = PROC / "graph_expanded.pt"
OUT_STATS  = PROC / "graph_expanded_stats.json"


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_base_graph() -> HeteroData:
    print(f"Loading base graph from {IN_GRAPH}...")
    data = torch.load(IN_GRAPH, map_location="cpu")
    for ntype in data.node_types:
        n = data[ntype].num_nodes
        print(f"  {ntype:<12}: {n:>7} nodes")
    for etype in data.edge_types:
        e = data[etype].edge_index.shape[1]
        print(f"  {str(etype):<50}: {e:>8} edges")
    return data


def safe_add_edges(data: HeteroData, src_type: str, rel: str,
                   dst_type: str, src: list, dst: list,
                   edge_attr=None) -> int:
    """Add edges to HeteroData, skipping if empty."""
    if not src:
        print(f"  Skipping ({src_type}, {rel}, {dst_type}) — no edges")
        return 0
    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)
    data[src_type, rel, dst_type].edge_index = torch.stack([src_t, dst_t])
    if edge_attr is not None:
        data[src_type, rel, dst_type].edge_attr = torch.tensor(
            edge_attr, dtype=torch.float).unsqueeze(1)
    n = len(src)
    print(f"  Added ({src_type}, {rel}, {dst_type}): {n} edges")
    return n


# ── 1. Pathway nodes + protein→pathway edges ─────────────────────────────────

def add_pathway_edges(data: HeteroData) -> int:
    """
    Adds:
        pathway node type (if KEGG available)
        (protein, member_of, pathway) edges
        (pathway, rev_member_of, protein) edges
    """
    print("\n[1] Pathway edges (KEGG)...")

    if not KEGG_FILE.exists():
        print(f"  {KEGG_FILE} not found — skipping pathway edges")
        print("  Run 01_fetch_kegg.py first to fetch KEGG pathway data")
        return 0

    with open(KEGG_FILE) as f:
        kegg = json.load(f)

    # Build protein gene_id → node_idx mapping from base graph
    # The base graph stores protein indices — we need to know which gene IDs
    # correspond to which node indices.
    # Load from the protein node mapping file if available.
    prot_mapping_file = PROC / "protein_node_mapping.csv"
    gene_to_idx = {}

    if prot_mapping_file.exists():
        with open(prot_mapping_file) as f:
            for row in csv.DictReader(f):
                gene_to_idx[row["gene_id"]] = int(row["node_idx"])
    else:
        # Fall back to building from PPI file
        print("  protein_node_mapping.csv not found — building from PPI file")
        ppi_file = RAW / "bio-decagon-ppi.csv"
        if ppi_file.exists():
            genes = set()
            with open(ppi_file) as f:
                for row in csv.DictReader(f):
                    genes.add(row["Gene 1"])
                    genes.add(row["Gene 2"])
            gene_to_idx = {g: i for i, g in enumerate(sorted(genes))}

    if not gene_to_idx:
        print("  Could not build gene→idx mapping — skipping pathway edges")
        return 0

    # Build pathway nodes and edges
    all_pathways     = sorted(kegg.keys())
    pw_to_idx        = {pw: i for i, pw in enumerate(all_pathways)}
    n_pathways       = len(all_pathways)

    src_prot, dst_pw = [], []
    covered_pws      = set()

    for pw_id, info in kegg.items():
        pw_idx = pw_to_idx[pw_id]
        for gene in info.get("genes", []):
            if gene in gene_to_idx:
                src_prot.append(gene_to_idx[gene])
                dst_pw.append(pw_idx)
                covered_pws.add(pw_id)

    # Deduplicate
    pairs     = list(set(zip(src_prot, dst_pw)))
    src_prot  = [p[0] for p in pairs]
    dst_pw    = [p[1] for p in pairs]

    # Add pathway nodes — use identity features (will be projected by InputProjection)
    data["pathway"].num_nodes = n_pathways
    data["pathway"].x = torch.zeros(n_pathways, 1)  # placeholder; projected by model

    n_fwd = safe_add_edges(data, "protein", "member_of",     "pathway", src_prot, dst_pw)
    n_rev = safe_add_edges(data, "pathway", "rev_member_of", "protein", dst_pw,   src_prot)

    print(f"  Pathways: {n_pathways} total, {len(covered_pws)} with coverage")
    return n_fwd + n_rev


# ── 2. SE ontology edges ──────────────────────────────────────────────────────

def add_se_ontology_edges(data: HeteroData) -> int:
    """
    Adds:
        (mono_se, se_isa, mono_se) edges

    Uses HPO if hp.obo is available; otherwise uses keyword-based
    phenotypic groups derived from SE names.
    """
    print("\n[2] SE ontology edges...")

    # Build SE node index from mono file
    mono_names = {}
    if not MONO_FILE.exists():
        print(f"  {MONO_FILE} not found — skipping SE ontology edges")
        return 0

    with open(MONO_FILE) as f:
        for row in csv.DictReader(f):
            mono_names[row["Individual Side Effect"]] = \
                row["Side Effect Name"].lower()

    se_idx = {cui: i for i, cui in enumerate(sorted(mono_names.keys()))}

    # Try HPO first
    if HPO_FILE.exists():
        edges = _se_edges_hpo(se_idx, mono_names)
        method = "HPO is-a"
    else:
        edges = _se_edges_keyword(se_idx, mono_names)
        method = "keyword phenotypic groups"

    if not edges:
        print(f"  No SE ontology edges built")
        return 0

    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    n   = safe_add_edges(data, "mono_se", "se_isa", "mono_se", src, dst)
    print(f"  SE ontology method: {method}")
    return n


def _se_edges_keyword(se_idx: dict, mono_names: dict) -> list:
    """
    Keyword-based phenotypic grouping fallback.
    Groups SE nodes by clinical phenotype inferred from SE name.
    """
    GROUPS = {
        "renal":           ["kidney","renal","nephro","creatinine","proteinuria",
                            "oliguria","glomerulo","urinary"],
        "hepatic":         ["liver","hepat","bilirubin","transaminase",
                            "cholestasis","jaundice","cirrhosis"],
        "cardiac":         ["cardiac","heart","myocardial","arrhythmia","tachycardia",
                            "bradycardia","angina","palpitation","coronary"],
        "haematological":  ["anaemia","anemia","thrombocytopenia","leukopenia",
                            "neutropenia","pancytopenia","haemorrhage","bleeding"],
        "neurological":    ["neuropathy","seizure","encephalopathy","headache",
                            "dizziness","tremor","ataxia","cognitive","confusion",
                            "paralysis","paresthesia"],
        "gastrointestinal":["nausea","vomiting","diarrhea","constipation","abdominal",
                            "colitis","gastric","pancreatitis","intestinal"],
        "respiratory":     ["pneumonia","dyspnoea","cough","pulmonary","bronchospasm",
                            "respiratory","pleural","hypoventilation"],
        "dermatological":  ["rash","pruritus","urticaria","dermatitis","alopecia",
                            "skin","photosensitivity","erythema","exanthema"],
        "musculoskeletal": ["myopathy","arthralgia","myalgia","rhabdomyolysis",
                            "bone","joint","tendon","muscle"],
        "endocrine":       ["hypothyroidism","hyperthyroid","diabetes","hyperglycaemia",
                            "adrenal","thyroid","hormone","cortisol"],
        "immune":          ["allergy","anaphylaxis","hypersensitivity","autoimmune",
                            "immune","lupus","vasculitis"],
        "infectious":      ["infection","sepsis","bacteremia","fungal","viral",
                            "opportunistic","abscess"],
        "metabolic":       ["electrolyte","hyponatremia","hypokalemia","hypocalcemia",
                            "acidosis","alkalosis","dehydration","asthenia","fatigue",
                            "weight"],
        "ocular":          ["eye","visual","optic","retina","cataract","glaucoma"],
        "psychiatric":     ["anxiety","depression","psychosis","hallucination",
                            "insomnia","agitation","mania"],
    }

    se_to_group = {}
    for cui, name in mono_names.items():
        for group, keywords in GROUPS.items():
            if any(kw in name for kw in keywords):
                se_to_group[cui] = group
                break

    group_to_nodes = defaultdict(list)
    for cui, group in se_to_group.items():
        group_to_nodes[group].append(se_idx[cui])

    MAX_PER_GROUP = 100  # cap to avoid O(n^2) explosion in large groups
    edges = []
    for group, members in group_to_nodes.items():
        capped = members[:MAX_PER_GROUP]
        for i in range(len(capped)):
            for j in range(i + 1, len(capped)):
                edges.append([capped[i], capped[j]])
                edges.append([capped[j], capped[i]])

    print(f"  Keyword groups: {len(group_to_nodes)} | "
          f"SEs assigned: {len(se_to_group)} / {len(mono_names)}")
    print(f"  Built {len(edges)} SE ontology edges")
    return edges


def _se_edges_hpo(se_idx: dict, mono_names: dict) -> list:
    """
    HPO-based is-a edges. Requires data/hp.obo.
    Maps UMLS CUIs to HPO terms via SE name matching,
    then traverses the is-a hierarchy up to depth 3.
    """
    print(f"  Parsing {HPO_FILE}...")

    # Parse HPO OBO file
    hpo_parents  = defaultdict(list)
    hpo_names    = {}
    current_id   = None
    current_name = None

    with open(HPO_FILE) as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id = None; current_name = None
            elif line.startswith("id: HP:"):
                current_id = line.split("id: ")[1].strip()
            elif line.startswith("name:") and current_id:
                current_name = line.split("name:")[1].strip().lower()
                hpo_names[current_id] = current_name
            elif line.startswith("is_a:") and current_id:
                parent = line.split("is_a:")[1].strip().split(" ")[0]
                if parent.startswith("HP:"):
                    hpo_parents[current_id].append(parent)

    # Match SE names to HPO terms by name similarity (simple substring)
    hpo_name_to_id = {v: k for k, v in hpo_names.items()}
    se_to_hpo = {}
    for cui, name in mono_names.items():
        if name in hpo_name_to_id:
            se_to_hpo[cui] = hpo_name_to_id[name]

    print(f"  SE→HPO name matches: {len(se_to_hpo)}")

    if len(se_to_hpo) < 10:
        print("  Too few matches — falling back to keyword groups")
        return _se_edges_keyword(se_idx, mono_names)

    # Build edges: for each SE, find other SEs that share an HPO ancestor
    hpo_to_ses = defaultdict(list)
    for cui, hpo_id in se_to_hpo.items():
        hpo_to_ses[hpo_id].append(se_idx[cui])
        # Walk ancestors up to depth 3
        frontier = [hpo_id]
        for _ in range(3):
            next_frontier = []
            for h in frontier:
                for parent in hpo_parents.get(h, []):
                    hpo_to_ses[parent].append(se_idx[cui])
                    next_frontier.append(parent)
            frontier = next_frontier

    edges = []
    seen  = set()
    for members in hpo_to_ses.values():
        unique = list(set(members))
        for i in range(len(unique)):
            for j in range(i + 1, min(len(unique), 50)):
                key = (unique[i], unique[j])
                if key not in seen:
                    seen.add(key)
                    seen.add((unique[j], unique[i]))
                    edges.append([unique[i], unique[j]])
                    edges.append([unique[j], unique[i]])

    print(f"  Built {len(edges)} HPO is-a edges")
    return edges


# ── 3. Drug structural similarity edges ──────────────────────────────────────

def add_structural_similarity_edges(data: HeteroData) -> int:
    """
    Adds:
        (drug, structurally_similar, drug) edges

    Uses RDKit ECFP4 if available; SMILES trigram fallback otherwise.
    """
    print("\n[3] Structural similarity edges (SMILES)...")

    if not SMILES_FILE.exists():
        print(f"  {SMILES_FILE} not found — skipping structural similarity edges")
        print("  Run 00b_fetch_smiles.py first")
        return 0

    # Load drug node mapping
    drug_mapping_file = PROC / "drug_node_mapping.csv"
    drug_idx = {}
    if drug_mapping_file.exists():
        with open(drug_mapping_file) as f:
            for row in csv.DictReader(f):
                drug_idx[row["drug_id"]] = int(row["node_idx"])
    else:
        # Build from SMILES file + graph node count
        print("  drug_node_mapping.csv not found — building drug_idx from SMILES file")
        with open(SMILES_FILE) as f:
            drugs = [row.get("drug_id", row.get("STITCH","")).strip()
                     for row in csv.DictReader(f)]
        drug_idx = {d: i for i, d in enumerate(sorted(set(drugs))) if d}

    # Load SMILES
    smiles_map = {}  # node_idx → smiles
    with open(SMILES_FILE) as f:
        for row in csv.DictReader(f):
            did    = row.get("drug_id", row.get("STITCH", "")).strip()
            smiles = row.get("smiles",  row.get("SMILES",  "")).strip()
            if did in drug_idx and smiles and smiles.upper() != "N/A":
                smiles_map[drug_idx[did]] = smiles

    print(f"  SMILES available for {len(smiles_map)} / {len(drug_idx)} drugs")

    if not smiles_map:
        print("  No SMILES loaded — skipping structural similarity")
        return 0

    # Compute fingerprints
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator, DataStructs

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fps = {}
        for idx, smi in smiles_map.items():
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps[idx] = gen.GetFingerprint(mol)

        def sim_fn(a, b):
            return DataStructs.TanimotoSimilarity(a, b)

        print(f"  Using RDKit ECFP4 fingerprints ({len(fps)} drugs)")
        fp_method = "ECFP4"

    except ImportError:
        print("  RDKit not available — using SMILES trigram fallback")
        print("  Install: pip install rdkit")
        fps = {}
        for idx, smi in smiles_map.items():
            clean = smi.replace("@","").replace("/","").replace("\\","")
            tgrams = frozenset(clean[i:i+3] for i in range(len(clean)-2))
            if tgrams:
                fps[idx] = tgrams

        def sim_fn(a, b):
            if not a or not b: return 0.0
            return len(a & b) / len(a | b)

        fp_method = "trigram_fallback"

    # Compute pairwise similarity
    THRESHOLD      = 0.6
    MAX_PER_DRUG   = 20
    indices        = sorted(fps.keys())
    n              = len(indices)
    candidates     = defaultdict(list)
    total_pairs    = n * (n - 1) // 2
    checked        = 0

    print(f"  Computing {total_pairs:,} pairwise similarities "
          f"(threshold={THRESHOLD})...")

    for i_pos in range(n):
        i = indices[i_pos]
        for j_pos in range(i_pos + 1, n):
            j   = indices[j_pos]
            sim = sim_fn(fps[i], fps[j])
            if sim >= THRESHOLD:
                candidates[i].append((sim, j))
                candidates[j].append((sim, i))
            checked += 1
            if checked % 100_000 == 0:
                pct = checked / total_pairs * 100
                print(f"    {checked:,}/{total_pairs:,} ({pct:.1f}%)")

    # Apply per-drug cap
    src, dst, weights = [], [], []
    seen = set()
    for i in indices:
        top_k = sorted(candidates[i], reverse=True)[:MAX_PER_DRUG]
        for score, j in top_k:
            if (i, j) not in seen:
                seen.add((i, j)); seen.add((j, i))
                src.extend([i, j])
                dst.extend([j, i])
                weights.extend([score, score])

    n_edges = safe_add_edges(
        data, "drug", "structurally_similar", "drug",
        src, dst, edge_attr=weights
    )
    print(f"  Fingerprint method: {fp_method}")
    print(f"  Unique similar pairs: {len(src)//2}")
    return n_edges


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("02b_build_expanded_graph.py — Priority 1 biological priors")
    print("=" * 60)
    print()
    print("Adding:")
    print("  1. Pathway nodes + protein→pathway edges  (KEGG)")
    print("  2. SE ontology edges                      (HPO or keyword groups)")
    print("  3. Drug structural similarity edges       (RDKit ECFP4 or fallback)")
    print()

    data = load_base_graph()

    stats = {
        "base": {
            "node_types": list(data.node_types),
            "edge_types": [str(et) for et in data.edge_types],
        },
        "edges_added": {}
    }

    n_pw  = add_pathway_edges(data)
    n_se  = add_se_ontology_edges(data)
    n_str = add_structural_similarity_edges(data)

    stats["edges_added"] = {
        "pathway_edges":              n_pw,
        "se_ontology_edges":          n_se,
        "structural_similarity_edges":n_str,
    }
    stats["expanded"] = {
        "node_types": list(data.node_types),
        "edge_types": [str(et) for et in data.edge_types],
    }

    # Save
    torch.save(data, OUT_GRAPH)
    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved expanded graph → {OUT_GRAPH}")
    print(f"Saved stats         → {OUT_STATS}")
    print()
    print("Node types in expanded graph:")
    for ntype in data.node_types:
        n = data[ntype].num_nodes
        print(f"  {ntype:<15}: {n:>8} nodes")
    print()
    print("Edge types in expanded graph:")
    for etype in data.edge_types:
        e = data[etype].edge_index.shape[1]
        print(f"  {str(etype):<55}: {e:>9} edges")

    print()
    print("Next step: update 04_train.py to load graph_expanded.pt")
    print("  Change: data = torch.load('data/processed/graph.pt')")
    print("  To:     data = torch.load('data/processed/graph_expanded.pt')")
    print()
    print("The HGT encoder will automatically learn new relation-specific")
    print("W_Q, W_K, W_V matrices for each new edge type.")
    print("Done.")


if __name__ == "__main__":
    main()

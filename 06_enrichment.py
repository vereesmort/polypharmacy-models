"""
06_enrichment.py
Validate biological coherence of learned drug embeddings via two analyses:

1. Target gene GO/pathway enrichment per drug cluster  (input side)
   — do drugs in the same cluster share biological target annotations?

2. Combo side effect disease class profiling per cluster  (output side) ← NEW
   — do drugs in the same cluster disproportionately cause side effects
     of the same disease class when combined with other drugs?

This cross-validation between input (target biology) and output (SE phenotype)
provides two independent lines of evidence for biologically meaningful embeddings.

Outputs:
    results/drug_embeddings.npy
    results/drug_embeddings_pca2d.npy
    results/cluster_labels.npy
    results/drug_clusters.csv
    results/enrichment_per_cluster/cluster_XX.csv   — GO/pathway terms
    results/enrichment_summary.json
    results/cluster_disease_class_profile.csv       ← NEW
    results/cluster_disease_class_summary.json      ← NEW
"""

import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from model import PolypharmacyHGT

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED      = Path("data/processed")
CHECKPOINTS    = Path("checkpoints")
RESULTS        = Path("results")
ENRICHMENT_DIR = RESULTS / "enrichment_per_cluster"
RAW            = Path("data/raw")
CATEGORIES_CSV = RAW / "bio-decagon-effectcategories.csv"
COMBO_FILE     = RAW / "bio-decagon-combo.csv"   # full file; or use chunks below
COMBO_CHUNKS   = [RAW / f"bio-decagon-combo-{i}.csv" for i in range(1, 9)]

ENRICHMENT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 15
SEED       = 42


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_effect_categories(path):
    if not path.exists():
        print(f"  Warning: {path} not found.")
        return {}
    cats = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            cats[row["Side Effect"]] = row["Disease Class"]
    print(f"  Loaded {len(cats)} SE disease class annotations.")
    return cats


def load_combo_se_per_pair(drug_idx, combo_file=None, combo_chunks=None):
    """
    Returns {(drug_i_node, drug_j_node): set of se_ids}
    Uses full combo file if available, else chunks.
    """
    pair_ses = defaultdict(set)
    idx = drug_idx

    def process_file(fpath):
        if not fpath.exists():
            return
        with open(fpath) as f:
            for row in csv.DictReader(f):
                d1, d2, se = row["STITCH 1"], row["STITCH 2"], row["Polypharmacy Side Effect"]
                if d1 in idx and d2 in idx:
                    key = (min(idx[d1], idx[d2]), max(idx[d1], idx[d2]))
                    pair_ses[key].add(se)

    if combo_file and Path(combo_file).exists():
        print(f"  Loading combo from {combo_file}...")
        process_file(Path(combo_file))
    else:
        print(f"  Loading combo from chunks...")
        for chunk in combo_chunks:
            process_file(chunk)

    print(f"  Loaded {len(pair_ses)} drug pairs with combo SE data.")
    return pair_ses


def load_model_and_embeddings(data, drug_pathway_map, num_se, num_pathways, device):
    ckpt = torch.load(CHECKPOINTS / "best_model.pt", map_location=device, weights_only=False)
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
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print("Computing drug embeddings...")
    z_drug = model.get_drug_embeddings(data, drug_pathway_map, device)
    return z_drug.cpu().numpy()


# ── Analysis 1: GO/pathway enrichment ─────────────────────────────────────────

def get_cluster_gene_sets(cluster_labels, drug_idx, drug_to_proteins):
    idx_to_drug = {v: k for k, v in drug_idx.items()}
    n_clusters  = max(cluster_labels) + 1
    cluster_genes = {c: set() for c in range(n_clusters)}
    for node_idx, cluster_id in enumerate(cluster_labels):
        drug_id = idx_to_drug.get(node_idx)
        if drug_id and drug_id in drug_to_proteins:
            cluster_genes[cluster_id].update(drug_to_proteins[drug_id])
    return cluster_genes


def run_gprofiler(gene_set, organism="hsapiens"):
    try:
        from gprofiler import GProfiler
    except ImportError:
        print("  gprofiler-official not installed.")
        return []
    if len(gene_set) < 3:
        return []
    gp = GProfiler(return_dataframe=False)
    try:
        return gp.profile(
            organism=organism,
            query=list(gene_set),
            sources=["GO:BP", "GO:MF", "KEGG", "REAC"],
            significance_threshold_method="fdr",
            user_threshold=0.05,
            no_evidences=True,
        )
    except Exception as e:
        print(f"  g:Profiler error: {e}")
        return []


# ── Analysis 2: Disease class profile per cluster ──────────────────────────────

def compute_cluster_disease_profiles(
    cluster_labels, drug_idx, pair_ses, se_categories, n_clusters
):
    """
    For each cluster, find all combo SE types that occur between drug pairs
    where BOTH drugs are in that cluster.
    Then compute what fraction of those SE types fall into each disease class.

    Returns:
        profiles: {cluster_id: Counter of disease_class -> count}
        pair_counts: {cluster_id: n_intra_cluster_pairs}
    """
    profiles    = {c: Counter() for c in range(n_clusters)}
    pair_counts = Counter()

    # Build cluster membership lookup
    drug_to_cluster = {node_idx: int(cl) for node_idx, cl in enumerate(cluster_labels)}

    for (d1_idx, d2_idx), ses in pair_ses.items():
        c1 = drug_to_cluster.get(d1_idx)
        c2 = drug_to_cluster.get(d2_idx)
        if c1 is None or c2 is None:
            continue
        # Only count intra-cluster pairs (both drugs in same cluster)
        if c1 != c2:
            continue
        pair_counts[c1] += 1
        for se in ses:
            cat = se_categories.get(se)
            if cat:
                profiles[c1][cat] += 1

    return profiles, pair_counts


def summarise_disease_profiles(profiles, pair_counts, n_clusters):
    """
    Returns list of rows for CSV output:
    cluster_id, disease_class, se_count, fraction_of_cluster_ses, n_pairs
    """
    rows = []
    for c in range(n_clusters):
        total = sum(profiles[c].values())
        if total == 0:
            continue
        for disease_class, count in profiles[c].most_common():
            rows.append({
                "cluster_id":           c,
                "disease_class":        disease_class,
                "se_count":             count,
                "fraction":             round(count / total, 4),
                "n_intra_cluster_pairs": pair_counts[c],
            })
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cpu")
    print("Loading data...")
    data  = torch.load(PROCESSED / "graph.pt",        weights_only=False)
    combo = torch.load(PROCESSED / "combo_edges.pt",  weights_only=False)
    num_se = len(combo["top_se_ids"])

    with open(PROCESSED / "pathway_memberships.pkl", "rb") as f:
        pathway_data = pickle.load(f)
    drug_pathway_map = pathway_data["drug_pathway_map"]
    num_pathways     = len(pathway_data["pathway_id_to_col"])

    with open(PROCESSED / "meta.json") as f:
        meta = json.load(f)
    drug_idx = meta["drug_idx"]

    # Load drug-protein targets
    drug_to_proteins = defaultdict(set)
    with open(RAW / "bio-decagon-targets.csv") as f:
        for row in csv.DictReader(f):
            drug_to_proteins[row["STITCH"]].add(row["Gene"])

    # Load disease class annotations
    se_categories = load_effect_categories(CATEGORIES_CSV)

    # ── Embeddings + clustering ────────────────────────────────────────────────
    z_drug = load_model_and_embeddings(data, drug_pathway_map, num_se, num_pathways, device)
    print(f"Drug embeddings shape: {z_drug.shape}")
    np.save(RESULTS / "drug_embeddings.npy", z_drug)

    print(f"K-means clustering into {N_CLUSTERS} clusters...")
    km             = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    cluster_labels = km.fit_predict(z_drug)
    np.save(RESULTS / "cluster_labels.npy", cluster_labels)

    # Save cluster assignments
    idx_to_drug = {v: k for k, v in drug_idx.items()}
    with open(RESULTS / "drug_clusters.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["drug_id", "node_idx", "cluster_id"])
        for node_idx, cluster_id in enumerate(cluster_labels):
            writer.writerow([idx_to_drug.get(node_idx, "unknown"), node_idx, int(cluster_id)])
    print(f"Saved cluster assignments → {RESULTS / 'drug_clusters.csv'}")

    # PCA 2D for plotting
    pca  = PCA(n_components=2, random_state=SEED)
    z_2d = pca.fit_transform(z_drug)
    np.save(RESULTS / "drug_embeddings_pca2d.npy", z_2d)
    print(f"PCA variance explained (2 PCs): {pca.explained_variance_ratio_.sum():.3f}")

    # ── Analysis 1: GO/pathway enrichment ─────────────────────────────────────
    cluster_genes = get_cluster_gene_sets(cluster_labels, drug_idx, drug_to_proteins)
    enrichment_summary = {}

    for c in range(N_CLUSTERS):
        genes   = cluster_genes[c]
        n_drugs = int((cluster_labels == c).sum())
        print(f"\nCluster {c}: {n_drugs} drugs, {len(genes)} unique target genes")

        if len(genes) < 3:
            print("  Too few genes for enrichment, skipping.")
            enrichment_summary[c] = {
                "n_drugs": n_drugs, "n_genes": len(genes), "n_significant": 0
            }
            continue

        results = run_gprofiler(genes)
        if results:
            cluster_file = ENRICHMENT_DIR / f"cluster_{c:02d}.csv"
            with open(cluster_file, "w", newline="") as f:
                fields = list(results[0].keys())
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(results)

            top_terms = [
                {"source": r.get("source"), "name": r.get("name"),
                 "p_value": r.get("p_value")}
                for r in sorted(results, key=lambda x: x.get("p_value", 1))[:5]
            ]
            enrichment_summary[c] = {
                "n_drugs":       n_drugs,
                "n_genes":       len(genes),
                "n_significant": len(results),
                "top_terms":     top_terms,
            }
            print(f"  Significant terms: {len(results)}")
            for t in top_terms[:3]:
                print(f"    [{t['source']}] {t['name']}  p={t['p_value']:.2e}")
        else:
            enrichment_summary[c] = {
                "n_drugs": n_drugs, "n_genes": len(genes), "n_significant": 0
            }

    with open(RESULTS / "enrichment_summary.json", "w") as f:
        json.dump(enrichment_summary, f, indent=2, default=str)
    print(f"\nSaved enrichment summary → {RESULTS / 'enrichment_summary.json'}")

    # ── Analysis 2: Disease class profile per cluster ──────────────────────────
    if se_categories:
        print("\nLoading combo SE data for disease class profiling...")
        pair_ses = load_combo_se_per_pair(
            drug_idx,
            combo_file=COMBO_FILE,
            combo_chunks=COMBO_CHUNKS,
        )

        print("Computing disease class profiles per cluster...")
        profiles, pair_counts = compute_cluster_disease_profiles(
            cluster_labels, drug_idx, pair_ses, se_categories, N_CLUSTERS
        )

        profile_rows = summarise_disease_profiles(profiles, pair_counts, N_CLUSTERS)

        with open(RESULTS / "cluster_disease_class_profile.csv", "w", newline="") as f:
            fields = ["cluster_id", "disease_class", "se_count",
                      "fraction", "n_intra_cluster_pairs"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(profile_rows)
        print(f"Saved disease class profile → {RESULTS / 'cluster_disease_class_profile.csv'}")

        # Print summary table
        print("\n── Disease class profile per cluster (top-2 classes) ───────────────")
        print(f"  {'Cluster':>7}  {'Drugs':>5}  {'Top disease class':<40}  {'Frac':>5}  "
              f"{'2nd disease class':<35}  {'Frac':>5}")
        print(f"  {'-'*7}  {'-'*5}  {'-'*40}  {'-'*5}  {'-'*35}  {'-'*5}")

        for c in range(N_CLUSTERS):
            n_drugs = int((cluster_labels == c).sum())
            top2    = profiles[c].most_common(2)
            total   = sum(profiles[c].values())
            if not top2:
                print(f"  {c:>7}  {n_drugs:>5}  {'(no annotated SE data)':<40}")
                continue
            first  = top2[0]
            second = top2[1] if len(top2) > 1 else ("—", 0)
            print(f"  {c:>7}  {n_drugs:>5}  {first[0]:<40}  "
                  f"{first[1]/total:>5.2f}  {second[0]:<35}  "
                  f"{second[1]/total:>5.2f}")

        # Save cluster-level summary JSON
        cluster_dc_summary = {}
        for c in range(N_CLUSTERS):
            total = sum(profiles[c].values())
            cluster_dc_summary[c] = {
                "n_intra_pairs":      pair_counts[c],
                "n_annotated_se":     total,
                "top_disease_classes": [
                    {"disease_class": dc, "count": cnt,
                     "fraction": round(cnt / total, 4) if total > 0 else 0}
                    for dc, cnt in profiles[c].most_common(5)
                ],
            }
        with open(RESULTS / "cluster_disease_class_summary.json", "w") as f:
            json.dump(cluster_dc_summary, f, indent=2)
        print(f"\nSaved cluster disease class summary → "
              f"{RESULTS / 'cluster_disease_class_summary.json'}")


if __name__ == "__main__":
    main()

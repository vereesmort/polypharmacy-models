"""
00c_feature_selection.py
Feature selection for drug input representations before graph construction.

Applies three complementary methods:

1. Frequency filtering (stop-word removal analogy)
   - Remove proteins/SEs targeted by too few drugs (rare = noise)
   - Remove proteins/SEs targeted by too many drugs (ubiquitous = no signal)

2. TF-IDF weighting
   - TF = 1 (binary presence in this drug)
   - IDF = log(N / df) where df = number of drugs sharing this feature
   - High TF-IDF = feature is present in this drug AND rare across others
   - Used both for ranking features and as edge weights in the graph

3. Variance filtering
   - Remove features with near-zero variance across all drugs
   - These are columns that are almost always 0 or almost always 1

Outputs:
    data/feature_selection/
        protein_selected.json     — list of kept protein gene IDs
        se_selected.json          — list of kept mono SE IDs
        drug_protein_tfidf.npz    — sparse TF-IDF weighted drug-protein matrix
        drug_se_tfidf.npz         — sparse TF-IDF weighted drug-SE matrix
        feature_stats.json        — summary statistics

Usage: Run before 02_build_graph.py. The graph builder will detect
       and load the filtered feature sets automatically.
"""

import csv
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import sparse

RAW    = Path("data/raw")
OUT    = Path("data/feature_selection")
OUT.mkdir(parents=True, exist_ok=True)

# ── Config — tune these based on your dataset analysis ────────────────────────
PROTEIN_MIN_FREQ   = 3     # remove proteins targeted by < 3 drugs (noise)
PROTEIN_MAX_FREQ   = 9999  # no upper cap needed (max in dataset is ~80)
PROTEIN_TOP_N      = None  # if set, keep only top-N by TF-IDF after freq filter

SE_MIN_FREQ        = 3     # remove SEs in < 3 drugs (noise)
SE_MAX_FREQ        = 200   # remove SEs in > 200 drugs (uninformative, ~31% of drugs)
SE_TOP_N           = 2000  # after freq filter, keep top-2000 by TF-IDF

MIN_VARIANCE       = 1e-4  # remove near-zero variance features


def load_data():
    drug_proteins = defaultdict(set)
    protein_drugs = defaultdict(set)
    with open(RAW / "bio-decagon-targets.csv") as f:
        for row in csv.DictReader(f):
            drug_proteins[row["STITCH"]].add(row["Gene"])
            protein_drugs[row["Gene"]].add(row["STITCH"])

    drug_ses = defaultdict(set)
    se_drugs = defaultdict(set)
    se_names = {}
    with open(RAW / "bio-decagon-mono.csv") as f:
        for row in csv.DictReader(f):
            drug_ses[row["STITCH"]].add(row["Individual Side Effect"])
            se_drugs[row["Individual Side Effect"]].add(row["STITCH"])
            se_names[row["Individual Side Effect"]] = row["Side Effect Name"]

    return drug_proteins, protein_drugs, drug_ses, se_drugs, se_names


def compute_tfidf(drug_features, feature_docs, all_drugs, top_n=None):
    """
    Compute TF-IDF scores for features.
    drug_features: {drug_id: set of feature_ids}
    feature_docs:  {feature_id: set of drug_ids}
    Returns: {feature_id: mean_tfidf_across_drugs} for ranking,
             and {drug_id: {feature_id: tfidf_weight}} for matrix
    """
    N = len(all_drugs)
    idf = {f: np.log(N / len(docs)) for f, docs in feature_docs.items()}

    # TF-IDF per drug (TF=1 for binary, so TF-IDF = IDF)
    feature_mean_tfidf = {f: idf[f] for f in feature_docs}

    # Drug-feature TF-IDF matrix
    drug_tfidf = {}
    for drug in all_drugs:
        drug_tfidf[drug] = {f: idf[f] for f in drug_features.get(drug, set())
                            if f in idf}

    return feature_mean_tfidf, drug_tfidf


def frequency_filter(feature_docs, min_freq, max_freq):
    """Keep features within [min_freq, max_freq] drug occurrences."""
    kept = {f for f, drugs in feature_docs.items()
            if min_freq <= len(drugs) <= max_freq}
    return kept


def variance_filter(drug_features, features, all_drugs, min_var=MIN_VARIANCE):
    """Remove features with variance below threshold."""
    kept = set()
    for f in features:
        col = np.array([1.0 if f in drug_features.get(d, set()) else 0.0
                        for d in all_drugs])
        if np.var(col) >= min_var:
            kept.add(f)
    return kept


def build_sparse_matrix(drug_features_weighted, all_drugs, all_features):
    """
    Build sparse TF-IDF matrix: rows=drugs, cols=features.
    drug_features_weighted: {drug_id: {feature_id: weight}}
    """
    drug_idx     = {d: i for i, d in enumerate(all_drugs)}
    feature_idx  = {f: i for i, f in enumerate(all_features)}

    rows, cols, data = [], [], []
    for drug, feat_weights in drug_features_weighted.items():
        if drug not in drug_idx:
            continue
        for feat, weight in feat_weights.items():
            if feat in feature_idx:
                rows.append(drug_idx[drug])
                cols.append(feature_idx[feat])
                data.append(weight)

    mat = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(all_drugs), len(all_features)),
        dtype=np.float32,
    )
    return mat, drug_idx, feature_idx


def main():
    print("Loading data...")
    drug_proteins, protein_drugs, drug_ses, se_drugs, se_names = load_data()

    all_drugs_p = sorted(drug_proteins.keys())
    all_drugs_s = sorted(drug_ses.keys())
    all_drugs   = sorted(set(all_drugs_p) | set(all_drugs_s))

    print(f"  Drugs with protein targets:  {len(all_drugs_p)}")
    print(f"  Drugs with mono SE:          {len(all_drugs_s)}")
    print(f"  Total unique drugs:          {len(all_drugs)}")
    print(f"  Raw proteins:                {len(protein_drugs)}")
    print(f"  Raw mono SEs:                {len(se_drugs)}")

    # ── Protein feature selection ──────────────────────────────────────────────
    print(f"\n── Protein feature selection ─────────────────────────")
    print(f"  Step 1: frequency filter [{PROTEIN_MIN_FREQ}, {PROTEIN_MAX_FREQ}]")
    kept_p = frequency_filter(protein_drugs, PROTEIN_MIN_FREQ, PROTEIN_MAX_FREQ)
    print(f"    Kept: {len(kept_p)} / {len(protein_drugs)} proteins")

    print(f"  Step 2: variance filter (min_var={MIN_VARIANCE})")
    kept_p = variance_filter(drug_proteins, kept_p, all_drugs_p)
    print(f"    Kept: {len(kept_p)} proteins after variance filter")

    print(f"  Step 3: TF-IDF ranking")
    protein_tfidf_scores, drug_protein_tfidf = compute_tfidf(
        drug_proteins,
        {p: protein_drugs[p] for p in kept_p},
        all_drugs_p
    )
    if PROTEIN_TOP_N:
        kept_p = set(sorted(kept_p, key=lambda x: -protein_tfidf_scores[x])[:PROTEIN_TOP_N])
        print(f"    Kept top-{PROTEIN_TOP_N}: {len(kept_p)} proteins")

    # Drug coverage after protein filtering
    p_covered = sum(1 for d in all_drugs_p if drug_proteins[d] & kept_p)
    print(f"  Drugs retaining >= 1 protein: {p_covered}/{len(all_drugs_p)}")

    # ── Mono SE feature selection ──────────────────────────────────────────────
    print(f"\n── Mono SE feature selection ──────────────────────────")
    print(f"  Step 1: frequency filter [{SE_MIN_FREQ}, {SE_MAX_FREQ}]")
    kept_s = frequency_filter(se_drugs, SE_MIN_FREQ, SE_MAX_FREQ)
    print(f"    Kept: {len(kept_s)} / {len(se_drugs)} SE nodes")

    print(f"  Step 2: variance filter (min_var={MIN_VARIANCE})")
    kept_s = variance_filter(drug_ses, kept_s, all_drugs_s)
    print(f"    Kept: {len(kept_s)} SE nodes after variance filter")

    print(f"  Step 3: TF-IDF ranking → keep top {SE_TOP_N}")
    se_tfidf_scores, drug_se_tfidf = compute_tfidf(
        drug_ses,
        {se: se_drugs[se] for se in kept_s},
        all_drugs_s
    )
    if SE_TOP_N:
        kept_s = set(sorted(kept_s, key=lambda x: -se_tfidf_scores[x])[:SE_TOP_N])
        print(f"    Kept top-{SE_TOP_N}: {len(kept_s)} SE nodes")

    s_covered = sum(1 for d in all_drugs_s if drug_ses[d] & kept_s)
    print(f"  Drugs retaining >= 1 SE: {s_covered}/{len(all_drugs_s)}")

    # ── Build sparse TF-IDF matrices ───────────────────────────────────────────
    print(f"\n── Building TF-IDF matrices ───────────────────────────")
    drug_protein_tfidf_filtered = {
        d: {p: w for p, w in weights.items() if p in kept_p}
        for d, weights in drug_protein_tfidf.items()
    }
    drug_se_tfidf_filtered = {
        d: {se: w for se, w in weights.items() if se in kept_s}
        for d, weights in drug_se_tfidf.items()
    }

    prot_mat, drug_idx_p, prot_idx = build_sparse_matrix(
        drug_protein_tfidf_filtered, all_drugs_p, sorted(kept_p)
    )
    se_mat, drug_idx_s, se_idx = build_sparse_matrix(
        drug_se_tfidf_filtered, all_drugs_s, sorted(kept_s)
    )

    print(f"  Protein matrix: {prot_mat.shape}  nnz={prot_mat.nnz}")
    print(f"  SE matrix:      {se_mat.shape}  nnz={se_mat.nnz}")
    print(f"  Protein matrix sparsity: {1 - prot_mat.nnz/(prot_mat.shape[0]*prot_mat.shape[1]):.3f}")
    print(f"  SE matrix sparsity:      {1 - se_mat.nnz/(se_mat.shape[0]*se_mat.shape[1]):.3f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    with open(OUT / "protein_selected.json", "w") as f:
        json.dump({
            "proteins": sorted(kept_p),
            "protein_idx": {p: i for i, p in enumerate(sorted(kept_p))},
            "n_selected": len(kept_p),
            "n_raw": len(protein_drugs),
        }, f)

    with open(OUT / "se_selected.json", "w") as f:
        json.dump({
            "se_ids": sorted(kept_s),
            "se_idx": {se: i for i, se in enumerate(sorted(kept_s))},
            "se_names": {se: se_names[se] for se in kept_s if se in se_names},
            "n_selected": len(kept_s),
            "n_raw": len(se_drugs),
        }, f)

    sparse.save_npz(OUT / "drug_protein_tfidf.npz", prot_mat)
    sparse.save_npz(OUT / "drug_se_tfidf.npz", se_mat)

    stats = {
        "protein_raw":          len(protein_drugs),
        "protein_selected":     len(kept_p),
        "protein_reduction_pct": round((1 - len(kept_p)/len(protein_drugs))*100, 1),
        "protein_drugs_covered": p_covered,
        "se_raw":               len(se_drugs),
        "se_selected":          len(kept_s),
        "se_reduction_pct":     round((1 - len(kept_s)/len(se_drugs))*100, 1),
        "se_drugs_covered":     s_covered,
        "protein_matrix_shape": list(prot_mat.shape),
        "se_matrix_shape":      list(se_mat.shape),
        "protein_sparsity":     round(1 - prot_mat.nnz/(prot_mat.shape[0]*prot_mat.shape[1]), 4),
        "se_sparsity":          round(1 - se_mat.nnz/(se_mat.shape[0]*se_mat.shape[1]), 4),
        "config": {
            "PROTEIN_MIN_FREQ": PROTEIN_MIN_FREQ,
            "PROTEIN_MAX_FREQ": PROTEIN_MAX_FREQ,
            "PROTEIN_TOP_N":    PROTEIN_TOP_N,
            "SE_MIN_FREQ":      SE_MIN_FREQ,
            "SE_MAX_FREQ":      SE_MAX_FREQ,
            "SE_TOP_N":         SE_TOP_N,
            "MIN_VARIANCE":     MIN_VARIANCE,
        }
    }
    with open(OUT / "feature_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n── Summary ────────────────────────────────────────────")
    print(f"  Proteins: {len(protein_drugs)} → {len(kept_p)} "
          f"({stats['protein_reduction_pct']}% reduction)")
    print(f"  SE nodes: {len(se_drugs)} → {len(kept_s)} "
          f"({stats['se_reduction_pct']}% reduction)")
    print(f"\nSaved to {OUT}/")
    print("  protein_selected.json, se_selected.json")
    print("  drug_protein_tfidf.npz, drug_se_tfidf.npz")
    print("  feature_stats.json")


if __name__ == "__main__":
    main()

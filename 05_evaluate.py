"""
05_evaluate.py
Evaluate trained model on the test split.

Metrics (following Zitnik et al. 2018 for comparability):
    - AUROC / AUPRC per SE type
    - Macro-average across all SE types
    - Frequency-binned averages (top-10, top-50)
    - Per disease class averages  ← NEW (requires bio-decagon-effectcategories.csv)

Outputs:
    results/metrics_per_se.csv          — per-SE AUROC, AUPRC, disease class
    results/metrics_per_disease_class.csv — mean AUROC/AUPRC per disease class
    results/metrics_summary.json        — overall + frequency-binned averages
"""

import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import PolypharmacyHGT

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED      = Path("data/processed")
CHECKPOINTS    = Path("checkpoints")
RESULTS        = Path("results")
CATEGORIES_CSV = Path("data/raw/bio-decagon-effectcategories.csv")
RESULTS.mkdir(exist_ok=True)

BATCH_SIZE = 512
TOP_N_SE   = 30   # must match what was used in training; set None for all


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_effect_categories(path):
    """Returns {se_id: disease_class}. Empty dict if file not found."""
    if not path.exists():
        print(f"  Warning: {path} not found — disease class analysis skipped.")
        return {}
    categories = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            categories[row["Side Effect"]] = row["Disease Class"]
    print(f"  Loaded {len(categories)} SE disease class annotations.")
    return categories


def load_model(checkpoint_path, data, num_se, num_pathways, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    model.eval()
    return model


def collect_scores_and_labels(model, data, splits, drug_pathway_map, device, top_n_se=None):
    test     = splits["test"]
    labels   = test["edge_labels"]
    if top_n_se is not None:
        labels = labels[:, :top_n_se]
    num_se     = labels.shape[1]
    neg_labels = torch.zeros(test["neg_edge_index"].shape[1], num_se)

    all_src = torch.cat([test["pos_edge_index"][0], test["neg_edge_index"][0]])
    all_dst = torch.cat([test["pos_edge_index"][1], test["neg_edge_index"][1]])
    all_lbl = torch.cat([labels, neg_labels])

    ds     = TensorDataset(all_src, all_dst, all_lbl)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    score_list, label_list = [], []
    with torch.no_grad():
        for src, dst, lbl in tqdm(loader, desc="Scoring test pairs"):
            src, dst   = src.to(device), dst.to(device)
            pair_index = torch.stack([src, dst])
            scores     = model(data, pair_index, drug_pathway_map, device)
            if top_n_se is not None:
                scores = scores[:, :top_n_se]
            score_list.append(scores.cpu())
            label_list.append(lbl)

    return torch.cat(score_list).numpy(), torch.cat(label_list).numpy()


def compute_per_se_metrics(scores, labels, top_se_ids, se_names, se_categories):
    """Compute AUROC and AUPRC per SE type, attach disease class."""
    results = []
    for r in range(scores.shape[1]):
        y_true  = labels[:, r]
        y_score = scores[:, r]
        if y_true.sum() == 0:
            continue
        se_id   = top_se_ids[r]
        auroc   = roc_auc_score(y_true, y_score)
        auprc   = average_precision_score(y_true, y_score)
        results.append({
            "se_id":          se_id,
            "se_name":        se_names.get(se_id, ""),
            "disease_class":  se_categories.get(se_id, "unannotated"),
            "n_pos":          int(y_true.sum()),
            "n_total":        int(len(y_true)),
            "pos_rate":       float(y_true.mean()),
            "auroc":          auroc,
            "auprc":          auprc,
        })
    return results


def compute_disease_class_metrics(per_se):
    """
    Aggregate AUROC and AUPRC by disease class.
    Returns list of dicts sorted by mean AUROC descending.
    """
    class_results = defaultdict(list)
    for r in per_se:
        class_results[r["disease_class"]].append(r)

    summary = []
    for disease_class, entries in class_results.items():
        aurocs = [e["auroc"] for e in entries]
        auprcs = [e["auprc"] for e in entries]
        summary.append({
            "disease_class":  disease_class,
            "n_se_types":     len(entries),
            "mean_auroc":     float(np.mean(aurocs)),
            "median_auroc":   float(np.median(aurocs)),
            "mean_auprc":     float(np.mean(auprcs)),
            "median_auprc":   float(np.median(auprcs)),
            "min_auroc":      float(np.min(aurocs)),
            "max_auroc":      float(np.max(aurocs)),
        })
    return sorted(summary, key=lambda x: -x["mean_auroc"])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    data   = torch.load(PROCESSED / "graph.pt",        weights_only=False).to(device)
    splits = torch.load(PROCESSED / "splits.pt",       weights_only=False)
    combo  = torch.load(PROCESSED / "combo_edges.pt",  weights_only=False)

    with open(PROCESSED / "pathway_memberships.pkl", "rb") as f:
        pathway_data = pickle.load(f)
    drug_pathway_map = pathway_data["drug_pathway_map"]
    num_pathways     = len(pathway_data["pathway_id_to_col"])

    top_se_ids = combo["top_se_ids"]
    se_names   = combo["se_names"]
    num_se     = len(top_se_ids) if TOP_N_SE is None else TOP_N_SE

    se_categories = load_effect_categories(CATEGORIES_CSV)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(CHECKPOINTS / "best_model.pt", data, num_se, num_pathways, device)
    print(f"Model loaded — evaluating {num_se} SE types")

    # ── Scores ────────────────────────────────────────────────────────────────
    print("Collecting scores on test split...")
    scores, labels = collect_scores_and_labels(
        model, data, splits, drug_pathway_map, device, top_n_se=TOP_N_SE
    )

    # ── Per-SE metrics ────────────────────────────────────────────────────────
    print("Computing per-SE metrics...")
    per_se = compute_per_se_metrics(
        scores, labels,
        top_se_ids[:num_se], se_names, se_categories
    )

    with open(RESULTS / "metrics_per_se.csv", "w", newline="") as f:
        fields = ["se_id", "se_name", "disease_class", "n_pos",
                  "n_total", "pos_rate", "auroc", "auprc"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_se)
    print(f"Saved per-SE metrics → {RESULTS / 'metrics_per_se.csv'}")

    # ── Disease class metrics ─────────────────────────────────────────────────
    if se_categories:
        print("Computing disease class metrics...")
        dc_metrics = compute_disease_class_metrics(per_se)

        with open(RESULTS / "metrics_per_disease_class.csv", "w", newline="") as f:
            fields = ["disease_class", "n_se_types", "mean_auroc", "median_auroc",
                      "mean_auprc", "median_auprc", "min_auroc", "max_auroc"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(dc_metrics)
        print(f"Saved disease class metrics → {RESULTS / 'metrics_per_disease_class.csv'}")

        print("\n── AUROC by disease class ───────────────────────────────────────")
        print(f"  {'Disease class':<40} {'n SE':>5}  {'AUROC mean':>10}  {'AUPRC mean':>10}")
        print(f"  {'-'*40}  {'-'*5}  {'-'*10}  {'-'*10}")
        annotated = [d for d in dc_metrics if d["disease_class"] != "unannotated"]
        for d in annotated:
            print(f"  {d['disease_class']:<40} {d['n_se_types']:>5}  "
                  f"{d['mean_auroc']:>10.4f}  {d['mean_auprc']:>10.4f}")
        unannotated = [d for d in dc_metrics if d["disease_class"] == "unannotated"]
        if unannotated:
            d = unannotated[0]
            print(f"  {'(unannotated)':<40} {d['n_se_types']:>5}  "
                  f"{d['mean_auroc']:>10.4f}  {d['mean_auprc']:>10.4f}")

    # ── Overall summary ───────────────────────────────────────────────────────
    aurocs       = [r["auroc"] for r in per_se]
    auprcs       = [r["auprc"] for r in per_se]
    per_se_freq  = sorted(per_se, key=lambda x: -x["n_pos"])

    def bin_avg(lst, metric, n):
        top = lst[:n]
        return float(np.mean([r[metric] for r in top])) if top else 0.0

    summary = {
        "n_se_evaluated": len(per_se),
        "macro_auroc":    float(np.mean(aurocs)),
        "macro_auprc":    float(np.mean(auprcs)),
        "median_auroc":   float(np.median(aurocs)),
        "median_auprc":   float(np.median(auprcs)),
        "top10_auroc":    bin_avg(per_se_freq, "auroc", 10),
        "top10_auprc":    bin_avg(per_se_freq, "auprc", 10),
        "top50_auroc":    bin_avg(per_se_freq, "auroc", 50),
        "top50_auprc":    bin_avg(per_se_freq, "auprc", 50),
    }

    with open(RESULTS / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── Overall summary ──────────────────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<25} {v:.4f}" if isinstance(v, float) else f"  {k:<25} {v}")


if __name__ == "__main__":
    main()

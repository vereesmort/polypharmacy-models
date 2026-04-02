# PolypharmacyHGT

Heterogeneous Graph Transformer for polypharmacy side-effect prediction.
MSc Thesis — Biologically Meaningful Representation Learning for Drug Combination Adverse Effects.

## Research Question

Do structured biological priors — pathway membership, phenotypic side-effect ontology,
and chemical structural similarity — improve the biological coherence of learned drug
representations for polypharmacy side-effect prediction, beyond what interaction data alone provides?

## Architecture

    Drug + Protein + Mono SE (+ Pathway nodes in expanded graph)
             |
        HGT Encoder  (type-specific W_Q, W_K, W_V per relation type)
             |
      PathwayAttentionPooling  (KEGG pathway-grouped protein aggregation)
             |
        FusionMLP  [h_drug || pathway_fp] -> z_drug
             |
      DEDICOM Decoder  score(i,j,r) = z_i . D_r . R . D_r . z_j

## Repository Structure

    00_node2vec_proteins.py              optional: pre-train protein embeddings
    00b_fetch_smiles.py                  fetch drug SMILES from PubChem
    00c_feature_selection.py             TF-IDF feature selection for mono SE
    01_fetch_kegg.py                     fetch KEGG pathway data
    01b_build_pathway_edges.py           build pathway nodes + protein->pathway edges
    01c_build_se_ontology_edges.py       build SE->SE ontology edges (HPO or keywords)
    01d_build_structural_similarity_edges.py  drug->drug Tanimoto similarity edges
    02_build_graph.py                    build base HeteroData graph
    02b_build_expanded_graph.py          add Priority 1 biological priors
    03_build_splits.py                   drug-level 80/10/10 stratified splits
    04_train.py                          full-graph training
    04_train_sampled.py                  HGT + neighbour sampling (scalable)
    05_evaluate.py                       AUROC / AUPRC evaluation
    06_enrichment.py                     GO/pathway enrichment per cluster
    07_attention_analysis.py             HGT attention weight analysis
    08_decoder_analysis.py               DEDICOM D_r weight extraction
    10_pca_scatter.py                    drug embedding PCA scatter figure
    11_channel_balance.py                attention channel balance figure
    12_dedicom_top_dim_bar.py            DEDICOM top-dim bar chart
    13_dim16_renal_analysis.py           dimension 16 renal toxicity analysis
    extract_channel_balance.py           standalone channel balance extractor
    model.py                             model definition (supports base + expanded graph)

## Data Setup

Download DECAGON from https://snap.stanford.edu/decagon/
Place these files in data/raw/:

    bio-decagon-combo.csv
    bio-decagon-mono.csv
    bio-decagon-ppi.csv
    bio-decagon-targets.csv
    bio-decagon-effectcategories.csv

## Installation

    pip install torch torch-geometric
    pip install rdkit-pypi          # ECFP4 fingerprints (recommended)
    pip install bioservices         # KEGG API
    pip install gprofiler-official  # GO enrichment
    pip install scikit-learn numpy matplotlib tqdm

## Pipeline — Base Graph

    python 01_fetch_kegg.py
    python 00b_fetch_smiles.py
    python 02_build_graph.py
    python 03_build_splits.py
    python 04_train.py              # full-graph, ~10 hr on T4
    python 05_evaluate.py

## Pipeline — Expanded Graph + Scalable Training

    # After base pipeline steps 1-4:
    python 02b_build_expanded_graph.py   # adds pathway, SE ontology, structural sim
    # In 04_train_sampled.py set USE_EXPANDED_GRAPH = True
    python 04_train_sampled.py           # HGT+sampling, ~2-3 hr on T4
    python 05_evaluate.py

## Key Results (Top-30 SE Types, Initial Run)

    AUROC (macro, leakage-free drug-level split): 0.84
    Comparable to DECAGON 0.874 under fair evaluation protocol
    Epochs to convergence: 40
    Model parameters: 935,150
    Training time: ~10 hours (Colab T4, full-graph)
                   ~2-3 hours (Colab T4, HGT+sampling)

## Biological Validation

    Kinase cluster (C5, n=66):         protein kinase activity p=10^-232
    Triptan co-cluster (C9):           Sumatriptan + Rizatriptan share HTR1A/1B/1D
    SE-only cluster (C10):             16/19 no-target drugs grouped by SE channel alone
    Attention channel separation:      100% SE channel for no-target drugs
    Renal toxicity decoder axis (d16): D_r[16]=0.938 (acute KF), D_r[16]=0.866 (KF)
    MedDRA taxonomy recovery:          7 pharmacological groups without clinical labels

## Graph Versions

    graph.pt          drug + protein + mono_se | 5 edge types  | base biological priors
    graph_expanded.pt + pathway nodes          | 9 edge types  | + KEGG + SE ontology + structural sim

## Scalability Comparison

    Full-graph HGT:  high memory, ~10 hr/run, AUROC baseline
    HGT + Sampling:  low memory,  ~2-3 hr/run, 0-3% AUROC gap

## Dataset Citation

Zitnik M, Agrawal M, Leskovec J (2018).
Modeling polypharmacy side effects with graph convolutional networks.
Bioinformatics 34(13): i457-i466.

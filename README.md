# Replication of Papers

Small, self-contained Python demos that reproduce the computational ideas behind two biology papers using synthetic data. The scripts are designed for quick inspection and teaching: they avoid external datasets and third-party dependencies, while preserving the shape of each analysis.

## Contents

| Script | Paper-inspired workflow | What it demonstrates |
| --- | --- | --- |
| `mtb_host_pathogen_qtl_demo.py` | Meade et al., 2023, host loci that modulate *Mycobacterium tuberculosis* fitness in BXD mice | Synthetic BXD genotypes, bacterial mutant fitness traits, single-marker QTL scans, permutation thresholds, and a shared chromosome 6 hotspot |
| `enzyme_plm_substrate_classifier_demo.py` | Jinich et al., 2022, predicting enzyme substrate structure with protein language models | Synthetic protein sequences, fixed-length sequence embeddings, binary classifiers, family-specific enzyme annotation tasks, and predictions for orphan proteins |

## Requirements

- Python 3.8 or newer
- No external Python packages

## Quick Start

Clone or enter this repository, then run either demo directly:

```bash
python3 mtb_host_pathogen_qtl_demo.py
python3 enzyme_plm_substrate_classifier_demo.py
```

Both scripts use only the Python standard library and print results to the terminal.

## Demo 1: Host-Pathogen QTL Scan

Run:

```bash
python3 mtb_host_pathogen_qtl_demo.py
```

The script simulates:

- BXD-like recombinant inbred mouse haplotypes
- *M. tuberculosis* transposon mutant fitness values
- Genome-wide LOD scans across synthetic mouse markers
- Permutation-derived significance thresholds
- A chromosome 6 hotspot shared by several bacterial mutant traits

For a faster exploratory run, reduce permutations:

```bash
python3 mtb_host_pathogen_qtl_demo.py --permutations 50
```

Useful options:

```bash
python3 mtb_host_pathogen_qtl_demo.py --seed 7 --permutations 500
```

Expected output includes a genome-wide scan summary, the top markers for hotspot traits, and a final hotspot check.

## Demo 2: Enzyme PLM-Style Substrate Classifier

Run:

```bash
python3 enzyme_plm_substrate_classifier_demo.py
```

The script simulates:

- Protein sequences with planted family and label motifs
- Compact fixed-length embeddings based on amino-acid composition, hashed 3-mers, and coarse physicochemical features
- Logistic regression classifiers implemented from scratch
- Two family-specific prediction tasks:
  - SDR cofactor preference: NAD vs NADP
  - SAM-dependent methyltransferase substrate scope: small molecule vs biopolymer
- Predictions for synthetic orphan enzymes

For a smaller training set:

```bash
python3 enzyme_plm_substrate_classifier_demo.py --n-per-class 20
```

Useful options:

```bash
python3 enzyme_plm_substrate_classifier_demo.py --seed 11 --n-per-class 60
```

Expected output includes train/test sizes, test accuracy, confusion counts, and orphan enzyme predictions.

## What These Demos Are Not

These are toy replications, not full scientific reproductions. They do not download or analyze the original datasets, and they intentionally replace heavier methods with readable standard-library approximations.

In particular:

- The QTL demo does not use the real BXD genotypes, TnSeq counts, R/qtl2 mixed models, LOCO kinship, or 10,000 permutations.
- The enzyme classifier demo does not use ESM transformer embeddings, UniProt records, RDKit fingerprints, or the curated datasets from the paper.

Use these scripts to understand the analysis patterns before moving to the full data and production-grade tooling.


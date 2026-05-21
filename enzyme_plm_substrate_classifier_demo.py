#!/usr/bin/env python3
"""
Toy replication demo for:

Jinich et al. "Predicting enzyme substrate chemical structure with protein
language models" bioRxiv, 2022.
https://www.biorxiv.org/content/10.1101/2022.09.28.509940v3.full

What this demo copies from the paper:
- Protein sequences are converted to fixed-length embeddings.
- Supervised classifiers predict enzyme labels from those embeddings.
- Two family-specific tasks are represented:
  1. SDR cofactor preference: NAD versus NADP.
  2. SAM-dependent methyltransferase substrate scope: small molecule versus
     biopolymer.
- Trained models are applied to "orphan" proteins to produce testable
  annotation hypotheses.

What this demo does not copy:
- It does not download ESM transformer embeddings, UniProt records, RDKit
  fingerprints, or the paper's curated SDR/SAM-MTase dataset. Instead it uses
  synthetic protein sequences with planted motifs and a small hashed k-mer
  embedding so the demo runs with only the Python standard library.

Run:
    python3 enzyme_plm_substrate_classifier_demo.py
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    family: str
    sequence: str
    label: str


@dataclass(frozen=True)
class BinaryModel:
    feature_means: List[float]
    feature_stds: List[float]
    weights: List[float]
    bias: float
    positive_label: str
    negative_label: str


def random_sequence(length: int, rng: random.Random, weights: Dict[str, float]) -> str:
    weighted_alphabet = []
    for aa in AMINO_ACIDS:
        weighted_alphabet.extend([aa] * int(weights.get(aa, 1.0) * 10))
    return "".join(rng.choice(weighted_alphabet) for _ in range(length))


def insert_motifs(sequence: str, motifs: Sequence[str], rng: random.Random) -> str:
    chars = list(sequence)
    for motif in motifs:
        start = rng.randint(0, len(chars) - len(motif))
        chars[start : start + len(motif)] = motif
    return "".join(chars)


def make_synthetic_dataset(seed: int, n_per_class: int) -> List[ProteinRecord]:
    rng = random.Random(seed)
    records: List[ProteinRecord] = []

    class_specs = [
        {
            "family": "SDR",
            "label": "NAD",
            "prefix": "SDR_NAD",
            "length": 178,
            "motifs": ["TGAAAGIG", "NNAG", "YXXXK".replace("X", "G")],
            "weights": {"A": 1.4, "G": 1.5, "V": 1.4, "D": 1.1, "N": 1.2},
        },
        {
            "family": "SDR",
            "label": "NADP",
            "prefix": "SDR_NADP",
            "length": 178,
            "motifs": ["TGSSSGIG", "RRK", "KTR", "YGGGK"],
            "weights": {"K": 1.5, "R": 1.6, "S": 1.3, "T": 1.2, "G": 1.2},
        },
        {
            "family": "SAM-MTase",
            "label": "small_molecule",
            "prefix": "SAM_SMALL",
            "length": 220,
            "motifs": ["VLDIGCGTG", "EEL", "HPP", "FDW"],
            "weights": {"F": 1.4, "W": 1.2, "E": 1.3, "L": 1.3, "V": 1.3},
        },
        {
            "family": "SAM-MTase",
            "label": "biopolymer",
            "prefix": "SAM_POLY",
            "length": 220,
            "motifs": ["VLDIGCGTG", "RKK", "KKR", "RGG"],
            "weights": {"K": 1.7, "R": 1.7, "G": 1.4, "Q": 1.2, "N": 1.2},
        },
    ]

    for spec in class_specs:
        for index in range(n_per_class):
            base = random_sequence(spec["length"], rng, spec["weights"])
            sequence = insert_motifs(base, spec["motifs"], rng)
            records.append(
                ProteinRecord(
                    protein_id=f"{spec['prefix']}_{index:03d}",
                    family=spec["family"],
                    sequence=sequence,
                    label=spec["label"],
                )
            )

    rng.shuffle(records)
    return records


def stable_kmer_hash(kmer: str, buckets: int) -> int:
    value = 2166136261
    for char in kmer:
        value ^= ord(char)
        value = (value * 16777619) % (2**32)
    return value % buckets


def protein_embedding(sequence: str, kmer_buckets: int = 96) -> List[float]:
    """
    Convert a sequence into a compact embedding.

    Real protein language models produce dense transformer embeddings. This
    lightweight substitute combines amino-acid composition, hashed 3-mer counts,
    and broad physicochemical fractions.
    """
    length = len(sequence)
    aa_composition = [sequence.count(aa) / length for aa in AMINO_ACIDS]

    kmer_counts = [0.0] * kmer_buckets
    total_kmers = max(1, length - 2)
    for index in range(total_kmers):
        bucket = stable_kmer_hash(sequence[index : index + 3], kmer_buckets)
        kmer_counts[bucket] += 1.0 / total_kmers

    positive = sum(sequence.count(aa) for aa in "KRH") / length
    negative = sum(sequence.count(aa) for aa in "DE") / length
    aromatic = sum(sequence.count(aa) for aa in "FWY") / length
    polar = sum(sequence.count(aa) for aa in "STNQ") / length
    glycine = sequence.count("G") / length

    return aa_composition + kmer_counts + [positive, negative, aromatic, polar, glycine]


def train_test_split(
    records: Sequence[ProteinRecord], rng: random.Random, test_fraction: float = 0.3
) -> Tuple[List[ProteinRecord], List[ProteinRecord]]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * (1.0 - test_fraction)))
    return shuffled[:split_at], shuffled[split_at:]


def standardize_features(
    matrix: Sequence[Sequence[float]],
) -> Tuple[List[List[float]], List[float], List[float]]:
    columns = list(zip(*matrix))
    means = [mean(column) for column in columns]
    stds = [pstdev(column) or 1.0 for column in columns]
    scaled = [
        [(value - means[index]) / stds[index] for index, value in enumerate(row)]
        for row in matrix
    ]
    return scaled, means, stds


def apply_standardization(
    row: Sequence[float], means: Sequence[float], stds: Sequence[float]
) -> List[float]:
    return [(value - means[index]) / stds[index] for index, value in enumerate(row)]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def sigmoid(value: float) -> float:
    if value < -35:
        return 0.0
    if value > 35:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def train_binary_logistic_regression(
    records: Sequence[ProteinRecord],
    positive_label: str,
    negative_label: str,
    epochs: int = 700,
    learning_rate: float = 0.25,
    l2: float = 0.01,
) -> BinaryModel:
    matrix = [protein_embedding(record.sequence) for record in records]
    x_scaled, means, stds = standardize_features(matrix)
    y = [1.0 if record.label == positive_label else 0.0 for record in records]

    weights = [0.0] * len(x_scaled[0])
    bias = 0.0
    n = len(records)

    for _ in range(epochs):
        grad_w = [0.0] * len(weights)
        grad_b = 0.0
        for row, target in zip(x_scaled, y):
            prediction = sigmoid(dot(weights, row) + bias)
            error = prediction - target
            grad_b += error
            for index, value in enumerate(row):
                grad_w[index] += error * value

        for index in range(len(weights)):
            grad_w[index] = grad_w[index] / n + l2 * weights[index]
            weights[index] -= learning_rate * grad_w[index]
        bias -= learning_rate * grad_b / n

    return BinaryModel(
        feature_means=means,
        feature_stds=stds,
        weights=weights,
        bias=bias,
        positive_label=positive_label,
        negative_label=negative_label,
    )


def predict_probability(model: BinaryModel, sequence: str) -> float:
    embedding = protein_embedding(sequence)
    scaled = apply_standardization(embedding, model.feature_means, model.feature_stds)
    return sigmoid(dot(model.weights, scaled) + model.bias)


def evaluate_model(
    model: BinaryModel, records: Sequence[ProteinRecord]
) -> Tuple[float, Dict[Tuple[str, str], int]]:
    confusion: Dict[Tuple[str, str], int] = {}
    correct = 0

    for record in records:
        probability = predict_probability(model, record.sequence)
        predicted = model.positive_label if probability >= 0.5 else model.negative_label
        confusion[(record.label, predicted)] = confusion.get((record.label, predicted), 0) + 1
        correct += int(predicted == record.label)

    return correct / len(records), confusion


def format_confusion(confusion: Dict[Tuple[str, str], int]) -> str:
    rows = []
    for (actual, predicted), count in sorted(confusion.items()):
        rows.append(f"    actual={actual:14s} predicted={predicted:14s} count={count}")
    return "\n".join(rows)


def family_records(records: Iterable[ProteinRecord], family: str) -> List[ProteinRecord]:
    return [record for record in records if record.family == family]


def make_orphan_sequences(seed: int) -> Dict[str, List[ProteinRecord]]:
    rng = random.Random(seed + 99)
    orphan_specs = [
        ("Mtb_Rv_sdr_orphan_A", "SDR", "unknown", 178, ["TGSSSGIG", "RRK", "KTR"], {"K": 1.5, "R": 1.5, "S": 1.3}),
        ("Mtb_Rv_sdr_orphan_B", "SDR", "unknown", 178, ["TGAAAGIG", "NNAG"], {"A": 1.4, "G": 1.4, "N": 1.2}),
        ("Mtb_Rv_sam_orphan_A", "SAM-MTase", "unknown", 220, ["VLDIGCGTG", "RKK", "KKR"], {"K": 1.7, "R": 1.5, "G": 1.3}),
        ("Mtb_Rv_sam_orphan_B", "SAM-MTase", "unknown", 220, ["VLDIGCGTG", "EEL", "FDW"], {"F": 1.4, "E": 1.3, "L": 1.3}),
    ]

    grouped: Dict[str, List[ProteinRecord]] = {"SDR": [], "SAM-MTase": []}
    for protein_id, family, label, length, motifs, weights in orphan_specs:
        sequence = insert_motifs(random_sequence(length, rng, weights), motifs, rng)
        grouped[family].append(ProteinRecord(protein_id, family, sequence, label))
    return grouped


def run_demo(seed: int, n_per_class: int) -> None:
    rng = random.Random(seed)
    records = make_synthetic_dataset(seed, n_per_class)

    print("Toy enzyme substrate/cofactor classifier demo")
    print("=" * 55)
    print(f"Synthetic proteins: {len(records)}")
    print("Embedding: amino-acid composition + hashed 3-mer counts")
    print()

    tasks = [
        ("SDR", "NADP", "NAD", "P(NADP cofactor preference)"),
        ("SAM-MTase", "biopolymer", "small_molecule", "P(biopolymer substrate)"),
    ]

    trained_models: Dict[str, BinaryModel] = {}
    for family, positive_label, negative_label, probability_name in tasks:
        subset = family_records(records, family)
        train, test = train_test_split(subset, rng)
        model = train_binary_logistic_regression(train, positive_label, negative_label)
        accuracy, confusion = evaluate_model(model, test)
        trained_models[family] = model

        print(f"{family} task: {negative_label} vs {positive_label}")
        print(f"  Train size: {len(train)}  Test size: {len(test)}")
        print(f"  Test accuracy: {accuracy:.3f}")
        print(format_confusion(confusion))
        print(f"  Output score: {probability_name}")
        print()

    print("Predictions for synthetic orphan enzymes")
    orphans = make_orphan_sequences(seed)
    for family, orphan_records in orphans.items():
        model = trained_models[family]
        for record in orphan_records:
            probability = predict_probability(model, record.sequence)
            predicted = model.positive_label if probability >= 0.5 else model.negative_label
            print(
                f"  {record.protein_id:20s} family={family:9s}"
                f" prediction={predicted:14s} probability={probability:.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=60,
        help="Synthetic labeled proteins per class.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(seed=args.seed, n_per_class=args.n_per_class)

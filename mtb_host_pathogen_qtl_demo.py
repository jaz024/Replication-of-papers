#!/usr/bin/env python3
"""
Toy replication demo for:

Meade et al. "Genome-wide screen identifies host loci that modulate
Mycobacterium tuberculosis fitness in immunodivergent mice"
G3: Genes|Genomes|Genetics, 2023.
https://academic.oup.com/g3journal/article/13/9/jkad147/7219625

What this demo copies from the paper:
- BXD-like recombinant inbred host genotypes with B6/D2 haplotype states.
- Mtb transposon mutant fitness values used as quantitative endophenotypes.
- Genome-wide QTL scans that calculate a LOD score per host marker.
- Permutation thresholds and a shared chromosome 6 hotspot for several
  bacterial mutant traits.

What this demo does not copy:
- It does not use the real mouse genotypes, TnSeq counts, R/qtl2 mixed model,
  LOCO kinship, or the paper's 10,000 permutations. It is a small, standard
  library-only simulation meant to show the computational idea.

Run:
    python3 mtb_host_pathogen_qtl_demo.py
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple


BXD_STRAINS = [
    "B6",
    "D2",
    "BXD9",
    "BXD29",
    "BXD39",
    "BXD40",
    "BXD48a",
    "BXD51",
    "BXD54",
    "BXD56",
    "BXD60",
    "BXD62",
    "BXD67",
    "BXD69",
    "BXD73",
    "BXD73b",
    "BXD77",
    "BXD79",
    "BXD90",
    "BXD93",
    "BXD102",
]


@dataclass(frozen=True)
class Marker:
    name: str
    chromosome: int
    position_mb: float


@dataclass(frozen=True)
class TraitResult:
    trait: str
    peak_marker: Marker
    peak_lod: float
    threshold: float
    allele_effect: float

    @property
    def significant(self) -> bool:
        return self.peak_lod >= self.threshold


def build_markers() -> List[Marker]:
    """Create a small synthetic mouse marker map with a chr6 marker at 80 Mb."""
    markers: List[Marker] = []
    for chromosome in range(1, 20):
        for position in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            markers.append(
                Marker(
                    name=f"chr{chromosome}_{position:03d}Mb",
                    chromosome=chromosome,
                    position_mb=float(position),
                )
            )
    return markers


def generate_bxd_genotypes(
    strains: Sequence[str], markers: Sequence[Marker], rng: random.Random
) -> Dict[str, List[int]]:
    """
    Generate BXD-like haplotype mosaics.

    Genotype state 0 means B6-like and state 1 means D2-like. The two parents
    are fixed, while each BXD strain is a simple block mosaic with occasional
    switches along each chromosome.
    """
    genotypes: Dict[str, List[int]] = {}
    genotypes["B6"] = [0 for _ in markers]
    genotypes["D2"] = [1 for _ in markers]

    by_chr: Dict[int, List[int]] = {}
    for index, marker in enumerate(markers):
        by_chr.setdefault(marker.chromosome, []).append(index)

    for strain in strains:
        if strain in {"B6", "D2"}:
            continue

        states = [0 for _ in markers]
        for indexes in by_chr.values():
            state = rng.randint(0, 1)
            for marker_index in indexes:
                if rng.random() < 0.12:
                    state = 1 - state
                states[marker_index] = state
        genotypes[strain] = states

    return genotypes


def marker_index(markers: Sequence[Marker], chromosome: int, position_mb: float) -> int:
    for index, marker in enumerate(markers):
        if marker.chromosome == chromosome and marker.position_mb == position_mb:
            return index
    raise ValueError(f"Missing marker chr{chromosome}:{position_mb}Mb")


def simulate_mtb_fitness_traits(
    strains: Sequence[str],
    markers: Sequence[Marker],
    genotypes: Dict[str, List[int]],
    rng: random.Random,
) -> Dict[str, List[float]]:
    """
    Simulate log2 fold-change fitness profiles for several Mtb mutants.

    Four traits share a causal chr6 locus near 80 Mb, mimicking the paper's
    hotspot around mak, rip2, perM, and espR. One trait has an independent chr11
    QTL and one trait is mostly noise.
    """
    chr6_hotspot = marker_index(markers, 6, 80.0)
    chr11_locus = marker_index(markers, 11, 50.0)
    trait_specs = {
        "Rv0127_mak": (chr6_hotspot, 0.82, -0.75, 0.18),
        "Rv0359_rip2": (chr6_hotspot, 0.74, -0.70, 0.20),
        "Rv0955_perM": (chr6_hotspot, -0.78, -0.35, 0.20),
        "Rv3849_espR": (chr6_hotspot, 0.68, -0.68, 0.20),
        "Rv1565_chr11_demo": (chr11_locus, -0.72, -0.58, 0.22),
        "Rv9999_noise_control": (None, 0.0, -0.60, 0.35),
    }

    traits: Dict[str, List[float]] = {}
    for trait, (causal_index, effect, baseline, noise_sd) in trait_specs.items():
        values: List[float] = []
        for strain in strains:
            genetic_effect = 0.0
            if causal_index is not None:
                genetic_effect = effect * genotypes[strain][causal_index]
            values.append(baseline + genetic_effect + rng.gauss(0, noise_sd))
        traits[trait] = values

    return traits


def residual_sum_of_squares(values: Sequence[float], predicted: Iterable[float]) -> float:
    return sum((observed - fitted) ** 2 for observed, fitted in zip(values, predicted))


def lod_score(values: Sequence[float], marker_genotypes: Sequence[int]) -> Tuple[float, float]:
    """
    Return a simple single-marker LOD score and D2-minus-B6 allele effect.

    The model compares y ~ 1 against y ~ marker_haplotype. In the paper this
    was done with R/qtl2 linear mixed models; this demo uses ordinary least
    squares so the math stays visible.
    """
    group0 = [value for value, state in zip(values, marker_genotypes) if state == 0]
    group1 = [value for value, state in zip(values, marker_genotypes) if state == 1]
    if len(group0) < 2 or len(group1) < 2:
        return 0.0, 0.0

    null_mean = mean(values)
    rss_null = residual_sum_of_squares(values, [null_mean] * len(values))
    mean0 = mean(group0)
    mean1 = mean(group1)
    fitted = [mean1 if state == 1 else mean0 for state in marker_genotypes]
    rss_marker = residual_sum_of_squares(values, fitted)

    if rss_marker <= 1e-12 or rss_null <= rss_marker:
        lod = 0.0
    else:
        lod = (len(values) / 2.0) * math.log10(rss_null / rss_marker)
    return lod, mean1 - mean0


def scan_trait(
    values: Sequence[float],
    markers: Sequence[Marker],
    genotypes: Dict[str, List[int]],
    strains: Sequence[str],
) -> Tuple[List[Tuple[Marker, float, float]], Tuple[Marker, float, float]]:
    scan: List[Tuple[Marker, float, float]] = []
    for marker_index_, marker in enumerate(markers):
        marker_states = [genotypes[strain][marker_index_] for strain in strains]
        lod, effect = lod_score(values, marker_states)
        scan.append((marker, lod, effect))

    peak = max(scan, key=lambda item: item[1])
    return scan, peak


def permutation_threshold(
    values: Sequence[float],
    markers: Sequence[Marker],
    genotypes: Dict[str, List[int]],
    strains: Sequence[str],
    rng: random.Random,
    permutations: int,
    quantile: float = 0.95,
) -> float:
    maxima: List[float] = []
    shuffled = list(values)
    for _ in range(permutations):
        rng.shuffle(shuffled)
        _, peak = scan_trait(shuffled, markers, genotypes, strains)
        maxima.append(peak[1])

    maxima.sort()
    index = min(len(maxima) - 1, int(round((len(maxima) - 1) * quantile)))
    return maxima[index]


def top_scan_rows(scan: Sequence[Tuple[Marker, float, float]], limit: int = 5) -> List[str]:
    rows = []
    for marker, lod, effect in sorted(scan, key=lambda item: item[1], reverse=True)[:limit]:
        rows.append(
            f"    chr{marker.chromosome:02d} {marker.position_mb:6.1f} Mb"
            f"  LOD={lod:5.2f}  effect(D2-B6)={effect:+.2f}"
        )
    return rows


def run_demo(seed: int, permutations: int) -> None:
    rng = random.Random(seed)
    markers = build_markers()
    genotypes = generate_bxd_genotypes(BXD_STRAINS, markers, rng)
    traits = simulate_mtb_fitness_traits(BXD_STRAINS, markers, genotypes, rng)

    print("Toy BXD x Mtb TnSeq host-pathogen QTL demo")
    print("=" * 55)
    print(f"Strains: {len(BXD_STRAINS)}  Markers: {len(markers)}")
    print(f"Permutation thresholding: {permutations} shuffles per trait")
    print()

    results: List[TraitResult] = []
    scans: Dict[str, List[Tuple[Marker, float, float]]] = {}

    for trait, values in traits.items():
        scan, (peak_marker, peak_lod, peak_effect) = scan_trait(
            values, markers, genotypes, BXD_STRAINS
        )
        threshold = permutation_threshold(
            values, markers, genotypes, BXD_STRAINS, rng, permutations
        )
        results.append(TraitResult(trait, peak_marker, peak_lod, threshold, peak_effect))
        scans[trait] = scan

    print("Genome-wide scan summary")
    print(
        "trait                  peak          LOD   threshold  effect(D2-B6)  call"
    )
    for result in sorted(results, key=lambda item: item.trait):
        marker = result.peak_marker
        call = "QTL" if result.significant else "not significant"
        print(
            f"{result.trait:22s}"
            f" chr{marker.chromosome:02d}:{marker.position_mb:5.1f}"
            f"  {result.peak_lod:5.2f}    {result.threshold:5.2f}"
            f"      {result.allele_effect:+6.2f}    {call}"
        )

    print()
    print("Top markers for the chr6-hotspot traits")
    for trait in ("Rv0127_mak", "Rv0359_rip2", "Rv0955_perM", "Rv3849_espR"):
        print(f"  {trait}")
        for row in top_scan_rows(scans[trait], limit=3):
            print(row)

    significant_peaks = [
        result
        for result in results
        if result.significant
        and result.peak_marker.chromosome == 6
        and 70.0 <= result.peak_marker.position_mb <= 90.0
    ]
    print()
    print(
        "Hotspot check: "
        f"{len(significant_peaks)} significant traits peak around chr6 70-90 Mb."
    )
    for result in significant_peaks:
        print(f"  - {result.trait} at {result.peak_marker.position_mb:.1f} Mb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--permutations",
        type=int,
        default=500,
        help="Number of permutations per trait. The paper used 10,000.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(seed=args.seed, permutations=args.permutations)

#!/usr/bin/env python3
"""Generate synthetic split-radix (rs) datapoints using a simple genetic algorithm.

This script DOES NOT generate polynomials.
It generates feature rows with the dataset structure you described:

Columns:
  raw_N, padded_N,
  frac_2i_nonzero, frac_2i1_nonzero,
  frac_4i_nonzero, frac_4i1_nonzero, frac_4i2_nonzero, frac_4i3_nonzero,
  output

Constraints enforced:
- raw_N is an integer in [1, max_raw_n]
- padded_N = next power of two >= raw_N
- Fractions are derived from integer nonzero counts for each class.
- Consistency:
    frac_2i_nonzero  == average(frac_4i_nonzero,  frac_4i2_nonzero)
    frac_2i1_nonzero == average(frac_4i1_nonzero, frac_4i3_nonzero)
  This is enforced by constructing 2*i and 2*i+1 from the mod-4 counts.

Note:
- Since we don't execute the actual FFT/multiplication counting, we label all generated points as 'rs'.
- The GA here is used to bias generation towards patterns that typically help split-radix:
  uneven distribution between (mod 4 == 1,3) vs (mod 4 == 0,2), plus moderate sparsity.

Usage:
  python3 ga_generate_rs_dataset.py --out rs_synthetic_10k.csv --rows 10000
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


@dataclass
class Individual:
    # chromosome
    raw_n: int
    # Nonzero counts in each residue class mod 4: positions {0,1,2,3}
    nz0: int
    nz1: int
    nz2: int
    nz3: int

    def padded_n(self) -> int:
        return next_pow2(self.raw_n)

    def denom4(self) -> int:
        # number of indices in each mod-4 class in [0..padded_n-1]
        return self.padded_n() // 4

    def repair(self) -> "Individual":
        # Ensure raw_n valid
        self.raw_n = max(1, self.raw_n)
        p = self.padded_n()
        if p < 4:
            # force minimum 4 so denominators behave
            self.raw_n = 4
            p = 4

        d4 = p // 4
        self.nz0 = clamp(self.nz0, 0, d4)
        self.nz1 = clamp(self.nz1, 0, d4)
        self.nz2 = clamp(self.nz2, 0, d4)
        self.nz3 = clamp(self.nz3, 0, d4)
        return self

    def fractions(self) -> Tuple[float, float, float, float, float, float]:
        d4 = self.denom4()
        # mod4 fractions
        f0 = self.nz0 / d4
        f1 = self.nz1 / d4
        f2 = self.nz2 / d4
        f3 = self.nz3 / d4
        # even/odd fractions from consistency rule
        f_even = 0.5 * (f0 + f2)
        f_odd = 0.5 * (f1 + f3)
        return f_even, f_odd, f0, f1, f2, f3

    def to_row(self) -> List[object]:
        f_even, f_odd, f0, f1, f2, f3 = self.fractions()
        return [
            self.raw_n,
            self.padded_n(),
            f_even,
            f_odd,
            f0,
            f1,
            f2,
            f3,
            "rs",
        ]


def fitness(ind: Individual) -> float:
    """Heuristic fitness to promote split-radix-like sparsity patterns.

    We don't have access to true mult counts here (no polynomial generation),
    so we optimize a proxy objective:
    - Moderate sparsity overall (not all zeros / not fully dense)
    - Strong imbalance between odd-quarter lanes (1,3) and even-quarter lanes (0,2)
      (this tends to change the number of twiddle multiplies actually used).

    Returns a score (higher is better).
    """
    ind = Individual(**ind.__dict__).repair()
    f_even, f_odd, f0, f1, f2, f3 = ind.fractions()

    # Overall sparsity target in [0.15, 0.55]
    sparsity = 0.5 * (f_even + f_odd)
    sparsity_score = 1.0 - abs(sparsity - 0.35) / 0.35  # peak near 0.35

    # Encourage odd quarters different from even quarters
    lane_even = 0.5 * (f0 + f2)
    lane_odd = 0.5 * (f1 + f3)
    imbalance = abs(lane_odd - lane_even)  # in [0,1]

    # Encourage also asymmetry between 1 and 3 (or 0 and 2)
    intra_odd = abs(f1 - f3)
    intra_even = abs(f0 - f2)

    # Penalize edge cases: all zeros or all ones
    extreme_penalty = 0.0
    if sparsity < 0.02 or sparsity > 0.98:
        extreme_penalty = 1.0

    # small preference for larger padded sizes so denominators can vary
    size_bonus = math.log2(ind.padded_n()) / 16.0

    return (
        2.0 * max(0.0, sparsity_score)
        + 2.5 * imbalance
        + 0.7 * intra_odd
        + 0.3 * intra_even
        + size_bonus
        - 3.0 * extreme_penalty
    )


def random_individual(max_raw_n: int, rng: random.Random) -> Individual:
    raw_n = rng.randint(1, max_raw_n)
    padded = next_pow2(raw_n)
    padded = max(4, padded)
    d4 = padded // 4

    # Start from a biased distribution: odd lanes tend a bit denser than even lanes
    # while keeping moderate overall sparsity.
    nz0 = rng.randint(0, d4)
    nz2 = rng.randint(0, d4)
    # bias: odd a bit larger
    nz1 = clamp(int(rng.gauss(mu=0.55 * d4, sigma=0.20 * d4)), 0, d4)
    nz3 = clamp(int(rng.gauss(mu=0.45 * d4, sigma=0.20 * d4)), 0, d4)

    return Individual(raw_n=raw_n, nz0=nz0, nz1=nz1, nz2=nz2, nz3=nz3).repair()


def crossover(a: Individual, b: Individual, rng: random.Random) -> Individual:
    # uniform crossover between genes
    child = Individual(
        raw_n=a.raw_n if rng.random() < 0.5 else b.raw_n,
        nz0=a.nz0 if rng.random() < 0.5 else b.nz0,
        nz1=a.nz1 if rng.random() < 0.5 else b.nz1,
        nz2=a.nz2 if rng.random() < 0.5 else b.nz2,
        nz3=a.nz3 if rng.random() < 0.5 else b.nz3,
    )
    return child.repair()


def mutate(ind: Individual, max_raw_n: int, mutation_rate: float, rng: random.Random) -> Individual:
    ind = Individual(**ind.__dict__)
    if rng.random() < mutation_rate:
        # tweak size with geometric-ish step
        step = rng.choice([-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64])
        ind.raw_n = clamp(ind.raw_n + step, 1, max_raw_n)

    # after raw_n might change, denominators change too
    ind.repair()
    d4 = ind.denom4()

    def mcount(x: int) -> int:
        if rng.random() < mutation_rate:
            # small gaussian move
            x = int(round(x + rng.gauss(0.0, 0.12 * d4)))
        return clamp(x, 0, d4)

    ind.nz0 = mcount(ind.nz0)
    ind.nz1 = mcount(ind.nz1)
    ind.nz2 = mcount(ind.nz2)
    ind.nz3 = mcount(ind.nz3)
    return ind.repair()


def evolve(
    *,
    out_rows: int,
    max_raw_n: int,
    seed: int,
    population_size: int,
    generations: int,
    elite_fraction: float,
    mutation_rate: float,
) -> List[Individual]:
    rng = random.Random(seed)

    pop: List[Individual] = [random_individual(max_raw_n, rng) for _ in range(population_size)]

    elite_n = max(2, int(population_size * elite_fraction))

    for _gen in range(generations):
        scored = [(fitness(ind), ind) for ind in pop]
        scored.sort(key=lambda t: t[0], reverse=True)

        elites = [ind for _score, ind in scored[:elite_n]]

        # tournament selection helper
        def pick_parent() -> Individual:
            k = 5
            best = None
            best_score = None
            for _ in range(k):
                cand = rng.choice(scored)
                if best is None or cand[0] > best_score:
                    best_score = cand[0]
                    best = cand[1]
            return best

        new_pop: List[Individual] = list(elites)
        while len(new_pop) < population_size:
            p1 = pick_parent()
            p2 = pick_parent()
            child = crossover(p1, p2, rng)
            child = mutate(child, max_raw_n, mutation_rate, rng)
            new_pop.append(child)

        pop = new_pop

    # Final sampling: keep best individuals and add mutated clones for diversity
    scored = [(fitness(ind), ind) for ind in pop]
    scored.sort(key=lambda t: t[0], reverse=True)
    base = [ind for _score, ind in scored[: min(population_size, 2000)]]

    result: List[Individual] = []
    while len(result) < out_rows:
        parent = rng.choice(base)
        child = mutate(parent, max_raw_n, mutation_rate=0.6, rng=rng)
        result.append(child)

    return result


def write_csv(path: str, individuals: List[Individual]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "raw_N",
                "padded_N",
                "frac_2i_nonzero",
                "frac_2i1_nonzero",
                "frac_4i_nonzero",
                "frac_4i1_nonzero",
                "frac_4i2_nonzero",
                "frac_4i3_nonzero",
                "output",
            ]
        )
        for ind in individuals:
            w.writerow(ind.to_row())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rs_synthetic_10k.csv", help="Output CSV file")
    ap.add_argument("--rows", type=int, default=10_000, help="Number of rs datapoints to generate")
    ap.add_argument("--max-raw-n", type=int, default=7000, help="Max raw_N")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--population", type=int, default=600, help="GA population size")
    ap.add_argument("--generations", type=int, default=40, help="GA generations")
    ap.add_argument("--elite", type=float, default=0.15, help="Elite fraction")
    ap.add_argument("--mutation", type=float, default=0.25, help="Mutation rate per gene")
    args = ap.parse_args()

    individuals = evolve(
        out_rows=args.rows,
        max_raw_n=args.max_raw_n,
        seed=args.seed,
        population_size=args.population,
        generations=args.generations,
        elite_fraction=args.elite,
        mutation_rate=args.mutation,
    )
    write_csv(args.out, individuals)
    print(f"Wrote {len(individuals)} rs synthetic rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""GA synthetic dataset generator for *both* r2 and rs labels.

This script uses `final_dataset.csv` as a reference distribution and generates
synthetic feature rows that:

- satisfy fraction consistency identities exactly
- produce *balanced* labels: 7000 rows labeled r2 and 7000 labeled rs

Important notes / assumptions (based on your last request):

1) We DO NOT emit `polynomial` and we DO NOT emit multiplication/time columns.
   Instead we emit the feature columns + an `output` label in {r2, rs}.

2) Labels come from `final_dataset.csv`.
   Because `r2` is extremely rare in the provided dataset (time_min_label has only
   54 rows), we train a simple prototype-based classifier on the feature space:

   - We take all rows where `time_min_label` ∈ {r2, rs}.
   - We compute per-class centroid in the 6-dimensional fraction space.
   - A synthetic point is labeled by whichever centroid is closer (Euclidean).

   This gives us a reproducible, data-driven way to define “r2-like” vs “rs-like”
   regions without running NTT timings.

3) Genetic algorithm objective:
   - maximize closeness to the target class centroid
   - keep overall density in a reasonable range (avoid all-zero / all-one extremes)

Output columns:
  raw_N,padded_N,
  frac_2i_nonzero,frac_2i1_nonzero,
  frac_4i_nonzero,frac_4i1_nonzero,frac_4i2_nonzero,frac_4i3_nonzero,
  output

"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


FRAC_COLS = [
    "frac_2i_nonzero",
    "frac_2i1_nonzero",
    "frac_4i_nonzero",
    "frac_4i1_nonzero",
    "frac_4i2_nonzero",
    "frac_4i3_nonzero",
]


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def l2_sq(a: Iterable[float], b: Iterable[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def mean_vec(vs: List[List[float]]) -> List[float]:
    if not vs:
        raise ValueError("empty vector list")
    m = len(vs[0])
    out = [0.0] * m
    for v in vs:
        for i in range(m):
            out[i] += v[i]
    for i in range(m):
        out[i] /= len(vs)
    return out


@dataclass
class Individual:
    # chromosome
    raw_n: int
    nz0: int
    nz1: int
    nz2: int
    nz3: int

    def padded_n(self) -> int:
        return max(4, next_pow2(self.raw_n))

    def denom4(self) -> int:
        return self.padded_n() // 4

    def repair(self) -> "Individual":
        self.raw_n = max(1, self.raw_n)
        p = self.padded_n()
        d4 = p // 4
        self.nz0 = clamp(self.nz0, 0, d4)
        self.nz1 = clamp(self.nz1, 0, d4)
        self.nz2 = clamp(self.nz2, 0, d4)
        self.nz3 = clamp(self.nz3, 0, d4)
        return self

    def fractions6(self) -> List[float]:
        d4 = self.denom4()
        if d4 == 0:
            return [0.0] * 6
        f0 = self.nz0 / d4
        f1 = self.nz1 / d4
        f2 = self.nz2 / d4
        f3 = self.nz3 / d4
        # enforced identities:
        f2i = 0.5 * (f0 + f2)
        f2i1 = 0.5 * (f1 + f3)
        return [f2i, f2i1, f0, f1, f2, f3]

    def to_row(self, label: str) -> List[object]:
        f2i, f2i1, f0, f1, f2, f3 = self.fractions6()
        return [self.raw_n, self.padded_n(), f2i, f2i1, f0, f1, f2, f3, label]


def load_centroids_from_final(final_csv: str) -> Dict[str, List[float]]:
    ref: Dict[str, List[List[float]]] = {"r2": [], "rs": []}
    with open(final_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            lbl = row.get("time_min_label")
            if lbl not in ("r2", "rs"):
                continue
            v = [float(row[c]) for c in FRAC_COLS]
            ref[lbl].append(v)

    if not ref["r2"] or not ref["rs"]:
        raise RuntimeError(
            "Need at least 1 row for each of time_min_label in {r2, rs} in final_dataset.csv"
        )

    return {"r2": mean_vec(ref["r2"]), "rs": mean_vec(ref["rs"]) }


def label_by_centroid(v6: List[float], centroids: Dict[str, List[float]]) -> str:
    d2 = l2_sq(v6, centroids["r2"])
    drs = l2_sq(v6, centroids["rs"])
    return "r2" if d2 <= drs else "rs"


def density_penalty(v6: List[float]) -> float:
    # discourage trivial all-0 / all-1 patterns
    f2i, f2i1, *_ = v6
    s = 0.5 * (f2i + f2i1)
    if s < 0.02 or s > 0.98:
        return 1.0
    return 0.0


def fitness_for_target(ind: Individual, target: str, centroids: Dict[str, List[float]]) -> float:
    ind = Individual(**ind.__dict__).repair()
    v6 = ind.fractions6()

    # closeness to target centroid
    dist = math.sqrt(l2_sq(v6, centroids[target]))
    closeness = 1.0 / (1.0 + dist)

    # ensure that centroid labeling also agrees with target (soft)
    pred = label_by_centroid(v6, centroids)
    agree = 1.0 if pred == target else 0.0

    # keep some size diversity (very mild)
    size_bonus = math.log2(ind.padded_n()) / 32.0

    pen = density_penalty(v6)

    return 4.0 * closeness + 1.5 * agree + size_bonus - 3.0 * pen


def random_individual(max_raw_n: int, rng: random.Random) -> Individual:
    raw_n = rng.randint(1, max_raw_n)
    padded = max(4, next_pow2(raw_n))
    d4 = padded // 4

    # random-ish, but with moderate density
    nz0 = clamp(int(rng.gauss(0.10 * d4, 0.12 * d4)), 0, d4)
    nz2 = clamp(int(rng.gauss(0.10 * d4, 0.12 * d4)), 0, d4)
    nz1 = clamp(int(rng.gauss(0.65 * d4, 0.20 * d4)), 0, d4)
    nz3 = clamp(int(rng.gauss(0.45 * d4, 0.20 * d4)), 0, d4)

    return Individual(raw_n=raw_n, nz0=nz0, nz1=nz1, nz2=nz2, nz3=nz3).repair()


def crossover(a: Individual, b: Individual, rng: random.Random) -> Individual:
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
        step = rng.choice([-256, -128, -64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64, 128, 256])
        ind.raw_n = clamp(ind.raw_n + step, 1, max_raw_n)

    ind.repair()
    d4 = ind.denom4()

    def mcount(x: int) -> int:
        if rng.random() < mutation_rate:
            x = int(round(x + rng.gauss(0.0, 0.18 * d4)))
        return clamp(x, 0, d4)

    ind.nz0 = mcount(ind.nz0)
    ind.nz1 = mcount(ind.nz1)
    ind.nz2 = mcount(ind.nz2)
    ind.nz3 = mcount(ind.nz3)

    return ind.repair()


def evolve_target(
    *,
    target: str,
    out_rows: int,
    max_raw_n: int,
    seed: int,
    population_size: int,
    generations: int,
    elite_fraction: float,
    mutation_rate: float,
    centroids: Dict[str, List[float]],
) -> List[Individual]:
    rng = random.Random(seed)
    pop: List[Individual] = [random_individual(max_raw_n, rng) for _ in range(population_size)]
    elite_n = max(2, int(population_size * elite_fraction))

    for _gen in range(generations):
        scored = [(fitness_for_target(ind, target, centroids), ind) for ind in pop]
        scored.sort(key=lambda t: t[0], reverse=True)
        elites = [ind for _s, ind in scored[:elite_n]]

        def pick_parent() -> Individual:
            k = 6
            best = None
            best_score = None
            for _ in range(k):
                s, cand = rng.choice(scored)
                if best is None or s > best_score:
                    best = cand
                    best_score = s
            return best

        new_pop: List[Individual] = list(elites)
        while len(new_pop) < population_size:
            p1 = pick_parent()
            p2 = pick_parent()
            child = crossover(p1, p2, rng)
            child = mutate(child, max_raw_n, mutation_rate, rng)
            new_pop.append(child)
        pop = new_pop

    # sample with extra mutation for diversity
    scored = [(fitness_for_target(ind, target, centroids), ind) for ind in pop]
    scored.sort(key=lambda t: t[0], reverse=True)
    base = [ind for _s, ind in scored[: min(len(scored), 2000)]]

    result: List[Individual] = []
    while len(result) < out_rows:
        parent = rng.choice(base)
        child = mutate(parent, max_raw_n, mutation_rate=0.7, rng=rng)
        # keep only those that actually classify as target
        if label_by_centroid(child.fractions6(), centroids) == target:
            result.append(child)

    return result


def write_csv(path: str, rows: List[List[object]]) -> None:
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
        w.writerows(rows)


def validate_out(path: str, want_r2: int, want_rs: int) -> None:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)

    from collections import Counter

    c = Counter(row["output"] for row in rows)
    assert c["r2"] == want_r2, c
    assert c["rs"] == want_rs, c

    bad_pow2 = 0
    bad_bounds = 0
    bad_cons = 0

    for row in rows:
        raw = int(row["raw_N"])
        padded = int(row["padded_N"])
        np2 = max(4, 1 if raw <= 1 else 1 << (raw - 1).bit_length())
        if padded != np2:
            bad_pow2 += 1

        f2i = float(row["frac_2i_nonzero"])
        f2i1 = float(row["frac_2i1_nonzero"])
        f0 = float(row["frac_4i_nonzero"])
        f1 = float(row["frac_4i1_nonzero"])
        f2 = float(row["frac_4i2_nonzero"])
        f3 = float(row["frac_4i3_nonzero"])

        if any((x < 0.0 or x > 1.0) for x in (f2i, f2i1, f0, f1, f2, f3)):
            bad_bounds += 1

        if f2i != 0.5 * (f0 + f2) or f2i1 != 0.5 * (f1 + f3):
            bad_cons += 1

    assert bad_pow2 == 0, bad_pow2
    assert bad_bounds == 0, bad_bounds
    assert bad_cons == 0, bad_cons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="final_dataset.csv", help="Reference dataset (final_dataset.csv)")
    ap.add_argument("--out", default="r2_rs_synthetic_7000.csv", help="Output CSV")
    ap.add_argument(
        "--rows",
        type=int,
        default=7000,
        help="Total rows to generate (mixed r2+rs, not per-class)",
    )
    ap.add_argument(
        "--r2-fraction",
        type=float,
        default=0.5,
        help="Target fraction of rows labeled r2 (0..1)",
    )
    ap.add_argument("--max-raw-n", type=int, default=7000, help="Max raw_N")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--population", type=int, default=800, help="GA population size")
    ap.add_argument("--generations", type=int, default=60, help="GA generations")
    ap.add_argument("--elite", type=float, default=0.18, help="Elite fraction")
    ap.add_argument("--mutation", type=float, default=0.28, help="Mutation rate")
    args = ap.parse_args()

    centroids = load_centroids_from_final(args.final)
    print('Centroids from final_dataset.csv (time_min_label):')
    print('  r2:', [round(x, 4) for x in centroids['r2']])
    print('  rs:', [round(x, 4) for x in centroids['rs']])

    r2_target = int(round(args.rows * max(0.0, min(1.0, args.r2_fraction))))
    rs_target = args.rows - r2_target

    r2 = evolve_target(
        target="r2",
        out_rows=r2_target,
        max_raw_n=args.max_raw_n,
        seed=args.seed + 1,
        population_size=args.population,
        generations=args.generations,
        elite_fraction=args.elite,
        mutation_rate=args.mutation,
        centroids=centroids,
    )
    rs = evolve_target(
        target="rs",
        out_rows=rs_target,
        max_raw_n=args.max_raw_n,
        seed=args.seed + 2,
        population_size=args.population,
        generations=args.generations,
        elite_fraction=args.elite,
        mutation_rate=args.mutation,
        centroids=centroids,
    )

    rows: List[List[object]] = []
    rows.extend([ind.to_row('r2') for ind in r2])
    rows.extend([ind.to_row('rs') for ind in rs])

    # deterministic shuffle
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    write_csv(args.out, rows)
    validate_out(args.out, want_r2=r2_target, want_rs=rs_target)
    print(f"Wrote {len(rows)} rows to {args.out} (r2={r2_target}, rs={rs_target})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

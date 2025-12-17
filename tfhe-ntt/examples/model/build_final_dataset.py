#!/usr/bin/env python3
"""Build final_dataset.csv by concatenating:

1) The existing real dataset: dataset_output.csv (21k rows)
2) The synthetic dataset: rs_synthetic_7k.csv (7k rows)

For synthetic rows, we generate a polynomial of length padded_N that matches the
requested nonzero fractions:
- Fractions are interpreted w.r.t padded_N.
- We use the mod-4 lane fractions to compute integer nonzero counts per lane.
- We place that many nonzeros in the corresponding indices (0 mod 4, 1 mod 4, ...).

We do NOT verify rs optimality here (per your instruction).

Output schema matches dataset_output.csv exactly:
raw_N,padded_N,polynomial,frac_2i_nonzero,frac_2i1_nonzero,frac_4i_nonzero,frac_4i1_nonzero,frac_4i2_nonzero,frac_4i3_nonzero,mult_r2,mult_r4,mult_rs,output

For synthetic rows:
- mult_r2/mult_r4/mult_rs are left blank (empty string) because we are not
  running Rust FFT scoring here.
- output remains 'rs'.

Usage:
  python3 build_final_dataset.py \
    --base dataset_output.csv \
    --synthetic rs_synthetic_7k.csv \
    --out final_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Dict, List, Tuple


FINAL_HEADER = [
    "raw_N",
    "padded_N",
    "polynomial",
    "frac_2i_nonzero",
    "frac_2i1_nonzero",
    "frac_4i_nonzero",
    "frac_4i1_nonzero",
    "frac_4i2_nonzero",
    "frac_4i3_nonzero",
    "mult_r2",
    "mult_r4",
    "mult_rs",
    "output",
]


def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def counts_from_fractions(padded_n: int, f0: float, f1: float, f2: float, f3: float) -> Tuple[int, int, int, int]:
    if padded_n % 4 != 0:
        raise ValueError("padded_N must be multiple of 4")
    denom = padded_n // 4

    def to_count(fr: float) -> int:
        return clamp(int(round(fr * denom)), 0, denom)

    return to_count(f0), to_count(f1), to_count(f2), to_count(f3)


def build_poly(padded_n: int, nz0: int, nz1: int, nz2: int, nz3: int, rng: random.Random) -> List[int]:
    poly = [0] * padded_n
    lanes = [list(range(k, padded_n, 4)) for k in range(4)]
    for lane in lanes:
        rng.shuffle(lane)

    def fill(lane_id: int, count: int) -> None:
        for idx in lanes[lane_id][:count]:
            poly[idx] = rng.randint(1, 50)

    fill(0, nz0)
    fill(1, nz1)
    fill(2, nz2)
    fill(3, nz3)
    return poly


def poly_to_field(poly: List[int]) -> str:
    # Match Rust generator: "[a b c ...]"
    return "[" + " ".join(str(x) for x in poly) + "]"


def copy_base(base_path: str, writer: csv.DictWriter) -> int:
    n = 0
    with open(base_path, newline="") as f:
        r = csv.DictReader(f)
        # Expect base to already be in FINAL_HEADER order, but produce by name.
        for row in r:
            out = {k: row.get(k, "") for k in FINAL_HEADER}
            writer.writerow(out)
            n += 1
    return n


def add_synthetic(synth_path: str, writer: csv.DictWriter, seed: int) -> int:
    rng = random.Random(seed)
    n = 0
    with open(synth_path, newline="") as f:
        r = csv.DictReader(f)
        required = {
            "raw_N",
            "padded_N",
            "frac_2i_nonzero",
            "frac_2i1_nonzero",
            "frac_4i_nonzero",
            "frac_4i1_nonzero",
            "frac_4i2_nonzero",
            "frac_4i3_nonzero",
            "output",
        }
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"Synthetic CSV missing columns: {sorted(missing)}")

        for i, row in enumerate(r):
            raw_n = int(row["raw_N"])
            padded_n = int(row["padded_N"])
            f0 = float(row["frac_4i_nonzero"])
            f1 = float(row["frac_4i1_nonzero"])
            f2 = float(row["frac_4i2_nonzero"])
            f3 = float(row["frac_4i3_nonzero"])

            nz0, nz1, nz2, nz3 = counts_from_fractions(padded_n, f0, f1, f2, f3)
            # Per-row deterministic variation
            local_rng = random.Random(seed + i)
            poly = build_poly(padded_n, nz0, nz1, nz2, nz3, local_rng)

            out: Dict[str, str] = {
                "raw_N": str(raw_n),
                "padded_N": str(padded_n),
                "polynomial": poly_to_field(poly),
                "frac_2i_nonzero": row["frac_2i_nonzero"],
                "frac_2i1_nonzero": row["frac_2i1_nonzero"],
                "frac_4i_nonzero": row["frac_4i_nonzero"],
                "frac_4i1_nonzero": row["frac_4i1_nonzero"],
                "frac_4i2_nonzero": row["frac_4i2_nonzero"],
                "frac_4i3_nonzero": row["frac_4i3_nonzero"],
                "mult_r2": "",
                "mult_r4": "",
                "mult_rs": "",
                "output": row["output"],
            }

            writer.writerow(out)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default=os.path.join(os.path.dirname(__file__), "dataset_output.csv"),
        help="Path to base dataset_output.csv",
    )
    ap.add_argument(
        "--synthetic",
        default=os.path.join(os.path.dirname(__file__), "rs_synthetic_7k.csv"),
        help="Path to synthetic rs_synthetic_7k.csv",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "final_dataset.csv"),
        help="Output final_dataset.csv",
    )
    ap.add_argument("--seed", type=int, default=123, help="Seed used for polynomial placement")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINAL_HEADER)
        w.writeheader()
        base_rows = copy_base(args.base, w)
        synth_rows = add_synthetic(args.synthetic, w, seed=args.seed)

    print(f"Wrote {base_rows + synth_rows} rows to {args.out} ({base_rows} base + {synth_rows} synthetic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

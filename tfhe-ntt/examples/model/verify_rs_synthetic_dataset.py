#!/usr/bin/env python3
"""Verify which synthetic 'rs' datapoints are actually rs-optimal (or tied).

Given an input CSV with columns:
  raw_N,padded_N,
  frac_2i_nonzero,frac_2i1_nonzero,
  frac_4i_nonzero,frac_4i1_nonzero,frac_4i2_nonzero,frac_4i3_nonzero,
  output
(this matches ga_generate_rs_dataset.py output)

We reconstruct a polynomial of length padded_N whose nonzero pattern matches the
requested *mod-4 lane* fractions (exactly when they are representable), then we
run the repo's Rust FFT implementations to count twiddle multiplications for:
  - radix-2
  - radix-4
  - split-radix

A row is considered "correct" if split-radix has the minimum multiplication
count, OR is tied for minimum.

Outputs:
- A filtered CSV of only correct rows with extra columns: mult_r2,mult_r4,mult_rs,is_correct
- A summary printed to stdout.

Implementation details:
- We compile a tiny Rust helper on the fly (once) under a temp dir, then call it
  as a subprocess to score batches of polynomials efficiently.

This script is meant to run from repo root, but uses absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


REPO_ROOT_DEFAULT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def _run(cmd: List[str], cwd: str | None = None) -> None:
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@dataclass
class Row:
    raw_n: int
    padded_n: int
    f0: float
    f1: float
    f2: float
    f3: float


def parse_rows(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        required = {
            "raw_N",
            "padded_N",
            "frac_4i_nonzero",
            "frac_4i1_nonzero",
            "frac_4i2_nonzero",
            "frac_4i3_nonzero",
        }
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

        for row in r:
            raw_n = int(row["raw_N"])
            padded_n = int(row["padded_N"])
            # sanity
            if padded_n != next_pow2(raw_n):
                # keep going but correct padded_n we build against
                padded_n = next_pow2(raw_n)

            rows.append(
                Row(
                    raw_n=raw_n,
                    padded_n=padded_n,
                    f0=float(row["frac_4i_nonzero"]),
                    f1=float(row["frac_4i1_nonzero"]),
                    f2=float(row["frac_4i2_nonzero"]),
                    f3=float(row["frac_4i3_nonzero"]),
                )
            )
    return rows


def counts_from_fractions(padded_n: int, f0: float, f1: float, f2: float, f3: float) -> Tuple[int, int, int, int]:
    """Convert lane fractions into integer nonzero counts per lane.

    Denominator per lane is padded_n/4.

    We try nearest integer (round), then clamp into [0, denom].
    """
    if padded_n % 4 != 0:
        raise ValueError("padded_n must be multiple of 4")
    denom = padded_n // 4

    def to_count(fr: float) -> int:
        # tolerate tiny float errors
        x = int(round(fr * denom))
        return max(0, min(denom, x))

    return to_count(f0), to_count(f1), to_count(f2), to_count(f3)


def build_poly(padded_n: int, nz0: int, nz1: int, nz2: int, nz3: int, seed: int) -> List[int]:
    """Construct a polynomial with exactly nz* nonzeros in each mod-4 class.

    Coefficients are nonzero mod p=65537 (we use small values 1..50).
    Nonzero positions are chosen deterministically from seed.
    """
    rng = random.Random(seed)
    poly = [0] * padded_n

    # indices by lane
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


def poly_to_dataset_field(poly: List[int]) -> str:
    """Match the Rust Dataset.rs formatting: a single CSV field like "[a b c ...]"."""
    return "[" + " ".join(str(x) for x in poly) + "]"


RUST_HELPER = r"""
use std::io::{self, Read};

use tfhe_ntt::custum_radix::{
    fft_radix2_recursive_mut,
    fft_radix4_recursive_mut,
    fft_split_radix_recursive_mut,
};
use tfhe_ntt::custum_radix::fwd_1::MultStats;

fn mul_mod(a: u32, b: u32, p: u32) -> u32 {
    ((a as u64 * b as u64) % (p as u64)) as u32
}

fn pow_mod(mut base: u32, mut exp: u32, p: u32) -> u32 {
    let mut res: u32 = 1;
    base %= p;
    while exp > 0 {
        if (exp & 1) == 1 {
            res = mul_mod(res, base, p);
        }
        base = mul_mod(base, base, p);
        exp >>= 1;
    }
    res
}

fn make_twiddles_from_root(root: u32, n: usize, p: u32) -> Vec<u32> {
    let mut tw = vec![0u32; n];
    let mut cur: u32 = 1;
    for k in 0..n {
        tw[k] = cur;
        cur = mul_mod(cur, root, p);
    }
    tw
}

fn compute_mults(poly: &[u32]) -> (usize, usize, usize) {
    let n = poly.len();
    let p: u32 = 65537;
    let g = 3;
    let root = pow_mod(g, (p - 1) / (n as u32), p);
    let twiddles = make_twiddles_from_root(root, n, p);

    let mut r2_stats = MultStats::default();
    let mut r4_stats = MultStats::default();
    let mut rs_stats = MultStats::default();

    let mut p2 = poly.to_vec();
    let mut p4 = poly.to_vec();
    let mut ps = poly.to_vec();

    fft_radix2_recursive_mut(&mut p2, &twiddles, p, &mut r2_stats);
    fft_radix4_recursive_mut(&mut p4, &twiddles, p, &mut r4_stats);
    fft_split_radix_recursive_mut(&mut ps, &twiddles, p, &mut rs_stats);

    (r2_stats.nonzero_mults, r4_stats.nonzero_mults, rs_stats.nonzero_mults)
}

fn main() {
    // Read JSON lines: each line is {"poly": [..u32..]}
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let mut out = String::new();
    for line in input.lines() {
        if line.trim().is_empty() { continue; }
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let arr = v.get("poly").unwrap().as_array().unwrap();
        let mut poly: Vec<u32> = Vec::with_capacity(arr.len());
        for x in arr {
            poly.push(x.as_u64().unwrap() as u32);
        }
        let (m2, m4, ms) = compute_mults(&poly);
        let best = if ms <= m2 && ms <= m4 { "rs" } else if m2 <= m4 { "r2" } else { "r4" };
        out.push_str(&format!("{{\"mult_r2\":{},\"mult_r4\":{},\"mult_rs\":{},\"best\":\"{}\"}}\n", m2, m4, ms, best));
    }

    print!("{}", out);
}
"""


def build_rust_scorer(repo_root: str) -> str:
    """Build a tiny Rust binary that scores polynomials using tfhe-ntt."""
    tmpdir = tempfile.mkdtemp(prefix="rs_verify_")
    cargo_toml = os.path.join(tmpdir, "Cargo.toml")
    src_dir = os.path.join(tmpdir, "src")
    os.makedirs(src_dir, exist_ok=True)
    main_rs = os.path.join(src_dir, "main.rs")

    with open(main_rs, "w") as f:
        f.write(RUST_HELPER)

    # Point dependency at workspace crate so it builds against current code.
    with open(cargo_toml, "w") as f:
        f.write(
            """[package]
name = "rs_scorer"
version = "0.1.0"
edition = "2021"

[dependencies]
serde_json = "1"
tfhe-ntt = { path = """ + json.dumps(os.path.join(repo_root, "tfhe-ntt")) + """ }
"""
        )

    _run(["cargo", "build", "--release"], cwd=tmpdir)
    bin_path = os.path.join(tmpdir, "target", "release", "rs_scorer")
    return bin_path


def score_batch(bin_path: str, polys: List[List[int]]) -> List[Dict[str, object]]:
    payload = "\n".join(json.dumps({"poly": p}) for p in polys) + "\n"
    p = subprocess.run(
        [bin_path],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"scorer failed: {p.stderr}\n{p.stdout}")
    out: List[Dict[str, object]] = []
    for line in p.stdout.splitlines():
        out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input synthetic CSV")
    ap.add_argument("--out", required=True, help="Output verified CSV (correct rs only)")
    ap.add_argument("--repo-root", default=REPO_ROOT_DEFAULT, help="Repo root path")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for Rust scorer")
    ap.add_argument("--seed", type=int, default=0, help="Seed base for polynomial construction")
    ap.add_argument("--keep-all", action="store_true", help="Keep all rows, add is_correct flag")
    args = ap.parse_args()

    rows = parse_rows(args.inp)
    scorer = build_rust_scorer(os.path.abspath(args.repo_root))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    total = 0
    correct = 0
    per_padded: Dict[int, Tuple[int, int]] = {}

    with open(args.inp, newline="") as fin, open(args.out, "w", newline="") as fout:
        rin = csv.DictReader(fin)
        field_in = rin.fieldnames or []

        extra = ["mult_r2", "mult_r4", "mult_rs", "best", "is_correct"]
        w = csv.DictWriter(fout, fieldnames=field_in + extra)
        w.writeheader()

        batch_rows: List[Dict[str, str]] = []
        batch_polys: List[List[int]] = []

        def flush_batch() -> None:
            nonlocal total, correct
            if not batch_rows:
                return
            scores = score_batch(scorer, batch_polys)
            for inrow, sc in zip(batch_rows, scores):
                total += 1
                padded = int(inrow["padded_N"])
                m2 = int(sc["mult_r2"])
                m4 = int(sc["mult_r4"])
                ms = int(sc["mult_rs"])
                best = str(sc["best"])
                minv = min(m2, m4, ms)
                is_correct = (ms == minv)

                t, c = per_padded.get(padded, (0, 0))
                t += 1
                c += 1 if is_correct else 0
                per_padded[padded] = (t, c)

                outrow = dict(inrow)
                outrow.update(
                    {
                        "mult_r2": m2,
                        "mult_r4": m4,
                        "mult_rs": ms,
                        "best": best,
                        "is_correct": "1" if is_correct else "0",
                    }
                )

                if args.keep_all or is_correct:
                    w.writerow(outrow)

                correct += 1 if is_correct else 0

            batch_rows.clear()
            batch_polys.clear()

        for inrow in rin:
            padded_n = int(inrow["padded_N"])
            f0 = float(inrow["frac_4i_nonzero"])
            f1 = float(inrow["frac_4i1_nonzero"])
            f2 = float(inrow["frac_4i2_nonzero"])
            f3 = float(inrow["frac_4i3_nonzero"])

            nz0, nz1, nz2, nz3 = counts_from_fractions(padded_n, f0, f1, f2, f3)

            seed = args.seed + total
            poly = build_poly(padded_n, nz0, nz1, nz2, nz3, seed=seed)

            batch_rows.append(inrow)
            batch_polys.append(poly)

            if len(batch_rows) >= args.batch:
                flush_batch()

        flush_batch()

    print(f"Total rows checked: {total}")
    print(f"Rows where rs is min (or tied): {correct}")
    if total:
        print(f"Accuracy: {correct/total:.4f}")

    top = sorted(per_padded.items(), key=lambda kv: kv[0])
    print("Per padded_N stats (padded_N: correct/total):")
    for padded, (t, c) in top:
        print(f"  {padded}: {c}/{t} ({(c/t):.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

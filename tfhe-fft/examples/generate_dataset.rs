///! Generate a dataset of polynomial FFT timings across different radix algorithms.
///!
///! Columns:
///!   N, Polynomial, 2*i, 2*i+1, 4*i, 4*i+1, 4*i+2, 4*i+3,
///!   8*i, 8*i+1, 8*i+2, 8*i+3, 8*i+4, 8*i+5, 8*i+6, 8*i+7,
///!   radix-2 time, radix-4 time, radix-8 time, radix-split time, Best algo
///!
///! - N: polynomial size (64 to 131072, powers of 2)
///! - Polynomial: semicolon-separated integer coefficients in [-1000000, 1000000]
///!   Each polynomial has a random sparsity (0%–90% of coefficients set to zero)
///!   so the non-zero count features have meaningful variation.
///! - 2*i, 2*i+1, ...: count of non-zero coefficients at those stride positions
///! - Times: average execution time in nanoseconds over 25 runs
///! - Best algo: r2 / r4 / r8 / rs (whichever had the lowest average time)
///!
///! For each size there are 1500 randomly generated polynomials.
///! Results are written to "dataset.csv".

use aligned_vec::CACHELINE_ALIGN;
use dyn_stack::PodStack;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use tfhe_fft::c64;
use tfhe_fft::ordered::FftAlgo;
use tfhe_fft::unordered::{Method, Plan, Radix2Plan, Radix4Plan, SplitRadixPlan};

/// Number of polynomials per FFT size.
const NUM_POLYS: usize = 1500;

/// Number of times to run each polynomial per algorithm (take average).
const RUNS_PER_POLY: usize = 25;

/// Base ordered-FFT size used by the outer radix decomposition.
const BASE_N: usize = 512;

/// Base algorithm for the ordered inner FFT.
const BASE_ALGO: FftAlgo = FftAlgo::Dif4;

/// Count non-zero coefficients at positions offset, offset+stride, offset+2*stride, ...
fn count_nonzero_at_stride(coeffs: &[i64], offset: usize, stride: usize) -> usize {
    coeffs
        .iter()
        .skip(offset)
        .step_by(stride)
        .filter(|&&c| c != 0)
        .count()
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // FFT sizes: 64, 128, 256, ..., 131072  (2^6 to 2^17)
    let sizes: Vec<usize> = (6..=17).map(|e| 1usize << e).collect();

    let csv_path = "dataset.csv";
    let file = File::create(csv_path).expect("cannot create CSV file");
    let mut csv = BufWriter::new(file);

    // -------- CSV header --------
    writeln!(
        csv,
        "N,Polynomial,\
         2*i,2*i+1,\
         4*i,4*i+1,4*i+2,4*i+3,\
         8*i,8*i+1,8*i+2,8*i+3,8*i+4,8*i+5,8*i+6,8*i+7,\
         radix-2 time,radix-4 time,radix-8 time,radix-split time,\
         Best algo"
    )
    .unwrap();

    println!("=== Dataset Generation ===");
    println!(
        "  {} polynomials per size, {} timing runs each",
        NUM_POLYS, RUNS_PER_POLY
    );
    println!("  Coefficients in [-1000000, 1000000]");
    println!("  Sizes: {:?}", sizes);
    println!("  Base algo: {:?}, base_n: {}", BASE_ALGO, BASE_N);
    println!();

    for &n in &sizes {
        println!("Processing N = {} ...", n);

        let base_n = if n <= BASE_N { n } else { BASE_N };

        // -------- build plans --------
        let plan_r2 = Radix2Plan::new(n, BASE_ALGO, base_n);
        let plan_r4 = Radix4Plan::new(n, BASE_ALGO, base_n);
        let plan_r8 = Plan::new(
            n,
            Method::UserProvided {
                base_algo: BASE_ALGO,
                base_n,
            },
        );
        let plan_sr = SplitRadixPlan::new(n, BASE_ALGO, base_n);

        // Scratch: take the max across all plans.
        let scratch_req = plan_r2
            .fft_scratch()
            .or(plan_r4.fft_scratch())
            .or(plan_r8.fft_scratch())
            .or(plan_sr.fft_scratch());

        let mut scratch_bytes = vec![0u8; scratch_req.size_bytes() + CACHELINE_ALIGN];

        // -------- warmup each plan --------
        {
            let warmup_poly: Vec<c64> = (0..n)
                .map(|i| c64 {
                    re: i as f64,
                    im: 0.0,
                })
                .collect();
            for _ in 0..5 {
                let mut buf = warmup_poly.clone();
                let mut stack = PodStack::new(&mut scratch_bytes);
                plan_r2.fwd(&mut buf, &mut stack);
            }
            for _ in 0..5 {
                let mut buf = warmup_poly.clone();
                let mut stack = PodStack::new(&mut scratch_bytes);
                plan_r4.fwd(&mut buf, &mut stack);
            }
            for _ in 0..5 {
                let mut buf = warmup_poly.clone();
                let mut stack = PodStack::new(&mut scratch_bytes);
                plan_r8.fwd(&mut buf, &mut stack);
            }
            for _ in 0..5 {
                let mut buf = warmup_poly.clone();
                let mut stack = PodStack::new(&mut scratch_bytes);
                plan_sr.fwd(&mut buf, &mut stack);
            }
        }

        // -------- generate and time each polynomial --------
        for poly_idx in 0..NUM_POLYS {
            if (poly_idx + 1) % 100 == 0 {
                println!("  N={}: polynomial {}/{}", n, poly_idx + 1, NUM_POLYS);
            }

            // Random sparsity: 0% to 90% of coefficients will be zero.
            let sparsity: f64 = rng.gen_range(0.0..0.9);

            // Random integer coefficients in [-1_000_000, 1_000_000]
            // with the chosen fraction randomly zeroed out.
            let coeffs: Vec<i64> = (0..n)
                .map(|_| {
                    if rng.gen::<f64>() < sparsity {
                        0i64
                    } else {
                        // Draw from [-1_000_000, -1] ∪ [1, 1_000_000] to avoid
                        // accidental zeros in the non-sparse portion.
                        let mag = rng.gen_range(1i64..=1_000_000i64);
                        if rng.gen::<bool>() { mag } else { -mag }
                    }
                })
                .collect();

            let poly: Vec<c64> = coeffs
                .iter()
                .map(|&c| c64 {
                    re: c as f64,
                    im: 0.0,
                })
                .collect();

            // ---- non-zero coefficient counts at stride positions ----
            // stride-2 positions
            let nz_2i = count_nonzero_at_stride(&coeffs, 0, 2);
            let nz_2i1 = count_nonzero_at_stride(&coeffs, 1, 2);
            // stride-4 positions
            let nz_4i = count_nonzero_at_stride(&coeffs, 0, 4);
            let nz_4i1 = count_nonzero_at_stride(&coeffs, 1, 4);
            let nz_4i2 = count_nonzero_at_stride(&coeffs, 2, 4);
            let nz_4i3 = count_nonzero_at_stride(&coeffs, 3, 4);
            // stride-8 positions
            let nz_8i = count_nonzero_at_stride(&coeffs, 0, 8);
            let nz_8i1 = count_nonzero_at_stride(&coeffs, 1, 8);
            let nz_8i2 = count_nonzero_at_stride(&coeffs, 2, 8);
            let nz_8i3 = count_nonzero_at_stride(&coeffs, 3, 8);
            let nz_8i4 = count_nonzero_at_stride(&coeffs, 4, 8);
            let nz_8i5 = count_nonzero_at_stride(&coeffs, 5, 8);
            let nz_8i6 = count_nonzero_at_stride(&coeffs, 6, 8);
            let nz_8i7 = count_nonzero_at_stride(&coeffs, 7, 8);

            // ---- time each algorithm (25 runs, average) ----
            let mut times_ns = [0u64; 4];

            // Radix-2
            {
                let mut total: u64 = 0;
                for _ in 0..RUNS_PER_POLY {
                    let mut buf = poly.clone();
                    let mut stack = PodStack::new(&mut scratch_bytes);
                    let start = Instant::now();
                    plan_r2.fwd(&mut buf, &mut stack);
                    let elapsed = start.elapsed().as_nanos() as u64;
                    std::hint::black_box(&buf);
                    total += elapsed;
                }
                times_ns[0] = total / RUNS_PER_POLY as u64;
            }

            // Radix-4
            {
                let mut total: u64 = 0;
                for _ in 0..RUNS_PER_POLY {
                    let mut buf = poly.clone();
                    let mut stack = PodStack::new(&mut scratch_bytes);
                    let start = Instant::now();
                    plan_r4.fwd(&mut buf, &mut stack);
                    let elapsed = start.elapsed().as_nanos() as u64;
                    std::hint::black_box(&buf);
                    total += elapsed;
                }
                times_ns[1] = total / RUNS_PER_POLY as u64;
            }

            // Radix-8 (default Plan)
            {
                let mut total: u64 = 0;
                for _ in 0..RUNS_PER_POLY {
                    let mut buf = poly.clone();
                    let mut stack = PodStack::new(&mut scratch_bytes);
                    let start = Instant::now();
                    plan_r8.fwd(&mut buf, &mut stack);
                    let elapsed = start.elapsed().as_nanos() as u64;
                    std::hint::black_box(&buf);
                    total += elapsed;
                }
                times_ns[2] = total / RUNS_PER_POLY as u64;
            }

            // Split-radix
            {
                let mut total: u64 = 0;
                for _ in 0..RUNS_PER_POLY {
                    let mut buf = poly.clone();
                    let mut stack = PodStack::new(&mut scratch_bytes);
                    let start = Instant::now();
                    plan_sr.fwd(&mut buf, &mut stack);
                    let elapsed = start.elapsed().as_nanos() as u64;
                    std::hint::black_box(&buf);
                    total += elapsed;
                }
                times_ns[3] = total / RUNS_PER_POLY as u64;
            }

            // ---- determine best algorithm ----
            let min_time = *times_ns.iter().min().unwrap();
            let best_idx = times_ns.iter().position(|&t| t == min_time).unwrap();
            let best_algo = match best_idx {
                0 => "r2",
                1 => "r4",
                2 => "r8",
                3 => "rs",
                _ => unreachable!(),
            };

            // ---- format polynomial as semicolon-separated coefficients ----
            let poly_str: String = coeffs
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(";");

            // ---- write CSV row ----
            writeln!(
                csv,
                "{},\"{}\",{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                n,
                poly_str,
                nz_2i,
                nz_2i1,
                nz_4i,
                nz_4i1,
                nz_4i2,
                nz_4i3,
                nz_8i,
                nz_8i1,
                nz_8i2,
                nz_8i3,
                nz_8i4,
                nz_8i5,
                nz_8i6,
                nz_8i7,
                times_ns[0],
                times_ns[1],
                times_ns[2],
                times_ns[3],
                best_algo,
            )
            .unwrap();
        }

        println!("  Done with N = {}", n);
    }

    csv.flush().unwrap();
    println!("\nDataset written to {csv_path}");
}

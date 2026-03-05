///! Compare radix-2, radix-4, radix-8 (current), and split-radix FFT algorithms
///! on 2000 random "diff" polynomials per size from 1024 to 262144 (powers of 2).
///!
///! For each polynomial size, it:
///!   1. Generates 2000 pairs of random polynomials (coefficients in [-1e6, 1e6]).
///!   2. Computes the difference polynomial (poly_a - poly_b) for each pair.
///!   3. Runs the forward FFT of each diff polynomial under each of the 4 algorithms.
///!   4. Records per-polynomial which algorithm was fastest.
///!   5. Writes a summary CSV ("diff_radix_compare.csv") and prints results to stdout.

use aligned_vec::CACHELINE_ALIGN;
use dyn_stack::PodStack;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use tfhe_fft::c64;
use tfhe_fft::ordered::FftAlgo;
use tfhe_fft::unordered::{Method, Plan, Radix2Plan, Radix4Plan, SplitRadixPlan};

/// Number of diff polynomials per FFT size.
const NUM_POLYS: usize = 2000;

/// Number of times to run each polynomial per algorithm (take average).
const RUNS_PER_POLY: usize = 20;

/// Base ordered-FFT size used by the outer radix decomposition.
/// Must be >= 32 and a power of two.  512 is a reasonable choice that
/// lets every outer strategy have enough levels to exercise its radix.
const BASE_N: usize = 512;

/// Base algorithm for the ordered inner FFT.
const BASE_ALGO: FftAlgo = FftAlgo::Dif4;

/// Names for reporting.
const ALGO_NAMES: [&str; 4] = ["radix2", "radix4", "radix8", "split_radix"];

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // FFT sizes: 1024, 2048, 4096, ..., 262144
    let sizes: Vec<usize> = (10..=15).map(|e| 1usize << e).collect();

    // Open output CSV.
    let csv_path = "diff_radix_compare.csv";
    let mut csv = File::create(csv_path).expect("cannot create CSV file");
    writeln!(
        csv,
        "fft_size,radix2_wins,radix4_wins,radix8_wins,split_radix_wins,\
         radix2_avg_ns,radix4_avg_ns,radix8_avg_ns,split_radix_avg_ns,\
         radix2_plan_ns,radix4_plan_ns,radix8_plan_ns,split_radix_plan_ns"
    )
    .unwrap();

    println!("=== Diff-Polynomial Radix Comparison ===");
    println!(
        "  {} polynomials per size, {} runs each, coefficients in [-1e6, 1e6]",
        NUM_POLYS, RUNS_PER_POLY
    );
    println!("  Base algo: {:?}, base_n: {}", BASE_ALGO, BASE_N);
    println!();

    for &n in &sizes {
        let base_n = if n <= BASE_N { n } else { BASE_N };

        // -------- build plans (timed) --------
        let start = Instant::now();
        let plan_r2 = Radix2Plan::new(n, BASE_ALGO, base_n);
        let plan_r2_create_ns = start.elapsed().as_nanos() as u64;

        let start = Instant::now();
        let plan_r4 = Radix4Plan::new(n, BASE_ALGO, base_n);
        let plan_r4_create_ns = start.elapsed().as_nanos() as u64;

        // radix-8 = default Plan with the same base
        let start = Instant::now();
        let plan_r8 = Plan::new(
            n,
            Method::UserProvided {
                base_algo: BASE_ALGO,
                base_n,
            },
        );
        let plan_r8_create_ns = start.elapsed().as_nanos() as u64;

        let start = Instant::now();
        let plan_sr = SplitRadixPlan::new(n, BASE_ALGO, base_n);
        let plan_sr_create_ns = start.elapsed().as_nanos() as u64;

        let plan_create_ns = [plan_r2_create_ns, plan_r4_create_ns, plan_r8_create_ns, plan_sr_create_ns];

        // Scratch requirement: take the max across all plans.
        let scratch_req = plan_r2
            .fft_scratch()
            .or(plan_r4.fft_scratch())
            .or(plan_r8.fft_scratch())
            .or(plan_sr.fft_scratch());

        // -------- generate 2000 diff polynomials --------
        let mut diff_polys: Vec<Vec<c64>> = Vec::with_capacity(NUM_POLYS);
        for _ in 0..NUM_POLYS {
            let poly: Vec<c64> = (0..n)
                .map(|_| {
                    let a_re: f64 = rng.gen_range(-1e6..1e6);
                    let a_im: f64 = 0.0; // real polynomials
                    let b_re: f64 = rng.gen_range(-1e6..1e6);
                    let b_im: f64 = 0.0;
                    c64 {
                        re: a_re - b_re,
                        im: a_im - b_im,
                    }
                })
                .collect();
            diff_polys.push(poly);
        }

        // -------- warmup each plan (a few runs) --------
        {
            let mut scratch_bytes = vec![0u8; scratch_req.size_bytes() + CACHELINE_ALIGN];
            let mut tmp = diff_polys[0].clone();

            for _ in 0..5 {
                let mut stack = PodStack::new(&mut scratch_bytes);
                plan_r2.fwd(&mut tmp, &mut stack);
            }
            let mut tmp = diff_polys[0].clone();
            for _ in 0..5 {
                let mut stack = PodStack::new(&mut scratch_bytes);
                plan_r4.fwd(&mut tmp, &mut stack);
            }
            let mut tmp = diff_polys[0].clone();
            for _ in 0..5 {
                let mut stack = PodStack::new(&mut scratch_bytes);
                plan_r8.fwd(&mut tmp, &mut stack);
            }
            let mut tmp = diff_polys[0].clone();
            for _ in 0..5 {
                let mut stack = PodStack::new(&mut scratch_bytes);
                plan_sr.fwd(&mut tmp, &mut stack);
            }
        }

        // -------- time each polynomial under each algo --------
        let mut wins = [0u64; 4]; // [r2, r4, r8, sr]
        let mut total_ns = [0u128; 4];

        let mut scratch_bytes = vec![0u8; scratch_req.size_bytes() + CACHELINE_ALIGN];

        for poly in &diff_polys {
            let mut times_ns = [0u64; 4];

            // Radix-2: run RUNS_PER_POLY times, take average
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

            // Radix-4: run RUNS_PER_POLY times, take average
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

            // Radix-8 (current): run RUNS_PER_POLY times, take average
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

            // Split-radix: run RUNS_PER_POLY times, take average
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

            // Which was fastest?
            let min_time = *times_ns.iter().min().unwrap();
            let best_idx = times_ns.iter().position(|&t| t == min_time).unwrap();
            wins[best_idx] += 1;

            for i in 0..4 {
                total_ns[i] += times_ns[i] as u128;
            }
        }

        let avg_ns: Vec<f64> = total_ns
            .iter()
            .map(|&t| t as f64 / NUM_POLYS as f64)
            .collect();

        // -------- report --------
        println!("FFT size = {n}");
        println!("  Plan creation times:");
        for i in 0..4 {
            println!(
                "    {:>12}: {:>10} ns",
                ALGO_NAMES[i], plan_create_ns[i]
            );
        }
        println!("  FFT execution:");
        for i in 0..4 {
            println!(
                "    {:>12}: wins = {:>5} / {NUM_POLYS}  |  avg = {:>10.1} ns",
                ALGO_NAMES[i], wins[i], avg_ns[i]
            );
        }
        println!();

        writeln!(
            csv,
            "{},{},{},{},{},{:.1},{:.1},{:.1},{:.1},{},{},{},{}",
            n,
            wins[0],
            wins[1],
            wins[2],
            wins[3],
            avg_ns[0],
            avg_ns[1],
            avg_ns[2],
            avg_ns[3],
            plan_create_ns[0],
            plan_create_ns[1],
            plan_create_ns[2],
            plan_create_ns[3],
        )
        .unwrap();
    }

    println!("Results written to {csv_path}");
}

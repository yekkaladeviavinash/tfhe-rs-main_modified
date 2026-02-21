///! Proof that the "winner" algorithm for a single polynomial is NOT deterministic.
///!
///! Takes ONE fixed random polynomial per FFT size, and re-runs it 1000 times
///! under each of the 4 algorithms (radix-2, radix-4, radix-8, split-radix).
///! Each "run" is the average of 20 FFT invocations.
///!
///! If algorithm choice were data-dependent, you'd see one algorithm winning
///! all 1000 times.  Instead, you'll see wins scattered across algorithms,
///! proving that per-polynomial variation is system noise.

use aligned_vec::CACHELINE_ALIGN;
use dyn_stack::PodStack;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe_fft::c64;
use tfhe_fft::ordered::FftAlgo;
use tfhe_fft::unordered::{Method, Plan, Radix2Plan, Radix4Plan, SplitRadixPlan};

/// Number of times to re-run the same polynomial (each run = avg of INNER_RUNS).
const OUTER_RUNS: usize = 1000;

/// Number of FFT invocations averaged per measurement.
const INNER_RUNS: usize = 20;

const BASE_N: usize = 512;
const BASE_ALGO: FftAlgo = FftAlgo::Dif4;
const ALGO_NAMES: [&str; 4] = ["radix2", "radix4", "radix8", "split_radix"];

fn main() {
    let mut rng = StdRng::seed_from_u64(123);

    // Test a few representative sizes
    let sizes: Vec<usize> = vec![1024, 4096, 8192, 16384];

    println!("=== Same-Polynomial Re-run Test ===");
    println!(
        "  1 fixed polynomial per size, re-run {} times ({} inner avg each)",
        OUTER_RUNS, INNER_RUNS
    );
    println!("  Base algo: {:?}, base_n: {}", BASE_ALGO, BASE_N);
    println!();

    for &n in &sizes {
        let base_n = if n <= BASE_N { n } else { BASE_N };

        // Build plans
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

        let scratch_req = plan_r2
            .fft_scratch()
            .or(plan_r4.fft_scratch())
            .or(plan_r8.fft_scratch())
            .or(plan_sr.fft_scratch());

        // Generate ONE fixed polynomial with coefficients in [-1e6, 1e6]
        let poly: Vec<c64> = (0..n)
            .map(|_| {
                let a: f64 = rng.gen_range(-1e6..1e6);
                let b: f64 = rng.gen_range(-1e6..1e6);
                c64 { re: a - b, im: 0.0 }
            })
            .collect();

        let mut scratch_bytes = vec![0u8; scratch_req.size_bytes() + CACHELINE_ALIGN];

        // Warmup
        for _ in 0..10 {
            let mut buf = poly.clone();
            let mut stack = PodStack::new(&mut scratch_bytes);
            plan_r2.fwd(&mut buf, &mut stack);
            let mut buf = poly.clone();
            let mut stack = PodStack::new(&mut scratch_bytes);
            plan_r4.fwd(&mut buf, &mut stack);
            let mut buf = poly.clone();
            let mut stack = PodStack::new(&mut scratch_bytes);
            plan_r8.fwd(&mut buf, &mut stack);
            let mut buf = poly.clone();
            let mut stack = PodStack::new(&mut scratch_bytes);
            plan_sr.fwd(&mut buf, &mut stack);
        }

        // Re-run the SAME polynomial OUTER_RUNS times
        let mut wins = [0u64; 4];
        let mut total_avg_ns = [0.0f64; 4];

        for _ in 0..OUTER_RUNS {
            let mut times_ns = [0u64; 4];

            // Radix-2: average of INNER_RUNS
            {
                let mut total: u64 = 0;
                for _ in 0..INNER_RUNS {
                    let mut buf = poly.clone();
                    let mut stack = PodStack::new(&mut scratch_bytes);
                    let start = Instant::now();
                    plan_r2.fwd(&mut buf, &mut stack);
                    let elapsed = start.elapsed().as_nanos() as u64;
                    std::hint::black_box(&buf);
                    total += elapsed;
                }
                times_ns[0] = total / INNER_RUNS as u64;
            }

            // Radix-4: average of INNER_RUNS
            {
                let mut total: u64 = 0;
                for _ in 0..INNER_RUNS {
                    let mut buf = poly.clone();
                    let mut stack = PodStack::new(&mut scratch_bytes);
                    let start = Instant::now();
                    plan_r4.fwd(&mut buf, &mut stack);
                    let elapsed = start.elapsed().as_nanos() as u64;
                    std::hint::black_box(&buf);
                    total += elapsed;
                }
                times_ns[1] = total / INNER_RUNS as u64;
            }

            // Radix-8: average of INNER_RUNS
            {
                let mut total: u64 = 0;
                for _ in 0..INNER_RUNS {
                    let mut buf = poly.clone();
                    let mut stack = PodStack::new(&mut scratch_bytes);
                    let start = Instant::now();
                    plan_r8.fwd(&mut buf, &mut stack);
                    let elapsed = start.elapsed().as_nanos() as u64;
                    std::hint::black_box(&buf);
                    total += elapsed;
                }
                times_ns[2] = total / INNER_RUNS as u64;
            }

            // Split-radix: average of INNER_RUNS
            {
                let mut total: u64 = 0;
                for _ in 0..INNER_RUNS {
                    let mut buf = poly.clone();
                    let mut stack = PodStack::new(&mut scratch_bytes);
                    let start = Instant::now();
                    plan_sr.fwd(&mut buf, &mut stack);
                    let elapsed = start.elapsed().as_nanos() as u64;
                    std::hint::black_box(&buf);
                    total += elapsed;
                }
                times_ns[3] = total / INNER_RUNS as u64;
            }

            // Which was fastest this run?
            let min_time = *times_ns.iter().min().unwrap();
            let best_idx = times_ns.iter().position(|&t| t == min_time).unwrap();
            wins[best_idx] += 1;

            for i in 0..4 {
                total_avg_ns[i] += times_ns[i] as f64;
            }
        }

        let grand_avg: Vec<f64> = total_avg_ns
            .iter()
            .map(|&t| t / OUTER_RUNS as f64)
            .collect();

        println!("FFT size = {n}  (same polynomial re-run {OUTER_RUNS} times)");
        for i in 0..4 {
            println!(
                "  {:>12}: wins = {:>5} / {OUTER_RUNS}  |  grand avg = {:>10.1} ns",
                ALGO_NAMES[i], wins[i], grand_avg[i]
            );
        }
        println!();
    }
}

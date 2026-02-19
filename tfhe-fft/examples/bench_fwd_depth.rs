///! Measures the execution time of the `fwd_depth` function
///! (called internally by `Plan::fwd`) for various FFT sizes.

use aligned_vec::CACHELINE_ALIGN;
use dyn_stack::PodStack;
use std::time::{Duration, Instant};
use tfhe_fft::c64;
use tfhe_fft::unordered::{Method, Plan};

/// Number of warmup iterations before timing.
const WARMUP_ITERS: u32 = 50;
/// Number of timed iterations to average over.
const BENCH_ITERS: u32 = 1000;

fn bench_fwd_depth(n: usize) {
    // Build the plan using the measurement-based method so it picks the fastest internal algo.
    let plan = Plan::new(n, Method::Measure(Duration::from_millis(10)));

    // Allocate the data buffer.
    let scratch_req = plan.fft_scratch();

    let mut buf = vec![c64 { re: 0.0, im: 0.0 }; n];

    // Fill with some non-trivial data so the compiler cannot optimise the call away.
    for (i, z) in buf.iter_mut().enumerate() {
        let angle = std::f64::consts::TAU * (i as f64) / (n as f64);
        *z = c64 {
            re: angle.cos(),
            im: angle.sin(),
        };
    }

    // ---- warmup ----
    for _ in 0..WARMUP_ITERS {
        let mut scratch_bytes = vec![0u8; scratch_req.size_bytes() + CACHELINE_ALIGN];
        let mut stack = PodStack::new(&mut scratch_bytes);
        let mut tmp = buf.clone();
        plan.fwd(&mut tmp, &mut stack);
        // prevent optimisation
        std::hint::black_box(&tmp);
    }

    // ---- timed runs ----
    let mut total = Duration::ZERO;
    let mut min_time = Duration::MAX;
    let mut max_time = Duration::ZERO;

    for _ in 0..BENCH_ITERS {
        let mut scratch_bytes = vec![0u8; scratch_req.size_bytes() + CACHELINE_ALIGN];
        let mut stack = PodStack::new(&mut scratch_bytes);
        let mut tmp = buf.clone();

        let start = Instant::now();
        plan.fwd(&mut tmp, &mut stack);
        let elapsed = start.elapsed();

        std::hint::black_box(&tmp);

        total += elapsed;
        if elapsed < min_time {
            min_time = elapsed;
        }
        if elapsed > max_time {
            max_time = elapsed;
        }
    }

    let avg = total / BENCH_ITERS;

    println!("FFT size = {n:>6} | algo = {:?}", plan.algo());
    println!("  iterations : {BENCH_ITERS}");
    println!("  avg        : {avg:>12.3?}");
    println!("  min        : {min_time:>12.3?}");
    println!("  max        : {max_time:>12.3?}");
    println!("  total      : {total:>12.3?}");
    println!();
}

fn main() {
    println!("=== fwd_depth (Plan::fwd) benchmark ===");
    println!();

    // Bench a range of power-of-two FFT sizes.
    for exp in 2..=18 {
        let n = 1 << exp; // 4, 8, 16, … , 16384
        bench_fwd_depth(n);
    }
}

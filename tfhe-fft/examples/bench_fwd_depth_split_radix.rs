///! Measures the execution time of the `fwd_depth_split_radix` function
///! (called internally by `SplitRadixPlan::fwd`) for various FFT sizes.

use aligned_vec::CACHELINE_ALIGN;
use dyn_stack::PodStack;
use std::time::{Duration, Instant};
use tfhe_fft::c64;
use tfhe_fft::ordered::FftAlgo;
use tfhe_fft::unordered::SplitRadixPlan;

/// Number of warmup iterations before timing.
const WARMUP_ITERS: u32 = 50;
/// Number of timed iterations to average over.
const BENCH_ITERS: u32 = 1000;

/// Candidate base sizes to try, mirroring what `Plan::new` with `Method::Measure` does.
const BASE_CANDIDATES: [usize; 2] = [512, 1024];

fn bench_fwd_depth_split_radix(n: usize) {
    let base_algo = FftAlgo::Dif4;

    // Pick the best base_n by quick measurement (same idea as Plan's Measure method).
    // For small n, base_n = n (the entire FFT is handled by the ordered base case).
    let plan = pick_fastest_plan(n, base_algo);
    let (_, base_n) = plan.algo();

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

    println!(
        "FFT size = {n:>6} | algo = {:?}, base_n = {base_n}",
        plan.algo()
    );
    println!("  iterations : {BENCH_ITERS}");
    println!("  avg        : {avg:>12.3?}");
    println!("  min        : {min_time:>12.3?}");
    println!("  max        : {max_time:>12.3?}");
    println!("  total      : {total:>12.3?}");
    println!();
}

/// Try each candidate base size and return the plan that runs fastest.
fn pick_fastest_plan(n: usize, base_algo: FftAlgo) -> SplitRadixPlan {
    const QUICK_ITERS: u32 = 200;

    let mut best_plan: Option<SplitRadixPlan> = None;
    let mut best_time = Duration::MAX;

    for &base_n in &BASE_CANDIDATES {
        if base_n > n {
            continue;
        }
        let plan = SplitRadixPlan::new(n, base_algo, base_n);
        let scratch_req = plan.fft_scratch();

        let mut buf = vec![c64 { re: 1.0, im: 0.0 }; n];

        // Quick warmup
        for _ in 0..10 {
            let mut scratch_bytes = vec![0u8; scratch_req.size_bytes() + CACHELINE_ALIGN];
            let mut stack = PodStack::new(&mut scratch_bytes);
            plan.fwd(&mut buf, &mut stack);
            std::hint::black_box(&buf);
        }

        // Quick timing
        let start = Instant::now();
        for _ in 0..QUICK_ITERS {
            let mut scratch_bytes = vec![0u8; scratch_req.size_bytes() + CACHELINE_ALIGN];
            let mut stack = PodStack::new(&mut scratch_bytes);
            plan.fwd(&mut buf, &mut stack);
            std::hint::black_box(&buf);
        }
        let elapsed = start.elapsed() / QUICK_ITERS;

        if elapsed < best_time {
            best_time = elapsed;
            best_plan = Some(plan);
        }
    }

    // If n is too small for any candidate base, use n itself as the base.
    best_plan.unwrap_or_else(|| SplitRadixPlan::new(n, base_algo, n))
}

fn main() {
    println!("=== fwd_depth_split_radix (SplitRadixPlan::fwd) benchmark ===");
    println!();

    // Bench a range of power-of-two FFT sizes.
    for exp in 2..=18 {
        let n = 1 << exp; // 4, 8, 16, … , 16384
        bench_fwd_depth_split_radix(n);
    }
}

use rand::random;
use std::collections::HashMap;
use std::time::Instant;
use tfhe_ntt::prime32::Plan;
use tfhe_ntt::prime32_r4::Plan_r4;

fn main() {
    // define suitable NTT prime
    let p: u32 = 1073479681;
    let mul = |x: u32, y: u32| ((x as u64 * y as u64) % p as u64) as u32;

    let sizes: [usize; 10] = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
    let mut plans: HashMap<usize, (Plan, Plan_r4)> = HashMap::new();

    let trials = 1000;
    let runs_per_trial = 30;
    let warmup_runs = 1;

    let mut r2_better = 0usize;
    let mut r4_better = 0usize;
    let mut ties = 0usize;
    let mut per_size_counts: HashMap<usize, (usize, usize, usize)> = HashMap::new();

    let mut last_size: Option<usize> = None;

    for _ in 0..trials {
        let mut size = sizes[random::<usize>() % sizes.len()];
        while Some(size) == last_size {
            size = sizes[random::<usize>() % sizes.len()];
        }
        last_size = Some(size);

        let (plan_r2, plan_r4) = plans.entry(size).or_insert_with(|| {
            (
                Plan::try_new(size, p).unwrap(),
                Plan_r4::try_new_r4(size, p).unwrap(),
            )
        });

        let lhs_poly: Vec<u32> = (0..size).map(|_| random::<u32>() % p).collect();
        let rhs_poly: Vec<u32> = (0..size).map(|_| random::<u32>() % p).collect();

        let mut r2_sum_ns: u128 = 0;
        let mut r4_sum_ns: u128 = 0;

        for run in 0..runs_per_trial {
            // radix-4
            let mut lhs_r4 = lhs_poly.clone();
            let mut rhs_r4 = rhs_poly.clone();
            let start_r4 = Instant::now();
            plan_r4.fwd_r4(&mut lhs_r4);
            plan_r4.fwd_r4(&mut rhs_r4);
            for i in 0..size {
                lhs_r4[i] = mul(lhs_r4[i], rhs_r4[i]);
            }
            plan_r4.inv_r4(&mut lhs_r4);
            plan_r4.normalize_r4(&mut lhs_r4);
            let elapsed_r4 = start_r4.elapsed().as_nanos();
            if run >= warmup_runs {
                r4_sum_ns += elapsed_r4;
            }
        }



        for run in 0..runs_per_trial {
            // radix-2
            let mut lhs_r2 = lhs_poly.clone();
            let mut rhs_r2 = rhs_poly.clone();
            let start_r2 = Instant::now();
            plan_r2.fwd(&mut lhs_r2);
            plan_r2.fwd(&mut rhs_r2);
            for i in 0..size {
                lhs_r2[i] = mul(lhs_r2[i], rhs_r2[i]);
            }
            plan_r2.inv(&mut lhs_r2);
            plan_r2.normalize(&mut lhs_r2);
            let elapsed_r2 = start_r2.elapsed().as_nanos();
            if run >= warmup_runs {
                r2_sum_ns += elapsed_r2;
            }
        }

        let effective_runs = (runs_per_trial - warmup_runs) as u128;
        let r2_avg_ns = r2_sum_ns / effective_runs;
        let r4_avg_ns = r4_sum_ns / effective_runs;

        let entry = per_size_counts.entry(size).or_insert((0, 0, 0));
        if r2_avg_ns < r4_avg_ns {
            r2_better += 1;
            entry.0 += 1;
        } else if r4_avg_ns < r2_avg_ns {
            r4_better += 1;
            entry.1 += 1;
        } else {
            ties += 1;
            entry.2 += 1;
        }
    }

    println!("Trials: {trials}");
    println!("Radix-2 better: {r2_better}");
    println!("Radix-4 better: {r4_better}");
    println!("Ties: {ties}");
    println!("Per-size results:");
    let mut sizes_sorted = sizes.to_vec();
    sizes_sorted.sort_unstable();
    for size in sizes_sorted {
        let (r2, r4, tie) = per_size_counts.get(&size).copied().unwrap_or((0, 0, 0));
        println!("  size {size}: r2={r2}, r4={r4}, ties={tie}");
    }
}

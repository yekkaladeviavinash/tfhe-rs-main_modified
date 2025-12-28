use rand::Rng;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use tfhe_ntt::custum_radix::{
    fft_radix2_recursive_mut, fft_radix4_recursive_mut, fft_split_radix_recursive_mut,
};
use tfhe_ntt::custum_radix::fwd_1::MultStats;

/// Generate timing+mult dataset for raw_N in 1..=MAX_RAW_N, with POLYS_PER_N samples each.
///
/// Output columns intentionally match `final_dataset_200_with_times.csv` schema.
const MAX_RAW_N: usize = 7000;
const POLYS_PER_N: usize = 3;

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
    assert!(n > 0 && n.is_power_of_two());
    let mut tw = vec![0u32; n];
    let mut cur: u32 = 1;
    for k in 0..n {
        tw[k] = cur;
        cur = mul_mod(cur, root, p);
    }
    tw
}

fn pad_to_pow2(mut v: Vec<u32>) -> Vec<u32> {
    let target = v.len().next_power_of_two();
    v.resize(target, 0);
    v
}

fn nonzero_fractions(poly: &[u32]) -> (f64, f64, [f64; 4]) {
    let n = poly.len();

    // n is always a power-of-two >= 1. For n < 4, denom4 becomes 0; we define fractions as 0.0.
    let denom2 = (n / 2) as f64;
    let denom4 = (n / 4) as f64;

    let mut even_nz = 0usize;
    let mut odd_nz = 0usize;
    let mut mod4_nz = [0usize; 4];

    for (idx, &coef) in poly.iter().enumerate() {
        if coef != 0 {
            if idx % 2 == 0 {
                even_nz += 1;
            } else {
                odd_nz += 1;
            }
            mod4_nz[idx % 4] += 1;
        }
    }

    let even_frac = if denom2 > 0.0 {
        even_nz as f64 / denom2
    } else {
        0.0
    };
    let odd_frac = if denom2 > 0.0 {
        odd_nz as f64 / denom2
    } else {
        0.0
    };

    let mod4_frac = if denom4 > 0.0 {
        [
            mod4_nz[0] as f64 / denom4,
            mod4_nz[1] as f64 / denom4,
            mod4_nz[2] as f64 / denom4,
            mod4_nz[3] as f64 / denom4,
        ]
    } else {
        [0.0; 4]
    };

    (even_frac, odd_frac, mod4_frac)
}

fn compute_mults(poly: &[u32], twiddles: &[u32], p: u32) -> (usize, usize, usize) {
    let mut r2_stats = MultStats::default();
    let mut r4_stats = MultStats::default();
    let mut rs_stats = MultStats::default();

    let mut p2 = poly.to_vec();
    let mut p4 = poly.to_vec();
    let mut ps = poly.to_vec();

    fft_radix2_recursive_mut(&mut p2, twiddles, p, &mut r2_stats);
    fft_radix4_recursive_mut(&mut p4, twiddles, p, &mut r4_stats);
    fft_split_radix_recursive_mut(&mut ps, twiddles, p, &mut rs_stats);

    (r2_stats.nonzero_mults, r4_stats.nonzero_mults, rs_stats.nonzero_mults)
}

fn pick_best_mult(m2: usize, m4: usize, ms: usize) -> &'static str {
    if m2 <= m4 && m2 <= ms {
        "r2"
    } else if m4 <= m2 && m4 <= ms {
        "r4"
    } else {
        "rs"
    }
}

fn time_forward_wall_avg_us(poly: &[u32], twiddles: &[u32], p: u32, algo: &str) -> f64 {
    // Run 30 times: discard first, average remaining 29.
    let mut sum = 0.0f64;
    let mut count = 0usize;

    for iter in 0..30 {
        let mut stats = MultStats::default();
        let mut buf = poly.to_vec();

        let start = Instant::now();
        match algo {
            "r2" => fft_radix2_recursive_mut(&mut buf, twiddles, p, &mut stats),
            "r4" => fft_radix4_recursive_mut(&mut buf, twiddles, p, &mut stats),
            "rs" => fft_split_radix_recursive_mut(&mut buf, twiddles, p, &mut stats),
            _ => unreachable!(),
        }
        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;

        if iter == 0 {
            continue;
        }
        sum += elapsed_us;
        count += 1;
    }

    sum / (count as f64)
}

fn min_time_label(t_r2: f64, t_r4: f64, t_rs: f64) -> &'static str {
    // Deterministic tie-breaker: r2, then r4, then rs.
    if t_r2 <= t_r4 && t_r2 <= t_rs {
        "r2"
    } else if t_r4 <= t_r2 && t_r4 <= t_rs {
        "r4"
    } else {
        "rs"
    }
}

fn main() -> std::io::Result<()> {
    let mut rng = rand::thread_rng();

    // Keep same naming convention as before but with correct size.
    let out_path = "final_dataset_300_with_times.csv";
    let file = File::create(out_path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "raw_N,padded_N,polynomial,frac_2i_nonzero,frac_2i1_nonzero,frac_4i_nonzero,frac_4i1_nonzero,frac_4i2_nonzero,frac_4i3_nonzero,mult_r2,mult_r4,mult_rs,output,time_r2_sys_avg,time_rs_sys_avg,time_r4_sys_avg,time_min_label,label_mult_eq_time"
    )?;

    let p: u32 = 65537;
    let g: u32 = 3;

    for raw_n in 1..=MAX_RAW_N {
        for _ in 0..POLYS_PER_N {
            // Same style as Dataset.rs: random coefficients in [0, 50)
            let raw_poly: Vec<u32> = (0..raw_n).map(|_| rng.gen_range(0..50)).collect();
            let poly = pad_to_pow2(raw_poly);
            let n = poly.len();

            let root = pow_mod(g, (p - 1) / (n as u32), p);
            let twiddles = make_twiddles_from_root(root, n, p);

            let (even_frac, odd_frac, mod4_frac) = nonzero_fractions(&poly);

            let (m2, m4, ms) = compute_mults(&poly, &twiddles, p);
            let best_mult = pick_best_mult(m2, m4, ms);

            let t_r2 = time_forward_wall_avg_us(&poly, &twiddles, p, "r2");
            let t_rs = time_forward_wall_avg_us(&poly, &twiddles, p, "rs");
            let t_r4 = time_forward_wall_avg_us(&poly, &twiddles, p, "r4");
            let best_time = min_time_label(t_r2, t_r4, t_rs);
            let label_ok = if best_mult == best_time { "yes" } else { "no" };

            let poly_str = format!(
                "[{}]",
                poly.iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );

            writeln!(
                writer,
                "{raw_n},{padded_n},\"{poly}\",{even:.6},{odd:.6},{m0:.6},{m1:.6},{m2f:.6},{m3:.6},{r2},{r4},{rs},{out},{t2:.3},{ts:.3},{t4:.3},{tmin},{ok}",
                raw_n = raw_n,
                padded_n = n,
                poly = poly_str,
                even = even_frac,
                odd = odd_frac,
                m0 = mod4_frac[0],
                m1 = mod4_frac[1],
                m2f = mod4_frac[2],
                m3 = mod4_frac[3],
                r2 = m2,
                r4 = m4,
                rs = ms,
                out = best_mult,
                t2 = t_r2,
                ts = t_rs,
                t4 = t_r4,
                tmin = best_time,
                ok = label_ok
            )?;
        }

        // Periodic flush so progress is written.
        if raw_n % 50 == 0 {
            writer.flush()?;
        }
    }

    writer.flush()?;
    eprintln!("Wrote {out_path}");
    Ok(())
}

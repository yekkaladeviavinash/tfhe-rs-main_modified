use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use tfhe_ntt::custum_radix::{
    fft_radix2_recursive_mut, fft_radix4_recursive_mut, fft_split_radix_recursive_mut,
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
    assert!(n > 0 && n.is_power_of_two());
    let mut tw = vec![0u32; n];
    let mut cur: u32 = 1;
    for k in 0..n {
        tw[k] = cur;
        cur = mul_mod(cur, root, p);
    }
    tw
}

fn nonzero_fractions(poly: &[u32]) -> (f64, f64, [f64; 4]) {
    let n = poly.len();
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

    let even_frac = even_nz as f64 / denom2;
    let odd_frac = odd_nz as f64 / denom2;
    let mod4_frac = [
        mod4_nz[0] as f64 / denom4,
        mod4_nz[1] as f64 / denom4,
        mod4_nz[2] as f64 / denom4,
        mod4_nz[3] as f64 / denom4,
    ];

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

fn poly_to_field(poly: &[u32]) -> String {
    format!(
        "[{}]",
        poly.iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    )
}

fn repeat_to_len(pattern: &[u32], n: usize) -> Vec<u32> {
    assert!(!pattern.is_empty());
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let rem = n - out.len();
        if rem >= pattern.len() {
            out.extend_from_slice(pattern);
        } else {
            out.extend_from_slice(&pattern[..rem]);
        }
    }
    out
}

fn make_case_poly(case_id: usize, n: usize, base64: &[u32]) -> Vec<u32> {
    match case_id {
        // Test Case 1: original polynomial (defined for N=64) repeated for larger N
        1 => repeat_to_len(base64, n),

        // Test Case 2: (index % 8 == 7) are zero
        2 => (0..n)
            .map(|i| {
                if i % 8 == 7 {
                    0
                } else {
                    (i as u32) + 1
                }
            })
            .collect(),

        // Test Case 3: (index % 4 == 3) are zero
        3 => (0..n)
            .map(|i| {
                if i % 4 == 3 {
                    0
                } else {
                    (i as u32) + 1
                }
            })
            .collect(),

        // Test Case 4: (index % 4 == 1) are zero
        4 => (0..n)
            .map(|i| {
                if i % 4 == 1 {
                    0
                } else {
                    (i as u32) + 1
                }
            })
            .collect(),

        // Test Case 5: all odd indices are zero
        5 => (0..n)
            .map(|i| {
                if i % 2 == 1 {
                    0
                } else {
                    // 64-case used values in [1..9]; keep small repeating values
                    ((i % 64) as u32 % 9) + 1
                }
            })
            .collect(),

        // Test Case 6: only indices % 4 == 0 are non-zero
        6 => (0..n)
            .map(|i| if i % 4 == 0 { ((i % 64) as u32 % 9) + 1 } else { 0 })
            .collect(),

        // Test Case 7: only indices % 4 == 2 are non-zero
        7 => (0..n)
            .map(|i| if i % 4 == 2 { ((i % 64) as u32 % 9) + 1 } else { 0 })
            .collect(),

        // Test Case 8: only indices % 4 == 1 are non-zero
        8 => (0..n)
            .map(|i| if i % 4 == 1 { ((i % 64) as u32 % 9) + 1 } else { 0 })
            .collect(),

        // Test Case 9: first half == second half
        9 => {
            let half = n / 2;
            let mut first: Vec<u32> = (0..half)
                .map(|i| {
                    if i % 4 == 1 {
                        // mimic [0,10,0,0], [0,20,0,0], ... pattern
                        (((i / 4) % 8) as u32 + 1) * 10
                    } else {
                        0
                    }
                })
                .collect();
            let mut out = first.clone();
            out.append(&mut first);
            out
        }

        // Test Case 10: only indices % 8 == 0 are non-zero
        10 => (0..n)
            .map(|i| if i % 8 == 0 { (((i / 8) % 16) as u32 + 1) * 10 } else { 0 })
            .collect(),

        // Test Case 11: only indices % 4 == 3 are non-zero
        11 => (0..n)
            .map(|i| if i % 4 == 3 { (((i / 4) % 32) as u32 + 1) * 10 } else { 0 })
            .collect(),

        // Test Case 12: only even indices are zero
        12 => (0..n).map(|i| if i % 2 == 0 { 0 } else { (i as u32 + 1) / 2 }).collect(),

        // Test Case 13: indices % 4 == 2 are zero, other positions non-zero
        13 => (0..n)
            .map(|i| {
                if i % 4 == 2 {
                    0
                } else {
                    if i % 4 == 0 {
                        // larger values on i%4==0 like 100,200,...
                        (((i / 4) % 64) as u32 + 1) * 100
                    } else {
                        // small values on other non-zero lanes
                        ((i % 64) as u32 % 32) + 1
                    }
                }
            })
            .collect(),

        // Test Case 14: indices % 4 in {0,1} are zero (pattern 0,0,1,1 repeated)
        14 => (0..n).map(|i| if i % 4 == 2 || i % 4 == 3 { 1 } else { 0 }).collect(),

        _ => unreachable!("unknown case id"),
    }
}

fn main() -> std::io::Result<()> {
    let base_n = 64usize;
    let base_tc1: Vec<u32> = vec![
        5, 11, 3, 12, 8, 13, 2, 14, 4, 15, 7, 16, 6, 17, 1, 18, 3, 19, 9, 20, 2, 21,
        5, 22, 7, 23, 4, 24, 1, 25, 8, 26, 9, 27, 6, 28, 3, 29, 5, 30, 2, 31, 8, 32,
        4, 33, 7, 34, 1, 35, 6, 36, 3, 37, 9, 38, 2, 39, 5, 40, 7, 41, 4, 42,
    ];
    assert_eq!(base_tc1.len(), base_n);

    // N values to generate (powers of two >=8 and <7000)
    let n_values: [usize; 10] = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

    let p: u32 = 65537;
    let g: u32 = 3;

    let case_names: [(&str, usize); 14] = [
        ("tc1_original", 1),
        ("tc2_idx_mod8_eq7_zero", 2),
        ("tc3_idx_mod4_eq3_zero", 3),
        ("tc4_idx_mod4_eq1_zero", 4),
        ("tc5_all_odd_zero", 5),
        ("tc6_only_idx_mod4_eq0_nonzero", 6),
        ("tc7_only_idx_mod4_eq2_nonzero", 7),
        ("tc8_only_idx_mod4_eq1_nonzero", 8),
        ("tc9_first_half_eq_second_half", 9),
        ("tc10_only_idx_mod8_eq0_nonzero", 10),
        ("tc11_only_idx_mod4_eq3_nonzero", 11),
        ("tc12_only_even_zero", 12),
        ("tc13_idx_mod4_eq2_zero", 13),
        ("tc14_idx_mod4_0_1_zero", 14),
    ];

    let out_path = "special_cases.csv";
    let file = File::create(out_path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "case,raw_N,padded_N,polynomial,frac_2i_nonzero,frac_2i1_nonzero,frac_4i_nonzero,frac_4i1_nonzero,frac_4i2_nonzero,frac_4i3_nonzero,mult_r2,mult_r4,mult_rs,output,time_r2_sys_avg,time_rs_sys_avg,time_r4_sys_avg,time_min_label,label_mult_eq_time"
    )?;

    for &n in &n_values {
        let root = pow_mod(g, (p - 1) / (n as u32), p);
        let twiddles = make_twiddles_from_root(root, n, p);

        for (case_name, case_id) in case_names {
            let mut poly = make_case_poly(case_id, n, &base_tc1);
            poly.resize(n, 0);

            let (even_frac, odd_frac, mod4_frac) = nonzero_fractions(&poly);
            let (m2, m4, ms) = compute_mults(&poly, &twiddles, p);
            let best_mult = pick_best_mult(m2, m4, ms);

            let t_r2 = time_forward_wall_avg_us(&poly, &twiddles, p, "r2");
            let t_rs = time_forward_wall_avg_us(&poly, &twiddles, p, "rs");
            let t_r4 = time_forward_wall_avg_us(&poly, &twiddles, p, "r4");
            let best_time = min_time_label(t_r2, t_r4, t_rs);
            let label_ok = if best_mult == best_time { "yes" } else { "no" };

            writeln!(
                writer,
                "{case}_n{n},{raw_n},{padded_n},\"{poly_field}\",{even:.6},{odd:.6},{m0:.6},{m1:.6},{m2f:.6},{m3:.6},{r2},{r4},{rs},{out},{t2:.3},{ts:.3},{t4:.3},{tmin},{ok}",
                case = case_name,
                n = n,
                raw_n = n,
                padded_n = n,
                poly_field = poly_to_field(&poly),
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

            // Flush periodically
            writer.flush()?;
        }
    }

    writer.flush()?;
    eprintln!("Wrote {out_path}");
    Ok(())
}

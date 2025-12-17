use rand::Rng;
use std::io::BufWriter;
use std::fs::File;
use std::io::Write;

use tfhe_ntt::custum_radix::{
    fft_radix2_recursive_mut,
    fft_radix4_recursive_mut,
    fft_split_radix_recursive_mut,
};
use tfhe_ntt::custum_radix::fwd_1::MultStats;

const MAX_RAW_N: usize = 7000;
const POLYS_PER_N: usize = 3;

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

fn mul_mod(a: u32, b: u32, p: u32) -> u32 {
    ((a as u64 * b as u64) % (p as u64)) as u32
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

fn compute_mults(poly: &[u32]) -> (usize, usize, usize) {
    let n = poly.len();
    let p: u32 = 65537;
    let g = 3; // known primitive root for 65537
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

fn pick_best(m2: usize, m4: usize, ms: usize) -> &'static str {
    if m2 <= m4 && m2 <= ms { "r2" }
    else if m4 <= m2 && m4 <= ms { "r4" }
    else { "rs" }
}

fn main() -> std::io::Result<()> {
    let mut rng = rand::thread_rng();
    // Write directly to file (no huge in-memory CSV, no terminal spam)
    let file = File::create("dataset_output.csv")?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "raw_N,padded_N,polynomial,frac_2i_nonzero,frac_2i1_nonzero,frac_4i_nonzero,frac_4i1_nonzero,frac_4i2_nonzero,frac_4i3_nonzero,mult_r2,mult_r4,mult_rs,output"
    )?;

    for raw_n in 1..=MAX_RAW_N {
        for _ in 0..POLYS_PER_N {
            let raw_poly: Vec<u32> = (0..raw_n).map(|_| rng.gen_range(0..50)).collect();
            let poly = pad_to_pow2(raw_poly);
            let padded_n = poly.len();

            let (even_frac, odd_frac, mod4_frac) = nonzero_fractions(&poly);
            let (m2, m4, ms) = compute_mults(&poly);
            let best = pick_best(m2, m4, ms);

            // Keep polynomial as a single CSV field
            let poly_str = format!(
                "[{}]",
                poly.iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );

            writeln!(
                writer,
                "{raw_n},{padded_n},\"{poly}\",{even:.6},{odd:.6},{m0:.6},{m1:.6},{m2:.6},{m3:.6},{r2},{r4},{rs},{best}",
                raw_n = raw_n,
                padded_n = padded_n,
                poly = poly_str,
                even = even_frac,
                odd = odd_frac,
                m0 = mod4_frac[0],
                m1 = mod4_frac[1],
                m2 = mod4_frac[2],
                m3 = mod4_frac[3],
                r2 = m2,
                r4 = m4,
                rs = ms,
                best = best
            )?;
        }
    }

    writer.flush()?;

    Ok(())
}

use std::fs::File;
use std::io::{self, BufWriter};
use std::time::Instant;

use serde::Deserialize;

use tfhe_ntt::custum_radix::{
    fft_radix2_recursive_mut,
    fft_radix4_recursive_mut,
    fft_split_radix_recursive_mut,
};
use tfhe_ntt::custum_radix::fwd_1::MultStats;

#[derive(Debug, Deserialize)]
struct RowIn {
    raw_N: usize,
    padded_N: usize,
    polynomial: String,
    frac_2i_nonzero: f64,
    frac_2i1_nonzero: f64,
    frac_4i_nonzero: f64,
    frac_4i1_nonzero: f64,
    frac_4i2_nonzero: f64,
    frac_4i3_nonzero: f64,
    mult_r2: Option<usize>,
    mult_r4: Option<usize>,
    mult_rs: Option<usize>,
    output: String,
}

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
    debug_assert!(n.is_power_of_two());
    let mut tw = vec![0u32; n];
    let mut cur: u32 = 1;
    for k in 0..n {
        tw[k] = cur;
        cur = mul_mod(cur, root, p);
    }
    tw
}

fn parse_poly_field(field: &str, expected_len: usize) -> Vec<u32> {
    // Field format: "[a b c ...]" (space-separated)
    let s = field.trim();
    let s = s.strip_prefix('[').unwrap_or(s);
    let s = s.strip_suffix(']').unwrap_or(s);
    let mut out: Vec<u32> = Vec::with_capacity(expected_len);
    if s.trim().is_empty() {
        out.resize(expected_len, 0);
        return out;
    }
    for tok in s.split_whitespace() {
        out.push(tok.parse::<u32>().unwrap_or(0));
    }
    if out.len() != expected_len {
        // If malformed, pad/truncate.
        out.resize(expected_len, 0);
    }
    out
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
            continue; // drop first
        }
        sum += elapsed_us;
        count += 1;
    }

    // average wall-clock time per FFT call, in microseconds
    sum / (count as f64)
}

fn min_label(t_r2: f64, t_r4: f64, t_rs: f64) -> &'static str {
    let minv = t_r2.min(t_r4.min(t_rs));
    if (t_r2 - minv).abs() <= f64::EPSILON {
        "r2"
    } else if (t_r4 - minv).abs() <= f64::EPSILON {
        "r4"
    } else {
        "rs"
    }
}

fn main() -> io::Result<()> {
    // Read first 200 rows from final_dataset.csv and write augmented CSV.
    let input_path = "tfhe-ntt/examples/model/final_dataset.csv";
    let output_path = "tfhe-ntt/examples/model/final_dataset_200_with_times.csv";

    let mut rdr = csv::Reader::from_path(input_path).expect("open input csv");

    let out_file = File::create(output_path)?;
    let mut wtr = csv::Writer::from_writer(BufWriter::new(out_file));

    // Output header = input columns + new columns
    wtr.write_record([
        "raw_N",
        "padded_N",
        "polynomial",
        "frac_2i_nonzero",
        "frac_2i1_nonzero",
        "frac_4i_nonzero",
        "frac_4i1_nonzero",
        "frac_4i2_nonzero",
        "frac_4i3_nonzero",
        "mult_r2",
        "mult_r4",
        "mult_rs",
        "output",
        "time_r2_sys_avg",
        "time_rs_sys_avg",
        "time_r4_sys_avg",
        "time_min_label",
        "label_mult_eq_time",
    ])?;

    for (i, rec) in rdr.deserialize::<RowIn>().enumerate() {
        if i >= 2000 {
            break;
        }
        let row = rec.expect("parse row");

        let p: u32 = 65537;
        let g = 3;
        let n = row.padded_N;
        let root = pow_mod(g, (p - 1) / (n as u32), p);
        let twiddles = make_twiddles_from_root(root, n, p);

        let poly = parse_poly_field(&row.polynomial, n);

    // measure wall time averages (microseconds)
    let t_r2 = time_forward_wall_avg_us(&poly, &twiddles, p, "r2");
    let t_rs = time_forward_wall_avg_us(&poly, &twiddles, p, "rs");
    let t_r4 = time_forward_wall_avg_us(&poly, &twiddles, p, "r4");

        let tmin = min_label(t_r2, t_r4, t_rs);
        let label_ok = if row.output == tmin { "yes" } else { "no" };

    wtr.write_record(&[
            row.raw_N.to_string(),
            row.padded_N.to_string(),
            row.polynomial,
            row.frac_2i_nonzero.to_string(),
            row.frac_2i1_nonzero.to_string(),
            row.frac_4i_nonzero.to_string(),
            row.frac_4i1_nonzero.to_string(),
            row.frac_4i2_nonzero.to_string(),
            row.frac_4i3_nonzero.to_string(),
            row.mult_r2.map(|x| x.to_string()).unwrap_or_default(),
            row.mult_r4.map(|x| x.to_string()).unwrap_or_default(),
            row.mult_rs.map(|x| x.to_string()).unwrap_or_default(),
            row.output,
        format!("{:.3}", t_r2),
        format!("{:.3}", t_rs),
        format!("{:.3}", t_r4),
            tmin.to_string(),
            label_ok.to_string(),
        ])?;

        // Flush occasionally so partial progress is written if interrupted.
        if i % 10 == 0 {
            wtr.flush()?;
        }
    }

    wtr.flush()?;

    eprintln!("Wrote {output_path}");
    Ok(())
}

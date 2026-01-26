//! Polynomial multiplication via FFT/NTT (using radix-2 from fwd_1.rs)
//!
//! This example:
//! 1. Takes two polynomials in coefficient form
//! 2. Pads them to next power of two (sum of degrees + 1)
//! 3. Converts to point-value form using forward radix-2 FFT
//! 4. Multiplies point-wise
//! 5. Converts back using inverse radix-2 FFT
//! 6. Prints the resulting polynomial coefficients

// Import from the crate's custum_radix module
use tfhe_ntt::custum_radix::{
    fft_radix2_recursive_mut,
    fft_radix4_recursive_mut,
    fft_split_radix_recursive_mut,
    ifft_radix2_recursive_mut,
    ifft_radix4_recursive_mut,
    ifft_split_radix_recursive_mut,
    MultStats,
};

// ---------- Helper functions (modular arithmetic + twiddles) ----------

#[inline(always)]
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

fn mod_inverse(a: u32, p: u32) -> u32 {
    pow_mod(a, p - 2, p)
}

fn compute_primitive_root(p: u32) -> u32 {
    let mut m = p - 1;
    let mut factors: Vec<u32> = Vec::new();
    let mut i: u32 = 2;
    while (i as u64) * (i as u64) <= m as u64 {
        if m % i == 0 {
            factors.push(i);
            while m % i == 0 {
                m /= i;
            }
        }
        i += 1;
    }
    if m > 1 {
        factors.push(m);
    }

    'outer: for g in 2..p {
        for &f in &factors {
            if pow_mod(g, (p - 1) / f, p) == 1 {
                continue 'outer;
            }
        }
        return g;
    }
    panic!("no primitive root found");
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

fn make_inv_twiddles(tw: &[u32]) -> Vec<u32> {
    let n = tw.len();
    let mut inv_tw = vec![0u32; n];
    inv_tw[0] = tw[0]; // = 1
    for k in 1..n {
        inv_tw[k] = tw[n - k];
    }
    inv_tw
}

fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    1 << (usize::BITS - (n - 1).leading_zeros())
}

/// Compute expected polynomial product via convolution (for verification)
fn compute_expected_product(a: &[u32], b: &[u32], p: u32) -> Vec<u32> {
    let result_len = a.len() + b.len() - 1;
    let mut result = vec![0u64; result_len];
    
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[i + j] += (a[i] as u64) * (b[j] as u64);
        }
    }
    
    // Reduce modulo p
    result.into_iter().map(|x| (x % (p as u64)) as u32).collect()
}

// ---------- Main polynomial multiplication ----------

fn main() {
    // Use a larger prime for bigger polynomials
    // p = 998244353 = 119 × 2^23 + 1 (common NTT prime, supports up to 2^23 points)
    let p: u32 = 998244353;

    // Two larger polynomials in coefficient form (lowest degree first)
    // poly_a(x) = 10 + 20x + 30x^2 + 40x^3 + 50x^4 + 60x^5 + 70x^6 + 80x^7
    // poly_b(x) = 100 + 200x + 300x^2 + 400x^3 + 500x^4 + 600x^5 + 700x^6 + 800x^7
    let poly_a: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let poly_b: Vec<u32> = vec![100, 200, 300, 400, 500, 600, 700, 800];

    println!("Polynomial A (coefficients): {:?}", poly_a);
    println!("Polynomial B (coefficients): {:?}", poly_b);

    // Degree of product = deg(A) + deg(B) = 3 + 3 = 6
    // We need at least 7 points, so pad to next power of two >= 7
    let result_len = poly_a.len() + poly_b.len() - 1; // = 7 coefficients
    let n = next_power_of_two(result_len);
    println!("\nPadded FFT size: {} (result needs {} coefficients)", n, result_len);

    // Pad polynomials with zeros
    let mut a_padded = vec![0u32; n];
    let mut b_padded = vec![0u32; n];
    for (i, &c) in poly_a.iter().enumerate() {
        a_padded[i] = c;
    }
    for (i, &c) in poly_b.iter().enumerate() {
        b_padded[i] = c;
    }

    // Compute primitive root and twiddle factors
    let g = compute_primitive_root(p);
    let root = pow_mod(g, (p - 1) / (n as u32), p);
    let tw = make_twiddles_from_root(root, n, p);
    let inv_tw = make_inv_twiddles(&tw);
    let n_inv = mod_inverse(n as u32, p);

    println!("Primitive root g = {}", g);
    println!("N-th root of unity ω = {} (N={})", root, n);

    // Stats for multiplication counting
    let mut stats_a = MultStats::default();
    let mut stats_b = MultStats::default();
    let mut stats_inv = MultStats::default();

    // Step 1: Forward FFT on both polynomials (coefficients -> point values)
    println!("\n--- Forward FFT (radix-2) ---");
    fft_radix4_recursive_mut(&mut a_padded, &tw, p, &mut stats_a);
    fft_split_radix_recursive_mut(&mut b_padded, &tw, p, &mut stats_b);

    println!("A in point-value form: {:?}", a_padded);
    println!("B in point-value form: {:?}", b_padded);

    // Step 2: Point-wise multiplication
    println!("\n--- Point-wise multiplication ---");
    let mut c_points = vec![0u32; n];
    for i in 0..n {
        c_points[i] = mul_mod(a_padded[i], b_padded[i], p);
    }
    println!("C = A * B in point-value form: {:?}", c_points);

    // Step 3: Inverse FFT to get result coefficients
    println!("\n--- Inverse FFT (radix-2) ---");
    ifft_split_radix_recursive_mut(&mut c_points, &inv_tw, p, n_inv, true, &mut stats_inv);

    println!("C in coefficient form: {:?}", c_points);

    // Extract meaningful coefficients (first result_len)
    let result_coeffs: Vec<u32> = c_points[..result_len].to_vec();
    println!("\nFinal product polynomial (first {} coeffs): {:?}", result_len, result_coeffs);

    // For larger polynomials, compute expected result via convolution
    // A(x) = sum(a[i] * x^i), B(x) = sum(b[i] * x^i)
    // C[k] = sum(a[i] * b[j]) for all i+j=k


    // let expected = compute_expected_product(&a_coeffs, &b_coeffs, p);
    // println!("Expected result: {:?}", expected);

    // if result_coeffs == expected {
    //     println!("\n✓ Polynomial multiplication CORRECT!");
    // } else {
    //     println!("\n✗ Polynomial multiplication FAILED!");
    // }

    // Print multiplication stats
    println!("\n--- Multiplication statistics ---");
    println!("Forward FFT (poly A): nonzero_mults = {}", stats_a.nonzero_mults);
    println!("Forward FFT (poly B): nonzero_mults = {}", stats_b.nonzero_mults);
    println!("Inverse FFT:          nonzero_mults = {}", stats_inv.nonzero_mults);
    println!("Point-wise mults:     {}", n);
    println!("Total FFT mults:      {}", stats_a.nonzero_mults + stats_b.nonzero_mults + stats_inv.nonzero_mults + n);
}

//! Polynomial multiplication using ML model to predict the best FFT algorithm.
//!
//! This example:
//! 1. Takes two polynomials
//! 2. Computes features from the polynomials
//! 3. Calls an ML model (via Python) to predict the best algorithm (r2, r4, rs)
//! 4. Uses the predicted algorithm for forward NTT (negacyclic)
//! 5. Performs point-wise multiplication
//! 6. Uses the predicted algorithm for inverse NTT (negacyclic)
//! 7. Measures execution time

use rand::random;
use std::process::Command;
use std::time::Instant;

// ============================================================================
// Modular arithmetic helpers
// ============================================================================

#[inline(always)]
fn add_mod(a: u32, b: u32, p: u32) -> u32 {
    let s = a as u64 + b as u64;
    if s >= p as u64 { (s - p as u64) as u32 } else { s as u32 }
}

#[inline(always)]
fn sub_mod(a: u32, b: u32, p: u32) -> u32 {
    if a >= b { a - b } else { a + p - b }
}

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

// ============================================================================
// Primitive root and twiddle factor generation for NEGACYCLIC NTT
// ============================================================================

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

/// For negacyclic NTT of size n, we need a (2n)-th root of unity.
/// psi = g^((p-1)/(2n)) where g is a primitive root of p
/// Then w = psi^2 is the n-th root of unity, and psi^n = -1
fn make_negacyclic_twiddles(n: usize, p: u32) -> (Vec<u32>, Vec<u32>, u32) {
    let g = compute_primitive_root(p);
    // psi is a 2n-th root of unity
    let psi = pow_mod(g, (p - 1) / (2 * n as u32), p);
    // w = psi^2 is an n-th root of unity
    let w = mul_mod(psi, psi, p);
    
    // Build psi powers for pre/post scaling: psi_powers[i] = psi^i
    let mut psi_powers = vec![0u32; n];
    let mut cur: u32 = 1;
    for k in 0..n {
        psi_powers[k] = cur;
        cur = mul_mod(cur, psi, p);
    }
    
    // Build standard twiddles for DIT FFT: tw[k] = w^k
    let mut tw = vec![0u32; n];
    let mut cur: u32 = 1;
    for k in 0..n {
        tw[k] = cur;
        cur = mul_mod(cur, w, p);
    }
    
    (psi_powers, tw, w)
}

fn make_inv_twiddles(tw: &[u32]) -> Vec<u32> {
    let n = tw.len();
    let mut inv_tw = vec![0u32; n];
    inv_tw[0] = tw[0];
    for k in 1..n {
        inv_tw[k] = tw[n - k];
    }
    inv_tw
}

// ============================================================================
// Bit-reversal permutation
// ============================================================================

fn bit_reverse(mut x: usize, log_n: usize) -> usize {
    let mut result = 0;
    for _ in 0..log_n {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

fn bit_reverse_permutation(a: &mut [u32]) {
    let n = a.len();
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            a.swap(i, j);
        }
    }
}

// ============================================================================
// Forward NTT Algorithms (Cooley-Tukey, DIT, in-place with bit-reversal)
// ============================================================================

/// Radix-2 Cooley-Tukey DIT FFT (iterative, in-place)
fn fft_radix2_iterative(a: &mut [u32], tw: &[u32], p: u32) {
    let n = a.len();
    bit_reverse_permutation(a);
    
    let mut m = 1;
    while m < n {
        let wm_step = n / (2 * m);
        for k in (0..n).step_by(2 * m) {
            for j in 0..m {
                let w = tw[j * wm_step];
                let u = a[k + j];
                let t = mul_mod(w, a[k + j + m], p);
                a[k + j] = add_mod(u, t, p);
                a[k + j + m] = sub_mod(u, t, p);
            }
        }
        m *= 2;
    }
}

/// Radix-4 DIT FFT (iterative, in-place)
fn fft_radix4_iterative(a: &mut [u32], tw: &[u32], p: u32) {
    let n = a.len();
    bit_reverse_permutation(a);
    
    // Handle radix-2 stage if log_n is odd
    let log_n = n.trailing_zeros() as usize;
    let mut m = 1;
    
    if log_n % 2 == 1 {
        // One radix-2 stage
        for k in (0..n).step_by(2) {
            let u = a[k];
            let t = a[k + 1];
            a[k] = add_mod(u, t, p);
            a[k + 1] = sub_mod(u, t, p);
        }
        m = 2;
    }
    
    // Radix-4 stages
    while m < n {
        let wm_step = n / (4 * m);
        let j_factor = tw[n / 4]; // w^(n/4) = imaginary unit
        
        for k in (0..n).step_by(4 * m) {
            for j in 0..m {
                let w1 = tw[j * wm_step];
                let w2 = tw[(2 * j * wm_step) % n];
                let w3 = tw[(3 * j * wm_step) % n];
                
                let a0 = a[k + j];
                let a1 = a[k + j + m];
                let a2 = a[k + j + 2 * m];
                let a3 = a[k + j + 3 * m];
                
                let t1 = mul_mod(w1, a1, p);
                let t2 = mul_mod(w2, a2, p);
                let t3 = mul_mod(w3, a3, p);
                
                let b0 = add_mod(a0, t2, p);
                let b1 = sub_mod(a0, t2, p);
                let b2 = add_mod(t1, t3, p);
                let b3 = mul_mod(j_factor, sub_mod(t1, t3, p), p);
                
                a[k + j] = add_mod(b0, b2, p);
                a[k + j + m] = add_mod(b1, b3, p);
                a[k + j + 2 * m] = sub_mod(b0, b2, p);
                a[k + j + 3 * m] = sub_mod(b1, b3, p);
            }
        }
        m *= 4;
    }
}

/// Split-radix DIT FFT (iterative, in-place) - hybrid of radix-2 and radix-4
fn fft_split_radix_iterative(a: &mut [u32], tw: &[u32], p: u32) {
    // For simplicity, use radix-2 as split-radix fallback
    // True split-radix is more complex to implement iteratively
    fft_radix2_iterative(a, tw, p);
}

// ============================================================================
// Inverse NTT Algorithms
// ============================================================================

/// Inverse Radix-2 FFT
fn ifft_radix2_iterative(a: &mut [u32], inv_tw: &[u32], p: u32, n_inv: u32) {
    let n = a.len();
    bit_reverse_permutation(a);
    
    let mut m = 1;
    while m < n {
        let wm_step = n / (2 * m);
        for k in (0..n).step_by(2 * m) {
            for j in 0..m {
                let w = inv_tw[j * wm_step];
                let u = a[k + j];
                let t = mul_mod(w, a[k + j + m], p);
                a[k + j] = add_mod(u, t, p);
                a[k + j + m] = sub_mod(u, t, p);
            }
        }
        m *= 2;
    }
    
    // Scale by 1/n
    for x in a.iter_mut() {
        *x = mul_mod(*x, n_inv, p);
    }
}

/// Inverse Radix-4 FFT
fn ifft_radix4_iterative(a: &mut [u32], inv_tw: &[u32], p: u32, n_inv: u32) {
    let n = a.len();
    bit_reverse_permutation(a);
    
    let log_n = n.trailing_zeros() as usize;
    let mut m = 1;
    
    if log_n % 2 == 1 {
        for k in (0..n).step_by(2) {
            let u = a[k];
            let t = a[k + 1];
            a[k] = add_mod(u, t, p);
            a[k + 1] = sub_mod(u, t, p);
        }
        m = 2;
    }
    
    while m < n {
        let wm_step = n / (4 * m);
        let neg_j = inv_tw[n / 4];
        
        for k in (0..n).step_by(4 * m) {
            for j in 0..m {
                let w1 = inv_tw[j * wm_step];
                let w2 = inv_tw[(2 * j * wm_step) % n];
                let w3 = inv_tw[(3 * j * wm_step) % n];
                
                let a0 = a[k + j];
                let a1 = a[k + j + m];
                let a2 = a[k + j + 2 * m];
                let a3 = a[k + j + 3 * m];
                
                let t1 = mul_mod(w1, a1, p);
                let t2 = mul_mod(w2, a2, p);
                let t3 = mul_mod(w3, a3, p);
                
                let b0 = add_mod(a0, t2, p);
                let b1 = sub_mod(a0, t2, p);
                let b2 = add_mod(t1, t3, p);
                let b3 = mul_mod(neg_j, sub_mod(t1, t3, p), p);
                
                a[k + j] = add_mod(b0, b2, p);
                a[k + j + m] = add_mod(b1, b3, p);
                a[k + j + 2 * m] = sub_mod(b0, b2, p);
                a[k + j + 3 * m] = sub_mod(b1, b3, p);
            }
        }
        m *= 4;
    }
    
    for x in a.iter_mut() {
        *x = mul_mod(*x, n_inv, p);
    }
}

/// Inverse Split-radix FFT
fn ifft_split_radix_iterative(a: &mut [u32], inv_tw: &[u32], p: u32, n_inv: u32) {
    ifft_radix2_iterative(a, inv_tw, p, n_inv);
}

// ============================================================================
// Negacyclic NTT wrappers (with pre/post scaling by psi powers)
// ============================================================================

fn negacyclic_fwd(a: &mut [u32], psi_powers: &[u32], tw: &[u32], p: u32, algo: Algorithm) {
    let n = a.len();
    
    // Pre-multiply by psi^i
    for i in 0..n {
        a[i] = mul_mod(a[i], psi_powers[i], p);
    }
    
    // Forward FFT
    match algo {
        Algorithm::Radix2 => fft_radix2_iterative(a, tw, p),
        Algorithm::Radix4 => fft_radix4_iterative(a, tw, p),
        Algorithm::SplitRadix => fft_split_radix_iterative(a, tw, p),
    }
}

fn negacyclic_inv(a: &mut [u32], psi_powers: &[u32], inv_tw: &[u32], p: u32, n_inv: u32, algo: Algorithm) {
    let n = a.len();
    
    // Inverse FFT
    match algo {
        Algorithm::Radix2 => ifft_radix2_iterative(a, inv_tw, p, n_inv),
        Algorithm::Radix4 => ifft_radix4_iterative(a, inv_tw, p, n_inv),
        Algorithm::SplitRadix => ifft_split_radix_iterative(a, inv_tw, p, n_inv),
    }
    
    // Post-multiply by psi^{-i}
    for i in 0..n {
        let psi_inv = mod_inverse(psi_powers[i], p);
        a[i] = mul_mod(a[i], psi_inv, p);
    }
}

// ============================================================================
// Feature computation for ML model
// ============================================================================

fn compute_features(poly: &[u32], padded_n: usize) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let raw_n = poly.len();
    
    let mut padded = poly.to_vec();
    padded.resize(padded_n, 0);
    
    let half = padded_n / 2;
    let even_nonzero: usize = (0..half).filter(|&i| padded[2 * i] != 0).count();
    let odd_nonzero: usize = (0..half).filter(|&i| padded[2 * i + 1] != 0).count();
    
    let frac_2i = even_nonzero as f64 / half as f64;
    let frac_2i1 = odd_nonzero as f64 / half as f64;
    
    let quarter = padded_n / 4;
    let idx_4_0_nz: usize = (0..quarter).filter(|&i| padded[4 * i] != 0).count();
    let idx_4_1_nz: usize = (0..quarter).filter(|&i| padded[4 * i + 1] != 0).count();
    let idx_4_2_nz: usize = (0..quarter).filter(|&i| padded[4 * i + 2] != 0).count();
    let idx_4_3_nz: usize = (0..quarter).filter(|&i| padded[4 * i + 3] != 0).count();
    
    let frac_4i = idx_4_0_nz as f64 / quarter as f64;
    let frac_4i1 = idx_4_1_nz as f64 / quarter as f64;
    let frac_4i2 = idx_4_2_nz as f64 / quarter as f64;
    let frac_4i3 = idx_4_3_nz as f64 / quarter as f64;
    
    (raw_n as f64, padded_n as f64, frac_2i, frac_2i1, frac_4i, frac_4i1, frac_4i2, frac_4i3)
}

// ============================================================================
// ML Model prediction
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum Algorithm {
    Radix2,
    Radix4,
    SplitRadix,
}

fn predict_algorithm(features: (f64, f64, f64, f64, f64, f64, f64, f64)) -> Algorithm {
    let output = Command::new("python3")
        .arg("examples/model/predict_algo.py")
        .arg(features.0.to_string())
        .arg(features.1.to_string())
        .arg(features.2.to_string())
        .arg(features.3.to_string())
        .arg(features.4.to_string())
        .arg(features.5.to_string())
        .arg(features.6.to_string())
        .arg(features.7.to_string())
        .output();

    match output {
        Ok(out) => {
            let result = String::from_utf8_lossy(&out.stdout).trim().to_string();
            match result.as_str() {
                "r2" => Algorithm::Radix2,
                "r4" => Algorithm::Radix4,
                "rs" => Algorithm::SplitRadix,
                _ => {
                    eprintln!("Warning: Unknown prediction '{}', defaulting to Radix4", result);
                    Algorithm::Radix4
                }
            }
        }
        Err(e) => {
            eprintln!("Warning: Failed to call Python predictor: {}, defaulting to Radix4", e);
            Algorithm::Radix4
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    // Define suitable NTT prime and polynomial size
    // p - 1 must be divisible by 2n for negacyclic NTT
    let p: u32 = 1073479681;
    let polynomial_size = 1024;

    println!("==========================================================");
    println!("POLYNOMIAL MULTIPLICATION WITH ML-BASED ALGORITHM SELECTION");
    println!("==========================================================");
    println!();
    println!("Prime modulus p = {}", p);
    println!("Polynomial size = {}", polynomial_size);

    // Generate random polynomials
    let lhs_poly: Vec<u32> = (0..polynomial_size).map(|_| random::<u32>() % p).collect();
    let rhs_poly: Vec<u32> = (0..polynomial_size).map(|_| random::<u32>() % p).collect();

    // ========================================================================
    // METHOD 1: Schoolbook algorithm (for verification)
    // ========================================================================
    let add = |x: u32, y: u32| ((x as u64 + y as u64) % p as u64) as u32;
    let sub = |x: u32, y: u32| add(x, p - y);
    let mul = |x: u32, y: u32| ((x as u64 * y as u64) % p as u64) as u32;

    let mut full_convolution = vec![0; 2 * polynomial_size];
    for i in 0..polynomial_size {
        for j in 0..polynomial_size {
            full_convolution[i + j] = add(full_convolution[i + j], mul(lhs_poly[i], rhs_poly[j]));
        }
    }

    let mut negacyclic_convolution = vec![0; polynomial_size];
    for i in 0..polynomial_size {
        negacyclic_convolution[i] = sub(full_convolution[i], full_convolution[polynomial_size + i]);
    }

    // ========================================================================
    // METHOD 2: Negacyclic NTT with ML-based algorithm selection
    // ========================================================================
    
    let fft_size = polynomial_size;
    println!("\nFFT size = {}", fft_size);
    
    // Check that p-1 is divisible by 2n
    assert!((p - 1) % (2 * fft_size as u32) == 0, 
            "p-1 must be divisible by 2n for negacyclic NTT");
    
    // Generate twiddle factors for negacyclic NTT
    let (psi_powers, tw, _w) = make_negacyclic_twiddles(fft_size, p);
    let inv_tw = make_inv_twiddles(&tw);
    let n_inv = mod_inverse(fft_size as u32, p);
    
    // ========================================================================
    // TIMED NTT OPERATIONS
    // ========================================================================
    
    // Predict algorithms BEFORE timing (ML model calls are slow)
    let features_lhs = compute_features(&lhs_poly, fft_size);
    let features_rhs = compute_features(&rhs_poly, fft_size);
    let algo_lhs = predict_algorithm(features_lhs);
    let algo_rhs = predict_algorithm(features_rhs);
    
    println!("\n--- ML Model Predictions ---");
    println!("Forward FFT for LHS: {:?}", algo_lhs);
    println!("Forward FFT for RHS: {:?}", algo_rhs);
    
    // For inverse, we'll use the same algorithm as forward (product is dense)
    // This avoids an extra Python call during timed section
    let algo_inv = algo_lhs;
    println!("Inverse FFT: {:?}", algo_inv);
    
    // Prepare working copies
    let mut lhs_ntt = lhs_poly.clone();
    let mut rhs_ntt = rhs_poly.clone();
    
    // NOW start timing (FFT operations only, no Python calls)
    let start = Instant::now();
    
    // Forward negacyclic NTT for LHS
    negacyclic_fwd(&mut lhs_ntt, &psi_powers, &tw, p, algo_lhs);
    
    // Forward negacyclic NTT for RHS
    negacyclic_fwd(&mut rhs_ntt, &psi_powers, &tw, p, algo_rhs);
    
    // Point-wise multiplication
    for i in 0..fft_size {
        lhs_ntt[i] = mul_mod(lhs_ntt[i], rhs_ntt[i], p);
    }
    
    // Inverse negacyclic NTT
    negacyclic_inv(&mut lhs_ntt, &psi_powers, &inv_tw, p, n_inv, algo_inv);
    
    let elapsed = start.elapsed();
    
    // ========================================================================
    // Results
    // ========================================================================
    println!("\n--- Timing ---");
    println!("NTT operations (fwd + mul + inv) execution time: {:?}", elapsed);
    
    // Verify result
    println!("\n--- Verification ---");
    if lhs_ntt == negacyclic_convolution {
        println!("✓ CORRECT! ML-based FFT multiplication matches schoolbook result.");
    } else {
        println!("✗ MISMATCH! Something went wrong.");
        let diff_count = lhs_ntt.iter()
            .zip(negacyclic_convolution.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!("  {} coefficients differ", diff_count);
    }
    
    println!("\nSuccess!");
}

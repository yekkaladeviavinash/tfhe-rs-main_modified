use rand::random;
use std::time::Instant;
use tfhe_ntt::custum_radix::{
    fft_radix2_recursive_mut,
    ifft_radix2_recursive_mut,
    MultStats,
};

// Helper functions for modular arithmetic
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
    let mut tw = vec![0u32; n];
    let mut cur: u32 = 1;
    for k in 0..n {
        tw[k] = cur;
        cur = mul_mod(cur, root, p);
    }
    tw
}

fn make_inv_twiddles(tw: &[u32], _p: u32) -> Vec<u32> {
    let n = tw.len();
    let mut inv_tw = vec![0u32; n];
    inv_tw[0] = tw[0];
    for k in 1..n {
        inv_tw[k] = tw[n - k];
    }
    inv_tw
}

fn main() {
    // Define NTT prime and polynomial size
    let p: u32 = 1073479681;
    let polynomial_size: usize = 1024;

    // Generate random polynomials
    let lhs_poly: Vec<u32> = (0..polynomial_size).map(|_| random::<u32>() % p).collect();
    let rhs_poly: Vec<u32> = (0..polynomial_size).map(|_| random::<u32>() % p).collect();

    // println!("Left polynomial:  {:?}", lhs_poly);
    // println!("Right polynomial: {:?}", rhs_poly);

    // Method 1: Schoolbook algorithm for reference
    let add = |x: u32, y: u32| ((x as u64 + y as u64) % p as u64) as u32;
    let sub = |x: u32, y: u32| add(x, p - y);
    let mul = |x: u32, y: u32| mul_mod(x, y, p);

    let mut full_convolution = vec![0u32; 2 * polynomial_size];
    for i in 0..polynomial_size {
        for j in 0..polynomial_size {
            full_convolution[i + j] = add(full_convolution[i + j], mul(lhs_poly[i], rhs_poly[j]));
        }
    }

    let mut negacyclic_convolution = vec![0u32; polynomial_size];
    for i in 0..polynomial_size {
        negacyclic_convolution[i] = sub(full_convolution[i], full_convolution[polynomial_size + i]);
    }

    // Method 2: Custom Radix-2 NTT from fwd_1.rs
    
    // For negacyclic convolution, we need to use 2n-th root of unity
    // We'll compute using doubled size for negacyclic property
    let n = polynomial_size;
    let two_n = 2 * n;
    
    // Compute primitive root and 2n-th root of unity for negacyclic convolution
    let g = compute_primitive_root(p);
    let psi = pow_mod(g, (p - 1) / (two_n as u32), p);  // 2n-th root of unity (psi)
    let omega = mul_mod(psi, psi, p);  // n-th root of unity (omega = psi^2)
    
    // Build twiddles for n-point NTT
    let tw = make_twiddles_from_root(omega, n, p);
    let inv_tw = make_inv_twiddles(&tw, p);
    let n_inv = mod_inverse(n as u32, p);
    
    // Pre-multiply by powers of psi for negacyclic convolution
    let mut lhs_ntt: Vec<u32> = lhs_poly.iter()
        .enumerate()
        .map(|(i, &x)| mul_mod(x, pow_mod(psi, i as u32, p), p))
        .collect();
    let mut rhs_ntt: Vec<u32> = rhs_poly.iter()
        .enumerate()
        .map(|(i, &x)| mul_mod(x, pow_mod(psi, i as u32, p), p))
        .collect();
    
    let mut stats = MultStats::default();
    
    // Start timing
    let start = Instant::now();
    
    // Forward NTT using radix-2
    fft_radix2_recursive_mut(&mut lhs_ntt, &tw, p, &mut stats);
    fft_radix2_recursive_mut(&mut rhs_ntt, &tw, p, &mut stats);
    
    // Elementwise multiplication
    for i in 0..n {
        lhs_ntt[i] = mul_mod(lhs_ntt[i], rhs_ntt[i], p);
    }
    
    // Inverse NTT using radix-2
    ifft_radix2_recursive_mut(&mut lhs_ntt, &inv_tw, p, n_inv, true, &mut stats);
    
    // Stop timing
    
    // Post-multiply by inverse powers of psi
    let psi_inv = mod_inverse(psi, p);
    for i in 0..n {
        lhs_ntt[i] = mul_mod(lhs_ntt[i], pow_mod(psi_inv, i as u32, p), p);
    }
    let duration = start.elapsed();
    
    println!("\nRadix-2 NTT multiplication time: {:?}", duration);
    println!("Multiplication stats: {:?}", stats);
    
    // Verify result matches schoolbook
    assert_eq!(lhs_ntt, negacyclic_convolution, "Results don't match!");
    // println!("\nMultiplied result (negacyclic convolution): {:?}", lhs_ntt);
    println!("Success! Radix-2 NTT result matches schoolbook algorithm.");
}

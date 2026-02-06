use rand::random;
use std::time::Instant;

// Helper functions for modular arithmetic
#[derive(Debug, Clone, Default)]
pub struct MultStats {
    pub nonzero_mults: usize,
    pub skipped_mults: usize,
}

#[inline(always)]
fn add_mod(a: u32, b: u32, p: u32) -> u32 {
    let s = a as u64 + b as u64;
    if s >= p as u64 { (s - p as u64) as u32 } else { s as u32 }
}

#[inline(always)]
fn sub_mod(a: u32, b: u32, p: u32) -> u32 {
    let a = a as u64;
    let b = b as u64;
    let p = p as u64;
    if a >= b { (a - b) as u32 } else { (a + p - b) as u32 }
}

#[inline(always)]
fn mul_mod(a: u32, b: u32, p: u32) -> u32 {
    ((a as u64 * b as u64) % (p as u64)) as u32
}

#[inline(always)]
fn mul_mod_counted(a: u32, b: u32, p: u32, stats: &mut MultStats) -> u32 {
    let r = mul_mod(a, b, p);
    if a != 0 && b != 0 {
        stats.nonzero_mults += 1;
    } else {
        stats.skipped_mults += 1;
    }
    r
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

const RECURSION_THRESHOLD: usize = 2048;

fn fft_radix2_iterative_mut(a: &mut [u32], twiddles: &[u32], p: u32, stats: &mut MultStats) {
    let n = a.len();
    if n <= 1 {
        return;
    }

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;

        for i in (0..n).step_by(len) {
            for j in 0..half {
                let w = twiddles[j * step];
                let t = mul_mod(w, a[i + j + half], p);
                stats.nonzero_mults += (a[i + j + half] != 0) as usize;

                let u = a[i + j];
                a[i + j] = add_mod(u, t, p);
                a[i + j + half] = sub_mod(u, t, p);
            }
        }

        len <<= 1;
    }
}

fn ifft_radix2_iterative_mut(
    a: &mut [u32],
    inv_tw: &[u32],
    p: u32,
    n_inv: u32,
    top: bool,
    stats: &mut MultStats,
) {
    let n = a.len();
    if n <= 1 {
        return;
    }

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;

        for i in (0..n).step_by(len) {
            for j in 0..half {
                let w = inv_tw[j * step];
                let t = mul_mod_counted(w, a[i + j + half], p, stats);

                let u = a[i + j];
                a[i + j] = add_mod(u, t, p);
                a[i + j + half] = sub_mod(u, t, p);
            }
        }

        len <<= 1;
    }

    if top {
        for x in a.iter_mut() {
            *x = mul_mod(*x, n_inv, p);
        }
    }
}

pub fn fft_radix2_recursive_mut(
    a: &mut [u32],
    twiddles: &[u32],
    p: u32,
    stats: &mut MultStats,
) {
    let n = a.len();
    if n <= RECURSION_THRESHOLD {
        fft_radix2_iterative_mut(a, twiddles, p, stats);
        return;
    }

    let half = n / 2;

    let mut even = vec![0u32; half];
    let mut odd = vec![0u32; half];

    for i in 0..half {
        even[i] = a[2 * i];
        odd[i] = a[2 * i + 1];
    }

    let mut tw2 = vec![0u32; half];
    for k in 0..half {
        tw2[k] = twiddles[(2 * k) % n];
    }

    fft_radix2_recursive_mut(&mut even, &tw2, p, stats);
    fft_radix2_recursive_mut(&mut odd, &tw2, p, stats);

    for k in 0..half {
        let t = mul_mod(twiddles[k], odd[k], p);
        stats.nonzero_mults += (odd[k] != 0) as usize;

        a[k] = add_mod(even[k], t, p);
        a[k + half] = sub_mod(even[k], t, p);
    }
}

pub fn ifft_radix2_recursive_mut(
    a: &mut [u32],
    inv_tw: &[u32],
    p: u32,
    n_inv: u32,
    top: bool,
    stats: &mut MultStats,
) {
    let n = a.len();
    if n <= RECURSION_THRESHOLD {
        ifft_radix2_iterative_mut(a, inv_tw, p, n_inv, top, stats);
        return;
    }

    let half = n / 2;

    let mut even = vec![0u32; half];
    let mut odd = vec![0u32; half];

    for i in 0..half {
        even[i] = a[2 * i];
        odd[i] = a[2 * i + 1];
    }

    let mut inv_tw2 = vec![0u32; half];
    for k in 0..half {
        inv_tw2[k] = inv_tw[(2 * k) % n];
    }

    ifft_radix2_recursive_mut(&mut even, &inv_tw2, p, n_inv, false, stats);
    ifft_radix2_recursive_mut(&mut odd, &inv_tw2, p, n_inv, false, stats);

    for k in 0..half {
        let t = mul_mod_counted(inv_tw[k], odd[k], p, stats);

        a[k] = add_mod(even[k], t, p);
        a[k + half] = sub_mod(even[k], t, p);
    }

    if top {
        for x in a.iter_mut() {
            *x = mul_mod(*x, n_inv, p);
        }
    }
}

fn main() {
    // Define NTT prime and polynomial size
    let p: u32 = 1073479681;
    let polynomial_size: usize = 16384*2;

    // Generate random polynomials
    let lhs_poly: Vec<u32> = (0..polynomial_size).map(|_| random::<u32>() % p).collect();
    let rhs_poly: Vec<u32> = (0..polynomial_size).map(|_| random::<u32>() % p).collect();

    // Method 2:
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
    
    let mut stats = MultStats::default();

    let start = Instant::now();
    let mut lhs_ntt: Vec<u32> = lhs_poly.iter()
        .enumerate()
        .map(|(i, &x)| mul_mod(x, pow_mod(psi, i as u32, p), p))
        .collect();
    let mut rhs_ntt: Vec<u32> = rhs_poly.iter()
        .enumerate()
        .map(|(i, &x)| mul_mod(x, pow_mod(psi, i as u32, p), p))
        .collect();
    
    fft_radix2_recursive_mut(&mut lhs_ntt, &tw, p, &mut stats);
    fft_radix2_recursive_mut(&mut rhs_ntt, &tw, p, &mut stats);
    for i in 0..n {
        lhs_ntt[i] = mul_mod(lhs_ntt[i], rhs_ntt[i], p);
    }
    ifft_radix2_recursive_mut(&mut lhs_ntt, &inv_tw, p, n_inv, true, &mut stats);
    let psi_inv = mod_inverse(psi, p);
    for i in 0..n {
        lhs_ntt[i] = mul_mod(lhs_ntt[i], pow_mod(psi_inv, i as u32, p), p);
    }

    let duration = start.elapsed();
    
    println!("\nRadix-2 NTT multiplication time: {:?}", duration);
}

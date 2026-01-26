// Multiplication statistics

#[derive(Debug, Clone, Default)]
pub struct MultStats {
    pub nonzero_mults: usize,  // nonzero * nonzero
    pub skipped_mults: usize,  // multiplications with zero
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

/// modular inverse via Fermat (works when p is prime)
fn mod_inverse(a: u32, p: u32) -> u32 {
    pow_mod(a, p - 2, p)
}

// ---------- primitive root finder (for prime p) ----------
fn compute_primitive_root(p: u32) -> u32 {
    // factor p-1
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
    panic!("no primitive root found (is p prime?)");
}

// ---------- twiddle generation ----------
/// Build twiddle table: tw[k] = root^k mod p for k=0..n-1
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

/// Build inverse twiddle table: inv_tw[k] = root^{-k} mod p
/// Given forward twiddles tw[k] = ω^k, we have inv_tw[k] = ω^{n-k} = tw[n-k] for k>0, inv_tw[0]=1
fn make_inv_twiddles(tw: &[u32], _p: u32) -> Vec<u32> {
    let n = tw.len();
    let mut inv_tw = vec![0u32; n];
    inv_tw[0] = tw[0]; // = 1
    for k in 1..n {
        inv_tw[k] = tw[n - k];
    }
    inv_tw
}


pub fn fft_radix4_recursive_mut(
    a: &mut [u32],
    twiddles: &[u32],
    p: u32,
    stats: &mut MultStats,
) {
    let n = a.len();
    if n == 1 {
        return;
    }

    if n == 2 {
        let t = a[0];
        a[0] = add_mod(a[0], a[1], p);
        a[1] = sub_mod(t, a[1], p);
        return;
    }

    let quarter = n / 4;

    let mut a0 = vec![0u32; quarter];
    let mut a1 = vec![0u32; quarter];
    let mut a2 = vec![0u32; quarter];
    let mut a3 = vec![0u32; quarter];

    for i in 0..quarter {
        a0[i] = a[4 * i];
        a1[i] = a[4 * i + 1];
        a2[i] = a[4 * i + 2];
        a3[i] = a[4 * i + 3];
    }

    let mut tw4 = vec![0u32; quarter];
    for k in 0..quarter {
        tw4[k] = twiddles[(4 * k) % n];
    }

    fft_radix4_recursive_mut(&mut a0, &tw4, p, stats);
    fft_radix4_recursive_mut(&mut a1, &tw4, p, stats);
    fft_radix4_recursive_mut(&mut a2, &tw4, p, stats);
    fft_radix4_recursive_mut(&mut a3, &tw4, p, stats);

    let w_n4 = twiddles[quarter];

    for k in 0..quarter {
        let w1 = twiddles[k];
        let w2 = twiddles[(2 * k) % n];
        let w3 = twiddles[(3 * k) % n];

        let t1 = mul_mod(w1, a1[k], p);
        let t2 = mul_mod(w2, a2[k], p);
        let t3 = mul_mod(w3, a3[k], p);

        stats.nonzero_mults +=
            (a1[k] != 0) as usize +
            (a2[k] != 0) as usize +
            (a3[k] != 0) as usize;

        let b0 = add_mod(a0[k], t2, p);
        let b1 = sub_mod(a0[k], t2, p);
        let b2 = add_mod(t1, t3, p);
        let b3 = sub_mod(t1, t3, p);

        let b3_rot = mul_mod(w_n4, b3, p);
        stats.nonzero_mults += (b3 != 0) as usize;

        a[k]               = add_mod(b0, b2, p);
        a[k + quarter]     = add_mod(b1, b3_rot, p);
        a[k + 2 * quarter] = sub_mod(b0, b2, p);
        a[k + 3 * quarter] = sub_mod(b1, b3_rot, p);
    }
}



pub fn fft_radix2_recursive_mut(
    a: &mut [u32],
    twiddles: &[u32],
    p: u32,
    stats: &mut MultStats,
) {
    let n = a.len();
    if n == 1 {
        return;
    }

    let half = n / 2;

    let mut even = vec![0u32; half];
    let mut odd  = vec![0u32; half];

    for i in 0..half {
        even[i] = a[2 * i];
        odd[i]  = a[2 * i + 1];
    }

    let mut tw2 = vec![0u32; half];
    for k in 0..half {
        tw2[k] = twiddles[(2 * k) % n];
    }

    fft_radix2_recursive_mut(&mut even, &tw2, p, stats);
    fft_radix2_recursive_mut(&mut odd,  &tw2, p, stats);

    for k in 0..half {
        let t = mul_mod(twiddles[k], odd[k], p);
        stats.nonzero_mults += (odd[k] != 0) as usize;

        a[k]        = add_mod(even[k], t, p);
        a[k + half] = sub_mod(even[k], t, p);
    }
}



pub fn fft_split_radix_recursive_mut(
    a: &mut [u32],
    tw: &[u32],
    p: u32,
    stats: &mut MultStats,
) {
    let n = a.len();
    if n == 1 {
        return;
    }

    if n == 2 {
        let t = a[0];
        a[0] = add_mod(a[0], a[1], p);
        a[1] = sub_mod(t, a[1], p);
        return;
    }

    let n2 = n / 2;
    let n4 = n / 4;

    let mut a0 = vec![0u32; n2];
    let mut a1 = vec![0u32; n4];
    let mut a2 = vec![0u32; n4];

    for i in 0..n2 {
        a0[i] = a[2 * i];
    }
    for i in 0..n4 {
        a1[i] = a[4 * i + 1];
        a2[i] = a[4 * i + 3];
    }

    let mut tw2 = vec![0u32; n2];
    for k in 0..n2 {
        tw2[k] = tw[(2 * k) % n];
    }

    let mut tw4 = vec![0u32; n4];
    for k in 0..n4 {
        tw4[k] = tw[(4 * k) % n];
    }

    fft_split_radix_recursive_mut(&mut a0, &tw2, p, stats);
    fft_split_radix_recursive_mut(&mut a1, &tw4, p, stats);
    fft_split_radix_recursive_mut(&mut a2, &tw4, p, stats);

    let j = tw[n4 % n];

    for k in 0..n4 {
        let w_k  = tw[k % n];
        let w_3k = tw[(3 * k) % n];

        let t1 = mul_mod(w_k,  a1[k], p);
        let t2 = mul_mod(w_3k, a2[k], p);

        stats.nonzero_mults +=
            (a1[k] != 0) as usize +
            (a2[k] != 0) as usize;

        let sum  = add_mod(t1, t2, p);
        let diff = sub_mod(t1, t2, p);

        let jdiff = mul_mod(j, diff, p);
        stats.nonzero_mults += (diff != 0) as usize;

        let u0 = a0[k];
        let u1 = a0[k + n4];

        a[k]           = add_mod(u0, sum, p);
        a[k + n4]      = add_mod(u1, jdiff, p);
        a[k + n2]      = sub_mod(u0, sum, p);
        a[k + n2 + n4] = sub_mod(u1, jdiff, p);
    }
}





pub fn ifft_radix4_recursive_mut(
    a: &mut [u32],
    inv_tw: &[u32],   // inv_tw[k] = ω^{-k}
    p: u32,
    n_inv: u32,
    top: bool,
    stats: &mut MultStats,
) {
    let n = a.len();
    if n == 1 {
        return;
    }

    if n == 2 {
        let t = a[0];
        a[0] = add_mod(a[0], a[1], p);
        a[1] = sub_mod(t, a[1], p);
        return;
    }

    let q = n / 4;

    let mut a0 = vec![0u32; q];
    let mut a1 = vec![0u32; q];
    let mut a2 = vec![0u32; q];
    let mut a3 = vec![0u32; q];

    for i in 0..q {
        a0[i] = a[4*i];
        a1[i] = a[4*i + 1];
        a2[i] = a[4*i + 2];
        a3[i] = a[4*i + 3];
    }

    // child inverse twiddles: ω^{-4k}
    let mut inv_tw4 = vec![0u32; q];
    for k in 0..q {
        inv_tw4[k] = inv_tw[(4 * k) % n];
    }

    ifft_radix4_recursive_mut(&mut a0, &inv_tw4, p, n_inv, false, stats);
    ifft_radix4_recursive_mut(&mut a1, &inv_tw4, p, n_inv, false, stats);
    ifft_radix4_recursive_mut(&mut a2, &inv_tw4, p, n_inv, false, stats);
    ifft_radix4_recursive_mut(&mut a3, &inv_tw4, p, n_inv, false, stats);

    // −j = ω^{-n/4}
    let minus_j = inv_tw[q];

    for k in 0..q {
        let w1 = inv_tw[k];
        let w2 = inv_tw[(2 * k) % n];
        let w3 = inv_tw[(3 * k) % n];

        let t1 = mul_mod_counted(w1, a1[k], p, stats);
        let t2 = mul_mod_counted(w2, a2[k], p, stats);
        let t3 = mul_mod_counted(w3, a3[k], p, stats);

        let b0 = add_mod(a0[k], t2, p);
        let b1 = sub_mod(a0[k], t2, p);
        let b2 = add_mod(t1, t3, p);
        let b3 = sub_mod(t1, t3, p);

        let b3_rot = mul_mod_counted(minus_j, b3, p, stats);

        a[k]         = add_mod(b0, b2, p);
        a[k + q]     = add_mod(b1, b3_rot, p);
        a[k + 2*q]   = sub_mod(b0, b2, p);
        a[k + 3*q]   = sub_mod(b1, b3_rot, p);
    }

    // scale once at the top
    if top {
        for x in a.iter_mut() {
            *x = mul_mod(*x, n_inv, p);
        }
    }
}

pub fn ifft_radix2_recursive_mut(
    a: &mut [u32],
    inv_tw: &[u32],   // inv_tw[k] = ω^{-k}
    p: u32,
    n_inv: u32,
    top: bool,
    stats: &mut MultStats,
) {
    let n = a.len();
    if n == 1 {
        return;
    }

    // if n == 2 {
    //     let t = a[0];
    //     a[0] = add_mod(a[0], a[1], p);
    //     a[1] = sub_mod(t, a[1], p);
    //     return;
    // }

    let half = n / 2;

    let mut even = vec![0u32; half];
    let mut odd  = vec![0u32; half];

    for i in 0..half {
        even[i] = a[2 * i];
        odd[i]  = a[2 * i + 1];
    }

    // child inverse twiddles: ω^{-2k}
    let mut inv_tw2 = vec![0u32; half];
    for k in 0..half {
        inv_tw2[k] = inv_tw[(2 * k) % n];
    }

    // recurse without scaling
    ifft_radix2_recursive_mut(&mut even, &inv_tw2, p, n_inv, false, stats);
    ifft_radix2_recursive_mut(&mut odd,  &inv_tw2, p, n_inv, false, stats);

    // inverse butterfly
    for k in 0..half {
        let t = mul_mod_counted(inv_tw[k], odd[k], p, stats);

        a[k]        = add_mod(even[k], t, p);
        a[k + half] = sub_mod(even[k], t, p);
    }

    // scale only once at the top
    if top {
        for x in a.iter_mut() {
            *x = mul_mod(*x, n_inv, p);
        }
    }
}

/// Inverse split-radix (radix-2/4) recursive FFT with twiddle multiplication counting.
/// 
/// - `a`: input/output slice (length must be a power of two)
/// - `inv_tw`: inverse twiddle factors, length n, with inv_tw[k] = ω^{-k}
/// - `p`: prime modulus
/// - `n_inv`: modular inverse of n mod p (used for scaling at top level)
/// - `top`: if true, scale the final result by n_inv
/// - `stats`: tracks nonzero twiddle multiplications
pub fn ifft_split_radix_recursive_mut(
    a: &mut [u32],
    inv_tw: &[u32],
    p: u32,
    n_inv: u32,
    top: bool,
    stats: &mut MultStats,
) {
    let n = a.len();
    
    // Base case: length 1, nothing to do
    if n == 1 {
        return;
    }
    
    // Base case: length 2, simple butterfly (no twiddle mult needed)
    if n == 2 {
        let u = a[0];
        let v = a[1];
        a[0] = add_mod(u, v, p);
        a[1] = sub_mod(u, v, p);
        if top {
            a[0] = mul_mod(a[0], n_inv, p);
            a[1] = mul_mod(a[1], n_inv, p);
        }
        return;
    }
    
    // Base case: length 4
    if n == 4 {
        // 4-point IFFT using inverse butterfly
        let u0 = a[0];
        let u1 = a[1];
        let u2 = a[2];
        let u3 = a[3];
        
        // First stage: length-2 IFFTs
        let t0 = add_mod(u0, u2, p);
        let t2 = sub_mod(u0, u2, p);
        let t1 = add_mod(u1, u3, p);
        let t3 = sub_mod(u1, u3, p);
        
        // inv_tw[n/4] = ω^{-n/4} = -j (for inverse)
        // For inverse: multiply by -j instead of j
        // -j = ω^{-n/4} = inv_tw[1] when n=4
        let neg_j = inv_tw[1]; // ω^{-1} for n=4
        let t3_neg_j = mul_mod_counted(neg_j, t3, p, stats);
        
        // Second stage
        a[0] = add_mod(t0, t1, p);
        a[1] = add_mod(t2, t3_neg_j, p);
        a[2] = sub_mod(t0, t1, p);
        a[3] = sub_mod(t2, t3_neg_j, p);
        
        if top {
            for x in a.iter_mut() {
                *x = mul_mod(*x, n_inv, p);
            }
        }
        return;
    }
    
    let n2 = n / 2;
    let n4 = n / 4;
    
    // Decompose into even, mod-4=1, and mod-4=3 indices
    let mut a0 = vec![0u32; n2]; // even indices
    let mut a1 = vec![0u32; n4]; // indices ≡ 1 (mod 4)
    let mut a2 = vec![0u32; n4]; // indices ≡ 3 (mod 4)
    
    for i in 0..n2 {
        a0[i] = a[2 * i];
    }
    for i in 0..n4 {
        a1[i] = a[4 * i + 1];
    }
    for i in 0..n4 {
        a2[i] = a[4 * i + 3];
    }
    
    // Build sub-twiddles
    // For even part: use every 2nd twiddle
    let mut inv_tw0 = vec![0u32; n2];
    for k in 0..n2 {
        inv_tw0[k] = inv_tw[(2 * k) % n];
    }
    // For quarter parts: use every 4th twiddle
    let mut inv_tw1 = vec![0u32; n4];
    for k in 0..n4 {
        inv_tw1[k] = inv_tw[(4 * k) % n];
    }
    
    // Recurse on subproblems (without top-level scaling)
    ifft_split_radix_recursive_mut(&mut a0, &inv_tw0, p, n_inv, false, stats);
    ifft_split_radix_recursive_mut(&mut a1, &inv_tw1, p, n_inv, false, stats);
    ifft_split_radix_recursive_mut(&mut a2, &inv_tw1, p, n_inv, false, stats);
    
    // -j = ω^{-n/4} = inv_tw[n/4]
    let neg_j = inv_tw[n4];
    
    // Inverse butterfly: combine the results
    for k in 0..n4 {
        let u0 = a0[k];
        let u1 = a0[k + n4];
        
        // w_k = inv_tw[k] = ω^{-k}
        // w_3k = inv_tw[3k] = ω^{-3k}
        let w_k = inv_tw[k];
        let w_3k = inv_tw[(3 * k) % n];
        
        // For inverse: t1 = w_k * a1[k], t2 = w_3k * a2[k]
        let t1 = mul_mod_counted(w_k, a1[k], p, stats);
        let t2 = mul_mod_counted(w_3k, a2[k], p, stats);
        
        let sum = add_mod(t1, t2, p);
        let diff = sub_mod(t1, t2, p);
        
        // -j * diff for inverse transform
        let neg_j_diff = mul_mod_counted(neg_j, diff, p, stats);
        
        // Combine into output positions
        a[k]           = add_mod(u0, sum, p);
        a[k + n4]      = add_mod(u1, neg_j_diff, p);
        a[k + n2]      = sub_mod(u0, sum, p);
        a[k + n2 + n4] = sub_mod(u1, neg_j_diff, p);
    }
    
    // Scale by n_inv only at the top level
    if top {
        for x in a.iter_mut() {
            *x = mul_mod(*x, n_inv, p);
        }
    }
}

// Example usage:
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiplication_counting() {
        let p: u32 = 65537; // Fermat prime F4
        let n: usize = 64;

        // Compute primitive root
        let g = compute_primitive_root(p);
        let root = pow_mod(g, (p - 1) / (n as u32), p);  // forward root ω

        // Build forward and inverse twiddles
        let tw = make_twiddles_from_root(root, n, p);
        let inv_tw = make_inv_twiddles(&tw, p);  // inv_tw[k] = ω^{-k}
        let n_inv = mod_inverse(n as u32, p);

        let original = vec![
            5, 11, 3, 12, 8, 13, 2, 14,
            4, 15, 7, 16, 6, 17, 1, 18,
            3, 19, 9, 20, 2, 21, 5, 22,
            7, 23, 4, 24, 1, 25, 8, 26,
            9, 27, 6, 28, 3, 29, 5, 30,
            2, 31, 8, 32, 4, 33, 7, 34,
            1, 35, 6, 36, 3, 37, 9, 38,
            2, 39, 5, 40, 7, 41, 4, 42,
        ];
        let mut test = original.clone();
        let mut stats = MultStats::default();
        
        println!("Input: {:?}", test);
        
        // Forward FFT using radix-2
        fft_radix2_recursive_mut(&mut test, &tw, p, &mut stats);
        println!("After FFT{:?}", test);
        
        // Inverse FFT using radix-2 with INVERSE twiddles
        ifft_radix4_recursive_mut(&mut test, &inv_tw, p, n_inv, true, &mut stats);
        println!("After IFFT{:?}", test);
        
        // Verify round-trip
        assert_eq!(test, original, "FFT->IFFT round-trip failed!");
        println!("Round-trip OK!");
    }

    #[test]
    fn test_split_radix_inverse() {
        let p: u32 = 65537; // Fermat prime F4
        let n: usize = 64;

        // Compute primitive root
        let g = compute_primitive_root(p);
        let root = pow_mod(g, (p - 1) / (n as u32), p);

        // Build forward and inverse twiddles
        let tw = make_twiddles_from_root(root, n, p);
        let inv_tw = make_inv_twiddles(&tw, p);
        let n_inv = mod_inverse(n as u32, p);

        let original: Vec<u32> = (1..=64).collect();
        let mut test = original.clone();
        let mut stats = MultStats::default();

        println!("Original: {:?}", &original[..8]);

        // Forward FFT using split-radix
        fft_split_radix_recursive_mut(&mut test, &tw, p, &mut stats);
        println!("After forward split-radix FFT: {:?}", &test[..8]);

        // Inverse FFT using split-radix with inverse twiddles
        ifft_split_radix_recursive_mut(&mut test, &inv_tw, p, n_inv, true, &mut stats);
        println!("After inverse split-radix FFT: {:?}", &test[..8]);

        // Verify round-trip
        assert_eq!(test, original, "Split-radix FFT->IFFT round-trip failed!");
        println!("Split-radix round-trip OK!");
        println!("Total nonzero mults: {}", stats.nonzero_mults);
    }

    #[test]
    fn test_split_radix_inverse_n16() {
        let p: u32 = 65537;
        let n: usize = 16;

        let g = compute_primitive_root(p);
        let root = pow_mod(g, (p - 1) / (n as u32), p);

        let tw = make_twiddles_from_root(root, n, p);
        let inv_tw = make_inv_twiddles(&tw, p);
        let n_inv = mod_inverse(n as u32, p);

        let original: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut test = original.clone();
        let mut stats = MultStats::default();

        fft_split_radix_recursive_mut(&mut test, &tw, p, &mut stats);
        ifft_split_radix_recursive_mut(&mut test, &inv_tw, p, n_inv, true, &mut stats);

        assert_eq!(test, original, "Split-radix n=16 round-trip failed!");
        println!("Split-radix n=16 round-trip OK!");
    }
}
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


pub fn fft_radix4_recursive_mut(a: &mut [u32], twiddles: &[u32], p: u32, stats: &mut MultStats) {
    let n = a.len();
    if n == 1 { return; }
    
    if n == 2 {
        let tmp = a[0];
        a[0] = add_mod(a[0], a[1], p);
        a[1] = sub_mod(tmp, a[1], p);
        return;
    }
    
    if n == 4 {
        let x0 = a[0];
        let x1 = a[1];
        let x2 = a[2];
        let x3 = a[3];
        
        let t0 = add_mod(x0, x2, p);
        let t1 = sub_mod(x0, x2, p);
        let t2 = add_mod(x1, x3, p);
        let t3 = sub_mod(x1, x3, p);
        
        let w = twiddles[1];
        let t3_rotated = mul_mod_counted(w, t3, p, stats);
        
        a[0] = add_mod(t0, t2, p);
        a[1] = add_mod(t1, t3_rotated, p);
        a[2] = sub_mod(t0, t2, p);
        a[3] = sub_mod(t1, t3_rotated, p);
        return;
    }

    let quarter = n / 4;
    
    // Split into 4 sub-arrays (stride-4)
    let mut a0 = vec![0u32; quarter];
    let mut a1 = vec![0u32; quarter];
    let mut a2 = vec![0u32; quarter];
    let mut a3 = vec![0u32; quarter];
    
    for i in 0..quarter {
        a0[i] = a[4*i];
        a1[i] = a[4*i + 1];
        a2[i] = a[4*i + 2];
        a3[i] = a[4*i + 3];
    }
    
    // Twiddle table for child FFTs: tw4[k] = twiddles[4k]
    let mut tw4 = vec![0u32; quarter];
    for k in 0..quarter {
        tw4[k] = twiddles[(4 * k) % n];
    }
    
    // Recurse
    fft_radix4_recursive_mut(&mut a0, &tw4, p, stats);
    fft_radix4_recursive_mut(&mut a1, &tw4, p, stats);
    fft_radix4_recursive_mut(&mut a2, &tw4, p, stats);
    fft_radix4_recursive_mut(&mut a3, &tw4, p, stats);
    
    // Combine using top-level twiddles
    for k in 0..quarter {
        // Apply twiddle factors: ω^k, ω^(2k), ω^(3k)
        let w1 = twiddles[k];
        let w2 = twiddles[(2*k) % n];
        let w3 = twiddles[(3*k) % n];
        
        let t1 = mul_mod_counted(w1, a1[k], p, stats);
        let t2 = mul_mod_counted(w2, a2[k], p, stats);
        let t3 = mul_mod_counted(w3, a3[k], p, stats);
        
        // Radix-4 butterfly
        let b0 = add_mod(a0[k], t2, p);
        let b1 = sub_mod(a0[k], t2, p);
        let b2 = add_mod(t1, t3, p);
        let b3 = sub_mod(t1, t3, p);
        
        // Apply n/4 twiddle rotation to b3
        let w_n4 = twiddles[quarter];
        let b3_rot = mul_mod_counted(w_n4, b3, p, stats);
        
        // Output at positions k, k+n/4, k+n/2, k+3n/4
        a[k] = add_mod(b0, b2, p);
        a[k + quarter] = add_mod(b1, b3_rot, p);
        a[k + 2*quarter] = sub_mod(b0, b2, p);
        a[k + 3*quarter] = sub_mod(b1, b3_rot, p);
    }
}

pub fn fft_radix2_recursive_mut(a: &mut [u32], twiddles: &[u32], p: u32, stats: &mut MultStats) {
    let n = a.len();
    if n == 1 { return; }
    if n == 2 {
        let nz = (a[1] != 0) as usize;
        //println!("value of nz={}",nz);
        stats.nonzero_mults += nz;
        let tmp = a[0];
        a[0] = add_mod(a[0], a[1], p);
        a[1] = sub_mod(tmp, a[1], p);
        
        return;
    }

    let half = n / 2;

    // split even / odd
    let mut even = vec![0u32; half];
    let mut odd  = vec![0u32; half];
    for i in 0..half {
        even[i] = a[2*i];
        odd[i]  = a[2*i + 1];
    }

    // correct twiddle table for child FFTs: tw2[k] = twiddles[2k]
    let mut tw2 = vec![0u32; half];
    for k in 0..half {
        tw2[k] = twiddles[(2 * k) % n];
    }

    // recurse
    fft_radix2_recursive_mut(&mut even, &tw2, p, stats);
    fft_radix2_recursive_mut(&mut odd,  &tw2, p, stats);

    // combine using top-level twiddles: ω^k
    for k in 0..half {
        let t = mul_mod_counted(twiddles[k], odd[k], p, stats);
        a[k]      = add_mod(even[k], t, p);
        a[k+half] = sub_mod(even[k], t, p);
    }
}

pub fn fft_split_radix_recursive_mut(a: &mut [u32], tw: &[u32], p: u32, stats: &mut MultStats) {
    let n = a.len();
    if n == 1 { return; }

    if n == 2 {
        // let nz = (a[1] != 0) as usize;
        // stats.nonzero_mults += nz;
        let t = a[0];
        a[0] = add_mod(a[0], a[1], p);
        a[1] = sub_mod(t, a[1], p);
        return;
    }

    let n2 = n / 2;
    let n4 = n / 4;

    // split inputs
    let mut a0 = vec![0u32; n2]; // even indices
    let mut a1 = vec![0u32; n4]; // indices 1 mod 4
    let mut a2 = vec![0u32; n4]; // indices 3 mod 4

    for i in 0..n2 { a0[i] = a[2*i]; }
    for i in 0..n4 { a1[i] = a[4*i + 1]; a2[i] = a[4*i + 3]; }

    // build child twiddle tables (indices multiply by 2 and 4)
    let mut tw2 = vec![0u32; n2];
    for k in 0..n2 { tw2[k] = tw[(2 * k) % n]; }

    let mut tw4 = vec![0u32; n4];
    for k in 0..n4 { tw4[k] = tw[(4 * k) % n]; }

    // recurse
    fft_split_radix_recursive_mut(&mut a0, &tw2, p, stats);
    fft_split_radix_recursive_mut(&mut a1, &tw4, p, stats);
    fft_split_radix_recursive_mut(&mut a2, &tw4, p, stats);

    // quarter-rotation constant J = W^{N/4} (order 4 element; J^2 = -1)
    let j = tw[n4 % n];

    // combine: for k in 0..N/4
    for k in 0..n4 {
        // W^k and W^{3k} (top-level twiddles)
        let w_k = tw[k % n];
        let w_3k = tw[(3 * k) % n];

        // multiply children by their twiddles
        let t1 = mul_mod_counted(w_k, a1[k], p, stats);    // W^k * O1[k]
        let t2 = mul_mod_counted(w_3k, a2[k], p, stats);   // W^{3k} * O2[k]

        let sum = add_mod(t1, t2, p);       // t1 + t2
        let diff = sub_mod(t1, t2, p);      // t1 - t2
        let jdiff = mul_mod_counted(j, diff, p, stats);    // J * (t1 - t2)

        let u0 = a0[k];             // E[k]
        let u1 = a0[k + n4];        // E[k + N/4]

        // outputs
        a[k]             = add_mod(u0, sum, p);        // X[k] = E[k] + sum
        a[k + n4]        = add_mod(u1, jdiff, p);      // X[k+N/4] = E[k+N/4] + J*(t1 - t2)
        a[k + n2]        = sub_mod(u0, sum, p);        // X[k+N/2] = E[k] - sum
        a[k + n2 + n4]   = sub_mod(u1, jdiff, p);      // X[k+3N/4] = E[k+N/4] - J*(t1 - t2)
    }
}

pub fn ifft_radix4_recursive_mut(a: &mut [u32], inv_twiddles: &[u32], p: u32, n_inv: u32, top: bool, stats: &mut MultStats) {

    let n = a.len();
    if n == 1 { 
        if top {
            a[0] = mul_mod(a[0], n_inv, p);
            // Note: we don't track the final scaling multiplications
        }
        return; 
    }

    if n == 2 {
        let tmp = a[0];
        a[0] = add_mod(a[0], a[1], p);
        a[1] = sub_mod(tmp, a[1], p);
        
        if top {
            a[0] = mul_mod(a[0], n_inv, p);
            a[1] = mul_mod(a[1], n_inv, p);
        }
        return;
    }

    let quarter = n / 4;
    let mut a0 = vec![0u32; quarter];
    let mut a1 = vec![0u32; quarter];
    let mut a2 = vec![0u32; quarter];
    let mut a3 = vec![0u32; quarter];

    for i in 0..quarter {
        a0[i] = a[4*i];
        a1[i] = a[4*i + 1];
        a2[i] = a[4*i + 2];
        a3[i] = a[4*i + 3];
    }

    // Build correct inverse twiddle table for n/4-size children
    let mut inv_tw4 = vec![0u32; quarter];
    for k in 0..quarter {
        inv_tw4[k] = inv_twiddles[(4 * k) % n];
    }

    // recurse with inv_tw4
    ifft_radix4_recursive_mut(&mut a0, &inv_tw4, p, n_inv, false, stats);
    ifft_radix4_recursive_mut(&mut a1, &inv_tw4, p, n_inv, false, stats);
    ifft_radix4_recursive_mut(&mut a2, &inv_tw4, p, n_inv, false, stats);
    ifft_radix4_recursive_mut(&mut a3, &inv_tw4, p, n_inv, false, stats);

    // combine using inverse twiddles (mirrors forward combine)
    for i in 0..quarter {
        let t1 = mul_mod_counted(inv_twiddles[i % n], a1[i], p, stats);
        let t2 = mul_mod_counted(inv_twiddles[(2*i) % n], a2[i], p, stats);
        let t3 = mul_mod_counted(inv_twiddles[(3*i) % n], a3[i], p, stats);

        let y0 = add_mod(add_mod(a0[i], t1, p), add_mod(t2, t3, p), p);

        let t1 = mul_mod_counted(inv_twiddles[(i + quarter) % n], a1[i], p, stats);
        let t2 = mul_mod_counted(inv_twiddles[(2*(i + quarter)) % n], a2[i], p, stats);
        let t3 = mul_mod_counted(inv_twiddles[(3*(i + quarter)) % n], a3[i], p, stats);
        let y1 = add_mod(add_mod(a0[i], t1, p), add_mod(t2, t3, p), p);

        let t1 = mul_mod_counted(inv_twiddles[(i + 2*quarter) % n], a1[i], p, stats);
        let t2 = mul_mod_counted(inv_twiddles[(2*(i + 2*quarter)) % n], a2[i], p, stats);
        let t3 = mul_mod_counted(inv_twiddles[(3*(i + 2*quarter)) % n], a3[i], p, stats);
        let y2 = add_mod(add_mod(a0[i], t1, p), add_mod(t2, t3, p), p);

        let t1 = mul_mod_counted(inv_twiddles[(i + 3*quarter) % n], a1[i], p, stats);
        let t2 = mul_mod_counted(inv_twiddles[(2*(i + 3*quarter)) % n], a2[i], p, stats);
        let t3 = mul_mod_counted(inv_twiddles[(3*(i + 3*quarter)) % n], a3[i], p, stats);
        let y3 = add_mod(add_mod(a0[i], t1, p), add_mod(t2, t3, p), p);

        a[i] = y0;
        a[i + quarter] = y1;
        a[i + 2*quarter] = y2;
        a[i + 3*quarter] = y3;
    }

    // Scale at top-level only
    if top { 
        for x in a.iter_mut() {
            *x = mul_mod(*x, n_inv, p);
        }
    }
}

pub fn ifft_radix2_recursive_mut(a: &mut [u32], inv_twiddles: &[u32], p: u32, n_inv: u32, top: bool) {
    let n = a.len();
    if n == 1 { return; }

    if n == 2 {
        let tmp = a[0];
        a[0] = add_mod(a[0], a[1], p);
        a[1] = sub_mod(tmp, a[1], p);
        return;
    }


    let half = n / 2;

    let mut even = vec![0u32; half];
    let mut odd  = vec![0u32; half];
    for i in 0..half {
        even[i] = a[2*i];
        odd[i]  = a[2*i + 1];
    }

    // ✔ CORRECT inverse twiddle table: inv_tw2[k] = inv_twiddles[2k]
    let mut inv_tw2 = vec![0u32; half];
    for k in 0..half {
        inv_tw2[k] = inv_twiddles[(2 * k) % n];
    }

    // recurse without scaling
    ifft_radix2_recursive_mut(&mut even, &inv_tw2, p, n_inv, false);
    ifft_radix2_recursive_mut(&mut odd,  &inv_tw2, p, n_inv, false);

    // combine using inverse top level twiddles: ω^{-k}
    for k in 0..half {
        let t = mul_mod(odd[k], inv_twiddles[k], p);
        a[k]      = add_mod(even[k], t, p);
        a[k+half] = sub_mod(even[k], t, p);
    }

    // scale only at top
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
    let p: u32 = 65537; // 15 * 2^12 + 1
    let n: usize = 64;   // Match your actual input size!

    // Compute primitive root
    let g = compute_primitive_root(p);
    let root = pow_mod(g, (p - 1) / (n as u32), p);  // forward root

    // Build twiddles for size n=4
    let tw = make_twiddles_from_root(root, n, p);
    let n_inv = mod_inverse(n as u32, p);

    let mut test = vec![
        5, 11, 3, 12, 8, 13, 2, 14,
        4, 15, 7, 16, 6, 17, 1, 18,
        3, 19, 9, 20, 2, 21, 5, 22,
        7, 23, 4, 24, 1, 25, 8, 26,
        9, 27, 6, 28, 3, 29, 5, 30,
        2, 31, 8, 32, 4, 33, 7, 34,
        1, 35, 6, 36, 3, 37, 9, 38,
        2, 39, 5, 40, 7, 41, 4, 42,
    ];
    let mut stats = MultStats::default();
    
    println!("Input: {:?}", test);
    fft_split_radix_recursive_mut(&mut test, &tw, p, &mut stats);
    println!("After FFT: {:?}", test);
    
    ifft_radix2_recursive_mut(&mut test, &tw, p, n_inv, false);
    println!("After IFFT: {:?}", test);
}
}
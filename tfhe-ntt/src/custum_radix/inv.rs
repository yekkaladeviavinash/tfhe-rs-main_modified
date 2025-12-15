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

/// Compute primitive n-th root and build twiddles
fn make_twiddles(n: usize, p: u32) -> Vec<u32> {
    assert!(n > 0 && n.is_power_of_two());
    // Ensure n divides p-1
    assert!((p - 1) as usize % n == 0, "n must divide p-1");

    let g = compute_primitive_root(p);
    let exp = ((p - 1) / (n as u32)) as u32;
    let root = pow_mod(g, exp, p);
    make_twiddles_from_root(root, n, p)
}

/// Inverse twiddles for INTT: inv[k] = tw[n-k] (and inv[0] = 1)
fn make_inv_twiddles(tw: &[u32], p: u32) -> Vec<u32> {
    let n = tw.len();
    let mut inv = vec![0u32; n];
    for k in 0..n {
        inv[k] = mod_inverse(tw[k], p);
    }
    inv
}


pub fn ifft_radix4_recursive(a: &mut [u32], inv_twiddles: &[u32], p: u32, n_inv: u32, top: bool) {
    let n = a.len();
    if n == 1 { return; }

    if n == 2 {
        // inverse base for radix-2
        let tmp = a[0];
        let inv2 = mod_inverse(2, p);
        a[0] = mul_mod(add_mod(a[0], a[1], p), inv2, p);
        a[1] = mul_mod(sub_mod(tmp, a[1], p), inv2, p);
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

    // Build inv_twiddles^4 for recursion
    let mut inv_tw4 = vec![0u32; n];
    for i in 0..n {
        inv_tw4[i] = pow_mod(inv_twiddles[i], 4, p);
    }

    // recurse with inv_tw4
    ifft_radix4_recursive(&mut a0, &inv_tw4, p, n_inv, false);
    ifft_radix4_recursive(&mut a1, &inv_tw4, p, n_inv, false);
    ifft_radix4_recursive(&mut a2, &inv_tw4, p, n_inv, false);
    ifft_radix4_recursive(&mut a3, &inv_tw4, p, n_inv, false);

    // combine using inverse twiddles (mirrors forward combine)
    for i in 0..quarter {
        let t1 = mul_mod(inv_twiddles[i % n], a1[i], p);
        let t2 = mul_mod(inv_twiddles[(2*i) % n], a2[i], p);
        let t3 = mul_mod(inv_twiddles[(3*i) % n], a3[i], p);

        let y0 = add_mod(add_mod(a0[i], t1, p), add_mod(t2, t3, p), p);

        let t1 = mul_mod(inv_twiddles[(i + quarter) % n], a1[i], p);
        let t2 = mul_mod(inv_twiddles[(2*(i + quarter)) % n], a2[i], p);
        let t3 = mul_mod(inv_twiddles[(3*(i + quarter)) % n], a3[i], p);
        let y1 = add_mod(add_mod(a0[i], t1, p), add_mod(t2, t3, p), p);

        let t1 = mul_mod(inv_twiddles[(i + 2*quarter) % n], a1[i], p);
        let t2 = mul_mod(inv_twiddles[(2*(i + 2*quarter)) % n], a2[i], p);
        let t3 = mul_mod(inv_twiddles[(3*(i + 2*quarter)) % n], a3[i], p);
        let y2 = add_mod(add_mod(a0[i], t1, p), add_mod(t2, t3, p), p);

        let t1 = mul_mod(inv_twiddles[(i + 3*quarter) % n], a1[i], p);
        let t2 = mul_mod(inv_twiddles[(2*(i + 3*quarter)) % n], a2[i], p);
        let t3 = mul_mod(inv_twiddles[(3*(i + 3*quarter)) % n], a3[i], p);
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

pub fn ifft_radix2_recursive(a: &mut [u32], inv_twiddles: &[u32], p: u32, n_inv: u32, top: bool) {
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
    ifft_radix2_recursive(&mut even, &inv_tw2, p, n_inv, false);
    ifft_radix2_recursive(&mut odd,  &inv_tw2, p, n_inv, false);

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

pub fn ifft_split_radix_recursive(
    a: &mut [u32],
    inv_tw: &[u32],
    p: u32,
    n_inv: u32,
    top: bool
) {
    let n = a.len();
    if n == 1 { return; }

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
        a0[i] = a[2*i];
    }
    for i in 0..n4 {
        a1[i] = a[4*i + 1];
        a2[i] = a[4*i + 3];
    }

    // inverse sub-twiddles
    let mut inv_tw2 = vec![0u32; n2];
    for k in 0..n2 {
        inv_tw2[k] = inv_tw[(2 * k) % n];
    }

    let mut inv_tw4 = vec![0u32; n4];
    for k in 0..n4 {
        inv_tw4[k] = inv_tw[(4 * k) % n];
    }

    // recursive calls (NO SCALING)
    ifft_split_radix_recursive(&mut a0, &inv_tw2, p, n_inv, false);
    ifft_split_radix_recursive(&mut a1, &inv_tw4, p, n_inv, false);
    ifft_split_radix_recursive(&mut a2, &inv_tw4, p, n_inv, false);

    // J = ω^{-N/4}
    let j = inv_tw[n / 4];

    // === INVERSE COMBINE ===
    for k in 0..n4 {
        let w_k  = inv_tw[k];
        let w_3k = inv_tw[(3 * k) % n];

        let t1 = mul_mod(a1[k], w_k,  p);
        let t2 = mul_mod(a2[k], w_3k, p);

        let sum  = add_mod(t1, t2, p);
        let diff = sub_mod(t1, t2, p);
        let jdiff = mul_mod(diff, j, p);   // J*(t1 − t2)

        let u0 = a0[k];
        let u1 = a0[k + n4];

        a[k]           = add_mod(u0, sum,  p);
        a[k + n4]      = add_mod(u1, jdiff, p);
        a[k + n2]      = sub_mod(u0, sum,  p);
        a[k + n2 + n4] = sub_mod(u1, jdiff, p);
    }

    // scale only at top
    if top {
        for x in a.iter_mut() {
            *x = mul_mod(*x, n_inv, p);
        }
    }
}
//! custum_radix/butterfly.rs
//!
//! Scalar butterfly primitives for the custom hybrid NTT backend.
//! - radix2_butterfly
//! - radix4_butterfly_modified
//! - split_radix_butterfly
//!
//! These are intentionally conservative (64-bit product + mod) to be correct.
//! Replace `mul_mod` with your Shoup/Barrett/SIMD variant later for speed.

#[inline(always)]
fn add_mod(a: u32, b: u32, p: u32) -> u32 {
    let s = a.wrapping_add(b);
    if s >= p { s - p } else { s }
}

#[inline(always)]
fn sub_mod(a: u32, b: u32, p: u32) -> u32 {
    // returns (a - b) mod p in [0, p)
    let s = a.wrapping_add(p).wrapping_sub(b);
    if s >= p { s - p } else { s }
}

#[inline(always)]
fn mul_mod(a: u32, b: u32, p: u32) -> u32 {
    // Simple, correct 64-bit reduction. Swap with optimized mul when available.
    let wide = (a as u64) * (b as u64);
    (wide % (p as u64)) as u32
}

/// Radix-2 butterfly (2-point)
#[inline(always)]
pub fn radix2_butterfly(z0: u32, z1: u32, w: u32, p: u32) -> (u32, u32) {
    let t = mul_mod(z1, w, p);
    let y0 = add_mod(z0, t, p);
    let y1 = sub_mod(z0, t, p);
    (y0, y1)
}

/// Modified radix-4 butterfly (4-point)
///
/// Inputs:
/// - z0,z1,z2,z3 : lanes
/// - w1,w2,w3 : twiddles for lanes 1..3
/// - R : quarter-turn constant (R^2 ≡ -1 mod p) for the stage
/// - p : modulus
///
/// Outputs: (y0,y1,y2,y3)
#[inline(always)]
pub fn radix4_butterfly_modified(
    z0: u32,
    z1: u32,
    z2: u32,
    z3: u32,
    w1: u32,
    w2: u32,
    w3: u32,
    p: u32,
) -> (u32, u32, u32, u32) {

    // apply twiddles
    let t1 = mul_mod(z1, w1, p);
    let t2 = mul_mod(z2, w2, p);
    let t3 = mul_mod(z3, w3, p);

    // radix-4 butterflies
    let a0 = add_mod(z0, t2, p);
    let a1 = sub_mod(z0, t2, p);

    let b0 = add_mod(t1, t3, p);
    let b1 = sub_mod(t1, t3, p);

    let y0 = add_mod(a0, b0, p);
    let y2 = sub_mod(a0, b0, p);
    let y1 = add_mod(a1, b1, p);
    let y3 = sub_mod(a1, b1, p);

    (y0, y1, y2, y3)
}

/// Split-radix butterfly (hybrid)
///
/// Layout: combine evens and odds; uses twiddles for odd lanes.
/// This implementation mirrors the pattern used in fwd.rs (w1 for lane1, w3 for lane3).
#[inline(always)]
pub fn split_radix_butterfly(
    z0: u32,
    z1: u32,
    z2: u32,
    z3: u32,
    w1: u32,
    _w2: u32, // unused in this simple mapping
    w3: u32,
    R: u32,
    p: u32,
) -> (u32, u32, u32, u32) {
    // evens
    let a = add_mod(z0, z2, p);
    let b = sub_mod(z0, z2, p);

    // odds with twiddles
    let t1 = mul_mod(z1, w1, p);
    let t3 = mul_mod(z3, w3, p);

    let c = add_mod(t1, t3, p);
    let d = sub_mod(t1, t3, p);

    let y0 = add_mod(a, c, p);
    let y2 = sub_mod(a, c, p);

    let Rd = mul_mod(d, R, p);
    let y1 = add_mod(b, Rd, p);
    let y3 = sub_mod(b, Rd, p);

    (y0, y1, y2, y3)
}

use crate::{
    bit_rev,
    fastdiv::{Div32, Div64},
    prime::is_prime64,
    roots::find_primitive_root64,
};
use aligned_vec::{avec, ABox};

#[allow(unused_imports)]
use pulp::*;

const RECURSION_THRESHOLD: usize = 2048;

#[path = "prime32/generic.rs"]
mod generic;

mod generic_r4;

mod shoup_r4_new;
mod less_than_30bit_r4;

#[path = "prime32/less_than_31bit.rs"]
mod less_than_31bit;
mod less_than_31bit_r4;

mod shoup {
    pub(crate) use super::shoup_r4_new::{
        fwd_breadth_first_avx2_r4 as fwd_breadth_first_avx2,
        fwd_breadth_first_avx512_r4 as fwd_breadth_first_avx512,
        fwd_breadth_first_scalar_r4 as fwd_breadth_first_scalar,
        fwd_depth_first_avx2_r4 as fwd_depth_first_avx2,
        fwd_depth_first_avx512_r4 as fwd_depth_first_avx512,
        fwd_depth_first_scalar_r4 as fwd_depth_first_scalar,
        inv_breadth_first_avx2_r4 as inv_breadth_first_avx2,
        inv_breadth_first_avx512_r4 as inv_breadth_first_avx512,
        inv_breadth_first_scalar_r4 as inv_breadth_first_scalar,
        inv_depth_first_avx2_r4 as inv_depth_first_avx2,
        inv_depth_first_avx512_r4 as inv_depth_first_avx512,
        inv_depth_first_scalar_r4 as inv_depth_first_scalar,
    };
}

fn init_negacyclic_twiddles_r4(p: u32, n: usize, twid: &mut [u32], inv_twid: &mut [u32]) {
    let div = Div32::new(p);
    let w = find_primitive_root64(Div64::new(p as u64), 2 * n as u64).unwrap() as u32;
    let mut k = 0;
    let mut wk = 1u32;

    let nbits = n.trailing_zeros();
    while k < n {
        let fwd_idx = bit_rev(nbits, k);

        twid[fwd_idx] = wk;

        let inv_idx = bit_rev(nbits, (n - k) % n);
        if k == 0 {
            inv_twid[inv_idx] = wk;
        } else {
            let x = p.wrapping_sub(wk);
            inv_twid[inv_idx] = x;
        }

        wk = Div32::rem_u64(wk as u64 * w as u64, div);
        k += 1;
    }
}

fn init_negacyclic_twiddles_shoup_r4(
    p: u32,
    n: usize,
    twid: &mut [u32],
    twid_shoup: &mut [u32],
    inv_twid: &mut [u32],
    inv_twid_shoup: &mut [u32],
) {
    let div = Div32::new(p);
    let w = find_primitive_root64(Div64::new(p as u64), 2 * n as u64).unwrap() as u32;
    let mut k = 0;
    let mut wk = 1u32;

    let nbits = n.trailing_zeros();
    while k < n {
        let fwd_idx = bit_rev(nbits, k);

        let wk_shoup = Div32::div_u64((wk as u64) << 32, div) as u32;
        twid[fwd_idx] = wk;
        twid_shoup[fwd_idx] = wk_shoup;

        let inv_idx = bit_rev(nbits, (n - k) % n);
        if k == 0 {
            inv_twid[inv_idx] = wk;
            inv_twid_shoup[inv_idx] = wk_shoup;
        } else {
            let x = p.wrapping_sub(wk);
            inv_twid[inv_idx] = x;
            inv_twid_shoup[inv_idx] = Div32::div_u64((x as u64) << 32, div) as u32;
        }

        wk = Div32::rem_u64(wk as u64 * w as u64, div);
        k += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "avx512")]
fn mul_assign_normalize_avx512_r4(
    simd: crate::V4,
    lhs: &mut [u32],
    rhs: &[u32],
    p: u32,
    p_barrett: u32,
    big_q: u32,
    n_inv_mod_p: u32,
    n_inv_mod_p_shoup: u32,
) {
    simd.vectorize(
        #[inline(always)]
        move || {
            let lhs = pulp::as_arrays_mut::<16, _>(lhs).0;
            let rhs = pulp::as_arrays::<16, _>(rhs).0;
            let big_q_m1 = simd.splat_u32x16(big_q - 1);
            let big_q_m1_complement = simd.splat_u32x16(32 - (big_q - 1));
            let n_inv_mod_p = simd.splat_u32x16(n_inv_mod_p);
            let n_inv_mod_p_shoup = simd.splat_u32x16(n_inv_mod_p_shoup);
            let p_barrett = simd.splat_u32x16(p_barrett);
            let p = simd.splat_u32x16(p);

            for (lhs_, rhs) in crate::izip!(lhs, rhs) {
                let lhs = cast(*lhs_);
                let rhs = cast(*rhs);

                let (lo, hi) = simd.widening_mul_u32x16(lhs, rhs);
                let c1 = simd.or_u32x16(
                    simd.shr_dyn_u32x16(lo, big_q_m1),
                    simd.shl_dyn_u32x16(hi, big_q_m1_complement),
                );
                let c3 = simd.widening_mul_u32x16(c1, p_barrett).1;
                let prod = simd.wrapping_sub_u32x16(lo, simd.wrapping_mul_u32x16(p, c3));

                let shoup_q = simd.widening_mul_u32x16(prod, n_inv_mod_p_shoup).1;
                let t = simd.wrapping_sub_u32x16(
                    simd.wrapping_mul_u32x16(prod, n_inv_mod_p),
                    simd.wrapping_mul_u32x16(shoup_q, p),
                );

                *lhs_ = cast(simd.small_mod_u32x16(p, t));
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn mul_assign_normalize_avx2_r4(
    simd: crate::V3,
    lhs: &mut [u32],
    rhs: &[u32],
    p: u32,
    p_barrett: u32,
    big_q: u32,
    n_inv_mod_p: u32,
    n_inv_mod_p_shoup: u32,
) {
    simd.vectorize(
        #[inline(always)]
        move || {
            let lhs = pulp::as_arrays_mut::<8, _>(lhs).0;
            let rhs = pulp::as_arrays::<8, _>(rhs).0;
            let big_q_m1 = simd.splat_u32x8(big_q - 1);
            let big_q_m1_complement = simd.splat_u32x8(32 - (big_q - 1));
            let n_inv_mod_p = simd.splat_u32x8(n_inv_mod_p);
            let n_inv_mod_p_shoup = simd.splat_u32x8(n_inv_mod_p_shoup);
            let p_barrett = simd.splat_u32x8(p_barrett);
            let p = simd.splat_u32x8(p);

            for (lhs_, rhs) in crate::izip!(lhs, rhs) {
                let lhs = cast(*lhs_);
                let rhs = cast(*rhs);

                let (lo, hi) = simd.widening_mul_u32x8(lhs, rhs);
                let c1 = simd.or_u32x8(
                    simd.shr_dyn_u32x8(lo, big_q_m1),
                    simd.shl_dyn_u32x8(hi, big_q_m1_complement),
                );
                let c3 = simd.widening_mul_u32x8(c1, p_barrett).1;
                let prod = simd.wrapping_sub_u32x8(lo, simd.wrapping_mul_u32x8(p, c3));

                let shoup_q = simd.widening_mul_u32x8(prod, n_inv_mod_p_shoup).1;
                let t = simd.wrapping_sub_u32x8(
                    simd.wrapping_mul_u32x8(prod, n_inv_mod_p),
                    simd.wrapping_mul_u32x8(shoup_q, p),
                );

                *lhs_ = cast(simd.small_mod_u32x8(p, t));
            }
        },
    );
}

fn mul_assign_normalize_scalar_r4(
    lhs: &mut [u32],
    rhs: &[u32],
    p: u32,
    p_barrett: u32,
    big_q: u32,
    n_inv_mod_p: u32,
    n_inv_mod_p_shoup: u32,
) {
    let big_q_m1 = big_q - 1;

    for (lhs_, rhs) in crate::izip!(lhs, rhs) {
        let lhs = *lhs_;
        let rhs = *rhs;

        let d = lhs as u64 * rhs as u64;
        let c1 = (d >> big_q_m1) as u32;
        let c3 = ((c1 as u64 * p_barrett as u64) >> 32) as u32;
        let prod = (d as u32).wrapping_sub(p.wrapping_mul(c3));
        let prod = prod.min(prod.wrapping_sub(p));

        let shoup_q = ((prod as u64 * n_inv_mod_p_shoup as u64) >> 32) as u32;
        let t = prod.wrapping_mul(n_inv_mod_p).wrapping_sub(shoup_q.wrapping_mul(p));

        *lhs_ = t.min(t.wrapping_sub(p));
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "avx512")]
fn normalize_avx512_r4(
    simd: crate::V4,
    values: &mut [u32],
    p: u32,
    n_inv_mod_p: u32,
    n_inv_mod_p_shoup: u32,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let values = pulp::as_arrays_mut::<16, _>(values).0;

            let n_inv_mod_p = simd.splat_u32x16(n_inv_mod_p);
            let n_inv_mod_p_shoup = simd.splat_u32x16(n_inv_mod_p_shoup);
            let p = simd.splat_u32x16(p);

            for val_ in values {
                let val = cast(*val_);

                let shoup_q = simd.widening_mul_u32x16(val, n_inv_mod_p_shoup).1;
                let t = simd.wrapping_sub_u32x16(
                    simd.wrapping_mul_u32x16(val, n_inv_mod_p),
                    simd.wrapping_mul_u32x16(shoup_q, p),
                );

                *val_ = cast(simd.small_mod_u32x16(p, t));
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn normalize_avx2_r4(
    simd: crate::V3,
    values: &mut [u32],
    p: u32,
    n_inv_mod_p: u32,
    n_inv_mod_p_shoup: u32,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let values = pulp::as_arrays_mut::<8, _>(values).0;

            let n_inv_mod_p = simd.splat_u32x8(n_inv_mod_p);
            let n_inv_mod_p_shoup = simd.splat_u32x8(n_inv_mod_p_shoup);
            let p = simd.splat_u32x8(p);

            for val_ in values {
                let val = cast(*val_);

                let shoup_q = simd.widening_mul_u32x8(val, n_inv_mod_p_shoup).1;
                let t = simd.wrapping_sub_u32x8(
                    simd.wrapping_mul_u32x8(val, n_inv_mod_p),
                    simd.wrapping_mul_u32x8(shoup_q, p),
                );

                *val_ = cast(simd.small_mod_u32x8(p, t));
            }
        },
    );
}

fn normalize_scalar_r4(values: &mut [u32], p: u32, n_inv_mod_p: u32, n_inv_mod_p_shoup: u32) {
    for values in values {
        let shoup_q = ((n_inv_mod_p_shoup as u64 * *values as u64) >> 32) as u32;
        let t = values
            .wrapping_mul(n_inv_mod_p)
            .wrapping_sub(shoup_q.wrapping_mul(p));
        *values = t.min(t.wrapping_sub(p));
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "avx512")]
fn mul_accumulate_avx512_r4(
    simd: crate::V4,
    acc: &mut [u32],
    lhs: &[u32],
    rhs: &[u32],
    p: u32,
    p_barrett: u32,
    big_q: u32,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let acc = pulp::as_arrays_mut::<16, _>(acc).0;
            let lhs = pulp::as_arrays::<16, _>(lhs).0;
            let rhs = pulp::as_arrays::<16, _>(rhs).0;

            let big_q_m1 = simd.splat_u32x16(big_q - 1);
            let big_q_m1_complement = simd.splat_u32x16(32 - (big_q - 1));
            let p_barrett = simd.splat_u32x16(p_barrett);
            let p = simd.splat_u32x16(p);

            for (acc, lhs, rhs) in crate::izip!(acc, lhs, rhs) {
                let lhs = cast(*lhs);
                let rhs = cast(*rhs);

                let (lo, hi) = simd.widening_mul_u32x16(lhs, rhs);
                let c1 = simd.or_u32x16(
                    simd.shr_dyn_u32x16(lo, big_q_m1),
                    simd.shl_dyn_u32x16(hi, big_q_m1_complement),
                );
                let c3 = simd.widening_mul_u32x16(c1, p_barrett).1;
                let prod = simd.wrapping_sub_u32x16(lo, simd.wrapping_mul_u32x16(p, c3));
                let prod = simd.small_mod_u32x16(p, prod);

                *acc = cast(simd.small_mod_u32x16(p, simd.wrapping_add_u32x16(prod, cast(*acc))));
            }
        },
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn mul_accumulate_avx2_r4(
    simd: crate::V3,
    acc: &mut [u32],
    lhs: &[u32],
    rhs: &[u32],
    p: u32,
    p_barrett: u32,
    big_q: u32,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let acc = pulp::as_arrays_mut::<8, _>(acc).0;
            let lhs = pulp::as_arrays::<8, _>(lhs).0;
            let rhs = pulp::as_arrays::<8, _>(rhs).0;

            let big_q_m1 = simd.splat_u32x8(big_q - 1);
            let big_q_m1_complement = simd.splat_u32x8(32 - (big_q - 1));
            let p_barrett = simd.splat_u32x8(p_barrett);
            let p = simd.splat_u32x8(p);

            for (acc, lhs, rhs) in crate::izip!(acc, lhs, rhs) {
                let lhs = cast(*lhs);
                let rhs = cast(*rhs);

                let (lo, hi) = simd.widening_mul_u32x8(lhs, rhs);
                let c1 = simd.or_u32x8(
                    simd.shr_dyn_u32x8(lo, big_q_m1),
                    simd.shl_dyn_u32x8(hi, big_q_m1_complement),
                );
                let c3 = simd.widening_mul_u32x8(c1, p_barrett).1;
                let prod = simd.wrapping_sub_u32x8(lo, simd.wrapping_mul_u32x8(p, c3));
                let prod = simd.small_mod_u32x8(p, prod);

                *acc = cast(simd.small_mod_u32x8(p, simd.wrapping_add_u32x8(prod, cast(*acc))));
            }
        },
    )
}

fn mul_accumulate_scalar_r4(
    acc: &mut [u32],
    lhs: &[u32],
    rhs: &[u32],
    p: u32,
    p_barrett: u32,
    big_q: u32,
) {
    let big_q_m1 = big_q - 1;

    for (acc, lhs, rhs) in crate::izip!(acc, lhs, rhs) {
        let lhs = *lhs;
        let rhs = *rhs;

        let d = lhs as u64 * rhs as u64;
        let c1 = (d >> big_q_m1) as u32;
        let c3 = ((c1 as u64 * p_barrett as u64) >> 32) as u32;
        let prod = (d as u32).wrapping_sub(p.wrapping_mul(c3));
        let prod = prod.min(prod.wrapping_sub(p));

        let acc_ = prod + *acc;
        *acc = acc_.min(acc_.wrapping_sub(p));
    }
}

struct BarrettInit32_r4 {
    big_q: u32,
    p_barrett: u32,
    requires_single_reduction_step: bool,
}

impl BarrettInit32_r4 {
    pub fn new_r4(modulus: u32) -> Self {
        let big_q = modulus.ilog2() + 1;
        let big_l = big_q + 31;
        let m_as_u64: u64 = modulus.into();
        let two_to_the_l = 1u64 << big_l;
        let (p_barrett, beta) = ((two_to_the_l / m_as_u64) as u32, (two_to_the_l % m_as_u64));

        let single_reduction_threshold = m_as_u64 - (1 << (big_q - 1));

        let requires_single_reduction_step = beta <= single_reduction_threshold;

        Self {
            big_q,
            p_barrett,
            requires_single_reduction_step,
        }
    }
}

/// Negacyclic NTT plan for 32bit primes (radix-4 variants).
#[derive(Clone)]
pub struct Plan_r4 {
    twid: ABox<[u32]>,
    twid_shoup: ABox<[u32]>,
    inv_twid: ABox<[u32]>,
    inv_twid_shoup: ABox<[u32]>,
    p: u32,
    p_div: Div32,
    can_use_fast_reduction_code: bool,
    p_barrett: u32,
    big_q: u32,
    n_inv_mod_p: u32,
    n_inv_mod_p_shoup: u32,
}

impl core::fmt::Debug for Plan_r4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Plan_r4")
            .field("ntt_size", &self.ntt_size_r4())
            .field("modulus", &self.modulus_r4())
            .finish()
    }
}

impl Plan_r4 {
    pub fn try_new_r4(polynomial_size: usize, modulus: u32) -> Option<Self> {
        let p_div = Div32::new(modulus);
        if polynomial_size < 32
            || !polynomial_size.is_power_of_two()
            || !is_prime64(modulus as u64)
            || find_primitive_root64(Div64::new(modulus as u64), 2 * polynomial_size as u64)
                .is_none()
        {
            None
        } else {
            let mut twid = avec![0u32; polynomial_size].into_boxed_slice();
            let mut inv_twid = avec![0u32; polynomial_size].into_boxed_slice();
            let (mut twid_shoup, mut inv_twid_shoup) = if modulus < (1u32 << 31) {
                (
                    avec![0u32; polynomial_size].into_boxed_slice(),
                    avec![0u32; polynomial_size].into_boxed_slice(),
                )
            } else {
                (avec![].into_boxed_slice(), avec![].into_boxed_slice())
            };

            if modulus < (1u32 << 31) {
                init_negacyclic_twiddles_shoup_r4(
                    modulus,
                    polynomial_size,
                    &mut twid,
                    &mut twid_shoup,
                    &mut inv_twid,
                    &mut inv_twid_shoup,
                );
            } else {
                init_negacyclic_twiddles_r4(modulus, polynomial_size, &mut twid, &mut inv_twid);
            }

            let n_inv_mod_p = crate::prime::exp_mod32(p_div, polynomial_size as u32, modulus - 2);
            let n_inv_mod_p_shoup = (((n_inv_mod_p as u64) << 32) / modulus as u64) as u32;

            let BarrettInit32_r4 {
                big_q,
                p_barrett,
                requires_single_reduction_step,
            } = BarrettInit32_r4::new_r4(modulus);

            let can_use_fast_reduction_code =
                (modulus < 1431655766) || (requires_single_reduction_step && modulus <= (1 << 31));

            Some(Self {
                twid,
                twid_shoup,
                inv_twid_shoup,
                inv_twid,
                p: modulus,
                p_div,
                can_use_fast_reduction_code,
                n_inv_mod_p,
                n_inv_mod_p_shoup,
                p_barrett,
                big_q,
            })
        }
    }

    pub(crate) fn p_div_r4(&self) -> Div32 {
        self.p_div
    }

    #[inline]
    pub fn ntt_size_r4(&self) -> usize {
        self.twid.len()
    }

    #[inline]
    pub fn modulus_r4(&self) -> u32 {
        self.p
    }

    #[inline]
    pub fn can_use_fast_reduction_code_r4(&self) -> bool {
        self.can_use_fast_reduction_code
    }

    pub fn fwd_r4(&self, buf: &mut [u32]) {
        assert_eq!(buf.len(), self.ntt_size_r4());
        let p = self.p;

        if p < (1u32 << 30) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "avx512")]
                if let Some(simd) = crate::V4::try_new() {
                    // println!("2");
                    less_than_30bit_r4::fwd_avx512_r4(simd, p, buf, &self.twid, &self.twid_shoup);
                    return;
                }
                if let Some(simd) = crate::V3::try_new() {
                    less_than_30bit_r4::fwd_avx2_r4(simd, p, buf, &self.twid, &self.twid_shoup);
                    return;
                }
            }
            less_than_30bit_r4::fwd_scalar_r4(p, buf, &self.twid, &self.twid_shoup);
        } else if p < (1u32 << 31) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "avx512")]
                if let Some(simd) = crate::V4::try_new() {
                    less_than_31bit_r4::fwd_avx512_r4(simd, p, buf, &self.twid, &self.twid_shoup);
                    return;
                }
                if let Some(simd) = crate::V3::try_new() {
                    less_than_31bit_r4::fwd_avx2_r4(simd, p, buf, &self.twid, &self.twid_shoup);
                    return;
                }
            }
            less_than_31bit_r4::fwd_scalar_r4(p, buf, &self.twid, &self.twid_shoup);
        } else {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[cfg(feature = "avx512")]
            if let Some(simd) = crate::V4::try_new() {
                generic_r4::fwd_avx512_r4(simd, buf, p, self.p_div, &self.twid);
                return;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(simd) = crate::V3::try_new() {
                generic_r4::fwd_avx2_r4(simd, buf, p, self.p_div, &self.twid);
                return;
            }
            generic_r4::fwd_scalar_r4(buf, p, self.p_div, &self.twid);
        }
    }

    pub fn inv_r4(&self, buf: &mut [u32]) {
        assert_eq!(buf.len(), self.ntt_size_r4());
        let p = self.p;

        if p < (1u32 << 30) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "avx512")]
                if let Some(simd) = crate::V4::try_new() {
                    less_than_30bit_r4::inv_avx512_r4(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                    return;
                }
                if let Some(simd) = crate::V3::try_new() {
                    less_than_30bit_r4::inv_avx2_r4(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                    return;
                }
            }
            less_than_30bit_r4::inv_scalar_r4(p, buf, &self.inv_twid, &self.inv_twid_shoup);
        } else if p < (1u32 << 31) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "avx512")]
                if let Some(simd) = crate::V4::try_new() {
                    less_than_31bit_r4::inv_avx512_r4(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                    return;
                }
                if let Some(simd) = crate::V3::try_new() {
                    less_than_31bit_r4::inv_avx2_r4(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                    return;
                }
            }
            less_than_31bit_r4::inv_scalar_r4(p, buf, &self.inv_twid, &self.inv_twid_shoup);
        } else {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[cfg(feature = "avx512")]
            if let Some(simd) = crate::V4::try_new() {
                generic_r4::inv_avx512_r4(simd, buf, p, self.p_div, &self.inv_twid);
                return;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(simd) = crate::V3::try_new() {
                generic_r4::inv_avx2_r4(simd, buf, p, self.p_div, &self.inv_twid);
                return;
            }
            generic_r4::inv_scalar_r4(buf, p, self.p_div, &self.inv_twid);
        }
    }

    pub fn mul_assign_normalize_r4(&self, lhs: &mut [u32], rhs: &[u32]) {
        if self.can_use_fast_reduction_code {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[cfg(feature = "avx512")]
            if let Some(simd) = crate::V4::try_new() {
                mul_assign_normalize_avx512_r4(
                    simd,
                    lhs,
                    rhs,
                    self.p,
                    self.p_barrett,
                    self.big_q,
                    self.n_inv_mod_p,
                    self.n_inv_mod_p_shoup,
                );
                return;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(simd) = crate::V3::try_new() {
                mul_assign_normalize_avx2_r4(
                    simd,
                    lhs,
                    rhs,
                    self.p,
                    self.p_barrett,
                    self.big_q,
                    self.n_inv_mod_p,
                    self.n_inv_mod_p_shoup,
                );
                return;
            }
            mul_assign_normalize_scalar_r4(
                lhs,
                rhs,
                self.p,
                self.p_barrett,
                self.big_q,
                self.n_inv_mod_p,
                self.n_inv_mod_p_shoup,
            );
        } else {
            let p_div = self.p_div;
            let n_inv_mod_p = self.n_inv_mod_p;
            for (lhs_, rhs) in crate::izip!(lhs, rhs) {
                let lhs = *lhs_;
                let rhs = *rhs;
                let prod = Div32::rem_u64(lhs as u64 * rhs as u64, p_div);
                let prod = Div32::rem_u64(prod as u64 * n_inv_mod_p as u64, p_div);
                *lhs_ = prod;
            }
        }
    }

    pub fn normalize_r4(&self, values: &mut [u32]) {
        if self.can_use_fast_reduction_code {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[cfg(feature = "avx512")]
            if let Some(simd) = crate::V4::try_new() {
                normalize_avx512_r4(
                    simd,
                    values,
                    self.p,
                    self.n_inv_mod_p,
                    self.n_inv_mod_p_shoup,
                );
                return;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(simd) = crate::V3::try_new() {
                normalize_avx2_r4(
                    simd,
                    values,
                    self.p,
                    self.n_inv_mod_p,
                    self.n_inv_mod_p_shoup,
                );
                return;
            }
            normalize_scalar_r4(values, self.p, self.n_inv_mod_p, self.n_inv_mod_p_shoup);
        } else {
            let p_div = self.p_div;
            let n_inv_mod_p = self.n_inv_mod_p;
            for values in values {
                let prod = Div32::rem_u64(*values as u64 * n_inv_mod_p as u64, p_div);
                *values = prod;
            }
        }
    }

    pub fn mul_accumulate_r4(&self, acc: &mut [u32], lhs: &[u32], rhs: &[u32]) {
        if self.can_use_fast_reduction_code {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[cfg(feature = "avx512")]
            if let Some(simd) = crate::V4::try_new() {
                mul_accumulate_avx512_r4(simd, acc, lhs, rhs, self.p, self.p_barrett, self.big_q);
                return;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(simd) = crate::V3::try_new() {
                mul_accumulate_avx2_r4(simd, acc, lhs, rhs, self.p, self.p_barrett, self.big_q);
                return;
            }
            mul_accumulate_scalar_r4(acc, lhs, rhs, self.p, self.p_barrett, self.big_q);
        } else {
            let p = self.p;
            let p_div = self.p_div;
            for (acc, lhs, rhs) in crate::izip!(acc, lhs, rhs) {
                let prod = generic_r4::mul_r4(p_div, *lhs, *rhs);
                *acc = generic_r4::add_r4(p, *acc, prod);
            }
        }
    }
}

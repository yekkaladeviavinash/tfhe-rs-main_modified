#!/usr/bin/env python3
"""
Polynomial multiplication using ML model to select the best FFT algorithm.

This script:
1. Takes two polynomials in vector (coefficient) form
2. Uses the trained ML model to decide which FFT algorithm to use (radix-2, radix-4, split-radix)
3. Converts both polynomials to point-value form using the selected algorithm
4. Multiplies the point-value forms element-wise
5. Uses the model again to select the best inverse FFT algorithm
6. Converts the result back to coefficient form

Usage:
  python poly_mul_with_model.py
"""

from __future__ import annotations
import os
import joblib
import numpy as np
from typing import Tuple, List

# -----------------------------------------------------------------------------
# Modular arithmetic helpers (matching the Rust implementation)
# -----------------------------------------------------------------------------

def add_mod(a: int, b: int, p: int) -> int:
    return (a + b) % p

def sub_mod(a: int, b: int, p: int) -> int:
    return (a - b + p) % p

def mul_mod(a: int, b: int, p: int) -> int:
    return (a * b) % p

def pow_mod(base: int, exp: int, p: int) -> int:
    result = 1
    base = base % p
    while exp > 0:
        if exp & 1:
            result = (result * base) % p
        exp >>= 1
        base = (base * base) % p
    return result

def mod_inverse(a: int, p: int) -> int:
    """Compute modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p"""
    return pow_mod(a, p - 2, p)

def compute_primitive_root(p: int) -> int:
    """Find a primitive root modulo p (generator of multiplicative group)."""
    if p == 2:
        return 1
    phi = p - 1
    # Factor phi
    factors = []
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    
    # Find primitive root
    for g in range(2, p):
        is_primitive = True
        for f in factors:
            if pow_mod(g, phi // f, p) == 1:
                is_primitive = False
                break
        if is_primitive:
            return g
    return 2

# -----------------------------------------------------------------------------
# Twiddle factor generation
# -----------------------------------------------------------------------------

def make_twiddles(root: int, n: int, p: int) -> List[int]:
    """Generate twiddle factors: tw[k] = root^k mod p"""
    tw = [1] * n
    for k in range(1, n):
        tw[k] = mul_mod(tw[k-1], root, p)
    return tw

def make_inv_twiddles(tw: List[int], n: int) -> List[int]:
    """Generate inverse twiddle factors: inv_tw[k] = tw[n-k] for k>0"""
    inv_tw = [0] * n
    inv_tw[0] = tw[0]  # tw[0] = 1
    for k in range(1, n):
        inv_tw[k] = tw[n - k]
    return inv_tw

# -----------------------------------------------------------------------------
# FFT Algorithms (Forward)
# -----------------------------------------------------------------------------

def fft_radix2_recursive(a: List[int], tw: List[int], p: int) -> List[int]:
    """Radix-2 Cooley-Tukey FFT"""
    n = len(a)
    if n == 1:
        return a[:]
    
    half = n // 2
    even = [a[2*i] for i in range(half)]
    odd = [a[2*i + 1] for i in range(half)]
    
    # Sub-twiddles (every 2nd element)
    tw2 = [tw[2*k % n] for k in range(half)]
    
    even_fft = fft_radix2_recursive(even, tw2, p)
    odd_fft = fft_radix2_recursive(odd, tw2, p)
    
    result = [0] * n
    for k in range(half):
        t = mul_mod(tw[k], odd_fft[k], p)
        result[k] = add_mod(even_fft[k], t, p)
        result[k + half] = sub_mod(even_fft[k], t, p)
    
    return result

def fft_radix4_recursive(a: List[int], tw: List[int], p: int) -> List[int]:
    """Radix-4 FFT"""
    n = len(a)
    if n == 1:
        return a[:]
    if n == 2:
        return [add_mod(a[0], a[1], p), sub_mod(a[0], a[1], p)]
    
    quarter = n // 4
    
    # Split into 4 parts
    a0 = [a[4*i] for i in range(quarter)]
    a1 = [a[4*i + 1] for i in range(quarter)]
    a2 = [a[4*i + 2] for i in range(quarter)]
    a3 = [a[4*i + 3] for i in range(quarter)]
    
    # Sub-twiddles
    tw4 = [tw[4*k % n] for k in range(quarter)]
    
    y0 = fft_radix4_recursive(a0, tw4, p)
    y1 = fft_radix4_recursive(a1, tw4, p)
    y2 = fft_radix4_recursive(a2, tw4, p)
    y3 = fft_radix4_recursive(a3, tw4, p)
    
    result = [0] * n
    j = tw[n // 4]  # ω^(n/4) = imaginary unit in this field
    
    for k in range(quarter):
        w1 = tw[k]
        w2 = tw[2*k % n]
        w3 = tw[3*k % n]
        
        t1 = mul_mod(w1, y1[k], p)
        t2 = mul_mod(w2, y2[k], p)
        t3 = mul_mod(w3, y3[k], p)
        
        u0 = add_mod(y0[k], t2, p)
        u1 = sub_mod(y0[k], t2, p)
        u2 = add_mod(t1, t3, p)
        u3 = mul_mod(j, sub_mod(t1, t3, p), p)
        
        result[k] = add_mod(u0, u2, p)
        result[k + quarter] = add_mod(u1, u3, p)
        result[k + 2*quarter] = sub_mod(u0, u2, p)
        result[k + 3*quarter] = sub_mod(u1, u3, p)
    
    return result

def fft_split_radix_recursive(a: List[int], tw: List[int], p: int) -> List[int]:
    """Split-radix (radix-2/4) FFT"""
    n = len(a)
    if n == 1:
        return a[:]
    if n == 2:
        return [add_mod(a[0], a[1], p), sub_mod(a[0], a[1], p)]
    
    n2 = n // 2
    n4 = n // 4
    
    # Decompose: even, 4k+1, 4k+3
    a0 = [a[2*i] for i in range(n2)]
    a1 = [a[4*i + 1] for i in range(n4)]
    a2 = [a[4*i + 3] for i in range(n4)]
    
    # Sub-twiddles
    tw0 = [tw[2*k % n] for k in range(n2)]
    tw1 = [tw[4*k % n] for k in range(n4)]
    
    y0 = fft_split_radix_recursive(a0, tw0, p)
    y1 = fft_split_radix_recursive(a1, tw1, p)
    y2 = fft_split_radix_recursive(a2, tw1, p)
    
    result = [0] * n
    j = tw[n4]  # ω^(n/4)
    
    for k in range(n4):
        u0 = y0[k]
        u1 = y0[k + n4]
        
        w_k = tw[k]
        w_3k = tw[3*k % n]
        
        t1 = mul_mod(w_k, y1[k], p)
        t2 = mul_mod(w_3k, y2[k], p)
        
        s = add_mod(t1, t2, p)
        d = sub_mod(t1, t2, p)
        jd = mul_mod(j, d, p)
        
        result[k] = add_mod(u0, s, p)
        result[k + n4] = sub_mod(u1, jd, p)
        result[k + n2] = sub_mod(u0, s, p)
        result[k + n2 + n4] = add_mod(u1, jd, p)
    
    return result

# -----------------------------------------------------------------------------
# Inverse FFT Algorithms
# -----------------------------------------------------------------------------

def ifft_radix2_recursive(a: List[int], inv_tw: List[int], p: int, n_inv: int, top: bool = True) -> List[int]:
    """Inverse Radix-2 FFT"""
    n = len(a)
    if n == 1:
        return a[:]
    
    half = n // 2
    even = [a[2*i] for i in range(half)]
    odd = [a[2*i + 1] for i in range(half)]
    
    inv_tw2 = [inv_tw[2*k % n] for k in range(half)]
    
    even_ifft = ifft_radix2_recursive(even, inv_tw2, p, n_inv, False)
    odd_ifft = ifft_radix2_recursive(odd, inv_tw2, p, n_inv, False)
    
    result = [0] * n
    for k in range(half):
        t = mul_mod(inv_tw[k], odd_ifft[k], p)
        result[k] = add_mod(even_ifft[k], t, p)
        result[k + half] = sub_mod(even_ifft[k], t, p)
    
    if top:
        result = [mul_mod(x, n_inv, p) for x in result]
    
    return result

def ifft_radix4_recursive(a: List[int], inv_tw: List[int], p: int, n_inv: int, top: bool = True) -> List[int]:
    """Inverse Radix-4 FFT"""
    n = len(a)
    if n == 1:
        return a[:]
    if n == 2:
        result = [add_mod(a[0], a[1], p), sub_mod(a[0], a[1], p)]
        if top:
            result = [mul_mod(x, n_inv, p) for x in result]
        return result
    
    quarter = n // 4
    
    a0 = [a[4*i] for i in range(quarter)]
    a1 = [a[4*i + 1] for i in range(quarter)]
    a2 = [a[4*i + 2] for i in range(quarter)]
    a3 = [a[4*i + 3] for i in range(quarter)]
    
    inv_tw4 = [inv_tw[4*k % n] for k in range(quarter)]
    
    y0 = ifft_radix4_recursive(a0, inv_tw4, p, n_inv, False)
    y1 = ifft_radix4_recursive(a1, inv_tw4, p, n_inv, False)
    y2 = ifft_radix4_recursive(a2, inv_tw4, p, n_inv, False)
    y3 = ifft_radix4_recursive(a3, inv_tw4, p, n_inv, False)
    
    result = [0] * n
    neg_j = inv_tw[n // 4]  # ω^(-n/4) for inverse
    
    for k in range(quarter):
        w1 = inv_tw[k]
        w2 = inv_tw[2*k % n]
        w3 = inv_tw[3*k % n]
        
        t1 = mul_mod(w1, y1[k], p)
        t2 = mul_mod(w2, y2[k], p)
        t3 = mul_mod(w3, y3[k], p)
        
        u0 = add_mod(y0[k], t2, p)
        u1 = sub_mod(y0[k], t2, p)
        u2 = add_mod(t1, t3, p)
        u3 = mul_mod(neg_j, sub_mod(t1, t3, p), p)
        
        result[k] = add_mod(u0, u2, p)
        result[k + quarter] = add_mod(u1, u3, p)
        result[k + 2*quarter] = sub_mod(u0, u2, p)
        result[k + 3*quarter] = sub_mod(u1, u3, p)
    
    if top:
        result = [mul_mod(x, n_inv, p) for x in result]
    
    return result

def ifft_split_radix_recursive(a: List[int], inv_tw: List[int], p: int, n_inv: int, top: bool = True) -> List[int]:
    """Inverse Split-radix FFT"""
    n = len(a)
    if n == 1:
        return a[:]
    if n == 2:
        result = [add_mod(a[0], a[1], p), sub_mod(a[0], a[1], p)]
        if top:
            result = [mul_mod(x, n_inv, p) for x in result]
        return result
    
    n2 = n // 2
    n4 = n // 4
    
    a0 = [a[2*i] for i in range(n2)]
    a1 = [a[4*i + 1] for i in range(n4)]
    a2 = [a[4*i + 3] for i in range(n4)]
    
    inv_tw0 = [inv_tw[2*k % n] for k in range(n2)]
    inv_tw1 = [inv_tw[4*k % n] for k in range(n4)]
    
    y0 = ifft_split_radix_recursive(a0, inv_tw0, p, n_inv, False)
    y1 = ifft_split_radix_recursive(a1, inv_tw1, p, n_inv, False)
    y2 = ifft_split_radix_recursive(a2, inv_tw1, p, n_inv, False)
    
    result = [0] * n
    neg_j = inv_tw[n4]
    
    for k in range(n4):
        u0 = y0[k]
        u1 = y0[k + n4]
        
        w_k = inv_tw[k]
        w_3k = inv_tw[3*k % n]
        
        t1 = mul_mod(w_k, y1[k], p)
        t2 = mul_mod(w_3k, y2[k], p)
        
        s = add_mod(t1, t2, p)
        d = sub_mod(t1, t2, p)
        neg_jd = mul_mod(neg_j, d, p)
        
        result[k] = add_mod(u0, s, p)
        result[k + n4] = add_mod(u1, neg_jd, p)
        result[k + n2] = sub_mod(u0, s, p)
        result[k + n2 + n4] = sub_mod(u1, neg_jd, p)
    
    if top:
        result = [mul_mod(x, n_inv, p) for x in result]
    
    return result

# -----------------------------------------------------------------------------
# Feature extraction for ML model
# -----------------------------------------------------------------------------

def compute_features(poly: List[int], padded_n: int) -> List[float]:
    """
    Compute the 8 features for the ML model:
    - raw_N: original polynomial length
    - padded_N: FFT size (power of 2)
    - frac_2i_nonzero: fraction of even indices that are nonzero
    - frac_2i1_nonzero: fraction of odd indices that are nonzero
    - frac_4i_nonzero: fraction of 4k indices that are nonzero
    - frac_4i1_nonzero: fraction of 4k+1 indices that are nonzero
    - frac_4i2_nonzero: fraction of 4k+2 indices that are nonzero
    - frac_4i3_nonzero: fraction of 4k+3 indices that are nonzero
    """
    raw_n = len(poly)
    
    # Pad to padded_n with zeros
    padded = poly + [0] * (padded_n - len(poly))
    
    # Even/odd splits
    even_indices = [padded[2*i] for i in range(padded_n // 2)]
    odd_indices = [padded[2*i + 1] for i in range(padded_n // 2)]
    
    frac_2i = sum(1 for x in even_indices if x != 0) / len(even_indices) if even_indices else 0
    frac_2i1 = sum(1 for x in odd_indices if x != 0) / len(odd_indices) if odd_indices else 0
    
    # Mod-4 splits
    idx_4_0 = [padded[4*i] for i in range(padded_n // 4)]
    idx_4_1 = [padded[4*i + 1] for i in range(padded_n // 4)]
    idx_4_2 = [padded[4*i + 2] for i in range(padded_n // 4)]
    idx_4_3 = [padded[4*i + 3] for i in range(padded_n // 4)]
    
    frac_4i = sum(1 for x in idx_4_0 if x != 0) / len(idx_4_0) if idx_4_0 else 0
    frac_4i1 = sum(1 for x in idx_4_1 if x != 0) / len(idx_4_1) if idx_4_1 else 0
    frac_4i2 = sum(1 for x in idx_4_2 if x != 0) / len(idx_4_2) if idx_4_2 else 0
    frac_4i3 = sum(1 for x in idx_4_3 if x != 0) / len(idx_4_3) if idx_4_3 else 0
    
    return [raw_n, padded_n, frac_2i, frac_2i1, frac_4i, frac_4i1, frac_4i2, frac_4i3]

def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n"""
    p = 1
    while p < n:
        p *= 2
    return p

# -----------------------------------------------------------------------------
# Main polynomial multiplication with model-based algorithm selection
# -----------------------------------------------------------------------------

def poly_multiply_with_model(poly_a: List[int], poly_b: List[int], p: int, 
                              model, label_encoder) -> Tuple[List[int], dict]:
    """
    Multiply two polynomials using FFT with ML model-based algorithm selection.
    
    Returns:
        result: coefficient vector of the product polynomial
        info: dictionary with algorithm choices and intermediate results
    """
    info = {}
    
    # Compute required FFT size
    result_len = len(poly_a) + len(poly_b) - 1
    fft_size = next_power_of_2(result_len)
    
    info['poly_a_len'] = len(poly_a)
    info['poly_b_len'] = len(poly_b)
    info['result_len'] = result_len
    info['fft_size'] = fft_size
    
    # Pad polynomials to FFT size
    padded_a = poly_a + [0] * (fft_size - len(poly_a))
    padded_b = poly_b + [0] * (fft_size - len(poly_b))
    
    # Compute primitive root and twiddle factors
    g = compute_primitive_root(p)
    root = pow_mod(g, (p - 1) // fft_size, p)
    tw = make_twiddles(root, fft_size, p)
    inv_tw = make_inv_twiddles(tw, fft_size)
    n_inv = mod_inverse(fft_size, p)
    
    info['primitive_root'] = g
    info['nth_root'] = root
    
    # --- Forward FFT for poly_a ---
    features_a = compute_features(poly_a, fft_size)
    pred_a = model.predict([features_a])[0]
    algo_a = label_encoder.inverse_transform([pred_a])[0] if hasattr(pred_a, '__int__') or isinstance(pred_a, (int, np.integer)) else pred_a
    info['algo_forward_a'] = algo_a
    
    if algo_a == 'r2':
        fft_a = fft_radix2_recursive(padded_a, tw, p)
    elif algo_a == 'r4':
        fft_a = fft_radix4_recursive(padded_a, tw, p)
    else:  # 'rs' - split radix
        fft_a = fft_split_radix_recursive(padded_a, tw, p)
    
    # --- Forward FFT for poly_b ---
    features_b = compute_features(poly_b, fft_size)
    pred_b = model.predict([features_b])[0]
    algo_b = label_encoder.inverse_transform([pred_b])[0] if hasattr(pred_b, '__int__') or isinstance(pred_b, (int, np.integer)) else pred_b
    info['algo_forward_b'] = algo_b
    
    if algo_b == 'r2':
        fft_b = fft_radix2_recursive(padded_b, tw, p)
    elif algo_b == 'r4':
        fft_b = fft_radix4_recursive(padded_b, tw, p)
    else:  # 'rs'
        fft_b = fft_split_radix_recursive(padded_b, tw, p)
    
    # --- Point-wise multiplication ---
    fft_c = [mul_mod(fft_a[i], fft_b[i], p) for i in range(fft_size)]
    
    # --- Inverse FFT for result ---
    # Use features of the product for inverse selection
    # (In practice, the product is dense, so features will be different)
    features_c = compute_features(fft_c, fft_size)
    pred_c = model.predict([features_c])[0]
    algo_c = label_encoder.inverse_transform([pred_c])[0] if hasattr(pred_c, '__int__') or isinstance(pred_c, (int, np.integer)) else pred_c
    info['algo_inverse'] = algo_c
    
    if algo_c == 'r2':
        result = ifft_radix2_recursive(fft_c, inv_tw, p, n_inv, True)
    elif algo_c == 'r4':
        result = ifft_radix4_recursive(fft_c, inv_tw, p, n_inv, True)
    else:  # 'rs'
        result = ifft_split_radix_recursive(fft_c, inv_tw, p, n_inv, True)
    
    # Trim to actual result length
    result = result[:result_len]
    info['result'] = result
    
    return result, info

def schoolbook_multiply(a: List[int], b: List[int], p: int) -> List[int]:
    """Reference: schoolbook polynomial multiplication for verification"""
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] = (result[i + j] + ai * bj) % p
    return result

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Prime modulus (Fermat prime F4)
    P = 65537
    
    # Load the trained model and label encoder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best_model.joblib")
    encoder_path = os.path.join(script_dir, "label_encoder.joblib")
    
    print("Loading trained model...")
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print(f"Model loaded: {type(model).__name__}")
    print(f"Labels: {list(label_encoder.classes_)}")
    
    # Define two polynomials in vector (coefficient) form
    # poly_a = a_0 + a_1*x + a_2*x^2 + ... 
    poly_a = [10, 20, 30, 40, 50, 60, 70, 80]       # 8 coefficients
    poly_b = [100, 200, 300, 400, 500, 600, 700, 800]  # 8 coefficients
    
    print("\n" + "="*60)
    print("POLYNOMIAL MULTIPLICATION WITH ML-BASED ALGORITHM SELECTION")
    print("="*60)
    
    print(f"\nPolynomial A (coefficients): {poly_a}")
    print(f"Polynomial B (coefficients): {poly_b}")
    print(f"Prime modulus p = {P}")
    
    # Perform multiplication with model-based algorithm selection
    result, info = poly_multiply_with_model(poly_a, poly_b, P, model, label_encoder)
    
    print(f"\n--- Algorithm Selection ---")
    print(f"FFT size: {info['fft_size']} (result needs {info['result_len']} coefficients)")
    print(f"Forward FFT for A: {info['algo_forward_a']}")
    print(f"Forward FFT for B: {info['algo_forward_b']}")
    print(f"Inverse FFT: {info['algo_inverse']}")
    
    print(f"\n--- Result ---")
    print(f"Product polynomial (coefficients): {result}")
    
    # Verify with schoolbook multiplication
    expected = schoolbook_multiply(poly_a, poly_b, P)
    print(f"\n--- Verification (schoolbook) ---")
    print(f"Expected: {expected}")
    
    if result == expected:
        print("\n✓ CORRECT! FFT multiplication matches schoolbook result.")
    else:
        print("\n✗ MISMATCH! Something went wrong.")
        print(f"Difference at indices: {[i for i in range(len(result)) if result[i] != expected[i]]}")
    
    # =========================================================================
    # TEST 10 DIFFERENT POLYNOMIAL PAIRS
    # =========================================================================
    print("\n" + "="*70)
    print("TESTING 10 DIFFERENT POLYNOMIAL PAIRS")
    print("="*70)
    
    import random
    random.seed(42)  # For reproducibility
    
    test_cases = [
        # (name, poly_a, poly_b)
        ("Dense small", 
         [1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 2, 0, 4, 0,6, 0, 8, 0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 2, 0, 4, 0, 6, 0, 8, 0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 2, 0, 4, 0, 6, 0, 8, 0, 1, 0, 3, 0 ,5, 0, 7, 0, 9, 0], 
         [5, 6, 7, 8]),
        
        ("Dense medium", 
         [10, 20, 30, 40, 50, 60, 70, 80], 
         [1, 2, 3, 4, 5, 6, 7, 8]),
        
        ("Sparse (every 2nd)", 
         [1, 0, 2, 0, 3, 0, 4, 0], 
         [5, 0, 6, 0, 7, 0, 8, 0]),
        
        ("Sparse (every 4th)", 
         [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0], 
         [5, 0, 0, 0, 6, 0, 0, 0]),
        
        ("Mixed sparse/dense", 
         [1, 2, 3, 4, 5, 6, 7, 8], 
         [1, 0, 0, 0, 2, 0, 0, 0]),
        
        ("Large dense (16 coeffs)", 
         list(range(1, 17)), 
         list(range(17, 33))),
        
        ("Large dense (32 coeffs)", 
         list(range(1, 33)), 
         list(range(33, 65))),
        
        ("Random small", 
         [random.randint(1, 100) for _ in range(8)], 
         [random.randint(1, 100) for _ in range(8)]),
        
        ("Random with zeros", 
         [random.choice([0, 0, random.randint(1, 50)]) for _ in range(16)], 
         [random.choice([0, 0, random.randint(1, 50)]) for _ in range(16)]),
        
        ("Single coefficients", 
         [7], 
         [11]),
    ]
    
    passed = 0
    failed = 0
    
    for i, (name, pa, pb) in enumerate(test_cases, 1):
        result_i, info_i = poly_multiply_with_model(pa, pb, P, model, label_encoder)
        expected_i = schoolbook_multiply(pa, pb, P)
        
        is_correct = (result_i == expected_i)
        status = "✓" if is_correct else "✗"
        
        if is_correct:
            passed += 1
        else:
            failed += 1
        
        print(f"\nTest {i:2d}: {name}")
        print(f"  poly_a ({len(pa):2d} coeffs): {pa[:8]}{'...' if len(pa) > 8 else ''}")
        print(f"  poly_b ({len(pb):2d} coeffs): {pb[:8]}{'...' if len(pb) > 8 else ''}")
        print(f"  FFT size: {info_i['fft_size']}")
        print(f"  Algorithms: fwd_a={info_i['algo_forward_a']}, fwd_b={info_i['algo_forward_b']}, inv={info_i['algo_inverse']}")
        print(f"  Result: {result_i[:8]}{'...' if len(result_i) > 8 else ''}")
        print(f"  {status} {'CORRECT' if is_correct else 'MISMATCH'}")
    
    print("\n" + "="*70)
    print(f"SUMMARY: {passed}/{len(test_cases)} tests passed, {failed} failed")
    print("="*70)
    
    # =========================================================================
    # TEST 1000 RANDOMLY GENERATED POLYNOMIALS - STATISTICS ONLY
    # =========================================================================
    print("\n" + "="*70)
    print("TESTING 1000 RANDOMLY GENERATED POLYNOMIAL PAIRS")
    print("="*70)
    
    from collections import Counter
    
    random.seed(2026)  # For reproducibility
    
    combo_counts = Counter()  # Count each (fwd_a, fwd_b, inv) combination
    passed_1000 = 0
    failed_1000 = 0
    
    for _ in range(1000):
        # Random polynomial sizes (4 to 64 coefficients)
        size_a = random.choice([4, 8, 16, 32, 64])
        size_b = random.choice([4, 8, 16, 32, 64])
        
        # Random sparsity: 0=dense, 1=50% zeros, 2=75% zeros
        sparsity = random.choice([0, 1, 2])
        
        if sparsity == 0:
            # Dense polynomials
            pa = [random.randint(1, 1000) for _ in range(size_a)]
            pb = [random.randint(1, 1000) for _ in range(size_b)]
        elif sparsity == 1:
            # 50% zeros
            pa = [random.choice([0, random.randint(1, 1000)]) for _ in range(size_a)]
            pb = [random.choice([0, random.randint(1, 1000)]) for _ in range(size_b)]
        else:
            # 75% zeros
            pa = [random.choice([0, 0, 0, random.randint(1, 1000)]) for _ in range(size_a)]
            pb = [random.choice([0, 0, 0, random.randint(1, 1000)]) for _ in range(size_b)]
        
        result_i, info_i = poly_multiply_with_model(pa, pb, P, model, label_encoder)
        expected_i = schoolbook_multiply(pa, pb, P)
        
        # Count algorithm combination
        combo = f"{info_i['algo_forward_a']}/{info_i['algo_forward_b']}/{info_i['algo_inverse']}"
        combo_counts[combo] += 1
        
        if result_i == expected_i:
            passed_1000 += 1
        else:
            failed_1000 += 1
    
    print(f"\nCorrectness: {passed_1000}/1000 passed, {failed_1000} failed")
    print("\n" + "-"*50)
    print("Algorithm Combination Statistics (fwd_a/fwd_b/inv)")
    print("-"*50)
    
    # Sort by count descending
    for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1]):
        pct = count / 10  # percentage out of 1000
        bar = "█" * int(pct / 2)  # visual bar
        print(f"  {combo:12s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print("-"*50)
    print(f"Total unique combinations: {len(combo_counts)}/27 possible")
    print("="*70)

if __name__ == "__main__":
    main()

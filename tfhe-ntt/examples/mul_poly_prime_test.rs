use rand::random;
use std::time::Instant;
use tfhe_ntt::prime32::Plan;
use tfhe_ntt::prime32_r4::Plan_r4;

fn main() {
    // define suitable NTT prime and polynomial size
    let p: u32 = 1073479681;
    let polynomial_size = 16384; // minimum supported size is 32 for SIMD

    // unwrapping is fine here because we know roots of unity exist for the combination
    // `(polynomial_size, p)`
    let lhs_poly: Vec<u32> = (0..polynomial_size).map(|_| random::<u32>() % p).collect();
    let rhs_poly: Vec<u32> = (0..polynomial_size).map(|_| random::<u32>() % p).collect();

    // println!("Left polynomial:  {:?}", lhs_poly);
    // println!("Right polynomial: {:?}", rhs_poly);

    // method 1: schoolbook algorithm
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

    // method 2: radix-2 NTT
    let plan_r2 = Plan::try_new(polynomial_size, p).unwrap();
    let mut lhs_r2 = lhs_poly.clone();
    let mut rhs_r2 = rhs_poly.clone();

    let start_r2 = Instant::now();
    plan_r2.fwd(&mut lhs_r2);
    plan_r2.fwd(&mut rhs_r2);
    for i in 0..polynomial_size {
        lhs_r2[i] = mul(lhs_r2[i], rhs_r2[i]);
    }
    plan_r2.inv(&mut lhs_r2);
    plan_r2.normalize(&mut lhs_r2);
    let duration_r2 = start_r2.elapsed();

    let lhs_r2_result = lhs_r2.clone();
    let negacyclic_expected = negacyclic_convolution.clone();

    if lhs_r2_result != negacyclic_expected {
        println!("\nWarning: radix-2 differs from schoolbook result");
    }

    // method 3: radix-4 NTT
    let plan_r4 = Plan_r4::try_new_r4(polynomial_size, p).unwrap();
    let mut lhs_r4 = lhs_poly;
    let mut rhs_r4 = rhs_poly;

    let start_r4 = Instant::now();
    // println!("1");
    plan_r4.fwd_r4(&mut lhs_r4);
    plan_r4.fwd_r4(&mut rhs_r4);
    for i in 0..polynomial_size {
        lhs_r4[i] = mul(lhs_r4[i], rhs_r4[i]);
    }
    plan_r4.inv_r4(&mut lhs_r4);
    plan_r4.normalize_r4(&mut lhs_r4);
    let duration_r4 = start_r4.elapsed();

    // compare results
    let match_r2_r4 = lhs_r2_result == lhs_r4;

    // println!("\nRadix-2 result: {:?}", lhs_r2_result);
    // println!("Radix-4 result: {:?}", lhs_r4);
    println!("Radix-2 duration (lines 40-46): {:?}", duration_r2);
    println!("Radix-4 duration (lines 60-66): {:?}", duration_r4);

    if match_r2_r4 {
        println!("Success: radix-2 and radix-4 match!");
    } else {
        println!("Mismatch: radix-2 and radix-4 differ.");
    }
}

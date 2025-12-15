
use rand::random;
use std::time::Instant;

use tfhe_ntt::prime32::Plan;
use tfhe_ntt::custum_radix::{
    fft_radix4_recursive,
    fft_radix2_recursive,
    fft_split_radix_recursive,
    ifft_radix4_recursive,
    ifft_radix2_recursive,
    ifft_split_radix_recursive,
    
};

/// Dummy enum for now
#[derive(Debug, Copy, Clone)]
enum NttRadix {
    Radix2,
    Radix4,
    SplitRadix,
}

/// Dummy AI model: for now always returns RADIX-4
fn ai_predict_radix(_lhs: &[u32], _rhs: &[u32]) -> NttRadix {
    NttRadix::Radix4
}


fn next_power_of_two(x: usize) -> usize {
    x.next_power_of_two()
}

fn mod_pow(mut base: u32, mut exp: u32, modu: u32) -> u32 {
    let mut result = 1u32;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result as u64 * base as u64 % modu as u64) as u32;
        }
        base = (base as u64 * base as u64 % modu as u64) as u32;
        exp >>= 1;
    }
    result
}

fn mod_inv(n: u32, p: u32) -> u32 {
    mod_pow(n, p - 2, p)
}

fn main() {
    let p: u32 = 1073479681;
    let n1 = 32;
    let n2 = 32;

    // Pad polynomials to degree d1+d2
    //let n = n1 + n2;
    let n = next_power_of_two(n1 + n2);
    let mut lhs: Vec<u32> = (0..n1).map(|_| random::<u32>() % p).collect();
    let mut rhs: Vec<u32> = (0..n2).map(|_| random::<u32>() % p).collect();
    lhs.resize(n, 0);
    rhs.resize(n, 0);

    


    // // PRIME32 PLAN baseline
    let plan = Plan::try_new(n, p).unwrap();
    let twiddles = plan.twiddles();
    let inv_twiddles = plan.inv_twiddles();


    let mut a1 = lhs.clone();
    let mut b1 = rhs.clone();
    println!("a1 before NTT: {:?}, and b1: {:?}", &a1, &b1);
    let t0 = Instant::now();
    plan.fwd(&mut a1);
    plan.fwd(&mut b1);
    //println!("value of a1 after NTT: {:?}", &a1);
    plan.mul_assign_normalize(&mut a1, &b1);
    plan.inv(&mut a1);
    let total_tfhe = t0.elapsed();




    // CUSTOM RADIX version
    let mut a2 = lhs.clone();
    let mut b2 = rhs.clone();
    //println!("a2 before NTT: {:?}, and b2: {:?}", &a2, &b2);
    let radix_choice = ai_predict_radix(&a2, &b2);
    let t1 = Instant::now();

    

    match radix_choice {
        NttRadix::Radix4 => {
            fft_radix4_recursive(&mut a2, &twiddles, p);
            fft_radix4_recursive(&mut b2, &twiddles, p);
        }
        NttRadix::Radix2 => {
            fft_radix2_recursive(&mut a2, &twiddles, p);
            fft_radix2_recursive(&mut b2, &twiddles, p);
        }
        NttRadix::SplitRadix => {
            fft_split_radix_recursive(&mut a2, &twiddles, p);
            fft_split_radix_recursive(&mut b2, &twiddles, p);
        }
    }
    //println!("value of a2 after NTT: {:?}", &a2);
    plan.mul_assign_normalize(&mut a2, &b2);
    //plan.inv(&mut a2);
    let n_inv = mod_inv(n as u32, p);


    match radix_choice {
        NttRadix::Radix2 => {
            ifft_radix2_recursive(&mut a2, &inv_twiddles, p , n_inv, true);
        }
        NttRadix::Radix4 => {
            ifft_radix4_recursive(&mut a2, &inv_twiddles, p , n_inv, true);
        }
        NttRadix::SplitRadix => {
            ifft_split_radix_recursive(&mut a2, &inv_twiddles, p , n_inv, true);
        }
    }
    let total_custom = t1.elapsed();

    //println!("a1: {:?}", &a1[..16]);
    //println!("a2: {:?}", &a2);

    println!("\n-- TFHE prime32::Plan -- TOTAL: {:>8?}", total_tfhe);
    println!("-- Custom Radix (AI predicted) -- TOTAL: {:>8?}, Radix used: {:?}", total_custom, radix_choice);

    if a1 == a2 {
        println!("\n✔ OUTPUT MATCHES (Custom Radix == Prime32)");
    } else {
        println!("\n✘ WARNING: output mismatch!");
    }


    //a cryptanalysis

    // let mut a3 = lhs.clone();
    // println!("value of a3 before NTT: {:?}", &a3);
    // plan.fwd(&mut a3);
    // plan.inv(&mut a3); 
    // println!("value of a3 after inverse NTT: {:?}", &a3);



    let mut a4 = lhs.clone();
    println!("value of a4 before NTT: {:?}", &a4);
    fft_radix4_recursive(&mut a4, &twiddles, p);
    ifft_radix4_recursive(&mut a4, &inv_twiddles, p, n_inv, true); 
    println!("value of a4 after inverse NTT: {:?}", &a4);
}


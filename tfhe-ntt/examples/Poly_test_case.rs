use std::time::Instant;
use rand::Rng;
use tfhe_ntt::custum_radix::{
    fft_radix4_recursive,
    fft_radix2_recursive,
    fft_split_radix_recursive,

    fft_radix4_recursive_mut,
    fft_radix2_recursive_mut,
    fft_split_radix_recursive_mut,

    ifft_radix4_recursive,
    ifft_radix2_recursive,
    ifft_split_radix_recursive,
};

use tfhe_ntt::custum_radix::fwd_1::MultStats;

// Import your FFT functions
// use crate::custom_radix::fwd::{radix2_fft, radix4_fft, split_radix_fft};
// use crate::custom_radix::inv::{radix2_ifft, radix4_ifft, split_radix_ifft};

// fn generate_polynomial(size: usize) -> Vec<i64> {
//     let mut rng = rand::thread_rng();
//     (0..size).map(|_| rng.gen_range(-10..10)).collect()
// }

// fn generate_odd_zero_polynomial(size: usize) -> Vec<i64> {
//     let mut rng = rand::thread_rng();
//     (0..size)
//         .map(|i| if i % 2 == 1 { 0 } else { rng.gen_range(-10..10) })
//         .collect()
// }



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


fn main() {
    println!("FFT Algorithm Benchmark\n");
    
    // //For N=4
    //case-1
    let mut poly3: Vec<u32> = vec![6, 5, 0, 11];

    // //case-2 
    // let mut poly3: Vec<u32> = vec![0, 0, 9, 0];

    // //case-3
    // let mut poly3: Vec<u32> = vec![0, 5, 0, 0];

    // //case-4
    // let mut poly3: Vec<u32> = vec![12, 0, 0, 0];


    // //case-5
    // let mut poly3: Vec<u32> = vec![0, 0, 0, 14];

    // //case-6
    // let mut poly3: Vec<u32> = vec![3, 11, 0, 0];

    // //case-7
    // let mut poly3: Vec<u32> = vec![7, 0, 4, 0];


    // //case-8
    // let mut poly3: Vec<u32> = vec![13, 0, 0, 6];


    // //case-9
    // let mut poly3: Vec<u32> = vec![0, 10, 8, 0];


    // //case-10
    // let mut poly3: Vec<u32> = vec![0, 2, 0, 15];


    // //case-11
    // let mut poly3: Vec<u32> = vec![0, 0, 7, 9];


    // //case-12
    // let mut poly3: Vec<u32> = vec![4, 3, 12, 0];


    // //case-13
    // let mut poly3: Vec<u32> = vec![6, 0, 5, 11];


    // //case-14
    // let mut poly3: Vec<u32> = vec![0, 9, 3, 7];


    // //case-15
    // let mut poly3: Vec<u32> = vec![5, 8, 1, 4];


    // //case-15
    // let mut poly3: Vec<u32> = vec![
    // 1, 7, 9, 8
    // ];

    //For  N=64
    //Test Case 1: Your original polynomial
        let mut poly3: Vec<u32> = vec![
        5, 11, 3, 12, 8, 13, 2, 14,
        4, 15, 7, 16, 6, 17, 1, 18,
        3, 19, 9, 20, 2, 21, 5, 22,
        7, 23, 4, 24, 1, 25, 8, 26,
        9, 27, 6, 28, 3, 29, 5, 30,
        2, 31, 8, 32, 4, 33, 7, 34,
        1, 35, 6, 36, 3, 37, 9, 38,
        2, 39, 5, 40, 7, 41, 4, 42,
    ];

    //Test Case 2: odd terms of odd terms of odd terms are zero (index % 8 == 7)
    // let mut poly3: Vec<u32> = vec![
    // 1, 2, 3, 4, 5, 6, 7, 0,        // index 7 = 0
    // 9, 10, 11, 12, 13, 14, 15, 0,  // index 15 = 0
    // 17, 18, 19, 20, 21, 22, 23, 0, // index 23 = 0
    // 25, 26, 27, 28, 29, 30, 31, 0, // index 31 = 0
    // 33, 34, 35, 36, 37, 38, 39, 0, // index 39 = 0
    // 41, 42, 43, 44, 45, 46, 47, 0, // index 47 = 0
    // 49, 50, 51, 52, 53, 54, 55, 0, // index 55 = 0
    // 57, 58, 59, 60, 61, 62, 63, 0, // index 63 = 0
    // ];

// //  Test Case 3: odd terms of odd terms are zero (index % 4 == 3)
//     let mut poly3: Vec<u32> = vec![
//     1, 2, 3, 0,      // 0–3
//     5, 6, 7, 0,      // 4–7
//     9, 10, 11, 0,    // 8–11
//     13, 14, 15, 0,   // 12–15
//     17, 18, 19, 0,   // 16–19
//     21, 22, 23, 0,   // 20–23
//     25, 26, 27, 0,   // 24–27
//     29, 30, 31, 0,   // 28–31
//     33, 34, 35, 0,   // 32–35
//     37, 38, 39, 0,   // 36–39
//     41, 42, 43, 0,   // 40–43
//     45, 46, 47, 0,   // 44–47
//     49, 50, 51, 0,   // 48–51
//     53, 54, 55, 0,   // 52–55
//     57, 58, 59, 0,   // 56–59
//     61, 62, 63, 0,   // 60–63
//   ];


    //Test Case 4: even terms of odd terms are zero (1,5,9,13,...)
    // let mut poly3: Vec<u32> = vec![
    //     1, 0, 3, 4,       // 0–3
    //     5, 0, 7, 8,       // 4–7
    //     9, 0, 11, 12,     // 8–11
    //     13, 0, 15, 16,    // 12–15
    //     17, 0, 19, 20,    // 16–19
    //     21, 0, 23, 24,    // 20–23
    //     25, 0, 27, 28,    // 24–27
    //     29, 0, 31, 32,    // 28–31
    //     33, 0, 35, 36,    // 32–35
    //     37, 0, 39, 40,    // 36–39
    //     41, 0, 43, 44,    // 40–43
    //     45, 0, 47, 48,    // 44–47
    //     49, 0, 51, 52,    // 48–51
    //     53, 0, 55, 56,    // 52–55
    //     57, 0, 59, 60,    // 56–59
    //     61, 0, 63, 64,    // 60–63
    // ];

//     //Test Case 5: odd terms are zero (1,3,5,7,9,11,13,...)
//     let mut poly3: Vec<u32> = vec![
//     5, 0, 3, 0, 8, 0, 2, 0,
//     4, 0, 7, 0, 6, 0, 1, 0,
//     3, 0, 9, 0, 2, 0, 5, 0,
//     7, 0, 4, 0, 1, 0, 8, 0,
//     9, 0, 6, 0, 3, 0, 5, 0,
//     2, 0, 8, 0, 4, 0, 7, 0,
//     1, 0, 6, 0, 3, 0, 9, 0,
//     2, 0, 5, 0, 7, 0, 4, 0,
//    ];

//     //Test Case 6: all odd terms are zero and odd terms of even terms are zero
//     let mut poly3: Vec<u32> = vec![
//     5, 0, 0, 0,
//     3, 0, 0, 0,
//     8, 0, 0, 0,
//     2, 0, 0, 0,
//     4, 0, 0, 0,
//     7, 0, 0, 0,
//     6, 0, 0, 0,
//     1, 0, 0, 0,
//     9, 0, 0, 0,
//     3, 0, 0, 0,
//     2, 0, 0, 0,
//     5, 0, 0, 0,
//     7, 0, 0, 0,
//     4, 0, 0, 0,
//     1, 0, 0, 0,
//     8, 0, 0, 0,
//    ];

//Test Case 7: all odd terms are zero and even terms of even terms are zero
// let mut poly3: Vec<u32> = vec![
//     0, 0, 5, 0,
//     0, 0, 3, 0,
//     0, 0, 8, 0,
//     0, 0, 2, 0,
//     0, 0, 4, 0,
//     0, 0, 7, 0,
//     0, 0, 6, 0,
//     0, 0, 1, 0,
//     0, 0, 9, 0,
//     0, 0, 3, 0,
//     0, 0, 2, 0,
//     0, 0, 5, 0,
//     0, 0, 7, 0,
//     0, 0, 4, 0,
//     0, 0, 1, 0,
//     0, 0, 8, 0,
// ];


// //Test Case 8: ai%4 ==1 are non zero
// let mut poly3: Vec<u32> = vec![
//     0, 5, 0, 0,
//     0, 3, 0, 0,
//     0, 8, 0, 0,
//     0, 2, 0, 0,
//     0, 4, 0, 0,
//     0, 7, 0, 0,
//     0, 6, 0, 0,
//     0, 1, 0, 0,
//     0, 9, 0, 0,
//     0, 3, 0, 0,
//     0, 2, 0, 0,
//     0, 5, 0, 0,
//     0, 7, 0, 0,
//     0, 4, 0, 0,
//     0, 1, 0, 0,
//     0, 8, 0, 0,
// ];

// //Test Case 9: ai and ai+N/2 are same
// let mut poly3: Vec<u32> = vec![
//     0,10,0,0,   0,20,0,0,
//     0,30,0,0,   0,40,0,0,
//     0,50,0,0,   0,60,0,0,
//     0,70,0,0,   0,80,0,0,

//     // second half = same as first half
//     0,10,0,0,   0,20,0,0,
//     0,30,0,0,   0,40,0,0,
//     0,50,0,0,   0,60,0,0,
//     0,70,0,0,   0,80,0,0,
// ];

// // //Test Case 10: i%8==0 position is non zero only
// let mut poly3: Vec<u32> = vec![
//     10, 0, 0, 0, 0, 0, 0, 0,   // 0..7   -> index 0 = 10
//     20, 0, 0, 0, 0, 0, 0, 0,   // 8..15  -> index 8 = 20
//     30, 0, 0, 0, 0, 0, 0, 0,   // 16..23 -> index 16 = 30
//     40, 0, 0, 0, 0, 0, 0, 0,   // 24..31 -> index 24 = 40
//     50, 0, 0, 0, 0, 0, 0, 0,   // 32..39 -> index 32 = 50
//     60, 0, 0, 0, 0, 0, 0, 0,   // 40..47 -> index 40 = 60
//     70, 0, 0, 0, 0, 0, 0, 0,   // 48..55 -> index 48 = 70
//     80, 0, 0, 0, 0, 0, 0, 0,   // 56..63 -> index 56 = 80
// ];

// //Test Case 11: i%4==3 position is non zero only
// let mut poly3: Vec<u32> = vec![
//     0,0,0,10,
//     0,0,0,20,
//     0,0,0,30,
//     0,0,0,40,
//     0,0,0,50,
//     0,0,0,60,
//     0,0,0,70,
//     0,0,0,80,

//     0,0,0,90,
//     0,0,0,100,
//     0,0,0,110,
//     0,0,0,120,
//     0,0,0,130,
//     0,0,0,140,
//     0,0,0,150,
//     0,0,0,160,
// ];

// //Test Case 12: only even terms are zero
// let mut poly3: Vec<u32> = vec![
//     0, 1, 0, 2, 0, 3, 0, 4,
//     0, 5, 0, 6, 0, 7, 0, 8,
//     0, 9, 0, 10, 0, 11, 0, 12,
//     0, 13, 0, 14, 0, 15, 0, 16,
//     0, 17, 0, 18, 0, 19, 0, 20,
//     0, 21, 0, 22, 0, 23, 0, 24,
//     0, 25, 0, 26, 0, 27, 0, 28,
//     0, 29, 0, 30, 0, 31, 0, 32,
// ];

// //Test Case 13: all odd terms are non zero and odd terms of even terms are zero
// let mut poly3: Vec<u32> = vec![
//     100, 1,   0,   2,
//     200, 3,   0,   4,
//     300, 5,   0,   6,
//     400, 7,   0,   8,

//     500, 9,   0,  10,
//     600,11,   0,  12,
//     700,13,   0,  14,
//     800,15,   0,  16,

//     900,17,   0,  18,
//     1000,19,  0,  20,
//     1100,21,  0,  22,
//     1200,23,  0,  24,

//     1300,25,  0,  26,
//     1400,27,  0,  28,
//     1500,29,  0,  30,
//     1600,31,  0,  32,
// ];

//Test Case 14: aeven of even terms are 0 and even of odd terms are 0
// let mut poly3: Vec<u32> = vec![
//     // 0–7
//     0,0,1,1,0,0,1,1,
//     // 8–15
//     0,0,1,1,0,0,1,1,
//     // 16–23
//     0,0,1,1,0,0,1,1,
//     // 24–31
//     0,0,1,1,0,0,1,1,
//     // 32–39
//     0,0,1,1,0,0,1,1,
//     // 40–47
//     0,0,1,1,0,0,1,1,
//     // 48–55
//     0,0,1,1,0,0,1,1,
//     // 56–63
//     0,0,1,1,0,0,1,1,
// ];





        let p: u32 = 2013;
        let n: usize = 64;
        let g = compute_primitive_root(p);
        let root = pow_mod(g, (p - 1) / (n as u32), p);
        let twiddles = make_twiddles_from_root(root, n, p);
        let mut stats = MultStats::default();
    
    // Benchmark each test case
    
        println!("Test Case {}: Size = {}", 64, poly3.len());


        // // Calculate average of last 10 runs
        // let avg_radix4 = radix4_times.iter().sum::<std::time::Duration>() / radix4_times.len() as u32;
        // println!("  Radix-4 average (10 runs): {:?}", avg_radix4);

        // //Split-Radix
        // let start = Instant::now();
        // fft_split_radix_recursive_mut(&mut poly3, &twiddles, p, &mut stats);
        // // let reconstructed = split_radix_ifft(&result);
        // let time_split = start.elapsed();
        // // println!("  Split-Radix:  {:?}µs", time_split.as_micros());
        // println!("  Split-Radix:  {:?}", time_split);
        
        //Radix-2
        let start = Instant::now();
        fft_radix2_recursive_mut(&mut poly3, &twiddles, p, &mut stats);
        // let reconstructed = radix2_ifft(&result);
        let time_radix2 = start.elapsed();
        //println!("  Radix-2:      {:?}µs", time_radix2.as_micros());
        println!("  Radix-2:      {:?}", time_radix2);

        // //Radix-4
        // let start = Instant::now();
        // fft_radix4_recursive_mut(&mut poly3, &twiddles, p, &mut stats);
        // // let reconstructed = radix4_ifft(&result);
        // let time_radix4 = start.elapsed();
        // // println!("  Radix-4:      {:?}µs", time_radix4.as_micros());
        // println!("  Radix-4:      {:?}", time_radix4);


        // let start = Instant::now();
        // let time_radix2 = start.elapsed();
        // println!("  Radix-2:      {:?}", time_radix2);



        println!("Results: {:?}", poly3);
        println!("Nonzero mults: {}", stats.nonzero_mults);
        //println!("Skipped mults: {}", stats.skipped_mults);
        
        
        
        println!();
    
}
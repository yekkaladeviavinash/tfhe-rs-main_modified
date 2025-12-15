use rand::random;
use tfhe_ntt::prime32::Plan;

// bring your custom NTT module
use tfhe_ntt::custum_radix::forward_ntt;

fn main() {
    // ------------------------------------
    // parameters
    // ------------------------------------
    let p: u32 = 1073479681;
    let polynomial_size = 32;

    // generate random polys
    let lhs_poly: Vec<u32> = (0..polynomial_size).map(|_| random::<u32>() % p).collect();
    let rhs_poly: Vec<u32> = (0..polynomial_size).map(|_| random::<u32>() % p).collect();

    // ------------------------------------
    // method 1: schoolbook negacyclic
    // ------------------------------------
    let add = |x: u32, y: u32| ((x as u64 + y as u64) % p as u64) as u32;
    let sub = |x: u32, y: u32| add(x, p - y);
    let mul = |x: u32, y: u32| ((x as u64 * y as u64) % p as u64) as u32;

    let mut full_convolution = vec![0; 2 * polynomial_size];
    for i in 0..polynomial_size {
        for j in 0..polynomial_size {
            full_convolution[i + j] =
                add(full_convolution[i + j], mul(lhs_poly[i], rhs_poly[j]));
        }
    }

    let mut negacyclic_convolution = vec![0; polynomial_size];
    for i in 0..polynomial_size {
        negacyclic_convolution[i] =
            sub(full_convolution[i], full_convolution[polynomial_size + i]);
    }

    // ------------------------------------
    // method 2: custom-radix NTT
    // ------------------------------------
    let plan = Plan::try_new(polynomial_size, p).unwrap();

    // clone polys
    let mut lhs_ntt = lhs_poly.clone();
    let mut rhs_ntt = rhs_poly.clone();

    // custom forward NTT
    // forward_ntt(
    //     &mut lhs_ntt,
    //     polynomial_size,
    //     p,
    //     &plan.twiddles(),   // use plan’s twiddles
    // );

    // forward_ntt(
    //     &mut rhs_ntt,
    //     polynomial_size,
    //     p,
    //     &plan.twiddles(),
    // );

    // 
    plan.fwd(&mut lhs_ntt);
    plan.fwd(&mut rhs_ntt);


    // elementwise multiply + normalization using plan's code
    plan.mul_assign_normalize(&mut lhs_ntt, &rhs_ntt);

    // standard inverse NTT from TFHE-rs
    plan.inv(&mut lhs_ntt);

    // compare
    assert_eq!(lhs_ntt, negacyclic_convolution);

    println!("Success! Custom radix NTT agrees with schoolbook result.");
}

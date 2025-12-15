use rand::Rng;

/// Generate a random polynomial of given size
fn generate_polynomial(size: usize) -> Vec<i64> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| rng.gen_range(-10..10)) // random small coefficients
        .collect()
}

/// Compute polynomial degree (highest non-zero index)
fn polynomial_degree(poly: &[i64]) -> usize {
    match poly.iter().rposition(|&x| x != 0) {
        Some(pos) => pos,
        None => 0, // all zero polynomial
    }
}

/// Compute sparsity = (#zeros / total_size)
fn polynomial_sparsity(poly: &[i64]) -> f64 {
    let zero_count = poly.iter().filter(|&&x| x == 0).count();
    zero_count as f64 / poly.len() as f64
}

/// Check if n is a power of 2
fn is_power_of_2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Check if n is a power of 4
fn is_power_of_4(n: usize) -> bool {
    is_power_of_2(n) && (n.trailing_zeros() % 2 == 0)
}

/// Distance to next power of 2
fn dist_to_next_pow2(n: usize) -> usize {
    if is_power_of_2(n) {
        return 0;
    }
    n.next_power_of_two() - n
}

/// Print properties of any polynomial
fn print_properties(name: &str, poly: &[i64]) {
    let size = poly.len();
    let degree = polynomial_degree(poly);
    let sparsity = polynomial_sparsity(poly);
    let dist = dist_to_next_pow2(size);
    let pow2 = is_power_of_2(size) as i32;
    let pow4 = is_power_of_4(size) as i32;

    println!("==============================");
    println!(" {} (size {})", name, size);
    println!("==============================");
    println!("Degree           : {}", degree);
    println!("Sparsity         : {:.4}", sparsity);
    println!("dist_to_next_pow2: {}", dist);
    println!("is_power_2       : {}", pow2);
    println!("is_power_4       : {}", pow4);
    println!();
}

/// Pad both polynomials to same FFT size
fn pad_for_convolution(a: &[i64], b: &[i64]) -> (Vec<i64>, Vec<i64>, usize) {
    let deg_a = polynomial_degree(a);
    let deg_b = polynomial_degree(b);

    // required convolution degree = deg_a + deg_b
    let final_degree = deg_a + deg_b;

    // required coefficient length = final_degree + 1
    let required_len = (final_degree + 1).next_power_of_two();

    // clone and resize
    let mut a_pad = a.to_vec();
    let mut b_pad = b.to_vec();
    a_pad.resize(required_len, 0);
    b_pad.resize(required_len, 0);

    (a_pad, b_pad, required_len)
}

fn main() {
    // ------------------------------------------
    // 1) Random polynomial A
    // ------------------------------------------
    let poly_a = generate_polynomial(120);

    // ------------------------------------------
    // 2) Your manually-given polynomial B
    // ------------------------------------------
    let poly_b: Vec<i64> = vec![
        136997769, 477663868, 1023283482, 510194523, 800871149, 455961778,
        394853285, 504271174, 761672772, 492941039, 232065973, 891930045,
        330156215, 697161753, 258056117, 537811432, 1050831244, 344598407,
        1011531881, 800932303, 450420035, 59259407, 392424004, 187445538,
        52390080, 21194045, 564680553, 609094682, 446883297, 528253075,
        1054301274, 44191645,
    ];

    // ------------------------------------------
    // 3) Print original properties
    // ------------------------------------------
    print_properties("Original Polynomial A", &poly_a);
    print_properties("Original Polynomial B", &poly_b);

    // ------------------------------------------
    // 4) Pad A and B for FFT multiplication
    // ------------------------------------------
    let (a_padded, b_padded, fft_len) = pad_for_convolution(&poly_a, &poly_b);

    println!("Required FFT Size = {}", fft_len);

    // ------------------------------------------
    // 5) Print padded properties
    // ------------------------------------------
    print_properties("Padded Polynomial A", &a_padded);
    print_properties("Padded Polynomial B", &b_padded);
    println!("a padded: {:?}", &a_padded);
    // Now you can pass: 
    //   size = fft_len
    //   sparsity = polynomial_sparsity(...)
    //   dist_to_next_pow2(fft_len)
    //   is_power_2(fft_len)
    //   is_power_4(fft_len)
    //
    // to your Python AI model.
}

use rand::Rng;
use std::process::Command;
use serde::{Deserialize, Serialize};

// ============ Polynomial Helper Functions ============

fn generate_polynomial(size: usize) -> Vec<i64> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-10..10)).collect()
}

fn polynomial_degree(poly: &[i64]) -> usize {
    poly.iter().rposition(|&x| x != 0).unwrap_or(0)
}

fn polynomial_sparsity(poly: &[i64]) -> f64 {
    let zero_count = poly.iter().filter(|&&x| x == 0).count();
    zero_count as f64 / poly.len() as f64
}

fn is_power_of_2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

fn is_power_of_4(n: usize) -> bool {
    is_power_of_2(n) && (n.trailing_zeros() % 2 == 0)
}

fn dist_to_next_pow2(n: usize) -> usize {
    if is_power_of_2(n) {
        return 0;
    }
    n.next_power_of_two() - n
}

// ============ ML Model Integration ============

#[derive(Serialize, Debug)]
struct PolynomialFeatures {
    polynomial_size: usize,
    sparsity: f64,
    dist_to_next_pow2: usize,
    is_power_2: i32,
    is_power_4: i32,
}

#[derive(Deserialize, Debug)]
struct MLPrediction {
    #[serde(rename = "Best Algorithm")]
    best_algorithm: String,
    
    #[serde(rename = "Confidence")]
    confidence: f64,
}

/// Call the Python ML model to predict the best FFT algorithm
fn predict_fft_algorithm(poly: &[i64]) -> Result<MLPrediction, Box<dyn std::error::Error>> {
    let size = poly.len();
    
    let features = PolynomialFeatures {
        polynomial_size: size,
        sparsity: polynomial_sparsity(poly),
        dist_to_next_pow2: dist_to_next_pow2(size),
        is_power_2: is_power_of_2(size) as i32,
        is_power_4: is_power_of_4(size) as i32,
    };
    
    println!("\n[DEBUG] Calling ML model with features:");
    println!("  Size: {}, Sparsity: {:.4}, Dist_to_pow2: {}, Power2: {}, Power4: {}",
        features.polynomial_size,
        features.sparsity,
        features.dist_to_next_pow2,
        features.is_power_2,
        features.is_power_4
    );

    let python_script = "/home/sanjib/NTT_radix/tfhe-rs-main_modified/ML_Model/predict_algorithm.py";
    
    // Build command arguments
    let output = Command::new("python3")
        .arg(python_script)  // Use absolute path
        .arg("--polynomial_size").arg(features.polynomial_size.to_string())
        .arg("--sparsity").arg(features.sparsity.to_string())
        .arg("--dist_to_next_pow2").arg(features.dist_to_next_pow2.to_string())
        .arg("--is_power_2").arg(features.is_power_2.to_string())
        .arg("--is_power_4").arg(features.is_power_4.to_string())
        .output()?;
    
    if !output.status.success() {
        let error_msg = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python prediction failed: {}", error_msg).into());
    }
    
    // Parse the output
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("[DEBUG] Python output:\n{}", stdout);
    
    // The Python script outputs JSON with the prediction
    let prediction: MLPrediction = serde_json::from_str(&stdout)?;
    
    Ok(prediction)
}

// ============ FFT Algorithm Implementations ============
// (You'll implement these based on your actual FFT code)

fn perform_radix2_fft(poly: &[i64]) {
    println!("  → Executing Radix-2 FFT on {} coefficients", poly.len());
    // Your actual Radix-2 implementation here
}

fn perform_split_radix_fft(poly: &[i64]) {
    println!("  → Executing Split-Radix FFT on {} coefficients", poly.len());
    // Your actual Split-Radix implementation here
}

fn perform_modified_radix4_fft(poly: &[i64]) {
    println!("  → Executing Modified-Radix-4 FFT on {} coefficients", poly.len());
    // Your actual Modified-Radix-4 implementation here
}

/// Execute the appropriate FFT based on ML prediction
fn execute_predicted_algorithm(poly: &[i64], algorithm: &str) {
    match algorithm {
        "Radix-2" => perform_radix2_fft(poly),
        "Split-Radix" | "Radix-Split" => perform_split_radix_fft(poly),
        "Modified-Radix-4" => perform_modified_radix4_fft(poly),
        _ => println!("  ⚠ Unknown algorithm: {}", algorithm),
    }
}

// ============ Main Program ============

fn main() {
    println!("========================================");
    println!("  NTT Multiplier - ML-Guided FFT");
    println!("========================================\n");
    
    // Generate two test polynomials
    let poly_a = generate_polynomial(120);
    let poly_b: Vec<i64> = vec![
    0, 5, 0, 0,
    0, 3, 0, 0,
    0, 8, 0, 0,
    0, 2, 0, 0,
    0, 4, 0, 0,
    0, 7, 0, 0,
    0, 6, 0, 0,
    0, 1, 0, 0,
    0, 9, 0, 0,
    0, 3, 0, 0,
    0, 2, 0, 0,
    0, 5, 0, 0,
    0, 7, 0, 0,
    0, 4, 0, 0,
    0, 1, 0, 0,
    0, 8, 0, 0,
];
    // let poly_b: Vec<i64> = vec![
    //     136997769, 477663868, 1023283482, 510194523, 800871149, 455961778,
    //     394853285, 504271174, 761672772, 492941039, 232065973, 891930045,
    //     330156215, 697161753, 258056117, 537811432, 1050831244, 344598407,
    //     1011531881, 800932303, 450420035, 59259407, 392424004, 187445538,
    //     52390080, 21194045, 564680553, 609094682, 446883297, 528253075,
    //     1054301274, 44191645,
    // ];
    
    println!("Polynomial A: {} coefficients", poly_a.len());
    println!("Polynomial B: {} coefficients", poly_b.len());
    
    // ========== Process Polynomial A ==========
    println!("\n{}", "=".repeat(50));
    println!("PROCESSING POLYNOMIAL A");
    println!("{}", "=".repeat(50));
    
    match predict_fft_algorithm(&poly_a) {
        Ok(prediction) => {
            println!("\n✓ ML Prediction for Polynomial A:");
            println!("  Algorithm: {}", prediction.best_algorithm);
            println!("  Confidence: {:.2}%\n", prediction.confidence * 100.0);
            
            execute_predicted_algorithm(&poly_a, &prediction.best_algorithm);
        }
        Err(e) => {
            eprintln!("✗ ML prediction failed for Polynomial A: {}", e);
            println!("  Falling back to Radix-2 (default)");
            perform_radix2_fft(&poly_a);
        }
    }
    
    // ========== Process Polynomial B ==========
    println!("\n{}", "=".repeat(50));
    println!("PROCESSING POLYNOMIAL B");
    println!("{}", "=".repeat(50));
    
    match predict_fft_algorithm(&poly_b) {
        Ok(prediction) => {
            println!("\n✓ ML Prediction for Polynomial B:");
            println!("  Algorithm: {}", prediction.best_algorithm);
            println!("  Confidence: {:.2}%\n", prediction.confidence * 100.0);
            
            execute_predicted_algorithm(&poly_b, &prediction.best_algorithm);
        }
        Err(e) => {
            eprintln!("✗ ML prediction failed for Polynomial B: {}", e);
            println!("  Falling back to Radix-2 (default)");
            perform_radix2_fft(&poly_b);
        }
    }
    
    println!("\n{}", "=".repeat(50));
    println!("Completed");
    println!("{}", "=".repeat(50));
}
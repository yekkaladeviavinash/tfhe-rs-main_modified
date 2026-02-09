#!/usr/bin/env python3
"""
Predict the best FFT algorithm for given polynomial features.

Usage:
    python predict_algo.py <raw_n> <padded_n> <frac_2i> <frac_2i1> <frac_4i> <frac_4i1> <frac_4i2> <frac_4i3>

Returns:
    Prints the predicted algorithm: 'r2', 'r4', or 'rs'
"""

import sys
import os
import joblib
import numpy as np

def main():
    if len(sys.argv) != 9:
        print("Usage: predict_algo.py <raw_n> <padded_n> <frac_2i> <frac_2i1> <frac_4i> <frac_4i1> <frac_4i2> <frac_4i3>", file=sys.stderr)
        sys.exit(1)
    
    # Parse features from command line
    raw_n = float(sys.argv[1])
    padded_n = float(sys.argv[2])
    frac_2i = float(sys.argv[3])
    frac_2i1 = float(sys.argv[4])
    frac_4i = float(sys.argv[5])
    frac_4i1 = float(sys.argv[6])
    frac_4i2 = float(sys.argv[7])
    frac_4i3 = float(sys.argv[8])
    
    features = [[raw_n, padded_n, frac_2i, frac_2i1, frac_4i, frac_4i1, frac_4i2, frac_4i3]]
    
    # Load model and label encoder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best_model.joblib")
    encoder_path = os.path.join(script_dir, "label_encoder.joblib")
    
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    
    # Predict
    pred = model.predict(features)[0]
    
    # Decode prediction
    if hasattr(pred, '__int__') or isinstance(pred, (int, np.integer)):
        algo = label_encoder.inverse_transform([pred])[0]
    else:
        algo = pred
    
    # Print result (will be captured by Rust)
    print(algo)

if __name__ == "__main__":
    main()

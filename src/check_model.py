"""
Quick check that the model works
"""
import os
import sys

# Проверка зависимостей
try:
    import joblib
    import numpy as np
    from config import MODEL_PKL, SCALER_X_PKL, SCALER_Y_PKL
except ImportError as e:
    print("Error: Dependencies not installed!")
    print(f"   {e}")
    print("\nInstall dependencies:")
    print("   python3 -m pip install -r requirements.txt")
    print("   or")
    print("   pip3 install -r requirements.txt")
    print("\n   Or run: bash scripts/install_dependencies.sh")
    sys.exit(1)


def check_model_files():
    """Checks for all required files"""
    required_files = [MODEL_PKL, SCALER_X_PKL, SCALER_Y_PKL]
    missing = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"Found: {file.name}")
        else:
            print(f"Missing: {file.name}")
            missing.append(file)
    
    return len(missing) == 0


def test_model_loading():
    """Checks that the model loads"""
    try:
        print("\nLoading model...")
        model = joblib.load(MODEL_PKL)
        scaler_X = joblib.load(SCALER_X_PKL)
        scaler_y = joblib.load(SCALER_Y_PKL)
        print("Model loaded successfully!")
        return model, scaler_X, scaler_y
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def test_prediction(model, scaler_X, scaler_y):
    """Tests prediction on an example"""
    try:
        print("\nTesting prediction...")
        
        # Test example
        test_params = np.array([[2.0, 0.7, 0.15]])  # mass, friction, inertia
        
        # Normalization
        test_scaled = scaler_X.transform(test_params)
        
        # Prediction
        pred_scaled = model.predict(test_scaled)
        
        # Denormalization
        pred = scaler_y.inverse_transform(pred_scaled)
        
        print(f"Prediction successful!")
        print(f"\n   Input robot parameters:")
        print(f"   - Mass: {test_params[0][0]:.2f}")
        print(f"   - Friction: {test_params[0][1]:.2f}")
        print(f"   - Inertia: {test_params[0][2]:.2f}")
        print(f"\n   Predicted PID parameters:")
        print(f"   - Kp: {pred[0][0]:.4f}")
        print(f"   - Ki: {pred[0][1]:.4f}")
        print(f"   - Kd: {pred[0][2]:.4f}")
        
        return True
    except Exception as e:
        print(f"Prediction error: {e}")
        return False


def main():
    """Main check function"""
    print("="*50)
    print("MODEL CHECK")
    print("="*50)
    
    # Check files
    if not check_model_files():
        print("\nNot all files found. Train the model first:")
        print("   python3 src/train_model.py")
        return
    
    # Load model
    model, scaler_X, scaler_y = test_model_loading()
    if model is None:
        return
    
    # Test prediction
    if test_prediction(model, scaler_X, scaler_y):
        print("\n" + "="*50)
        print("MODEL WORKING - Ready to use")
        print("="*50)
        print("\nHow to use:")
        print("   1. Quick prediction:")
        print("      python3 src/predict_pid.py 2.0 0.7 0.15")
        print("\n   2. Interactive mode:")
        print("      python3 src/predict_pid.py")
        print("\n   3. Full testing:")
        print("      python3 src/test_model.py")
        print("\n   4. Experiments for paper:")
        print("      python3 src/experiments.py")


if __name__ == "__main__":
    main()


"""
Быстрая проверка что модель работает
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
    print("   или")
    print("   pip3 install -r requirements.txt")
    print("\n   Или запусти: bash scripts/install_dependencies.sh")
    sys.exit(1)


def check_model_files():
    """Проверяет наличие всех необходимых файлов"""
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
    """Проверяет что модель загружается"""
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
    """Тестирует предсказание на примере"""
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
        print(f"\n   Входные параметры робота:")
        print(f"   - Масса: {test_params[0][0]:.2f}")
        print(f"   - Трение: {test_params[0][1]:.2f}")
        print(f"   - Инерция: {test_params[0][2]:.2f}")
        print(f"\n   Предсказанные PID параметры:")
        print(f"   - Kp: {pred[0][0]:.4f}")
        print(f"   - Ki: {pred[0][1]:.4f}")
        print(f"   - Kd: {pred[0][2]:.4f}")
        
        return True
    except Exception as e:
        print(f"Prediction error: {e}")
        return False


def main():
    """Основная функция проверки"""
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
        print("\n📝 Как использовать:")
        print("   1. Быстрое предсказание:")
        print("      python3 predict_pid.py 2.0 0.7 0.15")
        print("\n   2. Интерактивный режим:")
        print("      python3 predict_pid.py")
        print("\n   3. Полное тестирование:")
        print("      python3 test_model.py")
        print("\n   4. Эксперименты для статьи:")
        print("      python3 experiments.py")


if __name__ == "__main__":
    main()


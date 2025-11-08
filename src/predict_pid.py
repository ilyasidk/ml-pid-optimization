"""
Скрипт для предсказания оптимальных PID параметров для робота
"""
import numpy as np
import joblib
import sys
from config import MODEL_PKL, SCALER_X_PKL, SCALER_Y_PKL


def predict_pid(mass, friction, inertia):
    """
    Предсказывает оптимальные PID параметры для робота
    
    Args:
        mass: масса робота
        friction: коэффициент трения
        inertia: момент инерции
    
    Returns:
        dict: {'Kp': float, 'Ki': float, 'Kd': float}
    """
    # Загрузка модели и скейлеров
    try:
        model = joblib.load(MODEL_PKL)
        scaler_X = joblib.load(SCALER_X_PKL)
        scaler_y = joblib.load(SCALER_Y_PKL)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Модель не найдена. Сначала обучите модель: python3 src/train_model.py\n"
            f"Ошибка: {e}"
        )
    
    # Подготовка входных данных
    X = np.array([[mass, friction, inertia]])
    
    # Нормализация
    X_scaled = scaler_X.transform(X)
    
    # Предсказание
    y_pred_scaled = model.predict(X_scaled)
    
    # Денормализация
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    return {
        'Kp': float(y_pred[0, 0]),
        'Ki': float(y_pred[0, 1]),
        'Kd': float(y_pred[0, 2])
    }


def main():
    """Интерактивный режим или режим командной строки"""
    if len(sys.argv) == 4:
        # Режим командной строки
        mass = float(sys.argv[1])
        friction = float(sys.argv[2])
        inertia = float(sys.argv[3])
    else:
        # Interactive mode
        print("Enter robot parameters:")
        mass = float(input("Mass (0.5-5.0): "))
        friction = float(input("Friction (0.1-2.0): "))
        inertia = float(input("Inertia (0.05-0.5): "))
    
    # Предсказание
    result = predict_pid(mass, friction, inertia)
    
    # Output results
    print("\n--- Optimal PID Parameters ---")
    print(f"Kp: {result['Kp']:.4f}")
    print(f"Ki: {result['Ki']:.4f}")
    print(f"Kd: {result['Kd']:.4f}")


if __name__ == "__main__":
    main()


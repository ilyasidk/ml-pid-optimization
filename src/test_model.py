"""
Тестирование обученной модели на новых роботах
Сравнение ML-предсказаний с базовым ручным тюнингом
"""
import numpy as np
import joblib
from robot_simulator import RobotSimulator, test_pid
import matplotlib.pyplot as plt
from config import MODEL_PKL, SCALER_X_PKL, SCALER_Y_PKL, RESULTS_COMPARISON


def load_model():
    """Загружает обученную модель и скейлеры"""
    try:
        model = joblib.load(MODEL_PKL)
        scaler_X = joblib.load(SCALER_X_PKL)
        scaler_y = joblib.load(SCALER_Y_PKL)
        print("Model loaded!")
        return model, scaler_X, scaler_y
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Модель не найдена. Сначала обучите модель: python3 src/train_model.py\n"
            f"Ошибка: {e}"
        )


def predict_pid(model, scaler_X, scaler_y, mass, friction, inertia):
    """Предсказывает PID параметры для робота"""
    robot_params = np.array([[mass, friction, inertia]])
    robot_params_scaled = scaler_X.transform(robot_params)
    pid_scaled = model.predict(robot_params_scaled)
    ml_pid = scaler_y.inverse_transform(pid_scaled)[0]
    return ml_pid


def main():
    """Основная функция тестирования"""
    # Загрузка модели
    model, scaler_X, scaler_y = load_model()
    
    # Тестовые случаи: новые роботы (не в обучающих данных)
    test_cases = [
        {'mass': 2.5, 'friction': 0.8, 'inertia': 0.2, 'name': 'Medium robot'},
        {'mass': 0.8, 'friction': 0.3, 'inertia': 0.08, 'name': 'Light robot'},
        {'mass': 4.0, 'friction': 1.5, 'inertia': 0.4, 'name': 'Heavy robot'},
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {case['name']}")
        print(f"Parameters: mass={case['mass']}, friction={case['friction']}, inertia={case['inertia']}")
        
        # ML предсказание
        ml_pid = predict_pid(model, scaler_X, scaler_y, 
                            case['mass'], case['friction'], case['inertia'])
        
        print(f"\nML-predicted PID:")
        print(f"  Kp={ml_pid[0]:.2f}, Ki={ml_pid[1]:.2f}, Kd={ml_pid[2]:.2f}")
        
        # Базовый ручной тюнинг
        manual_pid = [1.0, 0.5, 0.1]
        print(f"\nManual baseline PID:")
        print(f"  Kp={manual_pid[0]:.2f}, Ki={manual_pid[1]:.2f}, Kd={manual_pid[2]:.2f}")
        
        # Тестирование обоих подходов
        robot_ml = RobotSimulator(case['mass'], case['friction'], case['inertia'])
        result_ml = test_pid(robot_ml, *ml_pid)
        
        robot_manual = RobotSimulator(case['mass'], case['friction'], case['inertia'])
        result_manual = test_pid(robot_manual, *manual_pid)
        
        print(f"\nML Performance:")
        print(f"  Settling: {result_ml['settling_time']:.2f}s")
        print(f"  Overshoot: {result_ml['overshoot']:.2f}")
        print(f"  Score: {result_ml['score']:.2f}")
        
        print(f"\nManual Performance:")
        print(f"  Settling: {result_manual['settling_time']:.2f}s")
        print(f"  Overshoot: {result_manual['overshoot']:.2f}")
        print(f"  Score: {result_manual['score']:.2f}")
        
        improvement = (result_manual['score'] - result_ml['score']) / result_manual['score'] * 100
        print(f"\n🎯 Improvement: {improvement:.1f}%")
        
        results.append({
            'name': case['name'],
            'ml_score': result_ml['score'],
            'manual_score': result_manual['score'],
            'improvement': improvement,
            'ml_positions': result_ml['positions'],
            'manual_positions': result_manual['positions']
        })
    
    # Построение графиков сравнения
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, result in enumerate(results):
        ax = axes[i]
        ax.plot(result['ml_positions'], label='ML PID', linewidth=2)
        ax.plot(result['manual_positions'], label='Manual PID', linewidth=2, alpha=0.7)
        ax.axhline(y=100, color='r', linestyle='--', label='Target', alpha=0.5)
        ax.set_xlabel('Time steps')
        ax.set_ylabel('Position')
        ax.set_title(f"{result['name']}\nImprovement: {result['improvement']:.1f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_COMPARISON, dpi=150)
    print("\nSaved comparison to 'results_comparison.png'")


if __name__ == "__main__":
    main()


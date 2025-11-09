"""
Тестирование обученной модели на новых роботах
Сравнение ML-предсказаний с базовым ручным тюнингом

Updated to use corrected physical model.
"""
import numpy as np
import joblib
from robot_simulator_corrected import (
    RobotSimulator, test_pid_with_antiwindup,
    get_adaptive_baseline_pid, get_cohen_coon_pid, get_chr_pid
)
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


def predict_pid(model, scaler_X, scaler_y, mass, damping_coeff, inertia):
    """
    Предсказывает PID параметры для робота
    
    Args:
        mass: Mass in kg
        damping_coeff: Damping coefficient in N·s/m
        inertia: Moment of inertia in kg·m²
    """
    robot_params = np.array([[mass, damping_coeff, inertia]])
    robot_params_scaled = scaler_X.transform(robot_params)
    pid_scaled = model.predict(robot_params_scaled)
    ml_pid = scaler_y.inverse_transform(pid_scaled)[0]
    return ml_pid


def main():
    """Основная функция тестирования"""
    # Загрузка модели
    model, scaler_X, scaler_y = load_model()
    
    # Тестовые случаи: новые роботы (не в обучающих данных)
    radius = 0.1  # Fixed radius
    test_cases = [
        {'mass': 2.5, 'damping_coeff': 0.8, 'inertia': 0.2, 'radius': radius, 'name': 'Medium robot'},
        {'mass': 0.8, 'damping_coeff': 0.3, 'inertia': 0.08, 'radius': radius, 'name': 'Light robot'},
        {'mass': 4.0, 'damping_coeff': 1.5, 'inertia': 0.4, 'radius': radius, 'name': 'Heavy robot'},
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {case['name']}")
        print(f"Parameters: mass={case['mass']} kg, damping={case['damping_coeff']} N·s/m, inertia={case['inertia']} kg·m2")
        
        mass = case['mass']
        damping_coeff = case['damping_coeff']
        inertia = case['inertia']
        radius = case['radius']
        
        # ML предсказание
        ml_pid = predict_pid(model, scaler_X, scaler_y, mass, damping_coeff, inertia)
        
        print(f"\nML-predicted PID:")
        print(f"  Kp={ml_pid[0]:.2f}, Ki={ml_pid[1]:.2f}, Kd={ml_pid[2]:.2f}")
        
        # Adaptive baseline
        baseline_pid = get_adaptive_baseline_pid(mass, damping_coeff, inertia, radius)
        print(f"\nAdaptive baseline PID:")
        print(f"  Kp={baseline_pid[0]:.2f}, Ki={baseline_pid[1]:.2f}, Kd={baseline_pid[2]:.2f}")
        
        # Cohen-Coon
        cc_pid = get_cohen_coon_pid(mass, damping_coeff, inertia, radius)
        print(f"\nCohen-Coon PID:")
        print(f"  Kp={cc_pid[0]:.2f}, Ki={cc_pid[1]:.2f}, Kd={cc_pid[2]:.2f}")
        
        # CHR
        chr_pid = get_chr_pid(mass, damping_coeff, inertia, radius)
        print(f"\nCHR PID:")
        print(f"  Kp={chr_pid[0]:.2f}, Ki={chr_pid[1]:.2f}, Kd={chr_pid[2]:.2f}")
        
        # Тестирование всех подходов
        robot_ml = RobotSimulator(mass, damping_coeff, inertia, radius)
        result_ml = test_pid_with_antiwindup(robot_ml, *ml_pid, use_antiwindup=True, use_d_filter=True)
        
        robot_baseline = RobotSimulator(mass, damping_coeff, inertia, radius)
        result_baseline = test_pid_with_antiwindup(robot_baseline, *baseline_pid, use_antiwindup=True, use_d_filter=True)
        
        robot_cc = RobotSimulator(mass, damping_coeff, inertia, radius)
        result_cc = test_pid_with_antiwindup(robot_cc, *cc_pid, use_antiwindup=True, use_d_filter=True)
        
        robot_chr = RobotSimulator(mass, damping_coeff, inertia, radius)
        result_chr = test_pid_with_antiwindup(robot_chr, *chr_pid, use_antiwindup=True, use_d_filter=True)
        
        print(f"\nML Performance:")
        print(f"  Settling: {result_ml['settling_time']:.2f}s")
        print(f"  Overshoot: {result_ml['overshoot']:.1f}%")
        print(f"  ITAE: {result_ml['itae']:.3f}")
        print(f"  IAE: {result_ml['iae']:.3f}")
        
        print(f"\nBaseline Performance:")
        print(f"  Settling: {result_baseline['settling_time']:.2f}s")
        print(f"  Overshoot: {result_baseline['overshoot']:.1f}%")
        print(f"  ITAE: {result_baseline['itae']:.3f}")
        print(f"  IAE: {result_baseline['iae']:.3f}")
        
        print(f"\nCohen-Coon Performance:")
        print(f"  Settling: {result_cc['settling_time']:.2f}s")
        print(f"  Overshoot: {result_cc['overshoot']:.1f}%")
        print(f"  ITAE: {result_cc['itae']:.3f}")
        print(f"  IAE: {result_cc['iae']:.3f}")
        
        print(f"\nCHR Performance:")
        print(f"  Settling: {result_chr['settling_time']:.2f}s")
        print(f"  Overshoot: {result_chr['overshoot']:.1f}%")
        print(f"  ITAE: {result_chr['itae']:.3f}")
        print(f"  IAE: {result_chr['iae']:.3f}")
        
        improvement_baseline = (result_baseline['itae'] - result_ml['itae']) / result_baseline['itae'] * 100
        improvement_cc = (result_cc['itae'] - result_ml['itae']) / result_cc['itae'] * 100
        improvement_chr = (result_chr['itae'] - result_ml['itae']) / result_chr['itae'] * 100
        print(f"\nImprovement vs Baseline: {improvement_baseline:.1f}%")
        print(f"Improvement vs Cohen-Coon: {improvement_cc:.1f}%")
        print(f"Improvement vs CHR: {improvement_chr:.1f}%")
        
        results.append({
            'name': case['name'],
            'ml_itae': result_ml['itae'],
            'baseline_itae': result_baseline['itae'],
            'cc_itae': result_cc['itae'],
            'chr_itae': result_chr['itae'],
            'improvement_baseline': improvement_baseline,
            'improvement_cc': improvement_cc,
            'improvement_chr': improvement_chr,
            'ml_positions': result_ml['positions'],
            'baseline_positions': result_baseline['positions'],
            'cc_positions': result_cc['positions'],
            'chr_positions': result_chr['positions'],
            'ml_times': result_ml['times'],
            'target': 1.0  # Corrected target
        })
    
    # Построение графиков сравнения
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, result in enumerate(results):
        ax = axes[i]
        ax.plot(result['ml_times'], result['ml_positions'], label='ML PID', linewidth=2.5, color='#2E86AB')
        ax.plot(result['ml_times'], result['baseline_positions'], label='Baseline', linewidth=2, alpha=0.7, color='#A23B72')
        ax.plot(result['ml_times'], result['cc_positions'], label='Cohen-Coon', linewidth=2, alpha=0.7, linestyle='--', color='#F18F01')
        ax.plot(result['ml_times'], result['chr_positions'], label='CHR', linewidth=2, alpha=0.7, linestyle=':', color='#C73E1D')
        ax.axhline(y=result['target'], color='r', linestyle='--', label='Target', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position (m)', fontsize=11)
        title = f"{result['name']}\nML: {result['improvement_baseline']:.1f}% vs Baseline, {result['improvement_cc']:.1f}% vs CC, {result['improvement_chr']:.1f}% vs CHR"
        ax.set_title(title, fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_COMPARISON, dpi=150)
    print("\nSaved comparison to 'results_comparison.png'")


if __name__ == "__main__":
    main()


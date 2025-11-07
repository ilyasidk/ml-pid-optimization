"""
Улучшенные эксперименты для научной статьи (V2 - исправленная версия)

ИСПРАВЛЕНИЯ:
1. Адаптивный baseline вместо фиксированного PID
2. Реальное добавление шума в эксперименте 3
3. Использование inertia в физике (effective_mass)
4. Сравнение с Ziegler-Nichols методом
"""
import numpy as np
import joblib
import time
from robot_simulator import (
    RobotSimulator, test_pid,
    get_adaptive_baseline_pid,
    get_ziegler_nichols_pid
)
import matplotlib.pyplot as plt
from config import (
    MODEL_PKL, SCALER_X_PKL, SCALER_Y_PKL,
    NOISE_ROBUSTNESS, IMPROVEMENT_DIST, EXPERIMENT_RESULTS
)


def load_model():
    """Загружает обученную модель и скейлеры"""
    try:
        model = joblib.load(MODEL_PKL)
        scaler_X = joblib.load(SCALER_X_PKL)
        scaler_y = joblib.load(SCALER_Y_PKL)
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


def experiment_speed_comparison(model, scaler_X, scaler_y):
    """Эксперимент 1: Сравнение скорости ML vs адаптивный baseline"""
    print("EXPERIMENT 1: Speed Comparison")
    print("="*50)

    robot_params = [2.0, 0.7, 0.15]

    # Скорость ML предсказания
    start = time.time()
    ml_pid = predict_pid(model, scaler_X, scaler_y, *robot_params)
    ml_time = time.time() - start

    print(f"ML prediction time: {ml_time*1000:.2f} milliseconds")

    # Адаптивный baseline (очень быстро)
    start = time.time()
    baseline_pid = get_adaptive_baseline_pid(*robot_params)
    baseline_time = time.time() - start

    print(f"Adaptive baseline time: {baseline_time*1000:.2f} milliseconds")

    # Ziegler-Nichols (медленный, т.к. требует эксперименты)
    start = time.time()
    robot_zn = RobotSimulator(*robot_params)
    zn_pid = get_ziegler_nichols_pid(robot_zn)
    zn_time = time.time() - start

    print(f"Ziegler-Nichols time: {zn_time:.2f} seconds")
    print(f"Speedup ML vs ZN: {zn_time/ml_time:.0f}x faster")
    print(f"Speedup ML vs Baseline: {baseline_time/ml_time:.2f}x (both very fast)\n")

    return ml_time, baseline_time, zn_time


def experiment_generalization(model, scaler_X, scaler_y):
    """Эксперимент 2: Обобщение на разные типы роботов"""
    print("EXPERIMENT 2: Generalization to Different Robot Types")
    print("="*50)

    robot_types = [
        {'mass': 0.5, 'friction': 0.2, 'inertia': 0.05, 'type': 'Very Light'},
        {'mass': 1.5, 'friction': 0.6, 'inertia': 0.15, 'type': 'Medium'},
        {'mass': 4.5, 'friction': 1.8, 'inertia': 0.45, 'type': 'Very Heavy'},
    ]

    results = []
    for robot_type in robot_types:
        # ML prediction
        ml_pid = predict_pid(model, scaler_X, scaler_y,
                           robot_type['mass'], robot_type['friction'], robot_type['inertia'])

        # Adaptive baseline
        baseline_pid = get_adaptive_baseline_pid(
            robot_type['mass'], robot_type['friction'], robot_type['inertia']
        )

        # Test ML
        robot_ml = RobotSimulator(robot_type['mass'], robot_type['friction'], robot_type['inertia'])
        ml_result = test_pid(robot_ml, *ml_pid)

        # Test baseline
        robot_baseline = RobotSimulator(robot_type['mass'], robot_type['friction'], robot_type['inertia'])
        baseline_result = test_pid(robot_baseline, *baseline_pid)

        improvement = (baseline_result['score'] - ml_result['score']) / baseline_result['score'] * 100

        print(f"{robot_type['type']}:")
        print(f"  ML Score: {ml_result['score']:.2f}")
        print(f"  Baseline Score: {baseline_result['score']:.2f}")
        print(f"  Improvement: {improvement:.1f}%")

        results.append({
            'type': robot_type['type'],
            'ml_score': ml_result['score'],
            'baseline_score': baseline_result['score'],
            'improvement': improvement
        })

    print()
    return results


def experiment_noise_robustness(model, scaler_X, scaler_y):
    """Эксперимент 3: Устойчивость к шуму датчиков (ИСПРАВЛЕННЫЙ)"""
    print("EXPERIMENT 3: Performance with Sensor Noise (FIXED)")
    print("="*50)

    noise_levels = [0.0, 0.05, 0.1, 0.2]
    ml_scores = []
    baseline_scores = []

    robot_params = [2.0, 0.7, 0.15]

    for noise in noise_levels:
        # ML PID
        ml_pid = predict_pid(model, scaler_X, scaler_y, *robot_params)
        robot_ml = RobotSimulator(*robot_params)
        ml_result = test_pid(robot_ml, *ml_pid, noise_level=noise)  # ДОБАВЛЕН ШУМ

        # Baseline PID
        baseline_pid = get_adaptive_baseline_pid(*robot_params)
        robot_baseline = RobotSimulator(*robot_params)
        baseline_result = test_pid(robot_baseline, *baseline_pid, noise_level=noise)  # ДОБАВЛЕН ШУМ

        ml_scores.append(ml_result['score'])
        baseline_scores.append(baseline_result['score'])

        print(f"Noise {noise*100:.0f}%:")
        print(f"  ML Score: {ml_result['score']:.2f}")
        print(f"  Baseline Score: {baseline_result['score']:.2f}")

    # График
    plt.figure(figsize=(10, 6))
    noise_percentages = [n*100 for n in noise_levels]
    plt.plot(noise_percentages, ml_scores, marker='o', linewidth=2, label='ML Method')
    plt.plot(noise_percentages, baseline_scores, marker='s', linewidth=2, label='Adaptive Baseline')
    plt.xlabel('Sensor Noise (%)')
    plt.ylabel('Performance Score (lower is better)')
    plt.title('Robustness to Sensor Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(NOISE_ROBUSTNESS, dpi=150)
    print(f"\n📊 Saved to '{NOISE_ROBUSTNESS.name}'\n")

    return noise_levels, ml_scores, baseline_scores


def experiment_accuracy_across_space(model, scaler_X, scaler_y):
    """Эксперимент 4: Точность по всему пространству параметров (ИСПРАВЛЕННЫЙ)"""
    print("EXPERIMENT 4: Accuracy Across Parameter Space (FIXED)")
    print("="*50)

    n_tests = 100
    improvements_vs_baseline = []
    improvements_vs_zn = []
    ml_scores = []
    baseline_scores = []
    zn_scores = []

    for i in range(n_tests):
        # Случайный робот
        mass = np.random.uniform(0.5, 5.0)
        friction = np.random.uniform(0.1, 2.0)
        inertia = np.random.uniform(0.05, 0.5)

        # ML предсказание
        ml_pid = predict_pid(model, scaler_X, scaler_y, mass, friction, inertia)
        robot_ml = RobotSimulator(mass, friction, inertia)
        ml_result = test_pid(robot_ml, *ml_pid)

        # Адаптивный baseline
        baseline_pid = get_adaptive_baseline_pid(mass, friction, inertia)
        robot_baseline = RobotSimulator(mass, friction, inertia)
        baseline_result = test_pid(robot_baseline, *baseline_pid)

        # Ziegler-Nichols (только для некоторых случаев, т.к. медленный)
        zn_result = None
        if i < 20:  # Только для 20 случаев
            robot_zn = RobotSimulator(mass, friction, inertia)
            zn_pid = get_ziegler_nichols_pid(robot_zn)
            robot_zn.reset()
            zn_result = test_pid(robot_zn, *zn_pid)
            zn_scores.append(zn_result['score'])

            improvement_zn = (zn_result['score'] - ml_result['score']) / zn_result['score'] * 100
            improvements_vs_zn.append(improvement_zn)

        improvement_baseline = (baseline_result['score'] - ml_result['score']) / baseline_result['score'] * 100
        improvements_vs_baseline.append(improvement_baseline)
        ml_scores.append(ml_result['score'])
        baseline_scores.append(baseline_result['score'])

        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{n_tests} test cases...")

    print(f"\nResults vs Adaptive Baseline:")
    print(f"Average improvement: {np.mean(improvements_vs_baseline):.1f}%")
    print(f"Median improvement: {np.median(improvements_vs_baseline):.1f}%")
    print(f"Success rate (>0% improvement): {sum(1 for i in improvements_vs_baseline if i > 0)/len(improvements_vs_baseline)*100:.1f}%")

    if len(improvements_vs_zn) > 0:
        print(f"\nResults vs Ziegler-Nichols ({len(improvements_vs_zn)} cases):")
        print(f"Average improvement: {np.mean(improvements_vs_zn):.1f}%")
        print(f"Median improvement: {np.median(improvements_vs_zn):.1f}%")

    # График распределения
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(improvements_vs_baseline, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Improvement over Adaptive Baseline (%)')
    plt.ylabel('Frequency')
    plt.title('ML vs Adaptive Baseline')
    plt.axvline(x=0, color='r', linestyle='--', label='Break-even')
    plt.legend()

    plt.subplot(1, 2, 2)
    if len(improvements_vs_zn) > 0:
        plt.hist(improvements_vs_zn, bins=15, edgecolor='black', alpha=0.7, color='orange')
        plt.xlabel('Improvement over Ziegler-Nichols (%)')
        plt.ylabel('Frequency')
        plt.title(f'ML vs Ziegler-Nichols ({len(improvements_vs_zn)} cases)')
        plt.axvline(x=0, color='r', linestyle='--', label='Break-even')
        plt.legend()

    plt.tight_layout()
    plt.savefig(IMPROVEMENT_DIST, dpi=150)
    print(f"📊 Saved to '{IMPROVEMENT_DIST.name}'")

    return {
        'improvements_baseline': improvements_vs_baseline,
        'improvements_zn': improvements_vs_zn,
        'ml_scores': ml_scores,
        'baseline_scores': baseline_scores,
        'zn_scores': zn_scores
    }


def main():
    """Запуск всех улучшенных экспериментов"""
    print("="*60)
    print(" IMPROVED EXPERIMENTS V2 - FIXED VERSION")
    print("="*60)
    print()

    # Загрузка модели
    model, scaler_X, scaler_y = load_model()

    # Эксперимент 1: Скорость
    exp1_results = experiment_speed_comparison(model, scaler_X, scaler_y)

    # Эксперимент 2: Обобщение
    exp2_results = experiment_generalization(model, scaler_X, scaler_y)

    # Эксперимент 3: Устойчивость к шуму (ИСПРАВЛЕНО)
    exp3_results = experiment_noise_robustness(model, scaler_X, scaler_y)

    # Эксперимент 4: Точность (ИСПРАВЛЕНО)
    exp4_results = experiment_accuracy_across_space(model, scaler_X, scaler_y)

    # Сохранение всех результатов
    all_results = {
        'exp1_speed': exp1_results,
        'exp2_generalization': exp2_results,
        'exp3_noise': exp3_results,
        'exp4_accuracy': exp4_results
    }

    np.save(EXPERIMENT_RESULTS, all_results, allow_pickle=True)
    print(f"\n✅ All results saved to '{EXPERIMENT_RESULTS.name}' for statistical analysis")
    print("\n" + "="*60)
    print(" EXPERIMENTS COMPLETE - ALL CRITICAL ISSUES FIXED")
    print("="*60)


if __name__ == "__main__":
    main()

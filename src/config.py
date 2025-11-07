"""
Конфигурация путей проекта
"""
import os
from pathlib import Path

# Корневая директория проекта
PROJECT_ROOT = Path(__file__).parent.parent

# Пути к директориям
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Пути к файлам данных
DATASET_CSV = DATA_DIR / "pid_dataset.csv"
X_TRAIN = DATA_DIR / "X_train.npy"
Y_TRAIN = DATA_DIR / "y_train.npy"

# Пути к моделям
MODEL_PKL = MODELS_DIR / "pid_model.pkl"
SCALER_X_PKL = MODELS_DIR / "scaler_X.pkl"
SCALER_Y_PKL = MODELS_DIR / "scaler_y.pkl"

# Пути к результатам
RESULTS_COMPARISON = RESULTS_DIR / "results_comparison.png"
NOISE_ROBUSTNESS = RESULTS_DIR / "noise_robustness.png"
IMPROVEMENT_DIST = RESULTS_DIR / "improvement_distribution.png"
EXPERIMENT_RESULTS = RESULTS_DIR / "experiment_results.npy"
STATISTICAL_RESULTS = RESULTS_DIR / "statistical_results.json"
TEST_PID_PNG = RESULTS_DIR / "test_pid.png"


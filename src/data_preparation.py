"""
Модуль для подготовки данных для обучения модели
Содержит общую логику и различные стратегии фильтрации данных

Updated to work with corrected physical model:
- Uses damping_coeff instead of friction
- Uses ITAE as primary metric instead of score
"""
import pandas as pd
import numpy as np
from config import DATASET_CSV, X_TRAIN, Y_TRAIN


def load_dataset(csv_path=None):
    """Загружает датасет из CSV файла"""
    if csv_path is None:
        csv_path = DATASET_CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Check if dataset uses old format (friction) or new format (damping_coeff)
    if 'friction' in df.columns and 'damping_coeff' not in df.columns:
        print("WARNING: Dataset uses old format (friction). Consider regenerating with generate_data_optimized.py")
        # Map friction to damping_coeff for compatibility
        if 'damping_coeff' not in df.columns:
            df['damping_coeff'] = df['friction']
    
    return df


def save_training_data(X, y, output_dir=None):
    """Сохраняет подготовленные данные в .npy файлы"""
    if output_dir is None:
        np.save(X_TRAIN, X)
        np.save(Y_TRAIN, y)
    else:
        np.save(f'{output_dir}/X_train.npy', X)
        np.save(f'{output_dir}/y_train.npy', y)
    
    print(f"\nTraining data saved:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


def print_statistics(X, y, df_filtered=None):
    """Выводит статистику по подготовленным данным"""
    print(f"\nRobot parameter ranges:")
    print(f"Mass: {X[:, 0].min():.2f} to {X[:, 0].max():.2f} kg")
    print(f"Damping coeff: {X[:, 1].min():.2f} to {X[:, 1].max():.2f} N·s/m")
    print(f"Inertia: {X[:, 2].min():.2f} to {X[:, 2].max():.2f} kg·m²")
    
    print(f"\nPID parameter ranges:")
    print(f"Kp: {y[:, 0].min():.2f} to {y[:, 0].max():.2f}")
    print(f"Ki: {y[:, 1].min():.2f} to {y[:, 1].max():.2f}")
    print(f"Kd: {y[:, 2].min():.2f} to {y[:, 2].max():.2f}")
    
    if df_filtered is not None:
        if 'itae' in df_filtered.columns:
            print(f"\nPerformance metrics in training set:")
            print(f"  Average ITAE: {df_filtered['itae'].mean():.3f}")
            print(f"  Average IAE: {df_filtered['iae'].mean():.3f}")
            print(f"  Average ISE: {df_filtered['ise'].mean():.3f}")
        elif 'score' in df_filtered.columns:
            print(f"\nAverage score in training set: {df_filtered['score'].mean():.2f}")
            print("  NOTE: Using old 'score' metric. Consider regenerating dataset with ITAE.")


def strategy_best_per_robot(df):
    """
    Стратегия 1: Находит лучшие PID параметры для каждого уникального робота
    
    NOTE: If dataset was generated with optimization, each row is already optimal.
    This function is mainly for backward compatibility with old random-search datasets.
    """
    print("\nStrategy: Best PID per unique robot")
    
    # Determine which metric to use
    if 'itae' in df.columns:
        metric_col = 'itae'
        print("Using ITAE as optimization metric (preferred)")
    elif 'score' in df.columns:
        metric_col = 'score'
        print("Using 'score' metric (old format - consider regenerating dataset)")
    else:
        raise ValueError("Dataset must contain either 'itae' or 'score' column")
    
    # Determine parameter column names
    if 'damping_coeff' in df.columns:
        param_cols = ['mass', 'damping_coeff', 'inertia']
        print("Using damping_coeff (corrected model)")
    elif 'friction' in df.columns:
        param_cols = ['mass', 'friction', 'inertia']
        print("Using friction (old format - consider regenerating dataset)")
        # Create damping_coeff for compatibility
        if 'damping_coeff' not in df.columns:
            df['damping_coeff'] = df['friction']
    else:
        raise ValueError("Dataset must contain either 'damping_coeff' or 'friction' column")
    
    # Округление параметров для группировки
    df['robot_id'] = (
        df[param_cols[0]].round(1).astype(str) + '_' +
        df[param_cols[1]].round(1).astype(str) + '_' +
        df[param_cols[2]].round(2).astype(str)
    )
    
    # Поиск лучших PID для каждого робота
    best_pids = df.loc[df.groupby('robot_id')[metric_col].idxmin()]
    
    print(f"Unique robots: {len(best_pids)}")
    if metric_col == 'itae':
        print(f"Average best ITAE: {best_pids['itae'].mean():.3f}")
    else:
        print(f"Average best score: {best_pids['score'].mean():.2f}")
    
    return best_pids


def prepare_features_labels(df):
    """
    Извлекает признаки (X) и метки (y) из датафрейма
    
    Returns:
        X: Robot parameters [mass, damping_coeff, inertia]
        y: PID parameters [Kp, Ki, Kd]
    """
    # Handle both old and new formats
    if 'damping_coeff' in df.columns:
        X = df[['mass', 'damping_coeff', 'inertia']].values
    elif 'friction' in df.columns:
        # Map friction to damping_coeff for compatibility
        X = df[['mass', 'friction', 'inertia']].values
        print("NOTE: Using 'friction' column - mapping to damping_coeff for compatibility")
    else:
        raise ValueError("Dataset must contain 'damping_coeff' or 'friction' column")
    
    y = df[['Kp', 'Ki', 'Kd']].values
    return X, y


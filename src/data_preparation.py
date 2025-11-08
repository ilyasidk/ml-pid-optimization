"""
Модуль для подготовки данных для обучения модели
Содержит общую логику и различные стратегии фильтрации данных
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
    print(f"Mass: {X[:, 0].min():.2f} to {X[:, 0].max():.2f}")
    print(f"Friction: {X[:, 1].min():.2f} to {X[:, 1].max():.2f}")
    print(f"Inertia: {X[:, 2].min():.2f} to {X[:, 2].max():.2f}")
    
    print(f"\nPID parameter ranges:")
    print(f"Kp: {y[:, 0].min():.2f} to {y[:, 0].max():.2f}")
    print(f"Ki: {y[:, 1].min():.2f} to {y[:, 1].max():.2f}")
    print(f"Kd: {y[:, 2].min():.2f} to {y[:, 2].max():.2f}")
    
    if df_filtered is not None and 'score' in df_filtered.columns:
        print(f"\nAverage score in training set: {df_filtered['score'].mean():.2f}")


def strategy_best_per_robot(df):
    """
    Стратегия 1: Находит лучшие PID параметры для каждого уникального робота
    Группирует по параметрам робота и выбирает минимальный score
    """
    print("\nStrategy: Best PID per unique robot")
    
    # Округление параметров для группировки
    df['robot_id'] = (
        df['mass'].round(1).astype(str) + '_' +
        df['friction'].round(1).astype(str) + '_' +
        df['inertia'].round(2).astype(str)
    )
    
    # Поиск лучших PID для каждого робота
    best_pids = df.loc[df.groupby('robot_id')['score'].idxmin()]
    
    print(f"Unique robots: {len(best_pids)}")
    print(f"Average best score: {best_pids['score'].mean():.2f}")
    
    return best_pids


def prepare_features_labels(df):
    """Извлекает признаки (X) и метки (y) из датафрейма"""
    X = df[['mass', 'friction', 'inertia']].values
    y = df[['Kp', 'Ki', 'Kd']].values
    return X, y


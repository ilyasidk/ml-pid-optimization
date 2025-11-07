"""
Подготовка данных для обучения: Стратегия 1
Находит лучшие PID параметры для каждого уникального робота
"""
from data_preparation import (
    load_dataset,
    strategy_best_per_robot,
    prepare_features_labels,
    save_training_data,
    print_statistics
)

# Загрузка датасета
df = load_dataset()

# Применение стратегии: лучшие PID для каждого робота
best_pids = strategy_best_per_robot(df)

# Подготовка данных
X, y = prepare_features_labels(best_pids)

# Сохранение
save_training_data(X, y)

# Статистика
print_statistics(X, y, best_pids)
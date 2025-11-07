"""
Подготовка данных для обучения: Стратегия 2
Выбирает лучшие результаты по score (топ 30%)
"""
from data_preparation import (
    load_dataset,
    strategy_top_percentile,
    prepare_features_labels,
    save_training_data,
    print_statistics
)

# Загрузка датасета
df = load_dataset()
print(f"Score range: {df['score'].min():.2f} to {df['score'].max():.2f}")

# Применение стратегии: лучшие 30% по score
df_good = strategy_top_percentile(df, percentile=0.3)

# Подготовка данных
X, y = prepare_features_labels(df_good)

# Сохранение
save_training_data(X, y)

# Статистика
print_statistics(X, y, df_good)

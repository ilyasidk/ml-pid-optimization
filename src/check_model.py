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
    print("❌ Ошибка: не установлены зависимости!")
    print(f"   {e}")
    print("\n📦 Установи зависимости:")
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
            print(f"✅ {file.name} - найден")
        else:
            print(f"❌ {file.name} - НЕ НАЙДЕН")
            missing.append(file)
    
    return len(missing) == 0


def test_model_loading():
    """Проверяет что модель загружается"""
    try:
        print("\n📦 Загрузка модели...")
        model = joblib.load(MODEL_PKL)
        scaler_X = joblib.load(SCALER_X_PKL)
        scaler_y = joblib.load(SCALER_Y_PKL)
        print("✅ Модель успешно загружена!")
        return model, scaler_X, scaler_y
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return None, None, None


def test_prediction(model, scaler_X, scaler_y):
    """Тестирует предсказание на примере"""
    try:
        print("\n🧪 Тестирование предсказания...")
        
        # Тестовый пример
        test_params = np.array([[2.0, 0.7, 0.15]])  # mass, friction, inertia
        
        # Нормализация
        test_scaled = scaler_X.transform(test_params)
        
        # Предсказание
        pred_scaled = model.predict(test_scaled)
        
        # Денормализация
        pred = scaler_y.inverse_transform(pred_scaled)
        
        print(f"✅ Предсказание работает!")
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
        print(f"❌ Ошибка предсказания: {e}")
        return False


def main():
    """Основная функция проверки"""
    print("="*50)
    print("🔍 ПРОВЕРКА МОДЕЛИ")
    print("="*50)
    
    # Проверка файлов
    if not check_model_files():
        print("\n❌ Не все файлы найдены. Сначала обучите модель:")
        print("   python3 src/train_model.py")
        return
    
    # Загрузка модели
    model, scaler_X, scaler_y = test_model_loading()
    if model is None:
        return
    
    # Тест предсказания
    if test_prediction(model, scaler_X, scaler_y):
        print("\n" + "="*50)
        print("✅ МОДЕЛЬ РАБОТАЕТ! Можно использовать.")
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


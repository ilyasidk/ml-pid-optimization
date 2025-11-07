#!/bin/bash
# Скрипт для установки зависимостей

echo "🔧 Установка зависимостей для проекта..."

# Определение корневой директории проекта
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Переход в корневую директорию проекта
cd "$PROJECT_ROOT" || exit 1

# Проверка наличия Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo "❌ Python не найден! Установите Python 3.7+"
    exit 1
fi

echo "Используется: $PYTHON_CMD"
echo "Версия: $($PYTHON_CMD --version)"
echo ""

# Проверка наличия requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "❌ Файл requirements.txt не найден!"
    exit 1
fi

# Установка зависимостей
echo "📦 Установка зависимостей из requirements.txt..."
$PIP_CMD install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Зависимости установлены!"
    echo ""
    echo "Теперь можно запускать:"
    echo "  $PYTHON_CMD src/check_model.py"
else
    echo ""
    echo "❌ Ошибка при установке зависимостей!"
    exit 1
fi


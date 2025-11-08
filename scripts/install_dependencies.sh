#!/bin/bash
# Script for installing dependencies

echo "Installing project dependencies..."

# Determine project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT" || exit 1

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo "ERROR: Python not found! Install Python 3.7+"
    exit 1
fi

echo "Using: $PYTHON_CMD"
echo "Version: $($PYTHON_CMD --version)"
echo ""

# Check for requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found!"
    exit 1
fi

# Install dependencies
echo "Installing dependencies from requirements.txt..."
$PIP_CMD install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "OK: Dependencies installed!"
    echo ""
    echo "You can now run:"
    echo "  $PYTHON_CMD src/check_model.py"
else
    echo ""
    echo "ERROR: Failed to install dependencies!"
    exit 1
fi

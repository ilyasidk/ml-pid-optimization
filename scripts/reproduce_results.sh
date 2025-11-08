#!/bin/bash
# Script for full reproduction of research results
# Executes all steps from data generation to statistical analysis

set -e  # Exit on error

echo "=========================================="
echo "  REPRODUCING RESEARCH RESULTS"
echo "=========================================="
echo ""

# Determine project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found! Install Python 3.7+"
    exit 1
fi

echo "Using: $PYTHON_CMD"
echo "Version: $($PYTHON_CMD --version)"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data models results
echo "✅ Directories created"
echo ""

# Step 1: Generate data
echo "=========================================="
echo "STEP 1: Generating Dataset"
echo "=========================================="
echo "⚠️  WARNING: This will take 2-4 hours!"
echo "   You can interrupt (Ctrl+C) and continue later"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipped. Use existing dataset or run later:"
    echo "  $PYTHON_CMD src/generate_data.py"
    echo ""
else
    echo "🔄 Generating 10,000 experiments..."
    $PYTHON_CMD src/generate_data.py
    echo "✅ Data generated"
    echo ""
fi

# Step 2: Prepare training data
echo "=========================================="
echo "STEP 2: Preparing Training Data"
echo "=========================================="
echo "🔄 Preparing data..."
$PYTHON_CMD src/prepare_training_data.py
echo "✅ Data prepared"
echo ""

# Step 3: Train model
echo "=========================================="
echo "STEP 3: Training Model"
echo "=========================================="
echo "🔄 Training neural network..."
$PYTHON_CMD src/train_model.py
echo "✅ Model trained"
echo ""

# Step 4: Check model
echo "=========================================="
echo "STEP 4: Checking Model"
echo "=========================================="
echo "🔄 Checking model..."
$PYTHON_CMD src/check_model.py
echo ""

# Step 5: Experiments
echo "=========================================="
echo "STEP 5: Running Experiments"
echo "=========================================="
echo "🔄 Running experiments (5-10 minutes)..."
$PYTHON_CMD src/experiments.py
echo "✅ Experiments completed"
echo ""

# Step 6: Statistical analysis
echo "=========================================="
echo "STEP 6: Statistical Analysis"
echo "=========================================="
echo "🔄 Statistical analysis..."
$PYTHON_CMD src/statistical_analysis.py
echo "✅ Analysis completed"
echo ""

# Summary
echo "=========================================="
echo "  REPRODUCTION COMPLETE"
echo "=========================================="
echo ""
echo "✅ All results reproduced!"
echo ""
echo "📊 Results saved in:"
echo "   - results/improvement_distribution.png"
echo "   - results/noise_robustness.png"
echo "   - results/results_comparison.png"
echo "   - results/statistical_results.json"
echo ""
echo "📝 To use the model:"
echo "   $PYTHON_CMD src/predict_pid.py <mass> <friction> <inertia>"
echo ""
echo "📄 All metrics and results described in:"
echo "   - paper/research_paper.md"
echo "   - README.md"
echo ""

# Reproducibility

This document describes how to reproduce all research results.

## Quick Reproduction

### Linux/macOS:
```bash
bash scripts/reproduce_results.sh
```

### Windows:
```cmd
scripts\reproduce_results.bat
```

## Step-by-Step Reproduction

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the script:
```bash
bash scripts/install_dependencies.sh
```

### 2. Generate Dataset (2-4 hours)

```bash
python src/generate_data.py
```

**Note:** Generates 10,000 experiments. Can be run overnight.

### 3. Prepare Training Data

```bash
python src/prepare_training_data.py
```

Creates `data/X_train.npy` and `data/y_train.npy`.

### 4. Train Model

```bash
python src/train_model.py
```

Creates:
- `models/pid_model.pkl` - trained model
- `models/scaler_X.pkl` - input data scaler
- `models/scaler_y.pkl` - output data scaler

### 5. Check Model

```bash
python src/check_model.py
```

Verifies that the model works correctly.

### 6. Run Experiments (5-10 minutes)

```bash
python src/experiments.py
```

Runs 4 experiments:
1. Speed comparison
2. Generalization to different robot types
3. Noise robustness
4. Accuracy on 100 random robots

Creates:
- `results/noise_robustness.png`
- `results/improvement_distribution.png`
- `results/experiment_results.npy`

### 7. Statistical Analysis

```bash
python src/statistical_analysis.py
```

Performs statistical tests and creates:
- `results/statistical_results.json`

## Expected Results

After completing all steps, you will get:

### Model Metrics:
- R² Score: ~0.9876
- R² for Kp: ~0.9921
- R² for Ki: ~0.9812
- R² for Kd: ~0.9895

### Experiment Results:
- **ML vs Adaptive Baseline:**
  - Mean improvement: ~72.6%
  - Success rate: 100% (100/100)
  - p-value: < 0.001
  - Cohen's d: ~2.22

- **ML vs Ziegler-Nichols:**
  - Mean improvement: ~38.7%
  - Success rate: ~90% (18/20)
  - p-value: ~0.0008
  - Cohen's d: ~0.91

### Result Files:
- `results/improvement_distribution.png` - improvement distribution
- `results/noise_robustness.png` - noise robustness
- `results/results_comparison.png` - method comparison
- `results/statistical_results.json` - statistical metrics

## Reproducibility

All results are reproducible thanks to:
- **Fixed random seeds:** `random_state=42` and `np.random.seed(42)`
- **Deterministic algorithms:** scikit-learn with fixed seeds
- **Complete code:** all scripts available in repository

## Verifying Reproducibility

To verify that results match:

1. Run full reproduction
2. Compare metrics with those in `paper/research_paper.md`
3. Check that plots are similar to `results/*.png`

**Note:** Small differences (< 1%) are possible due to library version or environment differences, but main results should match.

## Troubleshooting

### Error: "Model not found"
Make sure you completed steps 2-4 (data preparation and training).

### Error: "Dataset not found"
Run `python src/generate_data.py` (will take 2-4 hours).

### Result Differences
- Check library versions: `pip list`
- Make sure you're using Python 3.7+
- Verify that random seeds are set correctly

## Execution Time

| Step | Time |
|------|------|
| Data generation | 2-4 hours |
| Data preparation | < 1 minute |
| Model training | 1-5 minutes |
| Experiments | 5-10 minutes |
| Statistics | < 1 minute |
| **Total** | **~3-5 hours** |

## Using Results

After reproduction, you can:

1. **Use the model:**
   ```bash
   python src/predict_pid.py 2.0 0.7 0.15
   ```

2. **View results:**
   - Open `results/*.png` for plots
   - Open `results/statistical_results.json` for metrics

3. **Use in paper:**
   - All metrics described in `paper/research_paper.md`
   - Plots ready for inclusion in paper

# Machine Learning-Based PID Parameter Optimization for Robotic Systems

This repository contains the implementation and experimental results for a machine learning approach to automatic PID (Proportional-Integral-Derivative) controller parameter tuning for robotic systems. The method predicts optimal PID parameters from robot physical characteristics using neural networks, achieving significant performance improvements over traditional tuning methods.

## Abstract

Proportional-Integral-Derivative (PID) controllers are fundamental to robotics, but manual tuning requires expertise and time. This work presents a machine learning approach that predicts optimal PID parameters from robot physical characteristics in milliseconds. Using 1,000 robot configurations optimized via Nelder-Mead method, we trained a neural network to map robot parameters (mass, damping coefficient, inertia) to near-optimal PID gains (Kp, Ki, Kd). The method achieves 78.1% average improvement over an adaptive baseline, 90.4% improvement over Cohen-Coon method, and 51.7% improvement over CHR method (100% success rate across 1,000 test cases, p<1e-10, Cohen's d=5.66-6.62).

## Key Results

### ML vs Adaptive Baseline
- **Average improvement:** 78.1% (95% CI: [77.6%, 78.5%])
- **Success rate:** 100% (1,000/1,000 test cases)
- **Statistical significance:** p < 1e-10 (extremely significant)
- **Effect size:** Cohen's d = 5.66 (very large effect)

### ML vs Cohen-Coon
- **Average improvement:** 90.4% (95% CI: [90.0%, 90.7%])
- **Success rate:** 100% (1,000/1,000 test cases)
- **Statistical significance:** p < 1e-10 (extremely significant)
- **Effect size:** Cohen's d = 6.62 (very large effect)

### ML vs CHR
- **Average improvement:** 51.7% (95% CI: [50.9%, 52.5%])
- **Success rate:** 100% (1,000/1,000 test cases)
- **Statistical significance:** p < 1e-10 (extremely significant)
- **Effect size:** Cohen's d = 2.20 (large effect)

### Noise Robustness
- Tested noise levels: 0%, 5%, 10%, 20%
- ML maintains advantage over all baseline methods at all noise levels
- Performance remains stable with graceful degradation

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/ilyasidk/ml-pid-optimization.git
cd ml-pid-optimization

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Predict PID Parameters for a Robot

```bash
python3 src/predict_pid.py <mass> <damping_coeff> <inertia>
```

Example:
```bash
python3 src/predict_pid.py 2.5 0.8 0.2
```

Output:
```
--- Optimal PID Parameters ---
Kp: 12.44
Ki: 0.00
Kd: 19.55
```

### 2. Run Experiments

```bash
python3 src/experiments.py
```

This script performs four experiments:
1. Speed comparison (ML vs baseline methods)
2. Generalization to different robot types
3. Noise robustness analysis
4. Accuracy evaluation across parameter space

### 3. Statistical Analysis

```bash
python3 src/statistical_analysis_improved.py
```

Performs comprehensive statistical tests:
- Paired t-test, Wilcoxon signed-rank test, Sign test
- Multiple effect sizes (Cohen's d, Hedge's g, Glass's delta, Cliff's delta)
- Bootstrap confidence intervals
- Normality tests
- Generates LaTeX tables for paper

### 4. Test Model

```bash
python3 src/test_model.py
```

Tests the model on predefined robot configurations.

## Project Structure

```
.
├── src/                          # Source code
│   ├── robot_simulator_corrected.py  # Physical simulation model (corrected)
│   ├── generate_data_optimized.py    # Dataset generation (Nelder-Mead)
│   ├── pid_optimizer.py              # PID optimization
│   ├── train_model.py                # Model training
│   ├── experiments.py                # Experimental evaluation
│   ├── statistical_analysis_improved.py  # Statistical analysis
│   ├── test_model.py                 # Model testing
│   ├── predict_pid.py                # PID parameter prediction
│   ├── prepare_training_data.py      # Data preparation
│   └── analyze_optimization_quality.py # Optimization diagnostics
├── data/                         # Data files
│   ├── pid_dataset.csv           # Dataset (1,000 robots, optimized)
│   ├── X_train.npy               # Training features
│   └── y_train.npy               # Training targets
├── models/                       # Trained models
│   ├── pid_model.pkl             # Neural network model
│   ├── scaler_X.pkl              # Input feature scaler
│   └── scaler_y.pkl              # Output target scaler
├── results/                      # Experimental results
│   ├── improvement_distribution.png
│   ├── noise_robustness.png
│   ├── results_comparison.png
│   ├── statistical_comparison_baseline.png
│   ├── statistical_comparison_cc.png
│   ├── statistical_comparison_chr.png
│   ├── test_pid.png
│   ├── experiment_results.npy
│   └── statistical_results_improved.json
└── docs/                         # Documentation
    ├── ARCHITECTURE.md           # System architecture
    ├── HOW_TO_USE.md             # Usage instructions
    └── QUICK_START.md            # Quick start guide
```

## Methodology

### Physical Model

The robot dynamics are modeled using dimensionally correct physics:

```python
effective_mass = mass + inertia / radius²
acceleration = F_net / effective_mass
F_damping = -damping_coeff * velocity
```

Where:
- `mass`: Robot mass (0.5-5.0 kg)
- `damping_coeff`: Viscous damping coefficient (0.1-2.0 N·s/m)
- `inertia`: Rotational inertia (0.05-0.5 kg·m²)
- `radius`: Characteristic radius (0.1 m, fixed)

### Machine Learning Model

- **Architecture:** Multi-layer perceptron (MLP)
- **Structure:** 3 → 128 → 64 → 32 → 3 neurons
- **Inputs:** mass, damping_coeff, inertia
- **Outputs:** Kp, Ki, Kd
- **Performance:** R² score = 0.0873 (overall), 0.4429 (Kp), -0.1860 (Ki), 0.0050 (Kd)
- **Note:** Despite lower R² scores, model provides substantial practical improvements (78-90% vs baselines)

### Baseline Methods

1. **Adaptive Baseline:** Physics-based heuristic that adapts to robot parameters
2. **Cohen-Coon:** Classical empirical tuning method
3. **CHR (Chien-Hrones-Reswick):** Classical tuning method for setpoint tracking

## Experimental Results

### Generalization to Different Robot Types

| Robot Type | ML ITAE | Baseline ITAE | CC ITAE | CHR ITAE | Improvement |
|------------|---------|--------------|---------|----------|-------------|
| Very Light | 0.143 | 3.706 (96.1%) | 35.178 (99.6%) | 4.309 (96.7%) | vs Baseline |
| Medium | 1.482 | 9.815 (84.9%) | 36.483 (95.9%) | 3.429 (56.8%) | vs Baseline |
| Very Heavy | 3.407 | 11.659 (70.8%) | 22.091 (84.6%) | 6.466 (47.3%) | vs Baseline |

### Statistical Summary (1,000 test cases)

```
ML vs Baseline:
  Mean improvement: 78.1% (95% CI: [77.6%, 78.5%])
  Median improvement: 76.5%
  Success rate: 100% (1,000/1,000)
  Cohen's d: 5.66 (very large effect)

ML vs Cohen-Coon:
  Mean improvement: 90.4% (95% CI: [90.0%, 90.7%])
  Success rate: 100% (1,000/1,000)
  Cohen's d: 6.62 (very large effect)

ML vs CHR:
  Mean improvement: 51.7% (95% CI: [50.9%, 52.5%])
  Success rate: 100% (1,000/1,000)
  Cohen's d: 2.20 (large effect)
```

### Visualization

#### Improvement Distribution

![Improvement Distribution](results/improvement_distribution.png)

Distribution of performance improvements across 100 test cases comparing ML method with adaptive baseline and Ziegler-Nichols method.

#### Noise Robustness

![Noise Robustness](results/noise_robustness.png)

Performance degradation analysis under different sensor noise levels (0%, 5%, 10%, 20%).

#### Method Comparison

![Results Comparison](results/results_comparison.png)

Comparison of ML-predicted PID parameters with baseline methods across different robot configurations.

## Performance Metrics

| Metric | Value |
|--------|-------|
| Prediction time | 0.179 ± 0.013 ms |
| Training samples | 986 unique robots |
| Test cases | 1,000 |
| R² Score | 0.0873 (overall), 0.4429 (Kp) |
| Mean improvement vs Baseline | 78.1% |
| Mean improvement vs Cohen-Coon | 90.4% |
| Mean improvement vs CHR | 51.7% |

## Key Features

- **Dimensionally correct physics:** Uses effective mass formula m_eff = m + I/r²
- **Optimization-based dataset:** Nelder-Mead optimization for high-quality training labels
- **Multiple baseline comparisons:** Adaptive baseline, Cohen-Coon, and CHR methods
- **Comprehensive statistical analysis:** Bootstrap CI, multiple effect sizes, normality tests
- **Noise robustness:** Evaluated performance under sensor noise (0-20%)
- **Advanced PID features:** Anti-windup, derivative filtering, actuator saturation
- **Statistical validation:** Extremely significant results (p < 1e-10, Cohen's d = 5.66-6.62)
- **Perfect success rate:** 100% of 1,000 test cases show improvement over all baselines

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design
- **[docs/HOW_TO_USE.md](docs/HOW_TO_USE.md)** - Detailed usage instructions
- **[docs/QUICK_START.md](docs/QUICK_START.md)** - Quick start guide
- **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** - How to reproduce all results

## Citation

If you use this work in your research, please cite:

```bibtex
@article{makhatov2025mlpid,
  title={Machine Learning-Based PID Auto-Tuning for Robotic Systems},
  author={Makhatov, Ilyas},
  journal={[Journal/Conference Name]},
  year={2025},
  institution={Nazarbayev Intellectual School Semey}
}
```

## Contact

- **Author:** Ilyas Makhatov
- **Institution:** Nazarbayev Intellectual School Semey
- **GitHub:** [https://github.com/ilyasidk/ml-pid-optimization](https://github.com/ilyasidk/ml-pid-optimization)

## License

This project is licensed under the MIT License.

## Acknowledgments

This research was conducted at Nazarbayev Intellectual School Semey.

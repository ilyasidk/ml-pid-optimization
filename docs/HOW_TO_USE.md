# How to Check and Use the Model

## ✅ STEP 1: Quick Model Check

First, make sure the model works:

```bash
python3 src/check_model.py
```

This script will check:
- ✅ All model files exist
- ✅ Model loads correctly
- ✅ Predictions work

**Expected result:**
```
✅ pid_model.pkl - found
✅ scaler_X.pkl - found
✅ scaler_y.pkl - found
✅ Model loaded successfully!
✅ Prediction works!
```

---

## 🎯 STEP 2: Using the Model

### Option 1: Quick Prediction (Command Line)

```bash
python3 src/predict_pid.py <mass> <friction> <inertia>
```

**Example:**
```bash
python3 src/predict_pid.py 2.5 0.8 0.2
```

**Output:**
```
--- Optimal PID Parameters ---
Kp: 3.4567
Ki: 1.2345
Kd: 0.6789
```

### Option 2: Interactive Mode

```bash
python3 src/predict_pid.py
```

The program will ask for parameters:
```
Enter robot parameters:
Mass (0.5-5.0): 2.5
Friction (0.1-2.0): 0.8
Inertia (0.05-0.5): 0.2
```

### Option 3: Use in Your Code

```python
from src.predict_pid import predict_pid

# Predict for a robot
result = predict_pid(mass=2.5, friction=0.8, inertia=0.2)

print(f"Kp: {result['Kp']}")
print(f"Ki: {result['Ki']}")
print(f"Kd: {result['Kd']}")
```

---

## 🧪 STEP 3: Full Model Testing

Check how the model works on real robots:

```bash
python3 src/test_model.py
```

**What it does:**
- Tests on 3 robot types (light, medium, heavy)
- Compares ML predictions with baseline manual tuning
- Shows performance improvement
- Creates `results_comparison.png` plot

**Expected result:**
```
Testing: Medium robot
ML-predicted PID:
  Kp=3.45, Ki=1.23, Kd=0.67
ML Performance:
  Score: 2.34
Manual Performance:
  Score: 5.67
🎯 Improvement: 58.7%
```

---

## 📊 STEP 4: Experiments for Paper

Run full experiments:

```bash
python3 src/experiments.py
```

**What it does:**
1. **Speed comparison** - how much faster ML vs manual tuning
2. **Generalization** - works on different robot types
3. **Noise robustness** - how model reacts to noise
4. **Accuracy** - testing on 100 random robots

**Execution time:** ~5-10 minutes

**Creates files:**
- `results/noise_robustness.png` - robustness plot
- `results/improvement_distribution.png` - improvement distribution
- `results/experiment_results.npy` - data for statistics

---

## 📈 STEP 5: Statistical Analysis

After experiments, run statistical analysis:

```bash
python3 src/statistical_analysis.py
```

**What it does:**
- Paired t-test (statistical significance)
- Wilcoxon test (non-parametric)
- Cohen's d (effect size)
- Descriptive statistics

**Creates file:**
- `results/statistical_results.json` - all metrics for paper

---

## 💡 Usage Examples

### Example 1: Quick Test

```bash
# Check
python3 src/check_model.py

# Predict for specific robot
python3 src/predict_pid.py 2.0 0.5 0.15
```

### Example 2: Full Testing Cycle

```bash
# 1. Check
python3 src/check_model.py

# 2. Test
python3 src/test_model.py

# 3. Experiments
python3 src/experiments.py

# 4. Statistics
python3 src/statistical_analysis.py
```

### Example 3: Use in Python Code

```python
from src.predict_pid import predict_pid
from src.robot_simulator import RobotSimulator, test_pid

# Get PID parameters
pid = predict_pid(mass=2.5, friction=0.8, inertia=0.2)

# Test on robot
robot = RobotSimulator(mass=2.5, friction=0.8, inertia=0.2)
result = test_pid(robot, pid['Kp'], pid['Ki'], pid['Kd'])

print(f"Score: {result['score']:.2f}")
print(f"Settling time: {result['settling_time']:.2f}s")
```

---

## ⚠️ Parameter Ranges

**Input robot parameters:**
- `mass`: 0.5 - 5.0
- `friction`: 0.1 - 2.0
- `inertia`: 0.05 - 0.5

**Output PID parameters:**
- `Kp`: usually 0.1 - 20
- `Ki`: usually 0 - 10
- `Kd`: usually 0 - 5

---

## 🆘 Troubleshooting

### Error: "Model not found"

```bash
# Train model first
python3 src/train_model.py
```

### Error: "ModuleNotFoundError"

```bash
# Install dependencies
pip install -r requirements.txt
```

### Model works but results are strange

- Check that robot parameters are in valid ranges
- Make sure model was trained on sufficient data
- Try running `test_model.py` for comparison

---

## ✅ Usage Checklist

- [ ] Ran `check_model.py` - model works
- [ ] Tested on example: `python src/predict_pid.py 2.0 0.7 0.15`
- [ ] Ran full testing: `python src/test_model.py`
- [ ] Ran experiments: `python src/experiments.py`
- [ ] Got statistics: `python src/statistical_analysis.py`

**Done! Now you have all data for the paper! 🎉**

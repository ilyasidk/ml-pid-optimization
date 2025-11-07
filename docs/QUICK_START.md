# ⚡ QUICK START

## 🔧 Note: Use `python3` instead of `python`

On macOS and Linux, the `python` command may not work. Use `python3`:

```bash
python3 src/check_model.py
```

If you have Python installed via pyenv or other version managers, use the appropriate command for your system.

---

## ✅ STEP 1: Check Model

```bash
python3 src/check_model.py
```

Or if you have a virtual environment activated:

```bash
python src/check_model.py
```

---

## 🎯 STEP 2: Use Model

### Quick Prediction:

```bash
python3 src/predict_pid.py 2.5 0.8 0.2
```

### Interactive Mode:

```bash
python3 src/predict_pid.py
```

### Full Testing:

```bash
python3 src/test_model.py
```

### Experiments:

```bash
python3 src/experiments.py
```

### Statistics:

```bash
python3 src/statistical_analysis.py
```

---

## 📝 All Commands with python3:

```bash
# Check
python3 src/check_model.py

# Prediction
python3 src/predict_pid.py 2.0 0.7 0.15

# Testing
python3 src/test_model.py

# Experiments
python3 src/experiments.py

# Statistics
python3 src/statistical_analysis.py
```

---

## 🔄 Alternative: Create Alias (Optional)

If you want to use `python` instead of `python3`, add to your shell config (`~/.zshrc` for zsh or `~/.bashrc` for bash):

```bash
alias python='python3'
alias pip='pip3'
```

Then:
```bash
source ~/.zshrc  # or source ~/.bashrc
```

Now you can use just `python` and `pip`.

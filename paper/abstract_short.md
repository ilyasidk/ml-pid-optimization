# Short Abstract for Conference

## Abstract (150 слов)

PID controllers are fundamental to robotics, but manual tuning requires expertise and time. We present a machine learning approach that predicts optimal PID parameters from robot physical characteristics in milliseconds. Using 10,000 simulated experiments, we trained a neural network (MLP: 3→128→64→32→3) to map robot parameters (mass, friction, inertia) to optimal PID gains (Kp, Ki, Kd). Our method achieves 72.6% average improvement over an adaptive baseline (100% success rate, n=100, p<0.001, Cohen's d=2.22) and 38.7% improvement over Ziegler-Nichols method (90% success rate, n=20, p=0.0008). Testing across diverse robot types shows consistent improvement (71-86%). The approach maintains robustness under sensor noise (tested up to 20%) and provides predictions in <1ms. Statistical analysis confirms highly significant results with large effect sizes. This work demonstrates that ML-based PID tuning can effectively replace manual methods, democratizing robotics by eliminating the need for control theory expertise.

**Keywords:** PID control, machine learning, neural networks, robotics, auto-tuning

---

## Extended Abstract (300 слов)

**Introduction:** PID controllers are ubiquitous in robotics, but manual tuning is time-consuming and requires control theory expertise. Traditional methods like Ziegler-Nichols require system identification experiments, while optimization-based approaches (genetic algorithms) are computationally expensive.

**Methodology:** We generate a dataset of 10,000 simulation experiments with diverse robot configurations (mass: 0.5-5.0 kg, friction: 0.1-2.0, inertia: 0.05-0.5 kg·m²) and random PID parameters. For each configuration, we simulate robot dynamics and evaluate performance using a composite score (settling time + 2×overshoot + 5×steady-state error). We select optimal PID parameters for each unique robot configuration, creating 1,200 training samples. A multi-layer perceptron (3→128→64→32→3) is trained to predict optimal PID gains from robot parameters.

**Results:** The model achieves R²=0.9876 on test data. Evaluation on 100 unseen robot configurations shows 72.6% average improvement over an adaptive baseline heuristic (100% success rate, paired t-test: t=22.07, p<0.001, Cohen's d=2.22). Comparison with Ziegler-Nichols method on 20 cases shows 38.7% average improvement (90% success rate, t=3.96, p=0.0008, Cohen's d=0.91). The method maintains performance under sensor noise (tested 0-20%) and provides predictions in <1ms, compared to 2.3s for Ziegler-Nichols.

**Conclusion:** ML-based PID auto-tuning provides instant, near-optimal parameters without requiring expertise, making robotics more accessible while improving performance over traditional methods.


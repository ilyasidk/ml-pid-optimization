"""
Generate dataset with OPTIMIZED PID parameters using Nelder-Mead optimization.
Fixes the issue of using random search instead of proper optimization.

This generates ground truth optimal PID parameters for training ML models.
"""
import numpy as np
import pandas as pd
from robot_simulator_corrected import RobotSimulator
from pid_optimizer import optimize_pid_nelder_mead
from tqdm import tqdm
from config import DATASET_CSV
import time

# Set random seed for reproducibility
np.random.seed(42)

print("="*60)
print(" GENERATING OPTIMIZED PID DATASET")
print("="*60)
print("This uses Nelder-Mead optimization instead of random search.")
print("This will take MUCH longer (hours), but provides TRUE optimal labels.")
print("="*60)
print()

# Configuration
num_robots = 1000  # Number of unique robot configurations
max_time_per_robot = 5.0  # Maximum optimization time per robot (seconds) - увеличено для лучшей оптимизации
radius = 0.1  # Fixed characteristic radius

dataset = []
optimization_times = []

print(f"Generating optimized dataset for {num_robots} robots...")
print(f"Max optimization time per robot: {max_time_per_robot:.1f} seconds")
print(f"Estimated total time: ~{num_robots * max_time_per_robot / 60:.1f} minutes")
print()

start_total = time.time()

for i in tqdm(range(num_robots), desc="Optimizing robots"):
    # Random robot parameters
    mass = np.random.uniform(0.5, 5.0)
    damping_coeff = np.random.uniform(0.1, 2.0)
    inertia = np.random.uniform(0.05, 0.5)
    
    # Create robot
    robot = RobotSimulator(mass, damping_coeff, inertia, radius)
    
    # Optimize PID using Nelder-Mead (minimizing ITAE)
    opt_start = time.time()
    result = optimize_pid_nelder_mead(
        robot,
        objective='itae',  # Use ITAE as optimization criterion
        max_time=max_time_per_robot,
        duration=5.0
    )
    opt_time = time.time() - opt_start
    optimization_times.append(opt_time)
    
    # Store results
    dataset.append([
        mass, damping_coeff, inertia,  # robot params
        result['Kp'], result['Ki'], result['Kd'],  # OPTIMAL PID params
        result['settling_time'],
        result['overshoot'],
        result['ss_error'],
        result['itae'],  # Primary metric: ITAE
        result['iae'],   # Also store IAE
        result['ise'],   # And ISE
        opt_time  # Optimization time
    ])
    
    if (i + 1) % 50 == 0:
        avg_opt_time = np.mean(optimization_times)
        remaining = (num_robots - i - 1) * avg_opt_time
        print(f"  Progress: {i+1}/{num_robots}")
        print(f"  Avg opt time: {avg_opt_time:.2f}s, Est. remaining: {remaining/60:.1f} min")

total_time = time.time() - start_total

# Save to CSV
df = pd.DataFrame(dataset, columns=[
    'mass', 'damping_coeff', 'inertia',  # Note: damping_coeff, not friction!
    'Kp', 'Ki', 'Kd',
    'settling_time', 'overshoot', 'ss_error',
    'itae', 'iae', 'ise',  # Performance metrics
    'optimization_time'
])

df.to_csv(DATASET_CSV, index=False)

print(f"\n{'='*60}")
print(" DATASET GENERATION COMPLETE")
print("="*60)
print(f"Total robots: {len(df)}")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Average optimization time: {np.mean(optimization_times):.2f} seconds")
print(f"\nDataset statistics:")
print(f"  ITAE range: {df['itae'].min():.3f} to {df['itae'].max():.3f}")
print(f"  ITAE mean: {df['itae'].mean():.3f}, median: {df['itae'].median():.3f}")
print(f"  IAE range: {df['iae'].min():.3f} to {df['iae'].max():.3f}")
print(f"  ISE range: {df['ise'].min():.3f} to {df['ise'].max():.3f}")
print(f"\nSaved to: {DATASET_CSV}")


import numpy as np
import pandas as pd
from robot_simulator import RobotSimulator, test_pid

# Установка seed для воспроизводимости результатов
np.random.seed(42)
from tqdm import tqdm
from config import DATASET_CSV

print("Generating dataset...")
print("This will take 2-4 hours. Run overnight!")

dataset = []

# Generate 10,000 experiments
for i in tqdm(range(10000)):
    # Random robot parameters
    mass = np.random.uniform(0.5, 5.0)
    friction = np.random.uniform(0.1, 2.0)
    inertia = np.random.uniform(0.05, 0.5)
    
    # Random PID parameters
    Kp = np.random.uniform(0.1, 20)
    Ki = np.random.uniform(0, 10)
    Kd = np.random.uniform(0, 5)
    
    # Simulate
    robot = RobotSimulator(mass, friction, inertia)
    result = test_pid(robot, Kp, Ki, Kd)
    
    # Store results
    dataset.append([
        mass, friction, inertia,           # robot params
        Kp, Ki, Kd,                         # PID params
        result['settling_time'],
        result['overshoot'],
        result['ss_error'],
        result['score']                     # performance
    ])

# Save to CSV
df = pd.DataFrame(dataset, columns=[
    'mass', 'friction', 'inertia',
    'Kp', 'Ki', 'Kd',
    'settling_time', 'overshoot', 'ss_error', 'score'
])

df.to_csv(DATASET_CSV, index=False)
print(f"\nDataset saved: {len(df)} samples")
print(f"Score range: {df['score'].min():.2f} to {df['score'].max():.2f}")
"""
PID parameter optimization using various methods.
Provides ground truth optimal parameters for training ML models.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from robot_simulator_corrected import (
    RobotSimulator, test_pid_with_antiwindup,
    compute_iae, compute_itae, compute_ise
)


def objective_iae(pid_params, robot, duration=5.0, dt=0.01):
    """Objective function based on IAE (Integral Absolute Error)."""
    Kp, Ki, Kd = pid_params

    # Ensure positive gains
    if Kp < 0 or Ki < 0 or Kd < 0:
        return 1e6

    # Test controller
    result = test_pid_with_antiwindup(
        robot, Kp, Ki, Kd, duration=duration, dt=dt,
        use_antiwindup=True, use_d_filter=True
    )

    return result['iae']


def objective_itae(pid_params, robot, duration=5.0, dt=0.01):
    """Objective function based on ITAE (Integral Time-weighted Absolute Error)."""
    Kp, Ki, Kd = pid_params

    if Kp < 0 or Ki < 0 or Kd < 0:
        return 1e6

    result = test_pid_with_antiwindup(
        robot, Kp, Ki, Kd, duration=duration, dt=dt,
        use_antiwindup=True, use_d_filter=True
    )

    return result['itae']


def objective_ise(pid_params, robot, duration=5.0, dt=0.01):
    """Objective function based on ISE (Integral Squared Error)."""
    Kp, Ki, Kd = pid_params

    if Kp < 0 or Ki < 0 or Kd < 0:
        return 1e6

    result = test_pid_with_antiwindup(
        robot, Kp, Ki, Kd, duration=duration, dt=dt,
        use_antiwindup=True, use_d_filter=True
    )

    return result['ise']


def objective_combined(pid_params, robot, duration=5.0, dt=0.01,
                       w_iae=0.3, w_itae=0.3, w_overshoot=0.2, w_settling=0.2):
    """
    Combined objective with multiple criteria.

    Args:
        w_iae: Weight for IAE
        w_itae: Weight for ITAE
        w_overshoot: Weight for overshoot penalty
        w_settling: Weight for settling time
    """
    Kp, Ki, Kd = pid_params

    if Kp < 0 or Ki < 0 or Kd < 0:
        return 1e6

    result = test_pid_with_antiwindup(
        robot, Kp, Ki, Kd, duration=duration, dt=dt,
        use_antiwindup=True, use_d_filter=True
    )

    # Normalize metrics for fair weighting
    iae_norm = result['iae'] / robot.target  # Normalize by target
    itae_norm = result['itae'] / (duration * robot.target)
    overshoot_norm = result['overshoot'] / 100  # Already in percentage
    settling_norm = result['settling_time'] / duration

    # Combined score
    score = (w_iae * iae_norm +
             w_itae * itae_norm +
             w_overshoot * overshoot_norm +
             w_settling * settling_norm)

    return score


def optimize_pid_nelder_mead(robot, objective='itae', initial_guess=None,
                             max_time=2.0, duration=5.0):
    """
    Optimize PID parameters using Nelder-Mead simplex method.

    Args:
        robot: RobotSimulator instance
        objective: 'iae', 'itae', 'ise', or 'combined'
        initial_guess: Initial PID values [Kp, Ki, Kd]
        max_time: Maximum optimization time in seconds
        duration: Simulation duration for each evaluation

    Returns:
        Dictionary with optimal parameters and metrics
    """
    # Select objective function
    objectives = {
        'iae': objective_iae,
        'itae': objective_itae,
        'ise': objective_ise,
        'combined': objective_combined
    }
    obj_func = objectives.get(objective, objective_itae)

    # Initial guess - use adaptive baseline if not provided
    if initial_guess is None:
        from robot_simulator_corrected import get_adaptive_baseline_pid
        initial_guess = get_adaptive_baseline_pid(
            robot.mass, robot.damping_coeff, robot.inertia, robot.radius
        )

    # Bounds for parameters - расширены для лучшей оптимизации
    bounds = [(0.1, 100), (0, 50), (0, 20)]  # Kp, Ki, Kd

    # Clip initial guess to bounds
    initial_guess = np.clip(initial_guess, 
                           [b[0] for b in bounds], 
                           [b[1] for b in bounds])

    # Optimization with time limit
    start_time = time.time()

    def callback(xk):
        """Check time limit."""
        if time.time() - start_time > max_time:
            return True
        return False

    # Run optimization
    # Note: Nelder-Mead doesn't support bounds directly
    # We use a penalty function approach to enforce bounds
    def bounded_objective(x):
        # Penalty for violating bounds
        penalty = 0
        for i, (low, high) in enumerate(bounds):
            if x[i] < low:
                penalty += 1e6 * (low - x[i]) ** 2
            elif x[i] > high:
                penalty += 1e6 * (x[i] - high) ** 2
        return obj_func(x, robot, duration=duration) + penalty
    
    result = minimize(
        bounded_objective,
        initial_guess,
        method='Nelder-Mead',
        callback=callback,
        options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}  # Увеличено maxiter
    )
    
    # Clip result to bounds (safety check)
    result.x = np.clip(result.x, [b[0] for b in bounds], [b[1] for b in bounds])

    # Get final performance
    Kp_opt, Ki_opt, Kd_opt = result.x
    final_result = test_pid_with_antiwindup(
        robot, Kp_opt, Ki_opt, Kd_opt, duration=duration,
        use_antiwindup=True, use_d_filter=True
    )

    return {
        'Kp': Kp_opt,
        'Ki': Ki_opt,
        'Kd': Kd_opt,
        'iae': final_result['iae'],
        'itae': final_result['itae'],
        'ise': final_result['ise'],
        'settling_time': final_result['settling_time'],
        'overshoot': final_result['overshoot'],
        'ss_error': final_result['ss_error'],
        'optimization_time': time.time() - start_time,
        'success': result.success,
        'iterations': result.nit
    }


def optimize_pid_differential_evolution(robot, objective='itae', max_time=5.0,
                                       duration=5.0, population_size=15):
    """
    Optimize PID using Differential Evolution (global optimization).

    More robust but slower than Nelder-Mead.
    """
    objectives = {
        'iae': objective_iae,
        'itae': objective_itae,
        'ise': objective_ise,
        'combined': objective_combined
    }
    obj_func = objectives.get(objective, objective_itae)

    # Parameter bounds - расширены для лучшей оптимизации
    bounds = [(0.1, 100), (0, 50), (0, 20)]

    start_time = time.time()

    def callback(xk, convergence):
        """Check time limit."""
        if time.time() - start_time > max_time:
            return True
        return False

    # Run global optimization
    result = differential_evolution(
        lambda x: obj_func(x, robot, duration=duration),
        bounds,
        callback=callback,
        popsize=population_size,
        maxiter=100,
        tol=1e-4,
        seed=42
    )

    # Get final performance
    Kp_opt, Ki_opt, Kd_opt = result.x
    final_result = test_pid_with_antiwindup(
        robot, Kp_opt, Ki_opt, Kd_opt, duration=duration,
        use_antiwindup=True, use_d_filter=True
    )

    return {
        'Kp': Kp_opt,
        'Ki': Ki_opt,
        'Kd': Kd_opt,
        'iae': final_result['iae'],
        'itae': final_result['itae'],
        'ise': final_result['ise'],
        'settling_time': final_result['settling_time'],
        'overshoot': final_result['overshoot'],
        'ss_error': final_result['ss_error'],
        'optimization_time': time.time() - start_time,
        'success': result.success,
        'iterations': result.nit
    }


def compare_optimization_methods(robot, duration=5.0):
    """Compare different optimization methods."""

    print(f"\nOptimizing PID for robot: mass={robot.mass:.2f}, "
          f"damping={robot.damping_coeff:.2f}, inertia={robot.inertia:.3f}")
    print("-" * 60)

    methods = {
        'Nelder-Mead (ITAE)': lambda: optimize_pid_nelder_mead(
            robot, objective='itae', max_time=2.0, duration=duration
        ),
        'Nelder-Mead (IAE)': lambda: optimize_pid_nelder_mead(
            robot, objective='iae', max_time=2.0, duration=duration
        ),
        'Differential Evolution': lambda: optimize_pid_differential_evolution(
            robot, objective='itae', max_time=5.0, duration=duration
        )
    }

    results = {}
    for name, method in methods.items():
        print(f"\n{name}:")
        result = method()
        results[name] = result

        print(f"  Kp={result['Kp']:.3f}, Ki={result['Ki']:.3f}, Kd={result['Kd']:.3f}")
        print(f"  ITAE={result['itae']:.3f}, IAE={result['iae']:.3f}, ISE={result['ise']:.3f}")
        print(f"  Settling={result['settling_time']:.2f}s, Overshoot={result['overshoot']:.1f}%")
        print(f"  Optimization time: {result['optimization_time']:.2f}s")

    return results


def generate_optimal_pid_dataset(num_robots=100, max_time_per_robot=2.0):
    """
    Generate dataset of optimal PID parameters for various robots.

    Args:
        num_robots: Number of different robot configurations
        max_time_per_robot: Maximum optimization time per robot

    Returns:
        Lists of robot parameters and optimal PID values
    """
    np.random.seed(42)

    # Storage
    robot_params = []
    optimal_pids = []
    metrics = []

    print(f"Generating optimal PID dataset for {num_robots} robots...")
    print("This may take several minutes...")

    for i in range(num_robots):
        # Random robot parameters
        mass = np.random.uniform(0.5, 5.0)
        damping = np.random.uniform(0.1, 2.0)
        inertia = np.random.uniform(0.005, 0.5)
        radius = 0.1  # Fixed radius

        # Create robot
        robot = RobotSimulator(mass, damping, inertia, radius)

        # Optimize PID
        result = optimize_pid_nelder_mead(
            robot, objective='itae', max_time=max_time_per_robot
        )

        # Store results
        robot_params.append([mass, damping, inertia])
        optimal_pids.append([result['Kp'], result['Ki'], result['Kd']])
        metrics.append({
            'itae': result['itae'],
            'settling_time': result['settling_time'],
            'overshoot': result['overshoot']
        })

        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{num_robots} robots...")

    print("Dataset generation complete!")

    return {
        'robot_params': np.array(robot_params),
        'optimal_pids': np.array(optimal_pids),
        'metrics': metrics
    }


# Test the optimizer
if __name__ == "__main__":
    # Test robot
    robot = RobotSimulator(
        mass=2.0,
        damping_coeff=0.7,
        inertia=0.15,
        radius=0.1
    )

    # Compare methods
    results = compare_optimization_methods(robot, duration=5.0)

    # Generate small dataset
    print("\n" + "="*60)
    print("Generating small optimal dataset...")
    dataset = generate_optimal_pid_dataset(num_robots=10, max_time_per_robot=1.0)

    print(f"\nDataset shape:")
    print(f"  Robot parameters: {dataset['robot_params'].shape}")
    print(f"  Optimal PIDs: {dataset['optimal_pids'].shape}")

    # Show first few entries
    print("\nFirst 3 entries:")
    for i in range(min(3, len(dataset['robot_params']))):
        rp = dataset['robot_params'][i]
        op = dataset['optimal_pids'][i]
        m = dataset['metrics'][i]
        print(f"  Robot: mass={rp[0]:.2f}, damp={rp[1]:.2f}, inertia={rp[2]:.3f}")
        print(f"    Optimal PID: Kp={op[0]:.2f}, Ki={op[1]:.2f}, Kd={op[2]:.2f}")
        print(f"    ITAE={m['itae']:.3f}, Settling={m['settling_time']:.2f}s")
"""
Analyze optimization quality and detect boundary issues.
Helps diagnose problems with PID optimization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_optimization_results(csv_path):
    """
    Analyze if optimization is hitting boundaries too often.
    """
    # Load data
    df = pd.read_csv(csv_path)

    print("="*60)
    print("OPTIMIZATION QUALITY ANALYSIS")
    print("="*60)

    # Define original and new bounds
    old_bounds = {
        'Kp': (0.1, 50),
        'Ki': (0, 20),
        'Kd': (0, 10)
    }

    new_bounds = {
        'Kp': (0.1, 100),
        'Ki': (0, 50),
        'Kd': (0, 20)
    }

    # Check boundary hits for old bounds
    print("\n1. BOUNDARY HITS (Old Bounds)")
    print("-"*40)
    for param, (low, high) in old_bounds.items():
        at_low = np.sum(np.abs(df[param] - low) < 0.01)
        at_high = np.sum(np.abs(df[param] - high) < 0.01)
        total = len(df)

        pct_low = at_low / total * 100
        pct_high = at_high / total * 100

        print(f"{param}:")
        print(f"  At lower bound ({low}): {at_low}/{total} ({pct_low:.1f}%)")
        print(f"  At upper bound ({high}): {at_high}/{total} ({pct_high:.1f}%)")

        if pct_high > 20:
            print(f"  ⚠️ WARNING: {pct_high:.1f}% at upper bound - consider increasing!")

    # Statistics
    print("\n2. PARAMETER STATISTICS")
    print("-"*40)
    for param in ['Kp', 'Ki', 'Kd']:
        values = df[param]
        print(f"\n{param}:")
        print(f"  Mean:   {values.mean():.3f}")
        print(f"  Median: {values.median():.3f}")
        print(f"  Std:    {values.std():.3f}")
        print(f"  Min:    {values.min():.3f}")
        print(f"  Q25:    {values.quantile(0.25):.3f}")
        print(f"  Q75:    {values.quantile(0.75):.3f}")
        print(f"  Max:    {values.max():.3f}")

        # Check for log-normal distribution (common in PID)
        if param == 'Ki' and values.min() > 0:
            log_values = np.log10(values + 1e-10)
            log_std = log_values.std()
            if log_std > 2:
                print(f"  📊 Note: Ki has log-normal distribution (log-std={log_std:.2f})")

    # Performance metrics
    print("\n3. OPTIMIZATION PERFORMANCE")
    print("-"*40)
    if 'itae' in df.columns:
        print(f"ITAE - Mean: {df['itae'].mean():.3f}, Median: {df['itae'].median():.3f}")
    if 'settling_time' in df.columns:
        settled = np.sum(df['settling_time'] < 5.0) / len(df) * 100
        print(f"Settled within 5s: {settled:.1f}%")
    if 'overshoot' in df.columns:
        low_overshoot = np.sum(df['overshoot'] < 20) / len(df) * 100
        print(f"Overshoot < 20%: {low_overshoot:.1f}%")

    # Correlation analysis
    print("\n4. CORRELATION ANALYSIS")
    print("-"*40)
    robot_params = ['mass', 'damping_coeff', 'inertia']
    pid_params = ['Kp', 'Ki', 'Kd']

    if all(col in df.columns for col in robot_params):
        print("Correlations (Robot → PID):")
        for rp in robot_params:
            for pp in pid_params:
                corr = df[rp].corr(df[pp])
                if abs(corr) > 0.3:
                    print(f"  {rp} → {pp}: {corr:+.3f} {'⚠️' if abs(corr) > 0.7 else ''}")

    # Recommendations
    print("\n5. RECOMMENDATIONS")
    print("-"*40)

    recommendations = []

    # Check Kp
    if df['Kp'].quantile(0.75) >= old_bounds['Kp'][1] * 0.9:
        recommendations.append(f"• Increase Kp upper bound from {old_bounds['Kp'][1]} to {new_bounds['Kp'][1]}")

    # Check Ki
    if df['Ki'].std() / df['Ki'].mean() > 2:
        recommendations.append("• Ki has high variance - consider log-scale optimization")

    # Check Kd
    if df['Kd'].quantile(0.75) >= old_bounds['Kd'][1] * 0.9:
        recommendations.append(f"• Increase Kd upper bound from {old_bounds['Kd'][1]} to {new_bounds['Kd'][1]}")

    # Time recommendation
    if 'optimization_time' in df.columns:
        mean_time = df['optimization_time'].mean()
        if mean_time > 1.8:
            recommendations.append(f"• Optimization time averaging {mean_time:.1f}s - increase time limit")

    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✓ No issues detected")

    return df


def plot_optimization_analysis(df, save_path='optimization_analysis.png'):
    """Create diagnostic plots for optimization quality."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PID Optimization Analysis', fontsize=16)

    # 1. Parameter distributions
    ax = axes[0, 0]
    params = ['Kp', 'Ki', 'Kd']
    positions = [1, 2, 3]

    for i, param in enumerate(params):
        values = df[param].values
        bp = ax.boxplot(values, positions=[positions[i]], widths=0.6,
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor(f'C{i}')

    ax.set_xticks(positions)
    ax.set_xticklabels(params)
    ax.set_ylabel('Parameter Value')
    ax.set_title('PID Parameter Distributions')
    ax.grid(True, alpha=0.3)

    # 2. Log-scale Ki distribution
    ax = axes[0, 1]
    ki_values = df['Ki'].values
    ki_positive = ki_values[ki_values > 0]
    if len(ki_positive) > 0:
        ax.hist(np.log10(ki_positive + 1e-10), bins=30, edgecolor='black')
        ax.set_xlabel('log10(Ki)')
        ax.set_ylabel('Frequency')
        ax.set_title('Ki Distribution (log scale)')
    ax.grid(True, alpha=0.3)

    # 3. Boundary hit visualization
    ax = axes[0, 2]
    boundary_hits = []
    labels = []

    bounds = {'Kp': (0.1, 50), 'Ki': (0, 20), 'Kd': (0, 10)}

    for param, (low, high) in bounds.items():
        at_high = np.sum(np.abs(df[param] - high) < 0.01) / len(df) * 100
        boundary_hits.append(at_high)
        labels.append(f'{param} at {high}')

    bars = ax.bar(labels, boundary_hits, color=['red' if h > 20 else 'green' for h in boundary_hits])
    ax.set_ylabel('% at Upper Bound')
    ax.set_title('Boundary Hits')
    ax.axhline(y=20, color='orange', linestyle='--', label='20% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Performance metrics
    if 'itae' in df.columns:
        ax = axes[1, 0]
        ax.hist(df['itae'], bins=30, edgecolor='black')
        ax.set_xlabel('ITAE')
        ax.set_ylabel('Frequency')
        ax.set_title('ITAE Distribution')
        ax.axvline(x=df['itae'].median(), color='red', linestyle='--',
                   label=f'Median: {df["itae"].median():.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 5. Correlation heatmap
    if all(col in df.columns for col in ['mass', 'damping_coeff', 'inertia']):
        ax = axes[1, 1]
        robot_params = ['mass', 'damping_coeff', 'inertia']
        pid_params = ['Kp', 'Ki', 'Kd']

        corr_matrix = np.zeros((len(robot_params), len(pid_params)))
        for i, rp in enumerate(robot_params):
            for j, pp in enumerate(pid_params):
                corr_matrix[i, j] = df[rp].corr(df[pp])

        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(pid_params)))
        ax.set_xticklabels(pid_params)
        ax.set_yticks(range(len(robot_params)))
        ax.set_yticklabels(robot_params)
        ax.set_title('Robot → PID Correlations')

        # Add correlation values
        for i in range(len(robot_params)):
            for j in range(len(pid_params)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black")

        plt.colorbar(im, ax=ax)

    # 6. Optimization time
    if 'optimization_time' in df.columns:
        ax = axes[1, 2]
        ax.hist(df['optimization_time'], bins=30, edgecolor='black')
        ax.set_xlabel('Optimization Time (s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Optimization Time Distribution')
        ax.axvline(x=2.0, color='red', linestyle='--', label='2s limit')
        ax.axvline(x=df['optimization_time'].mean(), color='green',
                   linestyle='-', label=f'Mean: {df["optimization_time"].mean():.2f}s')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis plots to {save_path}")
    plt.show()

    return fig


if __name__ == "__main__":
    # Check for existing optimized data
    data_path = Path("data/pid_dataset_optimized.csv")

    if data_path.exists():
        print(f"Analyzing: {data_path}")
        df = analyze_optimization_results(data_path)

        # Create diagnostic plots
        plot_optimization_analysis(df, 'results/optimization_analysis.png')

        # Additional checks
        print("\n" + "="*60)
        print("QUICK FIXES IMPLEMENTED:")
        print("-"*40)
        print("✓ Bounds expanded: Kp(0.1-100), Ki(0-50), Kd(0-20)")
        print("✓ Time increased: 2s → 5s per robot")
        print("✓ Better initial guess from adaptive baseline")
        print("✓ Penalty function for bound constraints")

    else:
        print(f"File not found: {data_path}")
        print("Please run: python src/generate_data_optimized.py first")
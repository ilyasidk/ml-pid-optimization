"""
Improved statistical analysis with proper reporting.
Addresses reviewer concerns about statistical rigor.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
import json


def compute_bootstrap_ci(data, statistic=np.mean, confidence_level=0.95,
                         n_bootstrap=10000, random_state=42):
    """
    Compute bootstrap confidence intervals.

    Args:
        data: Array of values
        statistic: Function to compute statistic (mean, median, etc.)
        confidence_level: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with statistic value and CI bounds
    """
    rng = np.random.RandomState(random_state)

    # Wrap data for scipy bootstrap
    data_wrapped = (data,)

    # Define statistic function for bootstrap
    def stat_func(x):
        return statistic(x)

    # Perform bootstrap
    res = bootstrap(data_wrapped, stat_func,
                   n_resamples=n_bootstrap,
                   confidence_level=confidence_level,
                   random_state=rng)

    return {
        'statistic': statistic(data),
        'ci_low': res.confidence_interval.low,
        'ci_high': res.confidence_interval.high,
        'confidence_level': confidence_level
    }


def compute_effect_sizes(group1, group2):
    """
    Compute various effect size measures.

    Args:
        group1: First group (e.g., baseline scores)
        group2: Second group (e.g., ML scores)

    Returns:
        Dictionary with multiple effect size metrics
    """
    # Ensure arrays
    g1 = np.array(group1)
    g2 = np.array(group2)

    # Cohen's d
    mean_diff = np.mean(g2) - np.mean(g1)
    pooled_std = np.sqrt((np.var(g1, ddof=1) + np.var(g2, ddof=1)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    # Glass's delta (uses only control group std)
    glass_delta = mean_diff / np.std(g1, ddof=1) if np.std(g1, ddof=1) > 0 else 0

    # Hedge's g (corrected Cohen's d for small samples)
    n1, n2 = len(g1), len(g2)
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    hedges_g = cohens_d * correction

    # Probability of superiority (non-parametric)
    # P(randomly selected from g2 > randomly selected from g1)
    comparisons = []
    for v2 in g2:
        comparisons.extend([v2 > v1 for v1 in g1])
    prob_superiority = np.mean(comparisons)

    # Cliff's delta (non-parametric effect size)
    cliffs_delta = 2 * prob_superiority - 1

    return {
        'cohens_d': cohens_d,
        'glass_delta': glass_delta,
        'hedges_g': hedges_g,
        'prob_superiority': prob_superiority,
        'cliffs_delta': cliffs_delta
    }


def comprehensive_statistical_test(baseline_scores, ml_scores, alpha=0.05):
    """
    Comprehensive statistical testing with multiple tests.

    Args:
        baseline_scores: Baseline method scores
        ml_scores: ML method scores
        alpha: Significance level

    Returns:
        Dictionary with all test results
    """
    results = {}

    # Basic descriptive statistics
    results['descriptive'] = {
        'baseline': {
            'mean': np.mean(baseline_scores),
            'median': np.median(baseline_scores),
            'std': np.std(baseline_scores, ddof=1),
            'iqr': np.percentile(baseline_scores, 75) - np.percentile(baseline_scores, 25),
            'min': np.min(baseline_scores),
            'max': np.max(baseline_scores),
            'q25': np.percentile(baseline_scores, 25),
            'q75': np.percentile(baseline_scores, 75)
        },
        'ml': {
            'mean': np.mean(ml_scores),
            'median': np.median(ml_scores),
            'std': np.std(ml_scores, ddof=1),
            'iqr': np.percentile(ml_scores, 75) - np.percentile(ml_scores, 25),
            'min': np.min(ml_scores),
            'max': np.max(ml_scores),
            'q25': np.percentile(ml_scores, 25),
            'q75': np.percentile(ml_scores, 75)
        }
    }

    # Improvement metrics
    improvements = (baseline_scores - ml_scores) / baseline_scores * 100
    results['improvements'] = {
        'mean': np.mean(improvements),
        'median': np.median(improvements),
        'std': np.std(improvements, ddof=1),
        'min': np.min(improvements),
        'max': np.max(improvements),
        'positive_rate': np.mean(improvements > 0) * 100  # Success rate
    }

    # Normality tests
    _, baseline_normal_p = stats.shapiro(baseline_scores)
    _, ml_normal_p = stats.shapiro(ml_scores)
    _, diff_normal_p = stats.shapiro(baseline_scores - ml_scores)

    results['normality'] = {
        'baseline_p': baseline_normal_p,
        'ml_p': ml_normal_p,
        'difference_p': diff_normal_p,
        'is_normal': diff_normal_p > alpha
    }

    # Paired tests (for matched samples)
    if len(baseline_scores) == len(ml_scores):
        # Parametric test
        t_stat, t_p = stats.ttest_rel(baseline_scores, ml_scores)
        results['paired_t_test'] = {
            'statistic': t_stat,
            'p_value': t_p,
            'significant': t_p < alpha
        }

        # Non-parametric test
        w_stat, w_p = stats.wilcoxon(baseline_scores, ml_scores)
        results['wilcoxon'] = {
            'statistic': w_stat,
            'p_value': w_p,
            'significant': w_p < alpha
        }

        # Sign test (most robust)
        n_pos = np.sum(baseline_scores > ml_scores)
        n_neg = np.sum(ml_scores > baseline_scores)
        n_ties = np.sum(baseline_scores == ml_scores)
        sign_p = stats.binomtest(n_pos, n_pos + n_neg, p=0.5).pvalue if (n_pos + n_neg) > 0 else 1.0
        results['sign_test'] = {
            'positive': n_pos,
            'negative': n_neg,
            'ties': n_ties,
            'p_value': sign_p,
            'significant': sign_p < alpha
        }

    # Effect sizes
    results['effect_sizes'] = compute_effect_sizes(baseline_scores, ml_scores)

    # Bootstrap confidence intervals
    results['bootstrap'] = {
        'mean_improvement': compute_bootstrap_ci(improvements, np.mean),
        'median_improvement': compute_bootstrap_ci(improvements, np.median),
        'success_rate': compute_bootstrap_ci(improvements > 0, np.mean)
    }

    return results


def plot_statistical_comparison(baseline_scores, ml_scores, save_path=None):
    """Create comprehensive statistical comparison plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Statistical Comparison: Baseline vs ML', fontsize=16)

    # 1. Box plots
    ax = axes[0, 0]
    data = pd.DataFrame({
        'Baseline': baseline_scores,
        'ML': ml_scores
    })
    data.boxplot(ax=ax)
    ax.set_ylabel('Score (lower is better)')
    ax.set_title('Score Distribution')
    ax.grid(True, alpha=0.3)

    # 2. Violin plots
    ax = axes[0, 1]
    parts = ax.violinplot([baseline_scores, ml_scores],
                          positions=[1, 2], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Baseline', 'ML'])
    ax.set_ylabel('Score')
    ax.set_title('Violin Plot Comparison')
    ax.grid(True, alpha=0.3)

    # 3. Improvement histogram
    ax = axes[0, 2]
    improvements = (baseline_scores - ml_scores) / baseline_scores * 100
    ax.hist(improvements, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', label='No improvement')
    ax.axvline(x=np.median(improvements), color='green', linestyle='-',
               label=f'Median: {np.median(improvements):.1f}%')
    ax.set_xlabel('Improvement (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Improvement Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Paired scatter plot
    ax = axes[1, 0]
    ax.scatter(baseline_scores, ml_scores, alpha=0.6)
    min_val = min(np.min(baseline_scores), np.min(ml_scores))
    max_val = max(np.max(baseline_scores), np.max(ml_scores))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Equal performance')
    ax.set_xlabel('Baseline Score')
    ax.set_ylabel('ML Score')
    ax.set_title('Paired Performance (points below line = ML better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Q-Q plot for normality check
    ax = axes[1, 1]
    differences = baseline_scores - ml_scores
    stats.probplot(differences, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (check normality of differences)')
    ax.grid(True, alpha=0.3)

    # 6. Effect size visualization
    ax = axes[1, 2]
    effect_sizes = compute_effect_sizes(baseline_scores, ml_scores)
    effect_names = ['Cohen\'s d', 'Hedge\'s g', 'Glass Δ', 'Cliff\'s δ']
    effect_values = [effect_sizes['cohens_d'], effect_sizes['hedges_g'],
                     effect_sizes['glass_delta'], effect_sizes['cliffs_delta']]

    bars = ax.bar(effect_names, effect_values, color=['blue', 'green', 'orange', 'red'])
    ax.set_ylabel('Effect Size')
    ax.set_title('Effect Size Comparison')
    ax.axhline(y=0.2, color='gray', linestyle=':', label='Small')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Medium')
    ax.axhline(y=0.8, color='gray', linestyle='-', label='Large')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, effect_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved statistical plots to {save_path}")

    plt.close()  # Close figure instead of showing
    return fig


def generate_latex_table(results):
    """Generate LaTeX table for paper."""

    latex = r"""
\begin{table}[h]
\centering
\caption{Statistical Comparison Results}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Baseline} & \textbf{ML Model} \\
\hline
Mean Score & %.2f & %.2f \\
Median Score & %.2f & %.2f \\
Std. Deviation & %.2f & %.2f \\
IQR & %.2f & %.2f \\
\hline
\multicolumn{3}{c}{\textbf{Improvement Metrics}} \\
\hline
Mean Improvement & \multicolumn{2}{c}{%.1f\%% (95\%% CI: [%.1f, %.1f])} \\
Median Improvement & \multicolumn{2}{c}{%.1f\%% (95\%% CI: [%.1f, %.1f])} \\
Success Rate & \multicolumn{2}{c}{%.1f\%%} \\
\hline
\multicolumn{3}{c}{\textbf{Statistical Tests}} \\
\hline
Paired t-test & \multicolumn{2}{c}{t = %.2f, p %s} \\
Wilcoxon test & \multicolumn{2}{c}{W = %.1f, p %s} \\
\hline
\multicolumn{3}{c}{\textbf{Effect Sizes}} \\
\hline
Cohen's d & \multicolumn{2}{c}{%.2f} \\
Hedge's g & \multicolumn{2}{c}{%.2f} \\
Cliff's delta & \multicolumn{2}{c}{%.2f} \\
\hline
\end{tabular}
\end{table}
""" % (
        results['descriptive']['baseline']['mean'],
        results['descriptive']['ml']['mean'],
        results['descriptive']['baseline']['median'],
        results['descriptive']['ml']['median'],
        results['descriptive']['baseline']['std'],
        results['descriptive']['ml']['std'],
        results['descriptive']['baseline']['iqr'],
        results['descriptive']['ml']['iqr'],
        results['improvements']['mean'],
        results['bootstrap']['mean_improvement']['ci_low'],
        results['bootstrap']['mean_improvement']['ci_high'],
        results['improvements']['median'],
        results['bootstrap']['median_improvement']['ci_low'],
        results['bootstrap']['median_improvement']['ci_high'],
        results['improvements']['positive_rate'],
        results['paired_t_test']['statistic'],
        format_p_value(results['paired_t_test']['p_value']),
        results['wilcoxon']['statistic'],
        format_p_value(results['wilcoxon']['p_value']),
        results['effect_sizes']['cohens_d'],
        results['effect_sizes']['hedges_g'],
        results['effect_sizes']['cliffs_delta']
    )

    return latex


def format_p_value(p):
    """Format p-value for publication."""
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"= {p:.3f}"
    elif p < 0.05:
        return f"= {p:.3f}"
    else:
        return f"= {p:.3f} (ns)"


# Test the improved statistical analysis
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Try to load real experiment results
    project_root = Path(__file__).parent.parent
    experiment_results_path = project_root / "results" / "experiment_results.npy"
    
    if experiment_results_path.exists():
        print("=" * 60)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("Using REAL experiment data")
        print("=" * 60)
        
        # Load experiment results
        all_results = np.load(experiment_results_path, allow_pickle=True).item()
        exp4 = all_results.get('exp4_accuracy', {})
        
        if exp4:
            # Extract ITAE scores (lower is better)
            ml_itae = np.array(exp4.get('ml_itae', []))
            baseline_itae = np.array(exp4.get('baseline_itae', []))
            cc_itae = np.array(exp4.get('cc_itae', []))
            chr_itae = np.array(exp4.get('chr_itae', []))
            
            print(f"\nLoaded {len(ml_itae)} test cases from experiments")
            
            # Analyze ML vs Baseline
            print("\n" + "=" * 60)
            print("ANALYSIS 1: ML vs Adaptive Baseline")
            print("=" * 60)
            results_baseline = comprehensive_statistical_test(baseline_itae, ml_itae)
            
            # Analyze ML vs Cohen-Coon
            print("\n" + "=" * 60)
            print("ANALYSIS 2: ML vs Cohen-Coon")
            print("=" * 60)
            results_cc = comprehensive_statistical_test(cc_itae, ml_itae)
            
            # Analyze ML vs CHR
            print("\n" + "=" * 60)
            print("ANALYSIS 3: ML vs CHR")
            print("=" * 60)
            results_chr = comprehensive_statistical_test(chr_itae, ml_itae)
            
            # Store all results
            all_analyses = {
                'ml_vs_baseline': results_baseline,
                'ml_vs_cc': results_cc,
                'ml_vs_chr': results_chr
            }
            
            # Print results for each comparison
            for name, results in [('ML vs Baseline', results_baseline), 
                                  ('ML vs Cohen-Coon', results_cc),
                                  ('ML vs CHR', results_chr)]:
                print(f"\n{'='*60}")
                print(f"{name.upper()}")
                print("="*60)
                
                print("\n1. DESCRIPTIVE STATISTICS")
                print("-" * 40)
                baseline_name = name.split(' vs ')[1]
                print(f"{baseline_name}:")
                for key, val in results['descriptive']['baseline'].items():
                    print(f"  {key:10s}: {val:8.2f}")
                print("\nML Model:")
                for key, val in results['descriptive']['ml'].items():
                    print(f"  {key:10s}: {val:8.2f}")
                
                print("\n2. IMPROVEMENT METRICS")
                print("-" * 40)
                imp = results['improvements']
                print(f"Mean improvement:   {imp['mean']:.1f}%")
                print(f"Median improvement: {imp['median']:.1f}%")
                print(f"Success rate:       {imp['positive_rate']:.1f}%")
                print(f"Range:              [{imp['min']:.1f}%, {imp['max']:.1f}%]")
                
                print("\n3. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
                print("-" * 40)
                for metric_name, metric_data in results['bootstrap'].items():
                    print(f"{metric_name}:")
                    print(f"  Point estimate: {metric_data['statistic']:.3f}")
                    print(f"  95% CI: [{metric_data['ci_low']:.3f}, {metric_data['ci_high']:.3f}]")
                
                print("\n4. HYPOTHESIS TESTS")
                print("-" * 40)
                print(f"Normality of differences: p = {results['normality']['difference_p']:.4f}")
                print(f"  -> Use {'parametric' if results['normality']['is_normal'] else 'non-parametric'} tests\n")
                
                print(f"Paired t-test:")
                print(f"  t = {results['paired_t_test']['statistic']:.3f}")
                print(f"  p = {format_p_value(results['paired_t_test']['p_value'])}")
                
                print(f"\nWilcoxon signed-rank test:")
                print(f"  W = {results['wilcoxon']['statistic']:.1f}")
                print(f"  p = {format_p_value(results['wilcoxon']['p_value'])}")
                
                print(f"\nSign test:")
                st = results['sign_test']
                print(f"  Baseline better: {st['positive']} cases")
                print(f"  ML better:       {st['negative']} cases")
                print(f"  Ties:            {st['ties']} cases")
                print(f"  p = {format_p_value(st['p_value'])}")
                
                print("\n5. EFFECT SIZES")
                print("-" * 40)
                es = results['effect_sizes']
                print(f"Cohen's d:            {es['cohens_d']:.3f} {'(large)' if abs(es['cohens_d']) > 0.8 else '(medium)' if abs(es['cohens_d']) > 0.5 else '(small)'}")
                print(f"Hedge's g:            {es['hedges_g']:.3f}")
                print(f"Glass's delta:        {es['glass_delta']:.3f}")
                print(f"Cliff's delta:        {es['cliffs_delta']:.3f}")
                print(f"Prob. of superiority: {es['prob_superiority']:.3f}")
            
            # Generate plots for each comparison
            print("\n6. GENERATING VISUALIZATIONS...")
            print("-" * 40)
            results_dir = project_root / "results"
            results_dir.mkdir(exist_ok=True)
            
            plot_statistical_comparison(baseline_itae, ml_itae,
                                      save_path=str(results_dir / 'statistical_comparison_baseline.png'))
            plot_statistical_comparison(cc_itae, ml_itae,
                                      save_path=str(results_dir / 'statistical_comparison_cc.png'))
            plot_statistical_comparison(chr_itae, ml_itae,
                                      save_path=str(results_dir / 'statistical_comparison_chr.png'))
            
            # Generate LaTeX tables
            print("\n7. LATEX TABLES FOR PAPER")
            print("-" * 40)
            print("\n--- ML vs Baseline ---")
            print(generate_latex_table(results_baseline))
            print("\n--- ML vs Cohen-Coon ---")
            print(generate_latex_table(results_cc))
            print("\n--- ML vs CHR ---")
            print(generate_latex_table(results_chr))
            
            # Save results to JSON
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            output_path = results_dir / 'statistical_results_improved.json'
            with open(output_path, 'w') as f:
                json.dump(convert_numpy(all_analyses), f, indent=2)
            print(f"\nResults saved to {output_path}")
            
        else:
            print("ERROR: No exp4_accuracy data found in experiment_results.npy")
            sys.exit(1)
    else:
        # Fallback to synthetic data for testing
        print("=" * 60)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("Using SYNTHETIC test data (experiment_results.npy not found)")
        print("=" * 60)
        
        np.random.seed(42)
        n_samples = 100
        baseline_scores = np.random.gamma(shape=2, scale=50, size=n_samples)
        ml_improvement = np.random.uniform(0.3, 0.8, size=n_samples)
        ml_scores = baseline_scores * ml_improvement + np.random.normal(0, 5, size=n_samples)
        ml_scores = np.maximum(ml_scores, 1)
        
        results = comprehensive_statistical_test(baseline_scores, ml_scores)
        
        # Print results for synthetic data
        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 40)
        print("Baseline:")
        for key, val in results['descriptive']['baseline'].items():
            print(f"  {key:10s}: {val:8.2f}")
        print("\nML Model:")
        for key, val in results['descriptive']['ml'].items():
            print(f"  {key:10s}: {val:8.2f}")

        print("\n2. IMPROVEMENT METRICS")
        print("-" * 40)
        imp = results['improvements']
        print(f"Mean improvement:   {imp['mean']:.1f}%")
        print(f"Median improvement: {imp['median']:.1f}%")
        print(f"Success rate:       {imp['positive_rate']:.1f}%")
        print(f"Range:              [{imp['min']:.1f}%, {imp['max']:.1f}%]")

        print("\n3. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
        print("-" * 40)
        for metric_name, metric_data in results['bootstrap'].items():
            print(f"{metric_name}:")
            print(f"  Point estimate: {metric_data['statistic']:.3f}")
            print(f"  95% CI: [{metric_data['ci_low']:.3f}, {metric_data['ci_high']:.3f}]")

        print("\n4. HYPOTHESIS TESTS")
        print("-" * 40)
        print(f"Normality of differences: p = {results['normality']['difference_p']:.4f}")
        print(f"  -> Use {'parametric' if results['normality']['is_normal'] else 'non-parametric'} tests\n")

        print(f"Paired t-test:")
        print(f"  t = {results['paired_t_test']['statistic']:.3f}")
        print(f"  p = {format_p_value(results['paired_t_test']['p_value'])}")

        print(f"\nWilcoxon signed-rank test:")
        print(f"  W = {results['wilcoxon']['statistic']:.1f}")
        print(f"  p = {format_p_value(results['wilcoxon']['p_value'])}")

        print(f"\nSign test:")
        st = results['sign_test']
        print(f"  Baseline better: {st['positive']} cases")
        print(f"  ML better:       {st['negative']} cases")
        print(f"  Ties:            {st['ties']} cases")
        print(f"  p = {format_p_value(st['p_value'])}")

        print("\n5. EFFECT SIZES")
        print("-" * 40)
        es = results['effect_sizes']
        print(f"Cohen's d:            {es['cohens_d']:.3f} {'(large)' if abs(es['cohens_d']) > 0.8 else '(medium)' if abs(es['cohens_d']) > 0.5 else '(small)'}")
        print(f"Hedge's g:            {es['hedges_g']:.3f}")
        print(f"Glass's delta:        {es['glass_delta']:.3f}")
        print(f"Cliff's delta:        {es['cliffs_delta']:.3f}")
        print(f"Prob. of superiority: {es['prob_superiority']:.3f}")

        # Generate plots
        print("\n6. GENERATING VISUALIZATIONS...")
        print("-" * 40)
        plot_statistical_comparison(baseline_scores, ml_scores,
                                   save_path='statistical_comparison.png')

        # Generate LaTeX table
        print("\n7. LATEX TABLE FOR PAPER")
        print("-" * 40)
        latex_table = generate_latex_table(results)
        print(latex_table)

        # Save results to JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        with open('statistical_results_improved.json', 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
        print("\nResults saved to statistical_results_improved.json")
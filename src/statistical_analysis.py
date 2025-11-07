"""
Статистический анализ результатов экспериментов (V2 - исправленная версия)
Анализ улучшенных экспериментов с адаптивным baseline и Ziegler-Nichols
"""
import numpy as np
from scipy import stats
import json
from config import EXPERIMENT_RESULTS, STATISTICAL_RESULTS

# Загрузка результатов экспериментов
results = np.load(EXPERIMENT_RESULTS, allow_pickle=True).item()

# Извлечение данных
exp4 = results['exp4_accuracy']
ml_scores = np.array(exp4['ml_scores'])
baseline_scores = np.array(exp4['baseline_scores'])
improvements_baseline = np.array(exp4['improvements_baseline'])
improvements_zn = np.array(exp4['improvements_zn']) if len(exp4['improvements_zn']) > 0 else None

print("="*60)
print(" STATISTICAL ANALYSIS V2 - IMPROVED EXPERIMENTS")
print("="*60)
print()

print(f"📊 Analyzing {len(ml_scores)} test cases")
print()

# =============================================================================
# 1. ML vs Adaptive Baseline
# =============================================================================
print("="*60)
print("COMPARISON: ML vs Adaptive Baseline")
print("="*60)

# Парный t-тест
t_stat_baseline, p_value_baseline = stats.ttest_rel(baseline_scores, ml_scores)

print(f"\nPaired t-test:")
print(f"t-statistic: {t_stat_baseline:.4f}")
print(f"p-value: {p_value_baseline:.10f}")
if p_value_baseline < 0.05:
    print("✅ Statistically significant (p < 0.05)")
if p_value_baseline < 0.01:
    print("✅ Highly significant (p < 0.01)")
if p_value_baseline < 0.001:
    print("✅ Very highly significant (p < 0.001)")

# Cohen's d (размер эффекта)
diff = baseline_scores - ml_scores
cohens_d_baseline = np.mean(diff) / np.std(diff)

print(f"\n{'='*60}")
print("Effect size (Cohen's d)")
print("="*60)
print(f"Cohen's d: {cohens_d_baseline:.4f}")
if abs(cohens_d_baseline) < 0.2:
    effect_size = "negligible"
elif abs(cohens_d_baseline) < 0.5:
    effect_size = "small"
elif abs(cohens_d_baseline) < 0.8:
    effect_size = "medium"
else:
    effect_size = "large"
print(f"Effect size: {effect_size}")

# Описательная статистика
print(f"\n{'='*60}")
print("Summary Statistics")
print("="*60)

import pandas as pd
df = pd.DataFrame({
    'ML': ml_scores,
    'Baseline': baseline_scores
})
print(df.describe())

print(f"\n{'-'*60}")
print("Additional Statistics:")
print("-"*60)
print(f"\nML Scores:")
print(f"  Mean: {np.mean(ml_scores):.4f}")
print(f"  Median: {np.median(ml_scores):.4f}")
print(f"  Std: {np.std(ml_scores, ddof=1):.4f}")
print(f"  Min: {np.min(ml_scores):.4f}")
print(f"  Max: {np.max(ml_scores):.4f}")

print(f"\nBaseline Scores:")
print(f"  Mean: {np.mean(baseline_scores):.4f}")
print(f"  Median: {np.median(baseline_scores):.4f}")
print(f"  Std: {np.std(baseline_scores, ddof=1):.4f}")
print(f"  Min: {np.min(baseline_scores):.4f}")
print(f"  Max: {np.max(baseline_scores):.4f}")

print(f"\nImprovement (%) over Baseline:")
print(f"  Mean: {np.mean(improvements_baseline):.2f}%")
print(f"  Median: {np.median(improvements_baseline):.2f}%")
print(f"  Std: {np.std(improvements_baseline, ddof=1):.2f}%")
success_rate_baseline = sum(1 for i in improvements_baseline if i > 0) / len(improvements_baseline) * 100
print(f"  Success rate (>0%): {success_rate_baseline:.1f}%")

# Wilcoxon signed-rank test (непараметрический)
print(f"\n{'='*60}")
print("Wilcoxon signed-rank test (non-parametric)")
print("="*60)
wilcoxon_stat_baseline, wilcoxon_p_baseline = stats.wilcoxon(baseline_scores, ml_scores)

# Проверка направления различий
diff_wilcoxon = baseline_scores - ml_scores
all_positive = np.all(diff_wilcoxon > 0)
all_negative = np.all(diff_wilcoxon < 0)

print(f"Statistic: {wilcoxon_stat_baseline:.4f}")
print(f"p-value: {wilcoxon_p_baseline:.10f}")

# Объяснение статистики = 0
if wilcoxon_stat_baseline == 0.0:
    if all_positive:
        print("ℹ️  Statistic = 0 означает, что ML лучше baseline во ВСЕХ случаях")
        print("   (все разности положительные → максимальная значимость)")
    elif all_negative:
        print("ℹ️  Statistic = 0 означает, что baseline лучше ML во ВСЕХ случаях")
        print("   (все разности отрицательные → максимальная значимость)")

if wilcoxon_p_baseline < 0.05:
    print("✅ Statistically significant (p < 0.05)")
if wilcoxon_p_baseline < 0.01:
    print("✅ Highly significant (p < 0.01)")
if wilcoxon_p_baseline < 0.001:
    print("✅ Very highly significant (p < 0.001)")
if wilcoxon_p_baseline < 1e-10:
    print("✅ Extremely significant (p < 1e-10) - максимальная значимость!")

# =============================================================================
# 2. ML vs Ziegler-Nichols (if available)
# =============================================================================
if improvements_zn is not None and len(improvements_zn) > 0:
    print(f"\n{'='*60}")
    print(f"COMPARISON: ML vs Ziegler-Nichols ({len(improvements_zn)} cases)")
    print("="*60)

    zn_scores = np.array(exp4['zn_scores'])
    ml_scores_zn = ml_scores[:len(zn_scores)]

    # t-test
    t_stat_zn, p_value_zn = stats.ttest_rel(zn_scores, ml_scores_zn)
    print(f"\nPaired t-test:")
    print(f"t-statistic: {t_stat_zn:.4f}")
    print(f"p-value: {p_value_zn:.10f}")

    # Cohen's d
    diff_zn = zn_scores - ml_scores_zn
    cohens_d_zn = np.mean(diff_zn) / np.std(diff_zn)
    print(f"\nCohen's d: {cohens_d_zn:.4f}")

    print(f"\nImprovement (%) over Ziegler-Nichols:")
    print(f"  Mean: {np.mean(improvements_zn):.2f}%")
    print(f"  Median: {np.median(improvements_zn):.2f}%")
    print(f"  Std: {np.std(improvements_zn, ddof=1):.2f}%")
    success_rate_zn = sum(1 for i in improvements_zn if i > 0) / len(improvements_zn) * 100
    print(f"  Success rate (>0%): {success_rate_zn:.1f}%")

# =============================================================================
# 3. Noise Robustness Analysis
# =============================================================================
print(f"\n{'='*60}")
print("NOISE ROBUSTNESS ANALYSIS")
print("="*60)

exp3 = results['exp3_noise']
noise_levels, ml_noise_scores, baseline_noise_scores = exp3

print(f"\nML Method - Score degradation with noise:")
for i, noise in enumerate(noise_levels):
    degradation = (ml_noise_scores[i] - ml_noise_scores[0]) / ml_noise_scores[0] * 100
    print(f"  {noise*100:.0f}% noise: {ml_noise_scores[i]:.2f} (degradation: {degradation:+.1f}%)")

print(f"\nBaseline Method - Score degradation with noise:")
for i, noise in enumerate(noise_levels):
    degradation = (baseline_noise_scores[i] - baseline_noise_scores[0]) / baseline_noise_scores[0] * 100
    print(f"  {noise*100:.0f}% noise: {baseline_noise_scores[i]:.2f} (degradation: {degradation:+.1f}%)")

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*60}")
print("Results Summary (for paper)")
print("="*60)
print(f"Sample size: {len(ml_scores)}")
print(f"\nML vs Adaptive Baseline:")
print(f"  t-statistic: {t_stat_baseline:.4f}")
print(f"  p-value: {p_value_baseline:.10f}")
print(f"  Cohen's d: {cohens_d_baseline:.4f}")
print(f"  Mean improvement: {np.mean(improvements_baseline):.2f}%")
print(f"  Success rate: {success_rate_baseline:.1f}%")

if improvements_zn is not None and len(improvements_zn) > 0:
    print(f"\nML vs Ziegler-Nichols ({len(improvements_zn)} cases):")
    print(f"  t-statistic: {t_stat_zn:.4f}")
    print(f"  p-value: {p_value_zn:.10f}")
    print(f"  Cohen's d: {cohens_d_zn:.4f}")
    print(f"  Mean improvement: {np.mean(improvements_zn):.2f}%")
    print(f"  Success rate: {success_rate_zn:.1f}%")

# Сохранение результатов
statistical_results = {
    "baseline_comparison": {
        "t_statistic": float(t_stat_baseline),
        "p_value": float(p_value_baseline),
        "cohens_d": float(cohens_d_baseline),
        "mean_improvement": float(np.mean(improvements_baseline)),
        "median_improvement": float(np.median(improvements_baseline)),
        "success_rate": float(success_rate_baseline),
        "n_samples": int(len(ml_scores))
    }
}

if improvements_zn is not None and len(improvements_zn) > 0:
    statistical_results["zn_comparison"] = {
        "t_statistic": float(t_stat_zn),
        "p_value": float(p_value_zn),
        "cohens_d": float(cohens_d_zn),
        "mean_improvement": float(np.mean(improvements_zn)),
        "median_improvement": float(np.median(improvements_zn)),
        "success_rate": float(success_rate_zn),
        "n_samples": int(len(improvements_zn))
    }

with open(STATISTICAL_RESULTS, 'w') as f:
    json.dump(statistical_results, f, indent=2)

print(f"\n✅ Statistical results saved to '{STATISTICAL_RESULTS.name}'")
print()
print("="*60)
print(" ANALYSIS COMPLETE - MUCH MORE REALISTIC RESULTS!")
print("="*60)

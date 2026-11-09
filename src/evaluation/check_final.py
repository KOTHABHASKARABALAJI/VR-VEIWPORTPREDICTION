import os, json, pandas as pd

final_dir = 'd:/ml final project/results/day3/final'
print('=== Final Deliverables Directory ===')
for f in sorted(os.listdir(final_dir)):
    p = os.path.join(final_dir, f)
    if os.path.isdir(p):
        sub = os.listdir(p)
        print(f'  {f}/ ({len(sub)} files)')
    else:
        size_kb = os.path.getsize(p) / 1024
        print(f'  {f} ({size_kb:.1f} KB)')

print()
print('=== Budget Compliance Check ===')
df = pd.read_csv(os.path.join(final_dir, 'final_allocation_comparison.csv'))
violations = df[df['simulated_bitrate_kbps'] > df['bandwidth_budget_kbps'] + 0.001]
if len(violations) == 0:
    print('  PASS: No budget violations in any of the 24 rows.')
else:
    print('  FAIL:', violations[['bandwidth_budget_kbps','strategy','simulated_bitrate_kbps']])

print()
print('=== Key Metric Spot-Check ===')
df2 = pd.read_csv(os.path.join(final_dir, 'final_prediction_metrics_overall.csv'))
print(df2[['Model / Baseline','Spherical MAE (deg)','MAE Reduction vs Pers (%)']].to_string())

print()
print('=== Statistical Analysis Summary ===')
df3 = pd.read_csv(os.path.join(final_dir, 'statistical_analysis.csv'))
for _, r in df3.iterrows():
    line = f"  [{r['analysis']}]"
    if 'mean' in r.index and pd.notna(r.get('mean', float('nan'))):
        line += f" mean={float(r['mean']):.4f}"
    if 'ci_low_95' in r.index and pd.notna(r.get('ci_low_95', float('nan'))):
        line += f" 95%CI=[{float(r['ci_low_95']):.4f},{float(r['ci_high_95']):.4f}]"
    if 'p_value_one_sided' in r.index and pd.notna(r.get('p_value_one_sided', float('nan'))):
        line += f" p={float(r['p_value_one_sided']):.6f} t={float(r['t_statistic']):.4f} d={float(r['cohens_d']):.4f}"
    print(line)

print()
print('=== NaN/Inf Check on prediction CSV ===')
pred = pd.read_csv('d:/ml final project/results/predictions/final_model_test_predictions.csv')
key_cols = ['actual_yaw_deg','actual_pitch_deg','predicted_yaw_deg','predicted_pitch_deg','spherical_error_deg','persistence_sph_err_deg']
for c in key_cols:
    n_bad = pred[c].isna().sum() + (pred[c] == float('inf')).sum() + (pred[c] == float('-inf')).sum()
    print(f'  {c}: {n_bad} NaN/Inf values')


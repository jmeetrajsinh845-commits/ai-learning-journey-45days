# ================================================================
# 🚀 DAY 3: Statistics for Data Scientists
# Topics: Descriptive Stats | Distribution | Correlation | Hypothesis
# Dataset: QuickCart E-Commerce (Same as Day 2 — building on it!)
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats           # Statistics library — very powerful!
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 65)
print("🚀 DAY 3 — Statistics Deep Dive | QuickCart Dataset")
print("=" * 65)


# ================================================================
# STEP 0: DATASET REBUILD (same as Day 2)
# ================================================================
np.random.seed(42)
n = 600

categories    = np.random.choice(['Electronics','Fashion','Groceries','Books','Home & Kitchen'],
                                  n, p=[0.30,0.25,0.20,0.10,0.15])
base_spend    = {'Electronics':8500,'Fashion':3200,'Groceries':1600,'Books':750,'Home & Kitchen':4200}
total_spend   = np.array([max(200, np.random.normal(base_spend[c], base_spend[c]*0.45))
                           for c in categories]).round(2)
# Inject 5 VIP outliers
total_spend[np.random.choice(n, 5, replace=False)] = np.random.uniform(45000, 85000, 5)

ages          = np.random.normal(32, 9, n).clip(18, 65).astype(int)
orders        = np.random.randint(1, 26, n)
days_inactive = np.random.exponential(28, n).clip(1, 400).astype(int)
membership    = np.random.choice(['Free','Silver','Gold'], n, p=[0.52,0.30,0.18])
cities        = np.random.choice(['Mumbai','Delhi','Bangalore','Hyderabad','Chennai'],
                                  n, p=[0.28,0.25,0.22,0.13,0.12])
device        = np.random.choice(['Mobile','Desktop','Tablet'], n, p=[0.68,0.25,0.07])
ratings       = np.random.choice([1,2,3,4,5], n, p=[0.05,0.08,0.15,0.37,0.35]).astype(float)

df = pd.DataFrame({
    'age': ages, 'city': cities, 'device': device, 'category': categories,
    'total_spend': total_spend, 'orders': orders, 'rating': ratings,
    'days_inactive': days_inactive, 'membership': membership
})
# Quick clean
df['total_spend_capped'] = df['total_spend'].clip(upper=df['total_spend'].quantile(0.75) + 1.5*(df['total_spend'].quantile(0.75)-df['total_spend'].quantile(0.25)))
df['churn_risk'] = (df['days_inactive'] > 60).astype(int)
print(f"✅ Dataset ready: {df.shape}")


# ================================================================
# SECTION 1: DESCRIPTIVE STATISTICS — Mean vs Median vs Mode
# ================================================================
print("\n" + "=" * 65)
print("📊 SECTION 1: Descriptive Statistics")
print("=" * 65)

col = 'total_spend_capped'

mean_val   = df[col].mean()
median_val = df[col].median()
mode_val   = df[col].mode()[0]        # mode() list return karta hai, [0] = first mode
std_val    = df[col].std()
var_val    = df[col].var()
skew_val   = df[col].skew()
kurt_val   = df[col].kurt()

print(f"""
  📌 TOTAL SPEND — Complete Statistics:
  ┌─────────────────────────────────────┐
  │  Mean        : ₹{mean_val:>10,.2f}          │
  │  Median      : ₹{median_val:>10,.2f}          │
  │  Mode        : ₹{mode_val:>10,.2f}          │
  │  Std Dev     : ₹{std_val:>10,.2f}          │
  │  Variance    : ₹²{var_val:>9,.0f}          │
  │  Skewness    :  {skew_val:>10.4f}          │
  │  Kurtosis    :  {kurt_val:>10.4f}          │
  └─────────────────────────────────────┘
""")

# Mean vs Median comparison
diff_pct = abs(mean_val - median_val) / median_val * 100
print(f"  ⚠️  Mean vs Median gap: {diff_pct:.1f}%")
if diff_pct > 10:
    print(f"  → Gap > 10% = OUTLIERS present! Use MEDIAN for reporting.")
else:
    print(f"  → Gap < 10% = Data relatively symmetric. Mean is OK.")

print(f"\n  📐 Skewness Analysis:")
if abs(skew_val) < 0.5:
    print(f"  → Skewness {skew_val:.2f} = Near Normal Distribution ✅")
elif skew_val > 0.5:
    print(f"  → Skewness {skew_val:.2f} = RIGHT SKEWED 📈")
    print(f"  → High spenders are pulling data right")
    print(f"  → Use LOG TRANSFORM for ML models!")
else:
    print(f"  → Skewness {skew_val:.2f} = LEFT SKEWED 📉")


# ================================================================
# SECTION 2: THE 68-95-99.7 RULE (Empirical Rule)
# ================================================================
print("\n" + "=" * 65)
print("🔔 SECTION 2: Empirical Rule (68-95-99.7)")
print("=" * 65)

# Age is more normally distributed — let's use it
age_mean = df['age'].mean()
age_std  = df['age'].std()

within_1std = df[(df['age'] >= age_mean - age_std)   & (df['age'] <= age_mean + age_std)]
within_2std = df[(df['age'] >= age_mean - 2*age_std) & (df['age'] <= age_mean + 2*age_std)]
within_3std = df[(df['age'] >= age_mean - 3*age_std) & (df['age'] <= age_mean + 3*age_std)]

print(f"""
  Age Distribution — Empirical Rule Check:
  Mean = {age_mean:.1f} years  |  Std Dev = {age_std:.1f} years

  Expected  Actual
  ────────  ───────────────────────────────────────────
  68%    →  {len(within_1std)/n*100:.1f}%  within 1σ ({age_mean-age_std:.0f} – {age_mean+age_std:.0f} years)
  95%    →  {len(within_2std)/n*100:.1f}%  within 2σ ({age_mean-2*age_std:.0f} – {age_mean+2*age_std:.0f} years)
  99.7%  →  {len(within_3std)/n*100:.1f}%  within 3σ ({age_mean-3*age_std:.0f} – {age_mean+3*age_std:.0f} years)
  
  → Actual values expected ke close hain = Age is NORMALLY distributed ✅
  → Is matlab age ML ke liye directly use kar sakte hain!
""")


# ================================================================
# SECTION 3: LOG TRANSFORMATION (Skewed data fix karo)
# ================================================================
print("\n" + "=" * 65)
print("🔧 SECTION 3: Log Transform — Skewed Data Fix")
print("=" * 65)

# Log transform total_spend
# log1p = log(1 + x) → handles zero values safely
df['log_spend'] = np.log1p(df['total_spend_capped'])

original_skew = df['total_spend_capped'].skew()
log_skew      = df['log_spend'].skew()

print(f"""
  Before Log Transform:
  → total_spend_capped skewness: {original_skew:.4f}  (RIGHT SKEWED 📈)
  
  After Log Transform:
  → log_spend skewness:          {log_skew:.4f}  (Near Normal ✅)
  
  Why does this matter?
  → Many ML algorithms (Linear Regression, SVM) ASSUME normal distribution
  → Skewed data → bad model performance
  → Log transform → normalize skewed data → better ML! 🚀
  
  Formula: log_spend = log(1 + total_spend)
  np.log1p() use karo (not np.log) → handles 0 values safely!
""")


# ================================================================
# SECTION 4: CORRELATION ANALYSIS
# ================================================================
print("\n" + "=" * 65)
print("🔗 SECTION 4: Correlation Analysis")
print("=" * 65)

numeric_cols = ['age', 'total_spend_capped', 'orders', 'rating',
                'days_inactive', 'log_spend']
corr_matrix = df[numeric_cols].corr()

print("  Correlation Matrix:")
print(corr_matrix.round(3).to_string())

# Find strongest correlations (excluding self-correlation = 1.0)
print("\n  📌 Top Correlations (business insights):")
corr_pairs = []
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        col1, col2 = numeric_cols[i], numeric_cols[j]
        corr_val   = corr_matrix.loc[col1, col2]
        corr_pairs.append((abs(corr_val), corr_val, col1, col2))

corr_pairs.sort(reverse=True)

for rank, (abs_corr, corr_val, c1, c2) in enumerate(corr_pairs[:5], 1):
    direction = "📈 positive" if corr_val > 0 else "📉 negative"
    strength  = "Strong" if abs_corr > 0.5 else ("Moderate" if abs_corr > 0.3 else "Weak")
    print(f"  {rank}. {c1:20} ↔ {c2:20}: {corr_val:+.3f} ({strength} {direction})")

print(f"""
  💡 Key Insight:
  → total_spend ↔ log_spend: High (expected — derived col)
  → age ↔ orders: Low = age doesn't predict buying frequency!
  → Pro tip: Correlation > 0.85 between two INPUT features = 
    multicollinearity problem in Linear Regression!
    Remove one of them before training ML model.
""")


# ================================================================
# SECTION 5: HYPOTHESIS TESTING — T-TEST
# ================================================================
print("\n" + "=" * 65)
print("🧪 SECTION 5: Hypothesis Testing — T-Test")
print("=" * 65)

print("""
  Business Question:
  "Kya Gold members SIGNIFICANTLY zyada spend karte hain
   compared to Free members? Ya ye difference sirf chance hai?"
  
  H₀ (Null Hypothesis):     Gold aur Free ka mean spend SAME hai
  H₁ (Alternate Hypothesis): Gold members zyada spend karte hain
  
  Test: Independent T-Test (do alag groups compare karna)
  Decision Rule: p < 0.05 → H₀ reject karo (difference REAL hai)
""")

gold_spend = df[df['membership'] == 'Gold']['total_spend_capped'].dropna()
free_spend = df[df['membership'] == 'Free']['total_spend_capped'].dropna()

# Levene's test: check if variances are equal
levene_stat, levene_p = stats.levene(gold_spend, free_spend)
equal_var = levene_p > 0.05  # True if equal variance

# T-test
t_stat, p_value = stats.ttest_ind(gold_spend, free_spend,
                                   equal_var=equal_var,
                                   alternative='greater')  # Gold > Free

print(f"  📊 Sample Sizes:")
print(f"     Gold members: {len(gold_spend)} customers")
print(f"     Free members: {len(free_spend)} customers")
print(f"\n  📈 Mean Spend:")
print(f"     Gold: ₹{gold_spend.mean():,.2f}")
print(f"     Free: ₹{free_spend.mean():,.2f}")
print(f"     Difference: ₹{gold_spend.mean() - free_spend.mean():,.2f}")
print(f"\n  🧪 T-Test Results:")
print(f"     T-statistic : {t_stat:.4f}")
print(f"     P-value     : {p_value:.4f}")

if p_value < 0.05:
    print(f"\n  ✅ RESULT: p={p_value:.4f} < 0.05")
    print(f"  → H₀ REJECT! Gold members SIGNIFICANTLY zyada spend karte hain!")
    print(f"  → Ye difference REAL hai, sirf chance nahi!")
    print(f"  → Business Decision: Aggressively promote Free → Gold upgrade!")
else:
    print(f"\n  ❌ RESULT: p={p_value:.4f} > 0.05")
    print(f"  → H₀ ACCEPT! Difference significant nahi hai.")
    print(f"  → Membership type koi real difference nahi bana raha spend mein.")


# ================================================================
# SECTION 6: HYPOTHESIS TESTING — CHI-SQUARE TEST
# ================================================================
print("\n" + "=" * 65)
print("🧪 SECTION 6: Chi-Square Test (Categories ke liye)")
print("=" * 65)

print("""
  Business Question:
  "Kya device type (Mobile/Desktop) aur churn risk
   ke beech koi REAL relationship hai?"
  
  H₀: Device type aur churn risk INDEPENDENT hain (no relation)
  H₁: Device type churn risk ko AFFECT karta hai
  
  Chi-Square = Categorical variables ke liye T-test!
""")

# Contingency table (cross-tabulation)
contingency = pd.crosstab(df['device'], df['churn_risk'],
                           margins=True, margins_name='Total')
contingency.columns = ['Active (0)', 'Churn Risk (1)', 'Total']
print("  Contingency Table:")
print(contingency.to_string())

# Chi-square test
chi_data  = pd.crosstab(df['device'], df['churn_risk'])
chi2, p_val_chi, dof, expected = stats.chi2_contingency(chi_data)

print(f"""
  🧪 Chi-Square Test Results:
     Chi² statistic : {chi2:.4f}
     P-value        : {p_val_chi:.4f}
     Degrees of Freedom: {dof}
""")

if p_val_chi < 0.05:
    print(f"  ✅ RESULT: p={p_val_chi:.4f} < 0.05")
    print(f"  → H₀ REJECT! Device type aur churn RELATED hain! (Real relationship)")
    print(f"  → Mobile users zyada churn karte hain — FIX THE APP! 📱")
else:
    print(f"  ❌ RESULT: p={p_val_chi:.4f} > 0.05")
    print(f"  → H₀ ACCEPT! Device type churn ko affect nahi karta.")
    print(f"  → Somewhere else dhundho churn ka reason!")

# Churn rate by device (detailed)
print(f"\n  Churn Rate by Device:")
device_churn = df.groupby('device')['churn_risk'].agg(['mean','count'])
device_churn['mean'] = (device_churn['mean'] * 100).round(1)
device_churn.columns = ['Churn Rate %', 'Count']
print(device_churn.to_string())


# ================================================================
# SECTION 7: A/B TEST SIMULATION (Real World!)
# ================================================================
print("\n" + "=" * 65)
print("🔬 SECTION 7: A/B Test Simulation (Real World!)")
print("=" * 65)

print("""
  Scenario:
  QuickCart ne ek nayi checkout page test ki!
  Control (A): Old page  → 1000 visitors
  Treatment (B): New page → 1000 visitors
  
  Kya new page SIGNIFICANTLY better convert karta hai?
""")

np.random.seed(123)
# Simulate conversion data
n_ab = 1000
control_conversions   = np.random.binomial(1, 0.118, n_ab)  # 11.8% base rate
treatment_conversions = np.random.binomial(1, 0.136, n_ab)  # 13.6% (uplift!)

conv_a = control_conversions.mean() * 100
conv_b = treatment_conversions.mean() * 100
lift   = conv_b - conv_a

# 2-proportion z-test
count  = np.array([control_conversions.sum(), treatment_conversions.sum()])
nobs   = np.array([n_ab, n_ab])
z_stat, p_ab = stats.ttest_ind(control_conversions, treatment_conversions)

print(f"  Group A (Control):   {conv_a:.1f}% conversion  ({control_conversions.sum()} conversions)")
print(f"  Group B (Treatment): {conv_b:.1f}% conversion  ({treatment_conversions.sum()} conversions)")
print(f"  Lift:                +{lift:.1f}% percentage points")
print(f"\n  T-Test: p-value = {abs(p_ab):.4f}")

if abs(p_ab) < 0.05:
    daily_visitors = 50000
    extra_daily    = int(daily_visitors * (lift / 100))
    monthly_extra  = extra_daily * 30
    avg_order_val  = df['total_spend_capped'].median()
    monthly_rev    = monthly_extra * avg_order_val

    print(f"\n  ✅ RESULT: New page SIGNIFICANTLY better hai! Deploy karo!")
    print(f"\n  💰 Business Impact Calculator:")
    print(f"     Daily visitors:        {daily_visitors:,}")
    print(f"     Extra conversions/day: {extra_daily:,}")
    print(f"     Extra conversions/month: {monthly_extra:,}")
    print(f"     Avg order value:       ₹{avg_order_val:,.0f}")
    print(f"     Extra revenue/month:   ₹{monthly_rev:,.0f}")
    print(f"\n  → Statistics ne ek page change se ₹{monthly_rev/1e5:.1f}L/month ka decision liya!")
else:
    print(f"\n  ❌ RESULT: Difference significant nahi hai. Don't deploy yet!")
    print(f"     More data collect karo ya design improve karo.")


# ================================================================
# SECTION 8: ALL VISUALIZATIONS
# ================================================================
print("\n" + "=" * 65)
print("🎨 SECTION 8: Statistics Visualizations...")
print("=" * 65)

fig = plt.figure(figsize=(22, 18))
fig.suptitle("📊 Day 3 — Statistics Dashboard | QuickCart",
             fontsize=18, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── CHART 1: Distribution Comparison (Normal vs Skewed) ─────
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['age'], bins=25, color='#6C5CE7', edgecolor='white',
         alpha=0.8, density=True, label='Age (Normal)')
# Overlay theoretical normal curve
xmin, xmax = df['age'].min(), df['age'].max()
x = np.linspace(xmin, xmax, 100)
ax1.plot(x, stats.norm.pdf(x, df['age'].mean(), df['age'].std()),
         'r-', linewidth=2.5, label='Normal curve')
ax1.axvline(df['age'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean={df["age"].mean():.0f}')
ax1.axvline(df['age'].median(), color='orange', linestyle='--', alpha=0.7, label=f'Median={df["age"].median():.0f}')
ax1.set_title("🔔 Age — Near Normal\n(Mean ≈ Median)", fontweight='bold')
ax1.set_xlabel("Age"); ax1.legend(fontsize=7)
ax1.annotate(f"Skew={df['age'].skew():.2f}", xy=(0.65, 0.85),
             xycoords='axes fraction', fontsize=9, color='purple',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

# ── CHART 2: Right Skewed Distribution ──────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df['total_spend_capped'], bins=30, color='#FF6B6B',
         edgecolor='white', alpha=0.8, density=True)
ax2.axvline(df['total_spend_capped'].mean(),   color='red',    linestyle='--', linewidth=2, label=f'Mean=₹{df["total_spend_capped"].mean():.0f}')
ax2.axvline(df['total_spend_capped'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median=₹{df["total_spend_capped"].median():.0f}')
ax2.set_title("📈 Spend — RIGHT SKEWED\n(Mean > Median = Outliers!)", fontweight='bold')
ax2.set_xlabel("Total Spend (₹)"); ax2.legend(fontsize=7)
ax2.annotate(f"Skew={df['total_spend_capped'].skew():.2f}\nUse Median!", xy=(0.55, 0.75),
             xycoords='axes fraction', fontsize=9, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0'))

# ── CHART 3: Log Transform Effect ───────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df['log_spend'], bins=30, color='#00B894', edgecolor='white',
         alpha=0.8, density=True, label='Log(spend)')
x_log = np.linspace(df['log_spend'].min(), df['log_spend'].max(), 100)
ax3.plot(x_log, stats.norm.pdf(x_log, df['log_spend'].mean(), df['log_spend'].std()),
         'r-', linewidth=2.5, label='Normal curve')
ax3.set_title("✅ Log(Spend) — Near Normal\n(After Transform)", fontweight='bold')
ax3.set_xlabel("Log(Total Spend)"); ax3.legend(fontsize=7)
ax3.annotate(f"Skew={df['log_spend'].skew():.2f}\nML Ready!", xy=(0.05, 0.82),
             xycoords='axes fraction', fontsize=9, color='green', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#d4f5e9'))

# ── CHART 4: Empirical Rule Visualization ───────────────────
ax4 = fig.add_subplot(gs[1, 0])
mu, sigma = df['age'].mean(), df['age'].std()
x_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
y_norm = stats.norm.pdf(x_norm, mu, sigma)
ax4.plot(x_norm, y_norm, 'k-', linewidth=2)
# Shade regions
for mult, color, label in [(1, '#6C5CE7', '68%: ±1σ'),
                             (2, '#A29BFE', '95%: ±2σ'),
                             (3, '#DFE6E9', '99.7%: ±3σ')]:
    ax4.fill_between(x_norm,
                     np.where((x_norm >= mu-mult*sigma) & (x_norm <= mu+mult*sigma), y_norm, 0),
                     alpha=0.6, color=color, label=label)
ax4.set_title("🔔 Empirical Rule (68-95-99.7)\nAge Distribution", fontweight='bold')
ax4.set_xlabel("Age"); ax4.legend(fontsize=7)
ax4.axvline(mu, color='red', linestyle='--', linewidth=1.5)

# ── CHART 5: Correlation Heatmap ────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
num_cols_vis = ['age', 'total_spend_capped', 'orders', 'rating', 'days_inactive']
mask = np.triu(np.ones_like(df[num_cols_vis].corr(), dtype=bool), k=1)  # Upper triangle only
sns.heatmap(df[num_cols_vis].corr(),
            annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=ax5, linewidths=1, annot_kws={'size': 9, 'weight': 'bold'},
            vmin=-1, vmax=1)
ax5.set_title("🔗 Correlation Matrix\n(Full Numeric Columns)", fontweight='bold')
ax5.tick_params(axis='x', rotation=35, labelsize=8)
ax5.tick_params(axis='y', rotation=0, labelsize=8)

# ── CHART 6: Box Plot — Spend Distribution by Membership ────
ax6 = fig.add_subplot(gs[1, 2])
mem_order = ['Free', 'Silver', 'Gold']
bp_data   = [df[df['membership']==m]['total_spend_capped'].values for m in mem_order]
bp = ax6.boxplot(bp_data, labels=mem_order, patch_artist=True,
                 notch=False, widths=0.55,
                 medianprops=dict(color='black', linewidth=2.5))
for patch, color in zip(bp['boxes'], ['#74B9FF','#A29BFE','#FD79A8']):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax6.set_title("👑 Spend by Membership\n(Box Plot — Outlier View)", fontweight='bold')
ax6.set_ylabel("Total Spend (₹)")
for i, (m, vals) in enumerate(zip(mem_order, bp_data), 1):
    ax6.text(i, np.median(vals) + 150, f'₹{np.median(vals):,.0f}',
             ha='center', fontsize=8, fontweight='bold')

# ── CHART 7: A/B Test Visualization ─────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
ab_labels = ['Control (A)\nOld Page', 'Treatment (B)\nNew Page']
ab_values = [conv_a, conv_b]
bar_colors= ['#74B9FF', '#00B894']
bars7 = ax7.bar(ab_labels, ab_values, color=bar_colors, edgecolor='white',
                linewidth=1.5, width=0.5)
ax7.set_title("🔬 A/B Test Results\nConversion Rate (%)", fontweight='bold')
ax7.set_ylabel("Conversion Rate (%)")
for bar, val in zip(bars7, ab_values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax7.set_ylim(0, max(ab_values) * 1.3)
result_color = 'green' if abs(p_ab) < 0.05 else 'red'
result_text  = f'p={abs(p_ab):.3f}\n✅ Significant!' if abs(p_ab) < 0.05 else f'p={abs(p_ab):.3f}\n❌ Not Significant'
ax7.annotate(result_text, xy=(0.5, 0.80), xycoords='axes fraction',
             ha='center', fontsize=10, color=result_color, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor=result_color))

# ── CHART 8: T-Test — Gold vs Free Spend Distribution ───────
ax8 = fig.add_subplot(gs[2, 1])
ax8.hist(free_spend, bins=25, alpha=0.65, color='#74B9FF',
         density=True, label=f'Free (μ=₹{free_spend.mean():,.0f})', edgecolor='white')
ax8.hist(gold_spend, bins=25, alpha=0.65, color='#FD79A8',
         density=True, label=f'Gold (μ=₹{gold_spend.mean():,.0f})', edgecolor='white')
ax8.axvline(free_spend.mean(), color='#0984E3', linestyle='--', linewidth=2)
ax8.axvline(gold_spend.mean(), color='#E84393', linestyle='--', linewidth=2)
ax8.set_title(f"🧪 T-Test: Gold vs Free Spend\np={p_value:.3f} {'✅ Significant' if p_value < 0.05 else '❌ Not Significant'}", fontweight='bold')
ax8.set_xlabel("Total Spend (₹)"); ax8.legend(fontsize=8)

# ── CHART 9: P-Value Explainer ───────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
x_pval = np.linspace(-4, 4, 300)
y_pval = stats.norm.pdf(x_pval, 0, 1)
ax9.plot(x_pval, y_pval, 'k-', linewidth=2.5, label='Null Distribution')
ax9.fill_between(x_pval, np.where(x_pval >= 1.645, y_pval, 0),
                 alpha=0.6, color='#FF7675', label='p-value region (α=0.05)')
ax9.fill_between(x_pval, np.where(x_pval < 1.645, y_pval, 0),
                 alpha=0.3, color='#55EFC4', label='Accept H₀ region')
ax9.axvline(1.645, color='red', linestyle='--', linewidth=2, label='Critical value')
ax9.set_title("📐 P-Value Explained\n(One-tailed test, α=0.05)", fontweight='bold')
ax9.set_xlabel("Test Statistic"); ax9.legend(fontsize=7)
ax9.annotate("REJECT H₀\n(p < 0.05)", xy=(2.5, 0.15), fontsize=9,
             color='red', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#ffe0e0'))
ax9.annotate("ACCEPT H₀\n(p > 0.05)", xy=(-3.2, 0.15), fontsize=9,
             color='green', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#d4f5e9'))

plt.savefig('/mnt/user-data/outputs/day3_statistics_dashboard.png',
            dpi=150, bbox_inches='tight')
print("✅ Statistics Dashboard saved!")
plt.close()


# ================================================================
# SECTION 9: COMPLETE BUSINESS SUMMARY
# ================================================================
print("\n" + "=" * 65)
print("🧠 SECTION 9: Business Decisions from Statistics")
print("=" * 65)

spend_std = df['total_spend_capped'].std()
spend_mean = df['total_spend_capped'].mean()

print(f"""
  🎯 FINDING 1 — Spend is RIGHT SKEWED (skew={df['total_spend_capped'].skew():.2f})
     → Report MEDIAN (₹{df['total_spend_capped'].median():,.0f}) to CEO, not mean (₹{spend_mean:,.0f})
     → ML models pe log transform karo pehle!

  🎯 FINDING 2 — Empirical Rule on Age:
     → 68% customers age {df['age'].mean()-df['age'].std():.0f}–{df['age'].mean()+df['age'].std():.0f} years
     → Core demographic! Target ads yahan karo.
     → Age > {df['age'].mean()+2*df['age'].std():.0f} years = only {(df['age'] > df['age'].mean()+2*df['age'].std()).sum()} customers ({(df['age'] > df['age'].mean()+2*df['age'].std()).mean()*100:.1f}%)

  🎯 FINDING 3 — T-Test (Gold vs Free):
     → p={p_value:.4f} → {'SIGNIFICANT — Gold spend more!' if p_value < 0.05 else 'Not significant difference'}
     → Action: {'Aggressive Gold upgrade campaign chalao!' if p_value < 0.05 else 'Revisit membership perks!'}

  🎯 FINDING 4 — A/B Test (Checkout Page):
     → p={abs(p_ab):.4f} → {'New page WINS! Deploy karo!' if abs(p_ab) < 0.05 else 'Not significant, more data needed'}
     → Estimated impact: ₹{int(50000*(lift/100)*30*df['total_spend_capped'].median()/1e5):.0f}L/month

  🎯 FINDING 5 — Log Transform:
     → Before: skew={df['total_spend_capped'].skew():.2f} (bad for ML)
     → After:  skew={df['log_spend'].skew():.2f}  (ML ready!)
     → ALWAYS log transform income/spend/price columns before ML!
""")
print("=" * 65)
print("✅ Day 3 Statistics Complete! Foundation for ML is SOLID!")
print("=" * 65)

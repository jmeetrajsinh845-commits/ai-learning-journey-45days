# ================================================================
# 🚀 DAY 2: Deep EDA — QuickCart E-Commerce Dataset
# Topics: Univariate | Bivariate | Multivariate | Outliers
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# ── STEP 1: LOAD DATASET ─────────────────────────────────────
np.random.seed(42)
n = 600

customer_ids  = [f"C{str(i).zfill(3)}" for i in range(1, n+1)]
ages          = np.random.normal(32, 9, n).clip(18, 65).astype(int)
cities        = np.random.choice(['Mumbai','Delhi','Bangalore','Hyderabad','Chennai'],
                                  n, p=[0.28,0.25,0.22,0.13,0.12])
categories    = np.random.choice(['Electronics','Fashion','Groceries','Books','Home & Kitchen'],
                                  n, p=[0.30,0.25,0.20,0.10,0.15])
base_spend    = {'Electronics':8500,'Fashion':3200,'Groceries':1600,'Books':750,'Home & Kitchen':4200}
total_spend   = np.array([max(200, np.random.normal(base_spend[c], base_spend[c]*0.45))
                           for c in categories]).round(2)
orders        = np.random.randint(1, 26, n)
ratings       = np.random.choice([1,2,3,4,5], n, p=[0.05,0.08,0.15,0.37,0.35]).astype(float)
days_inactive = np.random.exponential(28, n).clip(1, 400).astype(int)
membership    = np.random.choice(['Free','Silver','Gold'], n, p=[0.52,0.30,0.18])
gender        = np.random.choice(['Male','Female','Other'], n, p=[0.52,0.45,0.03])
device        = np.random.choice(['Mobile','Desktop','Tablet'], n, p=[0.68,0.25,0.07])

# Inject missing values (realistic!)
ratings[np.random.choice(n, int(n*0.09), replace=False)] = np.nan
total_spend[np.random.choice(n, int(n*0.06), replace=False)] = np.nan
total_spend[np.random.choice(n, 5, replace=False)] = np.random.uniform(45000, 85000, 5)

df = pd.DataFrame({
    'customer_id': customer_ids, 'age': ages, 'city': cities,
    'gender': gender, 'device': device, 'category': categories,
    'total_spend': total_spend, 'orders': orders, 'rating': ratings,
    'days_inactive': days_inactive, 'membership': membership
})

# ── STEP 2: FIRST LOOK ───────────────────────────────────────
print(df.head())
print(df.shape)
df.info()
print(df.describe().round(2))

# ── STEP 3: MISSING VALUES ───────────────────────────────────
print(df.isnull().sum())

# ── STEP 4: CLEANING ─────────────────────────────────────────
df_clean = df.copy()
df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].median())
df_clean['total_spend'] = df_clean.groupby('category')['total_spend'].transform(
    lambda x: x.fillna(x.median()))
df_clean['spend_per_order'] = (df_clean['total_spend'] / df_clean['orders']).round(2)
df_clean['churn_risk'] = (df_clean['days_inactive'] > 60).astype(int)

# ── STEP 5: OUTLIER DETECTION (IQR Method) ───────────────────
Q1  = df_clean['total_spend'].quantile(0.25)
Q3  = df_clean['total_spend'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df_clean['total_spend_capped'] = df_clean['total_spend'].clip(upper=upper_bound)
print(f"Outliers capped above ₹{upper_bound:,.0f}")

# ── STEP 6: VISUALIZATIONS ───────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.suptitle("📊 QuickCart EDA Dashboard", fontsize=16, fontweight='bold')

# 1. Age Histogram (Univariate)
ax1 = fig.add_subplot(2, 3, 1)
ax1.hist(df_clean['age'], bins=20, color='#6C5CE7', edgecolor='white', alpha=0.85)
ax1.axvline(df_clean['age'].mean(), color='red', linestyle='--',
            label=f"Mean: {df_clean['age'].mean():.0f}")
ax1.set_title("👥 Age Distribution"); ax1.legend()

# 2. Revenue by Category (Bivariate)
ax2 = fig.add_subplot(2, 3, 2)
cat_rev = df_clean.groupby('category')['total_spend_capped'].sum().sort_values()
ax2.barh(cat_rev.index, cat_rev.values/1e6, color='#4ECDC4', edgecolor='white')
ax2.set_title("💰 Revenue by Category (₹M)")

# 3. Correlation Heatmap (Multivariate)
ax3 = fig.add_subplot(2, 3, 3)
num_cols = ['age', 'total_spend_capped', 'orders', 'rating', 'days_inactive']
sns.heatmap(df_clean[num_cols].corr().round(2), annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, ax=ax3, linewidths=0.5)
ax3.set_title("🔗 Correlation Heatmap")

# 4. Box Plot Spend by Membership
ax4 = fig.add_subplot(2, 3, 4)
bp_data = [df_clean[df_clean['membership']==m]['total_spend_capped'].dropna().values
           for m in ['Free', 'Silver', 'Gold']]
bp = ax4.boxplot(bp_data, labels=['Free','Silver','Gold'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#74B9FF', '#A29BFE', '#FD79A8']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax4.set_title("👑 Spend by Membership")

# 5. Churn Risk by City
ax5 = fig.add_subplot(2, 3, 5)
churn_city = df_clean.groupby('city')['churn_risk'].mean().mul(100).sort_values(ascending=False)
ax5.bar(churn_city.index, churn_city.values, color='#FF7675', edgecolor='white')
ax5.set_title("⚠️ Churn Risk % by City")
ax5.tick_params(axis='x', rotation=30)

# 6. Orders vs Spend Scatter
ax6 = fig.add_subplot(2, 3, 6)
colors_map = {'Free': '#74B9FF', 'Silver': '#A29BFE', 'Gold': '#FD79A8'}
for mem, grp in df_clean.groupby('membership'):
    ax6.scatter(grp['orders'], grp['total_spend_capped'],
                c=colors_map[mem], label=mem, alpha=0.5, s=25)
ax6.set_title("📦 Orders vs Spend"); ax6.legend()

plt.tight_layout()
plt.savefig('day2_eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA Complete!")

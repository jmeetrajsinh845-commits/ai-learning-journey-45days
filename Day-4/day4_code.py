# ================================================================
# 🚀 DAY 4: Feature Engineering — Complete Pipeline
# Topics: Encoding | Scaling | Domain Features | Selection
# Dataset: QuickCart (Building on Day 2 & 3!)
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 65)
print("🚀 DAY 4 — Feature Engineering | QuickCart Dataset")
print("=" * 65)

# ── STEP 0: REBUILD DATASET ──────────────────────────────────
np.random.seed(42)
n = 600

categories    = np.random.choice(
    ['Electronics','Fashion','Groceries','Books','Home & Kitchen'],
    n, p=[0.30,0.25,0.20,0.10,0.15])
base_spend    = {'Electronics':8500,'Fashion':3200,'Groceries':1600,
                 'Books':750,'Home & Kitchen':4200}
total_spend   = np.array([
    max(200, np.random.normal(base_spend[c], base_spend[c]*0.45))
    for c in categories]).round(2)
total_spend[np.random.choice(n, 5, replace=False)] = np.random.uniform(45000,85000,5)
ages          = np.random.normal(32, 9, n).clip(18, 65).astype(int)
orders        = np.random.randint(1, 26, n)
days_inactive = np.random.exponential(28, n).clip(1, 400).astype(int)
membership    = np.random.choice(['Free','Silver','Gold'], n, p=[0.52,0.30,0.18])
cities        = np.random.choice(
    ['Mumbai','Delhi','Bangalore','Hyderabad','Chennai'],
    n, p=[0.28,0.25,0.22,0.13,0.12])
device        = np.random.choice(['Mobile','Desktop','Tablet'], n, p=[0.68,0.25,0.07])
ratings       = np.random.choice([1,2,3,4,5], n, p=[0.05,0.08,0.15,0.37,0.35]).astype(float)
gender        = np.random.choice(['Male','Female','Other'], n, p=[0.52,0.45,0.03])

df = pd.DataFrame({
    'age':ages,'city':cities,'gender':gender,'device':device,
    'category':categories,'total_spend':total_spend,'orders':orders,
    'rating':ratings,'days_inactive':days_inactive,'membership':membership
})
df['total_spend'] = df['total_spend'].clip(
    upper=df['total_spend'].quantile(0.75)+1.5*(
          df['total_spend'].quantile(0.75)-df['total_spend'].quantile(0.25)))
df['rating'] = df['rating'].fillna(df['rating'].median())
print(f"✅ Base dataset ready: {df.shape}")

# ================================================================
# SECTION 1: DOMAIN-BASED FEATURE ENGINEERING
# ================================================================
print("\n" + "="*65)
print("🔧 SECTION 1: Domain-Based Feature Creation")
print("="*65)

df_feat = df.copy()

# F1: Spend per order
df_feat['spend_per_order'] = (df_feat['total_spend']/df_feat['orders']).round(2)
print(f"✅ F1: spend_per_order  range: ₹{df_feat['spend_per_order'].min():.0f}–₹{df_feat['spend_per_order'].max():.0f}")

# F2: Recency Score (0-100, higher = more recent)
df_feat['recency_score'] = ((1 - df_feat['days_inactive']/df_feat['days_inactive'].max())*100).round(1)
print(f"✅ F2: recency_score    mean: {df_feat['recency_score'].mean():.1f}")

# F3: Frequency Score (0-100)
df_feat['frequency_score'] = (df_feat['orders']/df_feat['orders'].max()*100).round(1)
print(f"✅ F3: frequency_score  mean: {df_feat['frequency_score'].mean():.1f}")

# F4: Monetary Score (0-100)
df_feat['monetary_score'] = (df_feat['total_spend']/df_feat['total_spend'].max()*100).round(1)
print(f"✅ F4: monetary_score   mean: {df_feat['monetary_score'].mean():.1f}")

# F5: RFM Score — THE GOLD STANDARD!
df_feat['rfm_score'] = (
    0.40*df_feat['recency_score'] +
    0.35*df_feat['monetary_score'] +
    0.25*df_feat['frequency_score']
).round(2)
print(f"✅ F5: rfm_score        range: {df_feat['rfm_score'].min():.1f}–{df_feat['rfm_score'].max():.1f}")

# F6: Customer Segment from RFM
df_feat['customer_segment'] = pd.cut(
    df_feat['rfm_score'],
    bins=[0,25,45,65,100],
    labels=['At Risk','Needs Attention','Loyal','Champion']
)
seg_counts = df_feat['customer_segment'].value_counts()
print(f"✅ F6: customer_segment")
for seg, cnt in seg_counts.items():
    print(f"   {seg:20}: {cnt} ({cnt/n*100:.1f}%)")

# F7: Churn Risk
df_feat['churn_risk'] = (df_feat['days_inactive'] > 60).astype(int)
print(f"✅ F7: churn_risk       {df_feat['churn_risk'].sum()} at risk ({df_feat['churn_risk'].mean()*100:.1f}%)")

# F8: High Value Flag
df_feat['is_high_value'] = (df_feat['total_spend'] >= df_feat['total_spend'].quantile(0.75)).astype(int)
print(f"✅ F8: is_high_value    {df_feat['is_high_value'].sum()} customers")

# F9: Engagement Score
df_feat['engagement_score'] = ((df_feat['rating']/5*50)+(df_feat['frequency_score']/2)).round(1)
print(f"✅ F9: engagement_score range: {df_feat['engagement_score'].min():.1f}–{df_feat['engagement_score'].max():.1f}")

# F10: Age Group
df_feat['age_group'] = pd.cut(
    df_feat['age'], bins=[17,25,35,45,65],
    labels=['Gen Z (18-25)','Millennial (26-35)','Gen X (36-45)','Boomer (46+)'])
print(f"✅ F10: age_group")
for grp, cnt in df_feat['age_group'].value_counts().sort_index().items():
    print(f"   {grp:25}: {cnt} ({cnt/n*100:.1f}%)")

print(f"\n📊 New features created: {df_feat.shape[1]-df.shape[1]}")

# ================================================================
# SECTION 2: CATEGORICAL ENCODING
# ================================================================
print("\n"+"="*65)
print("🔤 SECTION 2: Categorical Encoding")
print("="*65)

df_encoded = df_feat.copy()

# 2A: Label Encoding (ordered categories)
print("\n📌 2A: Label Encoding (ordered):")
df_encoded['membership_encoded'] = df_encoded['membership'].map({'Free':0,'Silver':1,'Gold':2})
df_encoded['segment_encoded']    = df_encoded['customer_segment'].map(
    {'At Risk':0,'Needs Attention':1,'Loyal':2,'Champion':3})
df_encoded['age_group_encoded']  = df_encoded['age_group'].map(
    {'Gen Z (18-25)':0,'Millennial (26-35)':1,'Gen X (36-45)':2,'Boomer (46+)':3})
print("   membership: Free→0, Silver→1, Gold→2 ✅")
print("   segment:    At Risk→0 ... Champion→3 ✅")
print("   age_group:  Gen Z→0 ... Boomer→3     ✅")

# 2B: One-Hot Encoding (no-order categories)
print("\n📌 2B: One-Hot Encoding (no order):")
before_cols = df_encoded.shape[1]
df_encoded  = pd.get_dummies(
    df_encoded, columns=['city','device','category','gender'],
    drop_first=True, dtype=int)
print(f"   Before: {before_cols} cols → After: {df_encoded.shape[1]} cols")
new_ohe = [c for c in df_encoded.columns if any(
           c.startswith(f"{col}_") for col in ['city','device','category','gender'])]
print(f"   New dummy columns ({len(new_ohe)}): {new_ohe}")

# 2C: Drop original text columns
df_encoded.drop(columns=['membership','customer_segment','age_group'], inplace=True)
print(f"\n   Final encoded shape: {df_encoded.shape}")

# ================================================================
# SECTION 3: FEATURE SCALING
# ================================================================
print("\n"+"="*65)
print("📏 SECTION 3: Feature Scaling")
print("="*65)

scale_cols = ['age','total_spend','orders','rating','days_inactive',
              'spend_per_order','recency_score','frequency_score',
              'monetary_score','rfm_score','engagement_score']

# StandardScaler
scaler_std    = StandardScaler()
df_std        = df_encoded.copy()
df_std[scale_cols] = scaler_std.fit_transform(df_encoded[scale_cols])

print("\n📌 StandardScaler results:")
for col in ['age','total_spend','rfm_score']:
    print(f"   {col:20}: mean={df_std[col].mean():.4f}≈0, std={df_std[col].std():.4f}≈1 ✅")

# MinMaxScaler
scaler_mm     = MinMaxScaler()
df_mm         = df_encoded.copy()
df_mm[scale_cols] = scaler_mm.fit_transform(df_encoded[scale_cols])

print("\n📌 MinMaxScaler results:")
for col in ['age','total_spend','rfm_score']:
    print(f"   {col:20}: min={df_mm[col].min():.3f}, max={df_mm[col].max():.3f} ✅")

# ================================================================
# SECTION 4: FEATURE SELECTION (Mutual Information)
# ================================================================
print("\n"+"="*65)
print("🎯 SECTION 4: Feature Selection (Mutual Information)")
print("="*65)

target = 'churn_risk'
feat_cols = [c for c in df_std.columns
             if c != target and df_std[c].dtype in ['float64','int64','int32']]
X = df_std[feat_cols].fillna(0)
y = df_std[target]

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'Feature':feat_cols,'MI Score':mi_scores}
                     ).sort_values('MI Score',ascending=False)

print("\nTop 12 Features for Churn Prediction:")
print("─"*50)
for _, row in mi_df.head(12).iterrows():
    bar   = "█" * int(row['MI Score']*200)
    star  = " ⭐" if row['MI Score'] > mi_df['MI Score'].quantile(0.75) else ""
    print(f"  {row['Feature']:28} {row['MI Score']:.4f} {bar}{star}")

top_features = mi_df.head(15)['Feature'].tolist()
ml_ready_df  = df_std[top_features+[target]].copy()
ml_ready_df.to_csv('/home/claude/ml_ready_dataset.csv', index=False)
print(f"\n✅ ML-ready dataset saved! Shape: {ml_ready_df.shape}")
print(f"   → Day 5 mein First ML Model banayenge on this! 🚀")

# ================================================================
# SECTION 5: VISUALIZATIONS
# ================================================================
print("\n"+"="*65)
print("🎨 SECTION 5: Building Dashboard...")
print("="*65)

seg_order  = ['At Risk','Needs Attention','Loyal','Champion']
seg_colors = ['#E74C3C','#E67E22','#3498DB','#2ECC71']

fig = plt.figure(figsize=(22,18))
fig.suptitle("📊 Day 4 — Feature Engineering Dashboard | QuickCart",
             fontsize=18, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

# Chart 1: Feature Importance
ax1 = fig.add_subplot(gs[0,:2])
top_mi = mi_df.head(12)
c_mi   = ['#2ECC71' if s > mi_df['MI Score'].quantile(0.75)
          else '#E67E22' if s > mi_df['MI Score'].quantile(0.5)
          else '#E74C3C' for s in top_mi['MI Score']]
ax1.barh(top_mi['Feature'][::-1], top_mi['MI Score'][::-1],
         color=c_mi[::-1], edgecolor='white')
ax1.axvline(0.005, color='red', linestyle='--', linewidth=1.5, label='Drop threshold')
ax1.set_title("🎯 Feature Importance for Churn Prediction (MI Score)", fontweight='bold')
ax1.set_xlabel("Mutual Information Score"); ax1.legend(fontsize=9)
for i,(_, row) in enumerate(top_mi[::-1].iterrows()):
    ax1.text(row['MI Score']+0.0003, i, f"{row['MI Score']:.4f}", va='center', fontsize=8)

# Chart 2: Customer Segments Pie
ax2 = fig.add_subplot(gs[0,2])
seg_c = df_feat['customer_segment'].value_counts().reindex(seg_order)
ax2.pie(seg_c.values, labels=seg_c.index, colors=seg_colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.75)
ax2.set_title("👥 Customer Segments\n(RFM Score)", fontweight='bold')

# Chart 3: RFM Distribution by Segment
ax3 = fig.add_subplot(gs[1,0])
for seg, color in zip(seg_order, seg_colors):
    subset = df_feat[df_feat['customer_segment']==seg]['rfm_score']
    if len(subset): ax3.hist(subset, bins=15, alpha=0.7, color=color,
                              label=seg, edgecolor='white')
ax3.set_title("📊 RFM Score by Segment", fontweight='bold')
ax3.set_xlabel("RFM Score"); ax3.legend(fontsize=7)

# Chart 4: Scaling comparison
ax4 = fig.add_subplot(gs[1,1])
sc_sample = scale_cols[:6]
before_r  = [df_encoded[c].max()-df_encoded[c].min() for c in sc_sample]
after_r   = [df_std[c].max()-df_std[c].min() for c in sc_sample]
x_pos     = np.arange(len(sc_sample))
ax4.bar(x_pos-0.2, before_r, 0.38, label='Before', color='#E74C3C', alpha=0.8)
ax4.bar(x_pos+0.2, after_r,  0.38, label='After',  color='#2ECC71', alpha=0.8)
ax4.set_title("📏 Feature Range Before vs After\nStandardScaler", fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([c.replace('_score','').replace('_',' ')
                     for c in sc_sample], rotation=35, ha='right', fontsize=8)
ax4.set_yscale('log'); ax4.legend(fontsize=8)

# Chart 5: Spend per order by segment
ax5 = fig.add_subplot(gs[1,2])
seg_sp = df_feat.groupby('customer_segment',observed=True)['spend_per_order'].mean().reindex(seg_order)
bars5  = ax5.bar(seg_sp.index, seg_sp.values, color=seg_colors, edgecolor='white')
ax5.set_title("💰 Avg Spend/Order by Segment", fontweight='bold')
ax5.tick_params(axis='x', rotation=25)
for bar, val in zip(bars5, seg_sp.values):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
             f'₹{val:,.0f}', ha='center', fontsize=9, fontweight='bold')

# Chart 6: Label encoding counts
ax6 = fig.add_subplot(gs[2,0])
mem_enc = df_encoded['membership_encoded'].value_counts().sort_index()
ax6.bar(['Free(0)','Silver(1)','Gold(2)'], mem_enc.values,
        color=['#74B9FF','#A29BFE','#FD79A8'], edgecolor='white')
ax6.set_title("🔤 Label Encoding\nMembership Distribution", fontweight='bold')
for i, val in enumerate(mem_enc.values):
    ax6.text(i, val+3, str(val), ha='center', fontsize=11, fontweight='bold')

# Chart 7: Churn by segment
ax7 = fig.add_subplot(gs[2,1])
c_seg = df_feat.groupby('customer_segment',observed=True)['churn_risk'].mean().mul(100).reindex(seg_order)
bars7 = ax7.bar(c_seg.index, c_seg.values,
                color=['#E74C3C' if v>15 else '#F39C12' if v>8 else '#2ECC71'
                       for v in c_seg.values], edgecolor='white')
ax7.axhline(df_feat['churn_risk'].mean()*100, color='navy', linestyle='--', linewidth=1.5)
ax7.set_title("⚠️  Churn % by Segment\n(RFM validation)", fontweight='bold')
ax7.tick_params(axis='x', rotation=25)
for bar, val in zip(bars7, c_seg.values):
    ax7.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

# Chart 8: RFM vs Engagement scatter
ax8 = fig.add_subplot(gs[2,2])
sc_map = {'At Risk':'#E74C3C','Needs Attention':'#E67E22',
          'Loyal':'#3498DB','Champion':'#2ECC71'}
for seg in seg_order:
    sub = df_feat[df_feat['customer_segment']==seg]
    ax8.scatter(sub['rfm_score'], sub['engagement_score'],
                c=sc_map[seg], label=seg, alpha=0.5, s=20)
ax8.set_title("🔗 RFM vs Engagement Score", fontweight='bold')
ax8.set_xlabel("RFM Score"); ax8.set_ylabel("Engagement Score")
ax8.legend(fontsize=7, markerscale=1.5)

plt.savefig('/mnt/user-data/outputs/day4_feature_engineering_dashboard.png',
            dpi=150, bbox_inches='tight')
print("✅ Dashboard saved!")
plt.close()

# ================================================================
# SECTION 6: BUSINESS SUMMARY
# ================================================================
print("\n"+"="*65)
print("🧠 SECTION 6: Business Summary")
print("="*65)

champ_spend = df_feat[df_feat['customer_segment']=='Champion']['total_spend'].mean()
risk_spend  = df_feat[df_feat['customer_segment']=='At Risk']['total_spend'].mean()
champ_pct   = (df_feat['customer_segment']=='Champion').mean()*100
risk_pct    = (df_feat['customer_segment']=='At Risk').mean()*100

print(f"""
  🎯 RFM Segmentation Results:
     Champions ({champ_pct:.1f}%): avg spend ₹{champ_spend:,.0f}
     At Risk   ({risk_pct:.1f}%):  avg spend ₹{risk_spend:,.0f}
     Spend gap: ₹{champ_spend-risk_spend:,.0f} (Champions spend MUCH more!)

  🎯 Top Churn Predictors:
     recency_score, days_inactive → "Absence = Churn"
     rfm_score → Composite signal (most powerful!)
     → Trigger alert: 30 days inactive → auto email!

  🎯 Dataset Journey (Days 1-4):
     Raw columns:        {df.shape[1]}
     After engineering:  {df_feat.shape[1]}
     After encoding:     {df_encoded.shape[1]}
     ML-ready features:  15 (best selected!)
     → DAY 5: First ML Model! 🚀🎉
""")
print("="*65)
print("✅ Day 4 Complete! ml_ready_dataset.csv saved!")
print("="*65)

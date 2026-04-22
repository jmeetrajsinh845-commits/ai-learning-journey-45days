# 📓 DAY 2 — EDA Study Notes (Hinglish)
## AI Learning Journey | QuickCart Dataset | 600 Customers

---

## 🔵 Concepts Learned

### EDA Kya Hai?
- Data ka "health checkup" karna = EDA
- Model banane se PEHLE data samajhna
- 70% of real data science = EDA + cleaning

### 3 Types of EDA
| Type | Matlab | Columns | Charts |
|------|--------|---------|--------|
| Univariate | Ek column akela | age, spend | Histogram, Bar |
| Bivariate | Do columns | age + spend | Scatter, Box |
| Multivariate | Teen+ columns | age+spend+city | Heatmap, Pair |

---

## 🟢 Key Python Commands

```python
df.head()           # Pehli 5 rows dekho
df.shape            # (rows, cols) — size
df.info()           # Dtypes + null count
df.describe()       # Stats: mean, std, min, max, quartiles
df.isnull().sum()   # Missing values count per column
df.copy()           # HAMESHA copy pe kaam karo!

# Missing value fill
df['col'].fillna(df['col'].median())    # Median fill
df.groupby('cat')['col'].transform(    # Category-wise fill
    lambda x: x.fillna(x.median()))

# Outlier detection (IQR)
Q1  = df['col'].quantile(0.25)
Q3  = df['col'].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
df['col'].clip(upper=upper)   # Cap outliers
```

---

## 💡 Business Insights Found

1. **Electronics = 55% revenue** → Focus campaigns here
2. **Mobile churn 12.2% vs Desktop 5.4%** → Fix mobile UX!
3. **63 customers at churn risk** → Win-back campaign needed
4. **100 customers <5 orders** → Early loyalty rewards karo
5. **Category-wise fill smarter** → Books ≠ Electronics value

---

## 🚨 Outlier Rules (IQR Method)
```
Lower = Q1 - 1.5 × IQR
Upper = Q3 + 1.5 × IQR

Options:
A) Remove  → Only if data entry error
B) Cap     → VIP customers / extreme but real
C) Keep    → Fraud detection (outlier = fraud signal!)
```

---

## ⚠️ Common Mistakes to Avoid
- ❌ df pe directly kaam karna → ✅ Always df.copy()
- ❌ Mean se fill karna outliers ke time → ✅ Median use karo
- ❌ Global median fill → ✅ Category-wise fill
- ❌ Visualization without business question → ✅ Har chart = 1 question

---

## 🤖 AI Prompts for EDA

```
Prompt 1 — Missing Values:
"Here is df.info() output: [paste output]
 Suggest the best strategy to handle missing values
 for each column based on its data type and % missing"

Prompt 2 — Business Insights:
"Here is df.describe() output: [paste output]
 I am analyzing an e-commerce dataset.
 What business insights can you derive from these statistics?"

Prompt 3 — Outlier Handling:
"Column total_spend has outliers above ₹12,580.
 These are VIP customers. Should I remove, cap, or keep?
 What are the tradeoffs for ML modeling?"

Prompt 4 — Visualization:
"I have customer data with: age, city, category, spend, orders.
 What 5 visualizations should I create to understand
 customer behavior? For each, say which EDA type it is."
```

---

## 💼 Career Value
- 80% interviews mein EDA questions hote hain
- Freelance EDA report = ₹15k–₹80k per project
- Every ML model starts with solid EDA
- Senior engineers judge you by EDA quality, not model choice

---

## 🔗 Next Steps (Day 3)
- Statistics deep dive (distributions, hypothesis testing)
- Feature Engineering (powerful new columns banana)
- Correlation analysis (which features matter for ML)

---
*Day 2 Complete ✅ | EDA Foundation Strong | Day 3 Ready 🚀*

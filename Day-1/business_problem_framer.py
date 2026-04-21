# ============================================
# 🎯 DAY 1: Business Problem Framing Template
# ============================================

# Think of this as your "checklist" before any project
# Like a pilot's pre-flight checklist ✈️

# ----- STEP 1: Define the Problem Dictionary -----
business_problem = {

    "company": "QuickCart (E-Commerce)",           # Who is the client?

    "complaint": "Sales dropped 20% this month",   # What are they saying?

    "real_question": "Which customers are about to stop buying from us?",  # The REAL question

    "metric": "Customer Churn Rate (%)",            # How do we measure success?

    "target_variable": "will_churn",               # What ML will predict
                                                    # 1 = will churn, 0 = won't

    "data_needed": [                               # What data we need
        "last_purchase_date",                      # When did they last buy?
        "purchase_frequency",                      # How often do they buy?
        "avg_order_value",                         # How much do they spend?
        "support_tickets",                         # Did they complain?
        "email_open_rate"                          # Are they engaging?
    ],

    "ml_approach": "Binary Classification",        # Type of ML problem

    "success_criteria": "Recall > 80%",            # We want to CATCH churners
                                                    # Missing one is costly!

    "business_impact": "Retain top 1000 customers = ₹50L revenue saved"  # Expected ROI
}

# ----- STEP 2: Print a Nicely Formatted Frame -----
print("=" * 55)
print("🎯 PROBLEM FRAMING DOCUMENT")
print("=" * 55)

for key, value in business_problem.items():
    # .replace("_", " ") makes "real_question" → "real question"
    # .upper() makes it ALL CAPS for headings
    key_display = key.replace("_", " ").upper()

    print(f"\n🔹 {key_display}:")

    # If the value is a list, print each item on new line
    if isinstance(value, list):
        for item in value:
            print(f"      ✅ {item}")
    else:
        print(f"      {value}")

print("\n" + "=" * 55)
print("✅ Problem Framing Complete! Ready for Data Collection.")
print("=" * 55)
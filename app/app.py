import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
from sklearn import set_config
 
set_config(transform_output="pandas")


# page config
st.set_page_config(
    page_title="Loan Default Predictor",
    layout="wide",
)

#custom style 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

.risk-card {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin-bottom: 12px;
}
.risk-high  { background: #FEE2E2; border: 2px solid #EF4444; }
.risk-low   { background: #DCFCE7; border: 2px solid #22C55E; }

.risk-label { font-size: 2rem; font-weight: 700; margin: 0; }
.risk-prob  { font-size: 1.1rem; color: #374151; margin-top: 4px; }

.insight-box {
    background: #F8FAFC;
    border-left: 4px solid #6366F1;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.95rem;
    color: #1E293B;
}
.insight-risk   { border-left-color: #EF4444; }
.insight-safe   { border-left-color: #22C55E; }
.insight-neutral{ border-left-color: #94A3B8; }

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: white;
    margin: 24px 0 12px;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)



# load model and SHAP

@st.cache_resource # to prevent reload everytime
def load_artifacts():
    model     = joblib.load("final_xgb_model.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    feat_names= joblib.load("feature_names.pkl")
    return model, explainer, feat_names
 #error message if file missing
try:
    model, explainer, feature_names = load_artifacts()
except FileNotFoundError:
    st.error("File Missing Error!")
    st.stop()


#input to DataFrame
def build_input(age, income, home, emp_len, intent, grade,
                loan_amnt, int_rate, default_on_file, cred_hist):
    loan_pct_income = round(loan_amnt / income, 4) if income > 0 else 0.0  
    return pd.DataFrame([{
        "person_age":                  age,
        "person_income":               income,
        "person_home_ownership":       home,
        "person_emp_length":           emp_len,
        "loan_intent":                 intent,
        "loan_grade":                  grade,
        "loan_amnt":                   loan_amnt,
        "loan_int_rate":               int_rate,
        "loan_percent_income":         loan_pct_income,
        "cb_person_default_on_file":   default_on_file,
        "cb_person_cred_hist_length":  cred_hist,
    }])



#plain-English SHAP explanations

FEATURE_LABELS = {
    "person_income":               "your income",
    "loan_percent_income":         "loan-to-income ratio",
    "loan_grade":                  "Loan Grade",
    "loan_amnt":                   "Loan Amount",
    "loan_int_rate":               "Interest Rate",
    "person_emp_length":           "Employment Length",
    "person_age":                  "Age",
    "cb_person_default_on_file":   "Past default history",
    "cb_person_cred_hist_length":  "credit history length",
    "person_home_ownership_OWN":   "home ownership (Own)",
    "person_home_ownership_RENT":  "home ownership (Rent)",
    "person_home_ownership_MORTGAGE": "home ownership (Mortgage)",
    "loan_intent_VENTURE":         "loan purpose (Venture)",
    "loan_intent_MEDICAL":         "loan purpose (Medical)",
    "loan_intent_HOMEIMPROVEMENT": "loan purpose (Home Improvement)",
    "loan_intent_DEBTCONSOLIDATION":"loan purpose (Debt Consolidation)",
    "loan_intent_EDUCATION":       "loan purpose (Education)",
    "loan_intent_PERSONAL":        "loan purpose (Personal)",
}

def shap_to_english(shap_vals, feature_names, X_row, top_n=4):
    
    pairs = sorted(
        zip(shap_vals, feature_names),
        key=lambda x: abs(x[0]),
        reverse=True
    )

    risk_factors = []
    safety_factors = []
    ohe_prefixes = [
        "person_home_ownership",
        "loan_intent"
    ]

# Skip inactive one-hot encoded features
    for val, feat in pairs:

        # proper OHE filtering
        if any(feat.startswith(p) for p in ohe_prefixes):
            if X_row.get(feat, 0) == 0:
                continue

        label = FEATURE_LABELS.get(feat, feat)

        if val > 0.05:
            risk_factors.append((val, label))
        elif val < -0.05:
            safety_factors.append((val, label))

        if len(risk_factors) >= top_n and len(safety_factors) >= top_n:
            break


    insights = []

    for val, feat in risk_factors:
        insights.append(("risk", f" Your **{feat}** is increasing the risk of default."))

    for val, feat in safety_factors:
        insights.append(("safe", f" Your **{feat}** is reducing the risk of default."))

    if not insights:
        insights.append(("neutral", "No single factor strongly dominates this prediction."))

    return insights



#  HEADER

st.markdown("# 💳 Loan Default Risk Predictor")
st.markdown(
    "Enter applicant details below. The model will predict default risk "
)
st.divider()



# sidebar input

st.sidebar.markdown("## 📋 Applicant Details")
st.sidebar.markdown("---")

st.sidebar.markdown("**Personal Information**")
age      = st.sidebar.slider("Age", 18, 100, 30)
income   = st.sidebar.number_input("Annual Income (₹ / $)", 4000, 6_000_000, 55000, step=1000)
home     = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
emp_len  = st.sidebar.slider("Employment Length (years)", 0, 50, 5)
cred_hist= st.sidebar.slider("Credit History Length (years)", 0, 30, 5)
default_file = st.sidebar.selectbox("Previous Default on File?", ["N", "Y"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Loan Details**")
loan_amnt = st.sidebar.number_input("Loan Amount", 500, 35000, 10000, step=500)
int_rate  = st.sidebar.slider("Interest Rate (%)", 5.0, 24.0, 11.0, step=0.1)
grade     = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
intent    = st.sidebar.selectbox("Loan Intent", [
    "EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
])

loan_pct  = round(loan_amnt / income, 4) if income > 0 else 0
st.sidebar.markdown(f"**Loan-to-Income Ratio** (auto): `{loan_pct:.2%}`")

predict_btn = st.sidebar.button(" Predict", use_container_width=True, type="primary")



# OUTPUT

if predict_btn:
    input_df = build_input(age, income, home, emp_len, intent, grade,
                           loan_amnt, int_rate, default_file, cred_hist)

    prob     = model.predict_proba(input_df)[0][1]
    pred     = int(prob >= 0.45)

    col1, col2 = st.columns([1, 2], gap="large")

    # left - verdict
    with col1:
        st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)

        if pred == 1:
            st.markdown(f"""
            <div class="risk-card risk-high">
                <p class="risk-label">⚠️ High Risk</p>
                <p class="risk-prob">Default Probability: <b>{prob:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-card risk-low">
                <p class="risk-label">✅ Low Risk</p>
                <p class="risk-prob">Default Probability: <b>{prob:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)

        # Probability bar
        st.markdown("**Risk Gauge**")
        fig_gauge, ax_g = plt.subplots(figsize=(4, 0.55))
        ax_g.barh(0, 1,       color="#DCFCE7", height=0.5)
        ax_g.barh(0, prob,    color="#EF4444" if pred else "#22C55E", height=0.5)
        ax_g.axvline(0.5, color="#94A3B8", lw=1.5, ls="--")
        ax_g.set_xlim(0, 1); ax_g.axis("off")
        ax_g.text(prob, 0, f" {prob:.1%}", va="center", fontsize=9,
                  color="#1E293B", fontweight="bold")
        fig_gauge.patch.set_alpha(0)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

        # inputs summary
        st.markdown('<p class="section-title">Input Summary</p>', unsafe_allow_html=True)
        summary = {
            "Loan Amount": f"${loan_amnt:,}",
            "Income":      f"${income:,}",
            "Grade":       grade,
            "Interest":    f"{int_rate}%",
            "Debt/Income": f"{loan_pct:.1%}",
        }
        for k, v in summary.items():
            st.markdown(f"**{k}:** {v}")

    # right - SHAP 
    with col2:
        st.markdown('<p class="section-title">Why This Prediction? (SHAP Analysis)</p>',
                    unsafe_allow_html=True)

        # Transform input through pipeline preprocessor
        preprocessor_step = model.named_steps['preprocessor']
        xgb_model_step    = model.named_steps['model']

        X_transformed = preprocessor_step.transform(input_df)
        X_transformed.columns = [c.split("__")[-1] for c in X_transformed.columns]

        shap_vals = explainer(X_transformed)

        # Waterfall plot
        fig_wf, ax_wf = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_vals[0], max_display=12, show=False)
        fig_wf.patch.set_facecolor("#FAFAFA")
        plt.title("Feature Contributions to This Prediction",
                  fontsize=11, fontweight="bold", pad=10)
        plt.tight_layout()
        st.pyplot(fig_wf, use_container_width=True)
        plt.close(fig_wf)

        # English explanation 
        st.markdown('<p class="section-title">What This Means for You</p>',
                    unsafe_allow_html=True)
        st.markdown(
            "The chart above shows which factors *pushed* the risk score **up** (red/right) "
            "or **down** (blue/left). Here's what that means in plain terms:"
        )

        sv = shap_vals[0].values
        fn = list(X_transformed.columns)
        X_row = X_transformed.iloc[0]
        insights = shap_to_english(sv, fn,X_row, top_n=4)
        for kind, text in insights:
            css_cls = {"risk": "insight-risk", "safe": "insight-safe"}.get(kind, "insight-neutral")
            st.markdown(f'<div class="insight-box {css_cls}">{text}</div>',
                        unsafe_allow_html=True)

        st.markdown("")
        if pred == 1:
            st.info(
                "💡 **What can improve this?** A lower loan amount, better loan grade, "
                "or a higher income relative to the loan would reduce default risk."
            )
        else:
            st.success(
                "💡 **Looking good.** The applicant's financial profile suggests "
                "a manageable repayment burden."
            )

else:
    # Landing state
    st.markdown("### 👈 Fill in the applicant details and click **Predict**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "XGBoost")
    with col2:
        st.metric("ROC-AUC", "0.9483")
    with col3:
        st.metric("Explainability", "SHAP Waterfall")

    st.markdown("""
    ---
    **How it works:**
    1. Enter the applicant's personal and loan details in the sidebar
    2. Click **Predict** to get the default risk score
    3. The SHAP waterfall chart shows exactly *which factors* drove the decision
    4. English insights translate the model output into actionable information
    """)

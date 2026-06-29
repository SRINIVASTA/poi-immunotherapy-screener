# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from data import generate_large_cohort_data

# 1. Interface Initialization
st.set_page_config(page_title="POI Immunotherapy Dashboard", page_icon="🧬", layout="wide")
st.title("🧬 Advanced POI Immunotherapy Diagnostics & Batch Screening Hub")
st.markdown("---")

# 2. Live Self-Training Mechanism (Trains on the 200-Patient Cohort)
@st.cache_resource
def train_and_cache_live_model():
    refine_data = generate_large_cohort_data(n_patients=200)
    treatment_group = refine_data[refine_data['Trial_Arm_Rituximab'] == 1].copy()
    
    # Train using the cohort features
    X = treatment_group[['Age', 'AMH_Baseline', 'Has_Addisons', 'Has_Hashimotos']]
    y = treatment_group['Ovarian_Reactivation']
    
    model = LogisticRegression()
    model.fit(X, y)
    return model, X, y

predictor_model, X_train, y_train = train_and_cache_live_model()
refine_data = generate_large_cohort_data(n_patients=200)
explainer = shap.LinearExplainer(predictor_model, X_train)

# 3. Workspace Navigation Layout Architecture (3 distinct Tabs)
tab1, tab2, tab3 = st.tabs([
    "📋 Tab 1: New Patient Intake Form", 
    "📊 Tab 2: Cohort Analytics & Validity", 
    "⚡ Tab 3: Screening Prediction Outcome"
])

# ==========================================
# TAB 1: NEW PATIENT INTAKE FORM (INPUTS TYPE)
# ==========================================
with tab1:
    st.header("📋 New Test Patient Diagnostic Entry")
    st.write("Provide custom variables below. The trained model will evaluate this input as an independent test sample.")
    
    col_input, col_info = st.columns([1.2, 0.8], gap="large")
    
    with col_input:
        with st.container(border=True):
            pt_id = st.text_input("Patient Identifier / Record #", "TEST-PT-01")
            
            # Custom input 1: Age
            age = st.slider("1. Input Patient Age (Years)", 18, 40, 27)
            
            # Custom input 2: AMH Baseline
            amh = st.slider("2. Input Baseline AMH Hormone Level (pmol/L)", 0.0, 3.5, 1.9, step=0.1)
            
            # Custom input 3: Autoimmune Checkboxes
            st.markdown("**3. Select Associated Autoimmune Diagnoses:**")
            addison_check = st.checkbox("Addison's Disease (Adrenal Autoimmunity)")
            hashimoto_check = st.checkbox("Hashimoto's Thyroiditis (Thyroid Autoimmunity)")
            
        # Convert user inputs to binary integers for testing
        is_addison = 1 if addison_check else 0
        is_hashimoto = 1 if hashimoto_check else 0
        
        # Save custom inputs to global session state for Tab 3 evaluation
        st.session_state['input_data'] = pd.DataFrame([{
            'Age': age, 
            'AMH_Baseline': amh, 
            'Has_Addisons': is_addison, 
            'Has_Hashimotos': is_hashimoto
        }])
        st.session_state['pt_id'] = pt_id
        
        st.success("✨ Custom test metrics locked into memory! Navigate to **Tab 3** to execute inference and view outcomes.")

    with col_info:
        st.info("""
        💡 **Clinical Testing Reference:**
        * **Age (18-40):** Tests the candidate boundaries based on the Karolinska study.
        * **AMH Level:** Evaluates dormant reserve volume. High values give the model strong responsive features.
        * **Autoimmune Flags:** Simulates the specific background profile targeting.
        """)

# ==========================================
# TAB 2: COHORT ANALYTICS & VALIDATION DATA
# ==========================================
with tab2:
    st.header("📁 200-Patient Training Dataset & Model Performance")
    
    col_metrics, col_roc = st.columns(2, gap="large")
    
    with col_metrics:
        st.subheader("📊 Cohort Distribution Summary")
        batch_df = refine_data.copy()
        X_batch = batch_df[['Age', 'AMH_Baseline', 'Has_Addisons', 'Has_Hashimotos']]
        batch_probabilities = predictor_model.predict_proba(X_batch)[:, 1]
        
        batch_df['Success_Probability_%'] = np.round(batch_probabilities * 100, 2)
        batch_df['Screening_Status'] = np.where(
            batch_df['Success_Probability_%'] >= 75, "Highly Recommended",
            np.where(batch_df['Success_Probability_%'] >= 40, "Borderline Review", "Not Recommended")
        )
        
        status_counts = batch_df['Screening_Status'].value_counts()
        for category in ["Highly Recommended", "Borderline Review", "Not Recommended"]:
            if category not in status_counts: status_counts[category] = 0
        status_counts = status_counts.loc[["Highly Recommended", "Borderline Review", "Not Recommended"]]
        
        # Donut Chart Rendering
        fig_pie, ax_pie = plt.subplots(figsize=(5, 3.2))
        colors_pie = ['#00cc66', '#ffaa00', '#ff4b4b']
        ax_pie.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=colors_pie)
        ax_pie.axis('equal')
        plt.tight_layout()
        st.pyplot(fig_pie)
        
    with col_roc:
        st.subheader("📈 Receiver Operating Characteristic (ROC)")
        y_prob = predictor_model.predict_proba(X_train)[:, 1]
        fpr, tpr, _ = roc_curve(y_train, y_prob)
        auc_score = roc_auc_score(y_train, y_prob)
        
        fig_roc, ax_roc = plt.subplots(figsize=(5, 3.2))
        ax_roc.plot(fpr, tpr, color='#ff6600', lw=2.5, label=f'Model ROC (AUC = {auc_score:.3f})')
        ax_roc.plot([0, 1], [0, 1], color='#112266', lw=1, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig_roc)
        
    st.markdown("---")
    st.subheader("🔍 Review Master Tabular Ledger (Baseline Training Profiles)")
    st.dataframe(batch_df, use_container_width=True)

# ==========================================
# TAB 3: SCREENING PREDICTION OUTCOME (TEST CASE SUMMARY)
# ==========================================
with tab3:
    st.header("⚡ Live Test Inference Outcome")
    
    # Read custom inputs directly from memory
    if 'input_data' in st.session_state:
        input_df = st.session_state['input_data']
        pt_id = st.session_state['pt_id']
        
        # Extract individual custom features using .iloc
        age_val = int(input_df['Age'].iloc[0])
        amh_val = float(input_df['AMH_Baseline'].iloc[0])
        addison_val = int(input_df['Has_Addisons'].iloc[0])
        hashimoto_val = int(input_df['Has_Hashimotos'].iloc[0])
        
        # Summary Card displaying what the user entered in Tab 1
        with st.container(border=True):
            st.markdown(f"#### 🔍 Custom Testing Parameters for: **{pt_id}**")
            st.markdown(f"**Age Parameter:** `{age_val} Years` | **AMH Blood Metric:** `{amh_val} pmol/L` | **Addison's Profile:** `{bool(addison_val)}` | **Hashimoto's Profile:** `{bool(hashimoto_val)}` ")
        
        # Compute testing instance inference against the trained model weights
        prob = predictor_model.predict_proba(input_df)[:, 1]
        # FIXED: Added [0] index accessor to safely extract the raw float scalar before text formatting
        prob_pct = float(prob[0] * 100)
        
        # Fire medical warning if both boxes are checked
        if addison_val == 1 and hashimoto_val == 1:
            st.warning("⚠️ **MEDICAL ALERT: Schmidt's Syndrome (APS-2) Detected.** Patient exhibits multiple autoimmune variables. Adrenal cortisol levels must be stabilized with steroid updates before addressing secondary thyroid components.")
            
        # Display the custom colored prediction box
        st.subheader("💡 Calculated Probability Outcome")
        if prob_pct >= 75:
            st.success(f"🟢 **RECOMMENDED TRIAL CANDIDATE** — Response Probability: **{prob_pct:.1f}%**")
        elif prob_pct >= 40:
            st.warning(f"🟡 **BORDERLINE CANDIDATE** — Response Probability: **{prob_pct:.1f}%**")
        else:
            st.error(f"🔴 **LOW RESPONSE COMPLIANCE** — Response Probability: **{prob_pct:.1f}%**")
            
        # Display the tailored explainable AI graphics
        st.markdown("---")
        st.subheader("🎯 Custom Feature Weighting Vectors (SHAP)")
        st.write("This chart details how much each of your custom variables shifted the prediction percentage value away from the baseline average score.")
        
        # FIXED SHAP array unpacking for standalone input instances
        shap_values = explainer(input_df)
        fig_shap, ax_shap = plt.subplots(figsize=(6, 2.5))
        y_pos = np.arange(len(X_train.columns))
        colors = ['#ff4b4b' if x < 0 else '#00cc66' for x in shap_values.values[0]]
        
        ax_shap.barh(y_pos, shap_values.values[0], color=colors, height=0.4)
        ax_shap.set_yticks(y_pos)
        ax_shap.set_yticklabels(['Age Parameter', 'AMH Blood Metric', "Has Addison's Profile", "Has Hashimoto's Profile"])
        ax_shap.axvline(0, color='black', lw=1, linestyle='--')
        ax_shap.set_xlabel('SHAP Value (Impact on Log-Odds)')
        plt.tight_layout()
        st.pyplot(fig_shap)
    else:
        st.warning("📋 No testing parameters detected. Please provide patient inputs inside **Tab 1** first.")

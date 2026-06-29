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

# 2. Live Self-Training Mechanism (Runs entirely in Web Cloud Memory)
@st.cache_resource
def train_and_cache_live_model():
    refine_data = generate_large_cohort_data(n_patients=200)
    treatment_group = refine_data[refine_data['Trial_Arm_Rituximab'] == 1].copy()
    
    # Train using both independent disease columns
    X = treatment_group[['Age', 'AMH_Baseline', 'Has_Addisons', 'Has_Hashimotos']]
    y = treatment_group['Ovarian_Reactivation']
    
    model = LogisticRegression()
    model.fit(X, y)
    return model, X, y

predictor_model, X_train, y_train = train_and_cache_live_model()
refine_data = generate_large_cohort_data(n_patients=200)
explainer = shap.LinearExplainer(predictor_model, X_train)

# 3. Streamlined Workspace Layout (2 Consolidated Tabs)
tab1, tab2 = st.tabs([
    "📋 Tab 1: Individual Patient Selection & SHAP", 
    "📁 Tab 2: Cohort Analytics & Validity"
])

# ==========================================
# TAB 1: ALL-IN-ONE SCREENER (LOOKUP + LIVE OUTPUTS)
# ==========================================
with tab1:
    st.header("📋 Patient Diagnostics Registry Lookup")
    st.write("Select an existing patient from the database cohort below. The model will calculate and display everything on this page instantly.")
    
    # Split the screen: Left side for selecting patient, Right side for showing outputs
    col_left_select, col_right_results = st.columns([1.1, 0.9], gap="large")
    
    with col_left_select:
        st.subheader("🔍 Cohort Selector")
        # Searchable drop-down panel containing all 200 patients from the cohort database
        patient_list = refine_data['Patient_ID'].tolist()
        selected_pt_id = st.selectbox("Search & Select Patient Record from Registry", patient_list)
        
        # Pull records dynamically based on selection instance
        patient_record = refine_data[refine_data['Patient_ID'] == selected_pt_id].iloc[0]
        
        # Map variables out of database row
        age = int(patient_record['Age'])
        amh = float(patient_record['AMH_Baseline'])
        is_addison = int(patient_record['Has_Addisons'])
        is_hashimoto = int(patient_record['Has_Hashimotos'])
        
        # Visual Card showing what is currently loaded for the user
        with st.container(border=True):
            st.markdown(f"### 📋 Profile Metrics for: **{selected_pt_id}**")
            st.write(f"• **Age:** {age} Years")
            st.write(f"• **Baseline AMH:** {amh} pmol/L")
            
            st.markdown("**Associated Autoimmune Diagnoses:**")
            st.checkbox("Addison's Disease (Adrenal)", value=bool(is_addison), disabled=True, key="chk_addison")
            st.checkbox("Hashimoto's Thyroiditis (Thyroid)", value=bool(is_hashimoto), disabled=True, key="chk_hashimoto")
            
        st.info("""
        💡 **Quick Clinical Reference:**
        * **Addison's Profile:** Positively weighted vector based on Karolinska trial births.
        * **AMH Blood Metric:** Reflects structural baseline egg reserves.
        """)

    with col_right_results:
        st.subheader("⚡ Model Diagnostics Output")
        
        # Format the parameters into an input dataframe for the machine learning model
        input_df = pd.DataFrame([{'Age': age, 'AMH_Baseline': amh, 'Has_Addisons': is_addison, 'Has_Hashimotos': is_hashimoto}])
        
        # Fire prediction mathematical equations on input state changes
        prob = predictor_model.predict_proba(input_df)[:, 1]
        prob_pct = float(prob[0] * 100)
        
        # Fire medical warning if both conditions are active
        if is_addison == 1 and is_hashimoto == 1:
            st.warning("⚠️ **MEDICAL ALERT: Schmidt's Syndrome (APS-2) Detected.** Ensure adrenal steroid replacements are stabilized before addressing thyroid components to avoid blood pressure collapse.")
            
        # Output the color-coded probability alert block
        if prob_pct >= 75:
            st.success(f"🟢 **RECOMMENDED TRIAL CANDIDATE** — Response Probability: **{prob_pct:.1f}%**")
        elif prob_pct >= 40:
            st.warning(f"🟡 **BORDERLINE CANDIDATE** — Response Probability: **{prob_pct:.1f}%**")
        else:
            st.error(f"🔴 **LOW RESPONSE COMPLIANCE** — Response Probability: **{prob_pct:.1f}%**")
            
        # Render the custom feature weighting graph directly beneath the score
        st.markdown("---")
        st.subheader("🎯 Custom Feature Weighting Vectors (SHAP)")
        
        shap_values = explainer(input_df)
        fig_shap, ax_shap = plt.subplots(figsize=(6, 2.3))
        y_pos = np.arange(len(X_train.columns))
        colors = ['#ff4b4b' if x < 0 else '#00cc66' for x in shap_values.values[0]]
        
        ax_shap.barh(y_pos, shap_values.values[0], color=colors, height=0.4)
        ax_shap.set_yticks(y_pos)
        ax_shap.set_yticklabels(['Age Parameter', 'AMH Blood Metric', "Has Addison's Profile", "Has Hashimoto's Profile"])
        ax_shap.axvline(0, color='black', lw=1, linestyle='--')
        ax_shap.set_xlabel('SHAP Value (Impact on Log-Odds)')
        plt.tight_layout()
        st.pyplot(fig_shap)

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
        
        fig_pie, ax_pie = plt.subplots(figsize=(5, 3.2))
        ax_pie.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=['#00cc66', '#ffaa00', '#ff4b4b'])
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

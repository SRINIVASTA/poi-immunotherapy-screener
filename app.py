# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from sklearn.metrics import roc_curve, roc_auc_score
from data import generate_large_cohort_data

# 1. Interface Initialization
st.set_page_config(page_title="POI Immunotherapy Dashboard", page_icon="🧬", layout="wide")
st.title("🧬 Advanced POI Immunotherapy Diagnostics & Batch Screening Hub")
st.markdown("---")

# Load cached pipeline binaries
@st.cache_resource
def load_clinical_model_assets():
    try:
        with open('model_assets.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("🚨 'model_assets.pkl' not found! Please execute: `python train.py` first.")
        st.stop()

# Instantiate data and analytics layers
payload = load_clinical_model_assets()
predictor_model = payload['model']
refine_data = generate_large_cohort_data(n_patients=200)

# Build features mapping matrices for analysis validation loops
treatment_group = refine_data[refine_data['Trial_Arm_Rituximab'] == 1].copy()
treatment_group['Is_Addisons'] = (treatment_group['Autoimmune_Profile'] == "Addison's").astype(int)
X_train = treatment_group[['Age', 'AMH_Baseline', 'Is_Addisons']]
y_train = treatment_group['Ovarian_Reactivation']

explainer = shap.LinearExplainer(predictor_model, X_train)

# 2. Workspace Navigation Layout Architecture
tab1, tab2 = st.tabs(["📋 Single Patient Diagnostics & SHAP", "📁 Default Batch File Processing (200 Cohort)"])

# ==========================================
# TAB 1: SINGLE PATIENT SCREENING ENGINE
# ==========================================
with tab1:
    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.header("📋 Patient Eligibility Diagnostics")
        st.write("Input custom patient parameters below. The model will calculate the output instantly.")
        
        with st.container(border=True):
            pt_id = st.text_input("Patient Identifier / Record #", "PT-9482")
            
            # --- USER INPUT CONTROLS ---
            age = st.slider("Patient Age (Years)", 18, 40, 27)
            amh = st.slider("Baseline AMH Hormone Level (pmol/L)", 0.0, 3.5, 1.9, step=0.1)
            condition = st.selectbox("Underlying Autoimmune Classification", ["Addison's Disease", "Hashimoto's Thyroiditis"])
        
        # Process the custom user inputs
        is_addisons = 1 if condition == "Addison's Disease" else 0
        input_df = pd.DataFrame([{'Age': age, 'AMH_Baseline': amh, 'Is_Addisons': is_addisons}])
        
        if st.button("⚡ Run Diagnostic Screening Evaluation", type="primary"):
            # Pass custom inputs directly to the trained model
            prob = predictor_model.predict_proba(input_df)[:, 1]
            prob_pct = prob[0] * 100
            
            # --- DYNAMIC MODEL OUTPUT ---
            st.subheader("💡 Diagnostic Outcomes")
            if prob_pct >= 75:
                st.success(f"🟢 **RECOMMENDED TRIAL CANDIDATE** — Response Probability: **{prob_pct:.1f}%**")
            elif prob_pct >= 40:
                st.warning(f"🟡 **BORDERLINE CANDIDATE** — Response Probability: **{prob_pct:.1f}%**")
            else:
                st.error(f"🔴 **LOW RESPONSE COMPLIANCE** — Response Probability: **{prob_pct:.1f}%**")
                
            # Dynamic SHAP chart corresponding to the unique inputs
            st.subheader("🎯 Feature Impact Visualization (SHAP)")
            shap_values = explainer(input_df)
            
            fig_shap, ax_shap = plt.subplots(figsize=(6, 2.2))
            y_pos = np.arange(len(X_train.columns))
            colors = ['#ff4b4b' if x < 0 else '#00cc66' for x in shap_values.values[0]]
            
            ax_shap.barh(y_pos, shap_values.values[0], color=colors, height=0.4)
            ax_shap.set_yticks(y_pos)
            ax_shap.set_yticklabels(['Age', 'AMH Level', "Is B-Cell Autoimmune"])
            ax_shap.axvline(0, color='black', lw=1, linestyle='--')
            ax_shap.set_xlabel('SHAP Value (Impact on Log-Odds)')
            plt.tight_layout()
            st.pyplot(fig_shap)

    with col2:
        st.header("📊 Model Metrics & Validation Data")
        y_prob = predictor_model.predict_proba(X_train)[:, 1]
        fpr, tpr, _ = roc_curve(y_train, y_prob)
        auc_score = roc_auc_score(y_train, y_prob)
        
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.plot(fpr, tpr, color='#ff6600', lw=2.5, label=f'Model ROC (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], color='#112266', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig)

# ==========================================
# TAB 2: SEAMLESS 200 PATIENT BATCH ENGINE
# ==========================================
with tab2:
    st.header("📁 Automatic 200-Patient Cohort Batch Screening Engine")
    st.write("This tab pulls your default dataset of 200 patients directly into the model for bulk processing without manual file uploads.")
    
    if st.button("🚀 Execute 200-Patient Batch Screening", type="primary"):
        batch_df = refine_data.copy()
        batch_df['Is_Addisons'] = (batch_df['Autoimmune_Profile'] == "Addison's").astype(int)
        
        X_batch = batch_df[['Age', 'AMH_Baseline', 'Is_Addisons']]
        batch_probabilities = predictor_model.predict_proba(X_batch)[:, 1]
        
        batch_df['Success_Probability_%'] = np.round(batch_probabilities * 100, 2)
        batch_df['Screening_Status'] = np.where(
            batch_df['Success_Probability_%'] >= 75, "Highly Recommended",
            np.where(batch_df['Success_Probability_%'] >= 40, "Borderline Review", "Not Recommended")
        )
        
        display_df = batch_df.drop(columns=['Is_Addisons'])
        st.success(f"✔ Completed automated analysis matrix for {len(display_df)} default patients!")
        
        graph_col, metric_col = st.columns([1, 1], gap="medium")
        status_counts = display_df['Screening_Status'].value_counts()
        
        for category in ["Highly Recommended", "Borderline Review", "Not Recommended"]:
            if category not in status_counts:
                status_counts[category] = 0
                
        status_counts = status_counts.loc[["Highly Recommended", "Borderline Review", "Not Recommended"]]
        
        with graph_col:
            st.subheader("📊 Cohort Distribution Summary")
            fig_pie, ax_pie = plt.subplots(figsize=(5, 3.5))
            colors_pie = ['#00cc66', '#ffaa00', '#ff4b4b']
            
            wedges, texts, autotexts = ax_pie.pie(
                status_counts.values, 
                labels=status_counts.index, 
                autopct='%1.1f%%',
                startangle=90, 
                colors=colors_pie,
                textprops=dict(color="black"),
                wedgeprops=dict(width=0.4, edgecolor='white')
            )
            
            plt.setp(autotexts, size=9, weight="bold")
            plt.setp(texts, size=9)
            ax_pie.axis('equal')  
            plt.tight_layout()
            st.pyplot(fig_pie)
            
        with metric_col:
            st.subheader("📋 Triage Counter Metrics")
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric(label="✅ Recommended", value=int(status_counts["Highly Recommended"]))
            kpi2.metric(label="⚠️ Borderline", value=int(status_counts["Borderline Review"]))
            kpi3.metric(label="❌ Not Recommended", value=int(status_counts["Not Recommended"]))
            
            st.info(f"💡 **Analysis Note:** Out of the 200 patients evaluated, **{status_counts['Highly Recommended']} candidates** have matching autoimmune profiles and necessary dormant ovarian follicle counts to warrant immediate clinical induction.")

        st.markdown("---")
        st.subheader("🔍 Detailed Batch Candidate Ledger")
        st.dataframe(display_df, use_container_width=True)
        
        output_csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Processed 200-Patient Cohort Report (.CSV)",
            data=output_csv,
            file_name="Automated_200_Patient_Screening_Report.csv",
            mime="text/csv"
        )

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
    # Automatically pulls 200 patients from our module data file
    refine_data = generate_large_cohort_data(n_patients=200)
    treatment_group = refine_data[refine_data['Trial_Arm_Rituximab'] == 1].copy()
    treatment_group['Is_Addisons'] = (treatment_group['Autoimmune_Profile'] == "Addison's").astype(int)
    
    X = treatment_group[['Age', 'AMH_Baseline', 'Is_Addisons']]
    y = treatment_group['Ovarian_Reactivation']
    
    # Model trains itself instantly on startup in the cloud
    model = LogisticRegression()
    model.fit(X, y)
    return model, X, y

# Instantiate trained elements directly into web variables
predictor_model, X_train, y_train = train_and_cache_live_model()
refine_data = generate_large_cohort_data(n_patients=200)
explainer = shap.LinearExplainer(predictor_model, X_train)

# 3. Workspace Navigation Layout Architecture
tab1, tab2 = st.tabs(["👥 Individual Patient Selection & SHAP", "📁 Default Batch File Processing (200 Cohort)"])

# ==========================================
# TAB 1: MULTIPLE PATIENT SELECTOR ENGINE
# ==========================================
with tab1:
    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.header("👥 Patient Diagnostics Registry Lookup")
        st.write("Select a patient from the database cohort to analyze their metrics and model outcomes instantly.")
        
        # Dropdown selection containing all 200 patients from the cohort database
        patient_list = refine_data['Patient_ID'].tolist()
        selected_pt_id = st.selectbox("🔍 Search & Select Patient Record", patient_list)
        
        patient_record = refine_data[refine_data['Patient_ID'] == selected_pt_id].iloc[0]
        
        # Map variables out of database row
        age = int(patient_record['Age'])
        amh = float(patient_record['AMH_Baseline'])
        profile_name = str(patient_record['Autoimmune_Profile'])
        
        # Format display name nicely for clinician reading interface
        display_condition = "Addison's Disease" if profile_name == "Addison's" else "Hashimoto's Thyroiditis"
        
        # Visual Summary Card displaying selected metrics
        with st.container(border=True):
            st.markdown(f"### 📋 Profile Metrics for **{selected_pt_id}**")
            st.write(f"• **Age:** {age} Years")
            st.write(f"• **Baseline AMH:** {amh} pmol/L")
            st.write(f"• **Autoimmune Classification:** {display_condition}")
            
            # Map features for calculation array matching matrix standards
            is_addisons = 1 if profile_name == "Addison's" else 0
            input_df = pd.DataFrame([{'Age': age, 'AMH_Baseline': amh, 'Is_Addisons': is_addisons}])

        # Calculate live prediction logic automatically on change selection
        prob = predictor_model.predict_proba(input_df)[:, 1]
        # FIXED: Added [0] index to pull the raw value before conversion to float
        prob_pct = float(prob[0] * 100) 
        
        st.subheader("💡 Diagnostic Outcomes")
        if prob_pct >= 75:
            st.success(f"🟢 **RECOMMENDED TRIAL CANDIDATE** — Response Probability: **{prob_pct:.1f}%**")
        elif prob_pct >= 40:
            st.warning(f"🟡 **BORDERLINE CANDIDATE** — Response Probability: **{prob_pct:.1f}%**")
        else:
            st.error(f"🔴 **LOW RESPONSE COMPLIANCE** — Response Probability: **{prob_pct:.1f}%**")
            
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
        
        graph_col, metric_col = st.columns(2, gap="medium")
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
                status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors_pie, textprops=dict(color="black"),
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
            st.info(f"💡 **Analysis Note:** Out of the 200 patients evaluated, **{status_counts['Highly Recommended']} candidates** warrant immediate clinical induction.")

        st.markdown("---")
        st.subheader("🔍 Detailed Batch Candidate Ledger")
        st.dataframe(display_df, use_container_width=True)
        
        output_csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Processed 200-Patient Cohort Report (.CSV)",
            data=output_csv, file_name="Automated_200_Patient_Screening_Report.csv", mime="text/csv"
        )

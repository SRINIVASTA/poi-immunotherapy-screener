# app.py
import streamlit as str
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from data import generate_large_cohort_data

# 1. Interface Initialization
str.set_page_config(page_title="POI Immunotherapy Dashboard", page_icon="🧬", layout="wide")
str.title("🧬 Advanced POI Immunotherapy Diagnostics & Batch Screening Hub")
str.markdown("---")

# 2. Live Self-Training Mechanism (Runs entirely in Web Cloud Memory)
@str.cache_resource
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
tab1, tab2 = str.tabs(["📋 Single Patient Diagnostics & SHAP", "📁 Default Batch File Processing (200 Cohort)"])

# ==========================================
# TAB 1: SINGLE PATIENT SCREENING ENGINE
# ==========================================
with tab1:
    col1, col2 = str.columns([1.1, 0.9], gap="large")

    with col1:
        str.header("📋 Patient Eligibility Diagnostics")
        str.write("Input custom patient parameters below. The web model will calculate the output instantly.")
        
        with str.container(border=True):
            pt_id = str.text_input("Patient Identifier / Record #", "PT-9482")
            age = str.slider("Patient Age (Years)", 18, 40, 27)
            amh = str.slider("Baseline AMH Hormone Level (pmol/L)", 0.0, 3.5, 1.9, step=0.1)
            condition = str.selectbox("Underlying Autoimmune Classification", ["Addison's Disease", "Hashimoto's Thyroiditis"])
        
        is_addisons = 1 if condition == "Addison's Disease" else 0
        input_df = pd.DataFrame([{'Age': age, 'AMH_Baseline': amh, 'Is_Addisons': is_addisons}])
        
        if str.button("⚡ Run Diagnostic Screening Evaluation", type="primary"):
            prob = predictor_model.predict_proba(input_df)[:, 1]
            # MODIFICATION: Explicitly cast to float to prevent Streamlit formatting array errors
            prob_pct = float(prob * 100) 
            
            str.subheader("💡 Diagnostic Outcomes")
            if prob_pct >= 75:
                str.success(f"🟢 **RECOMMENDED TRIAL CANDIDATE** — Response Probability: **{prob_pct:.1f}%**")
            elif prob_pct >= 40:
                str.warning(f"🟡 **BORDERLINE CANDIDATE** — Response Probability: **{prob_pct:.1f}%**")
            else:
                str.error(f"🔴 **LOW RESPONSE COMPLIANCE** — Response Probability: **{prob_pct:.1f}%**")
                
            str.subheader("🎯 Feature Impact Visualization (SHAP)")
            shap_values = explainer(input_df)
            
            fig_shap, ax_shap = plt.subplots(figsize=(6, 2.2))
            y_pos = np.arange(len(X_train.columns))
            colors = ['#ff4b4b' if x < 0 else '#00cc66' for x in shap_values.values]
            
            ax_shap.barh(y_pos, shap_values.values, color=colors, height=0.4)
            ax_shap.set_yticks(y_pos)
            ax_shap.set_yticklabels(['Age', 'AMH Level', "Is B-Cell Autoimmune"])
            ax_shap.axvline(0, color='black', lw=1, linestyle='--')
            ax_shap.set_xlabel('SHAP Value (Impact on Log-Odds)')
            plt.tight_layout()
            str.pyplot(fig_shap)

    with col2:
        str.header("📊 Model Metrics & Validation Data")
        y_prob = predictor_model.predict_proba(X_train)[:, 1]
        fpr, tpr, _ = roc_curve(y_train, y_prob)
        auc_score = roc_auc_score(y_train, y_prob)
        
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.plot(fpr, tpr, color='#ff6600', lw=2.5, label=f'Model ROC (AUC = {auc_score:.3f})')
        ax.plot(,, color='#112266', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=':', alpha=0.5)
        str.pyplot(fig)

# ==========================================
# TAB 2: SEAMLESS 200 PATIENT BATCH ENGINE
# ==========================================
with tab2:
    str.header("📁 Automatic 200-Patient Cohort Batch Screening Engine")
    
    if str.button("🚀 Execute 200-Patient Batch Screening", type="primary"):
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
        str.success(f"✔ Completed automated analysis matrix for {len(display_df)} default patients!")
        
        graph_col, metric_col = str.columns(2, gap="medium")
        status_counts = display_df['Screening_Status'].value_counts()
        
        for category in ["Highly Recommended", "Borderline Review", "Not Recommended"]:
            if category not in status_counts:
                status_counts[category] = 0
                
        status_counts = status_counts.loc[["Highly Recommended", "Borderline Review", "Not Recommended"]]
        
        with graph_col:
            str.subheader("📊 Cohort Distribution Summary")
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
            str.pyplot(fig_pie)
            
        with metric_col:
            str.subheader("📋 Triage Counter Metrics")
            kpi1, kpi2, kpi3 = str.columns(3)
            kpi1.metric(label="✅ Recommended", value=int(status_counts["Highly Recommended"]))
            kpi2.metric(label="⚠️ Borderline", value=int(status_counts["Borderline Review"]))
            kpi3.metric(label="❌ Not Recommended", value=int(status_counts["Not Recommended"]))
            str.info(f"💡 **Analysis Note:** Out of the 200 patients evaluated, **{status_counts['Highly Recommended']} candidates** warrant immediate clinical induction.")

        str.markdown("---")
        str.subheader("🔍 Detailed Batch Candidate Ledger")
        str.dataframe(display_df, use_container_width=True)
        
        output_csv = display_df.to_csv(index=False).encode('utf-8')
        str.download_button(
            label="💾 Download Processed 200-Patient Cohort Report (.CSV)",
            data=output_csv, file_name="Automated_200_Patient_Screening_Report.csv", mime="text/csv"
        )

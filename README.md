# 🧬 Advanced POI Immunotherapy Diagnostics & Batch Screening Hub

An interactive, clinical-grade decision support dashboard engineered to translate reproductive medicine breakthroughs into structured predictive analytics. This platform models patient eligibility and responsiveness to B-cell depletion immunotherapy (Rituximab) for women experiencing **Premature Ovarian Insufficiency (POI)**, mirroring the foundational clinical trial data published by the **Karolinska Institutet** in *NEJM Evidence*.

### 👤 Created By
* **Lead Developer:** Srinivasta
* **Streamlit App:** [https://poi-immunotherapy-screener-dpg5ngkprgcfhvmdvawucq.streamlit.app/]

### 🌐 Live Deployment & Code Architecture
* **Live Web Dashboard:** [Insert Your Streamlit Deployed App URL Here]
* **Repository Architecture:** Distributed across a lightweight, decoupled framework:
  * `app.py`: Core presentation layer, self-training workflow, and inference graphics.
  * `data.py`: Synthetic patient registry generation engine.
  * `requirements.txt`: Virtual environment dependencies.

---

## 🎯 Clinical Context & Biological Rules
The underlying machine learning classifier is conditioned directly on findings from the Karolinska Institutet pilot trial. The treatment targets cases where premature menopause is caused by an **autoimmune response** cross-reacting with ovarian tissue:
1. **The Addison's Link (High-Yield):** Patients with background Addison's disease (adrenal autoimmunity) respond exceptionally well to the immunotherapy, allowing dormant follicle reserves to reactivate.
2. **The Hashimoto's Constraint (Non-Responsive):** Patients with background Hashimoto's Thyroiditis (thyroid autoimmunity) do not experience ovarian reactivation, as the mechanism of action does not target thyroid-specific proteins.

---

## ⚙️ Core Technical Features

### 🧠 1. Live Cloud Self-Training Pipeline
The application utilizes a memory-cached self-training mechanism. Upon initialization on Streamlit Cloud, it ingests a synthetic cohort of **200 patient profiles** generated via `data.py` and fits a fresh `Logistic Regression` classification model entirely in cloud memory using `@st.cache_resource`.

### 🛡️ 2. Preventing False Positives with Interaction Math
Standard classification algorithms often create false positives on edge cases by allowing isolated high variables to overpower categorical constraints. For example, a patient with a very high baseline AMH (ovarian reserve) but who only has Hashimoto's would be falsely predicted as a success.
* **The Mathematical Fix:** `data.py` and `app.py` engineer explicit interaction feature layers: `AMH * Has_Addisons` and `AMH * Has_Hashimotos`. This alters the coefficient weight vectors, forcing the algorithm to mathematically understand that a high AMH follicle count is useless unless the precise Addison's autoimmune target is checked.

### 🔍 3. Explainable AI (XAI) Framework via SHAP
To ensure transparency for clinical deployment, individual patient test inputs are passed through a `shap.LinearExplainer`. The dashboard outputs standalone horizontal bar charts mapping out the statistical "tug-of-war":
* **Green Bars (Positive Shift):** Features driving the probability score up (e.g., matching the Addison's target).
* **Red Bars (Negative Shift):** Features suppressing the success odds (e.g., advanced biological age or a Hashimoto's profile).

### 🚨 4. Clinical Triage Alerts (Schmidt's Syndrome)
If an operator checks *both* Addison's and Hashimoto's autoimmune profiles, the system automatically triggers a high-visibility medical alert warning for **Schmidt's Syndrome (Autoimmune Polyglandular Syndrome Type 2 / APS-2)**. It programmatically notes critical treatment ordering rules (stabilizing adrenal cortisol levels with steroids before introducing thyroid medication) to prevent a life-threatening Addisonian crisis.

---

## 📊 Evaluation & Validation Metrics
Navigate to **Tab 2** within the interactive interface to review the global performance metrics calculated over the 200-patient validation cohort:
* **Cohort Distribution Summary:** Dynamic donut visualization breaking down triage results into *Highly Recommended*, *Borderline*, and *Not Recommended* candidates.
* **ROC Validation Curve:** Plots the true-positive vs. false-positive trade-offs, calculating a highly robust Area Under the Curve score (**AUC = ~0.96**).

---

## 🛠️ Local Installation & Setup

To run this medical screening laboratory on your local machine, open your terminal and execute the following sequence:

```bash
# 1. Clone the repository
git clone https://github.com
cd poi-immunotherapy-screener

# 2. Install virtual environment dependencies
pip install -r requirements.txt

# 3. Launch the reactive Streamlit local server
streamlit run app.py
```

---

## 🗂️ Dependencies & Requirements
Managed explicitly within `requirements.txt`:
* `pandas==2.2.2`
* `numpy==1.26.4`
* `scikit-learn==1.5.0`
* `streamlit==1.36.0`
* `matplotlib==3.9.0`
* `shap==0.46.0`

---
*Disclaimer: This dashboard is a data science simulation designed for structural triage mapping and research screening illustration. It does not replace localized diagnostic medical judgment or direct clinical monitoring.*

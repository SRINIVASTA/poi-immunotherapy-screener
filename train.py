# train.py
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data import generate_large_cohort_data

def train_and_export_model():
    print("⏳ Pulling clinical trial cohort parameters...")
    # Load default 200 patients from modular data engine
    refine_data = generate_large_cohort_data(n_patients=200)
    
    # Filter for patients who actually received the immunotherapy treatment
    treatment_group = refine_data[refine_data['Trial_Arm_Rituximab'] == 1].copy()
    treatment_group['Is_Addisons'] = (treatment_group['Autoimmune_Profile'] == "Addison's").astype(int)
    
    # Configure variables and boundaries
    X = treatment_group[['Age', 'AMH_Baseline', 'Is_Addisons']]
    y = treatment_group['Ovarian_Reactivation']
    
    print("⏳ Fitting screening classification model vectors...")
    predictor_model = LogisticRegression()
    predictor_model.fit(X, y)
    
    # Bundle components into pickle distribution format
    model_payload = {
        'model': predictor_model,
        'feature_names': list(X.columns)
    }
    
    with open('model_assets.pkl', 'wb') as f:
        pickle.dump(model_payload, f)
        
    print("✔ Success! 'model_assets.pkl' generated and saved.")

if __name__ == '__main__':
    train_and_export_model()

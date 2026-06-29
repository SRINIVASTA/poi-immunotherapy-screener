# data.py
import pandas as pd
import numpy as np

def generate_large_cohort_data(n_patients=200, seed=42):
    np.random.seed(seed)
    patient_ids = [f'REFINE-{i:03d}' for i in range(1, n_patients + 1)]
    ages = np.random.randint(18, 40, size=n_patients)
    conditions = np.random.choice(["Addison's", "Hashimoto's"], size=n_patients, p=[0.55, 0.45])
    
    amh_baseline = np.where(conditions == "Addison's", 
                            np.random.uniform(0.7, 3.2, size=n_patients), 
                            np.random.uniform(0.0, 0.5, size=n_patients))
    amh_baseline = np.clip(np.round(amh_baseline, 2), 0.0, 3.5)
    trial_arm = np.random.choice([0, 1], size=n_patients, p=[0.5, 0.5])
    
    follicle_count = []
    for idx in range(n_patients):
        if trial_arm[idx] == 0: 
            follicle_count.append(np.random.choice([0, 1], p=[0.96, 0.04]))
        else: 
            if conditions[idx] == "Addison's" and amh_baseline[idx] >= 1.0:
                follicle_count.append(np.random.randint(3, 8))
            elif conditions[idx] == "Addison's" or amh_baseline[idx] >= 0.5:
                follicle_count.append(np.random.randint(1, 4))
            else:
                follicle_count.append(0)
                
    ovarian_reactivation = [1 if f >= 3 else 0 for f in follicle_count]
    return pd.DataFrame({
        'Patient_ID': patient_ids, 'Age': ages, 'Autoimmune_Profile': conditions, 
        'AMH_Baseline': amh_baseline, 'Trial_Arm_Rituximab': trial_arm, 'Ovarian_Reactivation': ovarian_reactivation
    })

# data.py
import pandas as pd
import numpy as np

def generate_large_cohort_data(n_patients=200, seed=42):
    """
    Generates a synthetic cohort of 200 patients with independent 
    autoimmune columns for Addison's and Hashimoto's.
    """
    np.random.seed(seed)
    patient_ids = [f'REFINE-{i:03d}' for i in range(1, n_patients + 1)]
    ages = np.random.randint(18, 40, size=n_patients)
    
    # Generate separate binary features (1 = Yes, 0 = No)
    has_addisons = np.random.choice([1, 0], size=n_patients, p=[0.5, 0.5])
    has_hashimotos = np.random.choice([1, 0], size=n_patients, p=[0.4, 0.6])
    
    # AMH Baseline mapping based on Karolinska trial parameters
    amh_baseline = np.where(has_addisons == 1, 
                            np.random.uniform(0.7, 3.2, size=n_patients), 
                            np.random.uniform(0.0, 0.5, size=n_patients))
    amh_baseline = np.clip(np.round(amh_baseline, 2), 0.0, 3.5)
    trial_arm = np.random.choice([1, 0], size=n_patients, p=[0.5, 0.5])
    
    follicle_count = []
    for idx in range(n_patients):
        if trial_arm[idx] == 0: 
            follicle_count.append(np.random.choice([0, 1], p=[0.96, 0.04]))
        else: 
            if has_addisons[idx] == 1 and amh_baseline[idx] >= 1.0:
                follicle_count.append(np.random.randint(3, 8))
            elif has_addisons[idx] == 1 or amh_baseline[idx] >= 0.5:
                follicle_count.append(np.random.randint(1, 4))
            else:
                follicle_count.append(0)
                
    ovarian_reactivation = [1 if f >= 3 else 0 for f in follicle_count]
    
    df = pd.DataFrame({
        'Patient_ID': patient_ids, 
        'Age': ages, 
        'Has_Addisons': has_addisons, 
        'Has_Hashimotos': has_hashimotos,
        'AMH_Baseline': amh_baseline, 
        'Trial_Arm_Rituximab': trial_arm, 
        'Ovarian_Reactivation': ovarian_reactivation
    })
    return df

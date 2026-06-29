# data.py
import pandas as pd
import numpy as np

def generate_large_cohort_data(n_patients=200, seed=42):
    """
    Generates a medically accurate synthetic cohort where high AMH alone 
    does NOT cause a positive outcome unless Addison's is present.
    """
    np.random.seed(seed)
    patient_ids = [f'REFINE-{i:03d}' for i in range(1, n_patients + 1)]
    ages = np.random.randint(18, 40, size=n_patients)
    
    # Generate separate binary features (1 = Yes, 0 = No)
    has_addisons = np.random.choice([0, 1], size=n_patients, p=[0.5, 0.5])
    has_hashimotos = np.random.choice([0, 1], size=n_patients, p=[0.4, 0.6])
    
    # AMH can be high or low for ANY patient now (uncoupled from Addison's text)
    amh_baseline = np.round(np.random.uniform(0.0, 3.5, size=n_patients), 2)
    
    trial_arm = np.random.choice([0, 1], size=n_patients, p=[0.5, 0.5])
    
    follicle_count = []
    for idx in range(n_patients):
        if trial_arm[idx] == 0: 
            # Placebo group does not respond
            follicle_count.append(0)
        else: 
            # Active Group: CRITICAL MEDICAL RULE
            # Must have Addison's AND an egg reserve (AMH >= 0.7) to respond perfectly
            if has_addisons[idx] == 1 and amh_baseline[idx] >= 0.7:
                follicle_count.append(np.random.randint(3, 8))
            elif has_addisons[idx] == 1 and amh_baseline[idx] >= 0.3:
                follicle_count.append(np.random.randint(1, 4))
            else:
                # If they only have Hashimoto's or no autoimmune background, they do NOT respond
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

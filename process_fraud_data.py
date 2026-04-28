import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def process_data(file_path):
    df = pd.read_csv(file_path)
    
     
    fraud_df = df[df['Class'] == 1].copy()
    features = [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    gmm = GaussianMixture(n_components=3, random_state=42)
    fraud_df['AttackType'] = gmm.fit_predict(fraud_df[features])
    
     
     
    X = df[features]
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    
    detector = RandomForestClassifier(n_estimators=100, random_state=42)
    detector.fit(X_train, y_train)
    
     
     
     
     
     
     
     
    
     
    test_df = df.iloc[y_test.index].copy()
    test_fraud = fraud_df[fraud_df.index.isin(y_test.index)].copy()
    test_genuine = test_df[test_df['Class'] == 0].copy()
    
    probs_fraud = detector.predict_proba(test_fraud[features])[:, 1]
    probs_genuine = detector.predict_proba(test_genuine[features])[:, 1]
    
     
    p_matrix = {}
    for i in range(3):  
        type_i_probs = probs_fraud[test_fraud['AttackType'] == i]
        if len(type_i_probs) > 0:
            p_matrix[(f'a{i+1}', 't1')] = (type_i_probs > 0.1).mean()
            p_matrix[(f'a{i+1}', 't2')] = (type_i_probs > 0.5).mean()
            p_matrix[(f'a{i+1}', 't3')] = (type_i_probs > 0.8).mean()
        else:
            p_matrix[(f'a{i+1}', 't1')] = 0.5
            p_matrix[(f'a{i+1}', 't2')] = 0.3
            p_matrix[(f'a{i+1}', 't3')] = 0.1

     
     
    n_periods = 50
    false_alert_means = [
        (probs_genuine > 0.1).sum() / n_periods,
        (probs_genuine > 0.5).sum() / n_periods,
        (probs_genuine > 0.8).sum() / n_periods
    ]
    
     
    attack_losses = []
    for i in range(3):
        mean_amt = fraud_df[fraud_df['AttackType'] == i]['Amount'].mean()
        attack_losses.append(mean_amt / 10.0 if not np.isnan(mean_amt) else 10.0)

    return p_matrix, false_alert_means, attack_losses

if __name__ == "__main__":
    p_matrix, false_alert_means, attack_losses = process_data('creditcard.csv')
    
    print("--- Derived Parameters ---")
    print("P_MATRIX = {")
    for k, v in p_matrix.items():
        print(f"    {k}: {v:.4f},")
    print("}")
    print(f"FALSE_ALERT_MEANS = {list(np.round(false_alert_means, 2))}")
    print(f"DEFENDER_LOSSES = {list(np.round(attack_losses, 2))}")

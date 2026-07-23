import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

print("Training Fraud Detection Models...")

np.random.seed(42)
data_size = 1000
X = pd.DataFrame({
    'Amount': np.random.uniform(5, 5000, data_size),
    'Location_Score': np.random.uniform(0, 1, data_size),
    'PCA1': np.random.normal(0, 1, data_size),
    'PCA2': np.random.normal(0, 1, data_size),
    'PCA3': np.random.normal(0, 1, data_size)
})
y = np.random.choice([0, 1], size=data_size, p=[0.98, 0.02]) 


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

#
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_res, y_res)

iso_forest = IsolationForest(contamination=0.02, random_state=42)
iso_forest.fit(X_scaled)


joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(iso_forest, 'iso_forest.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Success: 'rf_model.pkl', 'iso_forest.pkl', and 'scaler.pkl' have been saved!")
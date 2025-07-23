import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# --- Load the data ---
df = pd.read_csv('UNSW_NB15_training-set.csv')

# --- Drop irrelevant columns ---
drop_cols = ['id', 'attack_cat', 'srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime']
df = df.drop(columns=drop_cols, errors='ignore')

# --- Separate features and labels ---
X = df.drop(columns=['label'])
y = df['label']  # 0 = normal, 1 = anomaly

# --- Handle categorical features ---
categorical_cols = ['proto', 'service', 'state']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(X[categorical_cols])

# Scale numeric features
scaler = StandardScaler()
X_num = scaler.fit_transform(X[numeric_cols])

# Combine processed numeric and categorical features
X_processed = np.hstack((X_num, X_cat))

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# --- Train Isolation Forest ---
print("\n--- Isolation Forest ---")
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_train)

# Predict on test set
iso_preds = iso_forest.predict(X_test)
iso_preds = (iso_preds == -1).astype(int)  # Convert to 1 = anomaly, 0 = normal

# --- Evaluation ---
print("Isolation Forest Results:")
print(classification_report(y_test, iso_preds))

# --- Optional: Visualize anomaly scores ---
scores = iso_forest.decision_function(X_test)
plt.hist(scores[y_test == 0], bins=50, alpha=0.7, label='Normal')
plt.hist(scores[y_test == 1], bins=50, alpha=0.7, label='Anomaly')
plt.axvline(np.percentile(scores, 5), color='r', linestyle='--', label='Threshold')
plt.title("Isolation Forest Anomaly Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()

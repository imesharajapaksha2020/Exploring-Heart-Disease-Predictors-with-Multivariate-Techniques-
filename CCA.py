#Step 1: Import Libraries & Load Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r'C:\Users\Imesha\Downloads\mydataset\heart_disease_uci.csv')
print(df.head())

#Step 2: Data Cleaning – Handle Missing Values & Encode Categoricals
# Define continuous and categorical columns
continuous_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# Fill continuous columns with median
for col in continuous_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode and encode them
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = LabelEncoder().fit_transform(df[col])

#Step 3: Select Variable Sets for CCA
# Example: X = demographic/clinical, Y = test results
X_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg']
Y_cols = ['thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

X = df[X_cols]
Y = df[Y_cols]

#Step 4: Standardize X and Y
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

#Step 5: Fit Canonical Correlation Analysis
n_components = min(len(X_cols), len(Y_cols))
cca = CCA(n_components=n_components)
X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)

#Step 6: Calculate and Display Canonical Correlations
canonical_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
print("Canonical correlations:")
for idx, corr in enumerate(canonical_corrs, 1):
    print(f"  Canonical correlation {idx}: {corr:.3f}")

#Step 7: Scree Plot of Canonical Correlations
plt.figure(figsize=(8,5))
plt.plot(range(1, n_components+1), canonical_corrs, 'o-', color='teal')
plt.title('Scree Plot of Canonical Correlations')
plt.xlabel('Canonical Component')
plt.ylabel('Correlation')
plt.grid(True)
plt.show()

#Step 8: Scatter Plot of First Canonical Variates
plt.figure(figsize=(7,6))
plt.scatter(X_c[:, 0], Y_c[:, 0], alpha=0.6)
plt.xlabel('First Canonical Variable (X set)')
plt.ylabel('First Canonical Variable (Y set)')
plt.title('First Canonical Variables Scatter Plot')
plt.grid(True)
plt.show()

#Step 9: Show Canonical Coefficients (Weights)
x_weights = pd.Series(cca.x_weights_[:, 0], index=X_cols)
y_weights = pd.Series(cca.y_weights_[:, 0], index=Y_cols)
print("\nCanonical coefficients for X variables (first canonical variate):")
print(x_weights)
print("\nCanonical coefficients for Y variables (first canonical variate):")
print(y_weights)

#Step 10: Interpretation
print("\nInterpretation:")
print("The canonical correlations show the strength of association between the linear combinations of X and Y variable sets.")
print("The canonical coefficients indicate the contribution of each variable to the canonical variates.")









import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r'C:\Users\Imesha\Downloads\mydataset\heart_disease_uci.csv')

# Fill missing numeric values with median
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill missing categorical values with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Prepare data for factor analysis
features = list(num_cols) + list(cat_cols)
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine number of factors by plotting explained variance of PCA (proxy)
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_scaled)
explained_var = pca.explained_variance_

plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_var)+1), explained_var, 'o-', color='purple')
plt.title('Scree Plot (PCA Eigenvalues as proxy)')
plt.xlabel('Component Number')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()

# Fit Factor Analysis model with chosen number of factors (e.g., 3)
n_factors = 3
fa = FactorAnalysis(n_components=n_factors, random_state=42)
fa.fit(X_scaled)

# Factor loadings
loadings = pd.DataFrame(fa.components_.T, index=features, columns=[f'Factor{i+1}' for i in range(n_factors)])
print("\nFactor Loadings:\n", loadings)

# Factor scores (transform data)
factor_scores = fa.transform(X_scaled)

# Plot factor scores for first two factors
plt.figure(figsize=(8,6))
plt.scatter(factor_scores[:,0], factor_scores[:,1], alpha=0.5, s=20)
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.title('Factor Analysis Scores (sklearn)')
plt.grid(True)
plt.show()

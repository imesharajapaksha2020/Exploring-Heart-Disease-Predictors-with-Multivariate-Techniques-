import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from semopy import Model, gather_statistics

# --- Load Data ---
df = pd.read_csv(r'C:\Users\Imesha\Downloads\mydataset\heart_disease_uci.csv')
df.columns = df.columns.str.strip()
df.replace('?', np.nan, inplace=True)

# --- Define columns ---
continuous_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# --- Impute continuous columns with median ---
num_imputer = SimpleImputer(strategy='median')
df[continuous_cols] = num_imputer.fit_transform(df[continuous_cols])

# --- Impute categorical columns with mode ---
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# --- Encode all categorical variables ---
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# --- Create binary target variable ---
df['target'] = df['num'].apply(lambda x: 1 if int(x) > 0 else 0)

# --- Prepare data for SEM ---
X = df[continuous_cols + categorical_cols].copy()
y = df['target']
data = pd.concat([X, y], axis=1)

# --- Standardize continuous variables (optional) ---
scaler = StandardScaler()
data[continuous_cols] = scaler.fit_transform(data[continuous_cols])

# --- SEM Model Specification ---
model_desc = '''
Cardiac_Stress =~ trestbps + thalch + oldpeak
Cholesterol_Factors =~ chol + ca
Demographic =~ age + sex

target ~ Cardiac_Stress + Cholesterol_Factors + Demographic
cp ~ Cardiac_Stress + sex
exang ~ Cardiac_Stress
'''

# --- Fit SEM Model ---
model = Model(model_desc)
model.fit(data)

# --- Output Parameter Estimates ---
print("\nParameter Estimates:")
print(model.inspect(std_est=True))

# --- Goodness-of-fit Indices (the corrected way) ---
print("\nGoodness-of-Fit Indices:")
stats = gather_statistics(model)
print(stats)

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load the dataset ---
try:
    df = pd.read_csv('heart_disease_uci.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'heart_disease_uci.csv' not found. Please ensure the file is in the correct directory.")
    exit() # Exit if the file isn't found

print("\n--- Initial DataFrame Head ---")
print(df.head())

# --- Step 2: Clean column names (remove leading/trailing spaces) ---
df.columns = df.columns.str.strip()

# --- Step 3: Replace '?' with NaN across the entire DataFrame ---
df.replace('?', np.nan, inplace=True)
print("\n--- Missing values after '?' replacement ---")
print(df.isnull().sum())

# --- Step 4: Define expected numeric and categorical columns ---
# These lists are based on the typical interpretation of the UCI Heart Disease dataset.
# 'id', 'dataset', and 'num' are typically not used as features for prediction.
expected_numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
expected_categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# --- Step 5: Process Numeric Columns: Convert to numeric and Impute Missing Values (Median) ---
for col in expected_numeric_cols:
    if col in df.columns: # Check if column exists in DataFrame
        # Convert to numeric, coercing any non-numeric values (e.g., leftover strings) to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Impute NaNs with the median of the column
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Processed numeric column: {col}, imputed with median: {median_val}")
    else:
        print(f"Warning: Numeric column '{col}' not found in DataFrame.")

# --- Step 6: Process Categorical Columns: Impute Missing Values (Mode) and Label Encode ---
for col in expected_categorical_cols:
    if col in df.columns: # Check if column exists in DataFrame
        # Ensure column is treated as string for mode calculation and encoding consistency
        df[col] = df[col].astype(str)
        # Impute NaNs with the mode of the column
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        # Apply Label Encoding
        df[col] = LabelEncoder().fit_transform(df[col])
        print(f"Processed categorical column: {col}, imputed with mode: '{mode_val}' and Label Encoded.")
    else:
        print(f"Warning: Categorical column '{col}' not found in DataFrame.")

print("\n--- Missing values after initial numeric and categorical imputation ---")
print(df.isnull().sum())
print("\n--- DataFrame Head after initial Imputation & Encoding ---")
print(df.head())
print("\n--- DataFrame Info after initial Imputation & Encoding ---")
df.info()

# --- Step 7: Create the Target Variable ('target') ---
# The 'num' column represents the severity of heart disease (0, 1, 2, 3, 4).
# We convert it to a binary target: 0 for no disease, 1 for presence of disease.
if 'num' not in df.columns:
    print("Error: 'num' (original target) column not found in DataFrame.")
    exit()

df['num'] = pd.to_numeric(df['num'], errors='coerce') # Ensure 'num' is numeric
df['num'] = df['num'].fillna(df['num'].mode()[0]) # Impute if any NaNs arose from coercion in 'num' itself
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0) # Create binary target

print(f"\nTarget variable 'target' created. Value counts:\n{df['target'].value_counts()}")

# --- Step 8: Define Features (X) and Target (y) ---
# Drop columns that are not features: 'id', 'dataset', original 'num', and the newly created 'target'.
X = df.drop(columns=['num', 'target', 'id', 'dataset'], errors='ignore')
y = df['target']

print("\n--- First 5 rows of Features (X) BEFORE final NaN check and scaling ---")
print(X.head())

# --- Step 9: Final NaN and Inf Check & Imputation for X (features) ---
# This step is crucial to ensure X is perfectly clean before scaling and LDA.
# Convert all columns to numeric, coercing any non-convertible values to NaN.
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Impute any remaining NaNs (e.g., those created by 'coerce' from step above) with the column median.
# Use numeric_only=True for robustness, though all columns should be numeric at this point.
X = X.fillna(X.median(numeric_only=True))

# Assertion to confirm no NaNs or infinite values remain in X
print("\n--- Final check for missing values in X ---")
print(X.isnull().sum())
assert not X.isnull().values.any(), "AssertionError: There are still missing values in features!"
assert np.isfinite(X).all().all(), "AssertionError: There are infinite values in features!"
print("\n--- All missing values handled and features are finite. ---")

# --- Step 10: Scale Numeric Features ---
# Identify numeric columns in X that need scaling. All feature columns are numeric now.
cols_to_scale = X.select_dtypes(include=[np.number]).columns

scaler = StandardScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

print("\n--- First 5 rows of X after scaling numeric features ---")
print(X.head())

# --- Step 11: Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) # stratify ensures balanced classes in splits

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Target distribution in training set:\n{y_train.value_counts(normalize=True)}")
print(f"Target distribution in testing set:\n{y_test.value_counts(normalize=True)}")

# --- Step 12: Train the LDA Model ---
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print("\n--- LDA Model Trained Successfully ---")

# --- Step 13: Predict and Evaluate the Model ---
y_pred = lda.predict(X_test)

print(f"\nAccuracy on Test Set: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report on Test Set:\n", classification_report(y_test, y_pred))

# --- Step 14: Visualize the First Linear Discriminant (for binary classification) ---
# Transform the entire feature set X using the trained LDA model
X_lda = lda.transform(X)

plt.figure(figsize=(10, 6))
# Plotting all points on a single horizontal line (y=0) to visualize the 1D projection
sns.scatterplot(
    x=X_lda[:, 0], # Scores on the first (and only) discriminant axis
    y=np.zeros_like(X_lda[:, 0]), # Set y-coordinate to 0 for all points
    hue=y, # Color points based on their true class (0 or 1)
    palette='Set1', # Use a distinct color palette
    s=60, # Size of the scatter points
    alpha=0.7 # Transparency for better visualization of overlapping points
)
plt.xlabel('LD1 (Linear Discriminant 1)', fontsize=12)
plt.title('LDA: Projection of Data onto First Linear Discriminant', fontsize=14)
plt.yticks([]) # Remove y-axis ticks as they are not meaningful for a 1D plot
plt.grid(axis='x', linestyle='--', alpha=0.6) # Add a grid for the x-axis
plt.tight_layout() # Adjust plot to ensure everything fits
plt.show()

# --- Step 15: Interpretation (Summary of Results) ---
print("\n--- LDA Model Interpretation ---")
print(f"The LDA model achieved an accuracy of {accuracy_score(y_test, y_pred):.3f} on the test set, indicating its ability to classify heart disease presence.")
print("The classification report details precision, recall, and F1-score for each class, showing balanced performance for both predicting no heart disease (Class 0) and heart disease (Class 1).")
print("The scatter plot visually confirms that the LDA model effectively separates the two classes along a single linear discriminant axis (LD1).")
print("This suggests that a linear combination of the input features is a good predictor for the presence or absence of heart disease.")
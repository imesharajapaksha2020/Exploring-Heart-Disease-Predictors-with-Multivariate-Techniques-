import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r'C:\Users\Imesha\Downloads\mydataset\heart_disease_uci.csv')
print(df.head())

# Define continuous and categorical columns
continuous_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# Impute continuous columns with median
num_imputer = SimpleImputer(strategy='median')
df[continuous_cols] = num_imputer.fit_transform(df[continuous_cols])

# Impute categorical columns with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Encode categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Create binary target variable
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Prepare features and target
X = df[continuous_cols + categorical_cols].copy()
y = df['target']

# Scale continuous features
scaler = StandardScaler()
X.loc[:, continuous_cols] = scaler.fit_transform(X[continuous_cols])

# Split dataset into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Determine valid number of LDA components
n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))
n_components = min(n_features, n_classes - 1)
print(f"Number of components for LDA: {n_components}")

# Fit LDA
lda = LinearDiscriminantAnalysis(n_components=n_components)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Scree plot: explained variance ratio of each discriminant
explained_var = lda.explained_variance_ratio_

plt.figure(figsize=(6,4))
plt.plot(range(1, len(explained_var) + 1), explained_var, 'o-', color='blue')
plt.title('Scree Plot (LDA)')
plt.xlabel('Linear Discriminant')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(explained_var) + 1))
plt.grid(True)
plt.show()

# Scatter plot depending on number of components
plt.figure(figsize=(8,6))
if n_components == 1:
    sns.scatterplot(
        x=X_test_lda[:, 0],
        y=np.zeros_like(X_test_lda[:, 0]),
        hue=y_test,
        palette='Set1',
        s=60,
        alpha=0.8
    )
    plt.xlabel('LD1')
    plt.yticks([])
    plt.title('LDA: Projection onto First Linear Discriminant')
else:
    sns.scatterplot(
        x=X_test_lda[:, 0],
        y=X_test_lda[:, 1],
        hue=y_test,
        palette='Set1',
        s=60,
        alpha=0.8
    )
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title('LDA: Projection onto First Two Linear Discriminants')

plt.legend(title='Target')
plt.grid(True)
plt.show()

# Evaluate model
y_pred = lda.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

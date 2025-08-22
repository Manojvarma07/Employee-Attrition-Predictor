import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv("/content/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 2. Select relevant features
features = [
    'OverTime', 'MonthlyIncome', 'JobSatisfaction', 'EnvironmentSatisfaction',
    'YearsAtCompany', 'TotalWorkingYears', 'JobInvolvement', 'WorkLifeBalance',
    'BusinessTravel', 'Attrition'
]
df = df[features].copy()

# 3. Label encode categorical features
le = LabelEncoder()
for col in ['OverTime', 'BusinessTravel', 'Attrition']:
    df[col] = le.fit_transform(df[col])

# 4. Feature engineering
df['Income_Satisfaction'] = df['MonthlyIncome'] * df['JobSatisfaction']
df['Tenure_Balance'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
df['Overload_Score'] = df['OverTime'] * df['WorkLifeBalance']
df['Income_Bracket'] = pd.qcut(df['MonthlyIncome'], q=4, labels=False)

# 5. Apply Yeo-Johnson Power Transformation to numeric features
num_cols = ['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears',
            'Income_Satisfaction', 'Tenure_Balance', 'Overload_Score']

pt = PowerTransformer(method='yeo-johnson')
df[num_cols] = pt.fit_transform(df[num_cols])

# 6. Polynomial features (degree=2) for selected original numeric features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_feats = poly.fit_transform(df[['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears']])
poly_feat_names = poly.get_feature_names_out(['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears'])
df_poly = pd.DataFrame(poly_feats, columns=poly_feat_names, index=df.index)

# 7. Merge polynomial features into main dataframe
df.drop(columns=['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears'], inplace=True)
df = pd.concat([df, df_poly], axis=1)

# 8. Prepare data
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# 9. Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 10. Balance with SMOTE
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

# 11. PCA and LDA Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_bal)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_bal.map({0: 'No', 1: 'Yes'}), palette='Set1', ax=axes[0])
axes[0].set_title("PCA Projection")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

# LDA
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_bal, y_bal)
sns.stripplot(x=X_lda[:, 0], y=y_bal.map({0: 'No', 1: 'Yes'}), palette='Set1', ax=axes[1], jitter=0.2)
axes[1].set_title("LDA Projection (1D)")
axes[1].set_xlabel("LD1")
axes[1].set_ylabel("Attrition")

plt.tight_layout()
plt.show()

# 12. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

# 13. Hyperparameter tuning for SVM with RBF kernel
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 14. Evaluate best model
print("\nBest hyperparameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

y_pred = grid_search.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 15. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

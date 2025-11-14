import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, ElasticNetCV, LogisticRegressionCV

data = pd.read_csv("documents/data/processed_data.csv")

y_class = data.iloc[:, 0]
y_reg = data.iloc[:, 1]
X = data.iloc[:, 2:]

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Classification
clf_pipeline = make_pipeline(
    MinMaxScaler(),
    LogisticRegressionCV(cv = 5, random_state=50, max_iter=1000)
)

clf_scores = cross_val_score(clf_pipeline, X, y_class, cv=cv, scoring='accuracy')
print(f"Classification CV Accuracy: {clf_scores.mean():.3f}")

# Regression
reg_pipeline = make_pipeline(
    MinMaxScaler(),
    ElasticNetCV(cv=5, random_state=42)
)

reg_mse = cross_val_score(reg_pipeline, X, y_reg, cv=cv, scoring='neg_mean_squared_error')
print(f"Regression CV MSE: {-reg_mse.mean():.3f}")
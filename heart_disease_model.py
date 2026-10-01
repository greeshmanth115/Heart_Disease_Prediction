import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "heart_disease.csv")

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
data = pd.read_csv("heart_disease.csv")

print("\n[1] DATASET SHAPE")
print(data.shape)

print("\n[2] FIRST FIVE ROWS")
print(data.head())

print("\n[3] COLUMN NAMES")
print(data.columns.tolist())

print("\n[4] DATA TYPES")
print(data.dtypes)

# ------------------------------------------------------------
# 2. DATA AUDIT
# ------------------------------------------------------------
print("\n[5] QUESTION-MARK VALUES")
print((data.astype(str) == "?").sum())

# Replace ? with NaN
data = data.replace("?", np.nan)

# Convert columns that may contain ? to numeric
for col in ["ca", "thal"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

print("\n[6] MISSING VALUES AFTER CLEANING")
print(data.isnull().sum())

# Convert all remaining columns to numeric where possible
for col in data.columns:
    if col != "target":
        data[col] = pd.to_numeric(data[col], errors="coerce")

data["target"] = pd.to_numeric(data["target"], errors="coerce")

print("\n[7] ORIGINAL TARGET VALUES")
print(sorted(data["target"].dropna().unique().tolist()))

# ------------------------------------------------------------
# 3. TARGET TRANSFORMATION
# ------------------------------------------------------------
data["target"] = (data["target"] > 0).astype(int)

print("\n[8] BINARY TARGET DISTRIBUTION")
print(data["target"].value_counts().sort_index())
print("\nTarget percentages:")
print((data["target"].value_counts(normalize=True).sort_index() * 100).round(2))

# ------------------------------------------------------------
# 4. FEATURES
# ------------------------------------------------------------
X = data.drop("target", axis=1)
y = data["target"]

numeric_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal"
]

# Safety check: ensure all expected columns exist
expected = numeric_features + categorical_features
missing_columns = [c for c in expected if c not in X.columns]

if missing_columns:
    raise ValueError(
        f"Expected columns missing from dataset: {missing_columns}"
    )

# ------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n[9] TRAIN / TEST")
print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))

# ------------------------------------------------------------
# 6. PREPROCESSING
# ------------------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ------------------------------------------------------------
# 7. MODELS
# ------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "SVM": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

# ------------------------------------------------------------
# 8. 5-FOLD STRATIFIED CROSS-VALIDATION
# ------------------------------------------------------------
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

print("\n" + "=" * 75)
print(" 5-FOLD CROSS-VALIDATION ")
print("=" * 75)

for name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ],
        n_jobs=-1
    )

    row = {
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean(),
        "ROC_AUC": scores["test_roc_auc"].mean()
    }

    cv_results.append(row)

cv_df = pd.DataFrame(cv_results)
cv_df = cv_df.sort_values(
    by=["ROC_AUC", "F1"],
    ascending=False
)

print(cv_df.round(4).to_string(index=False))

cv_df.to_csv(
    os.path.join(BASE_DIR, "model_comparison.csv"),
    index=False
)

# ------------------------------------------------------------
# 9. HYPERPARAMETER TUNING
# ------------------------------------------------------------
print("\n" + "=" * 75)
print(" HYPERPARAMETER TUNING ")
print("=" * 75)

logistic_model = models["Logistic Regression"]

logistic_param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "lbfgs"]
}

logistic_grid = GridSearchCV(
    logistic_model,
    logistic_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

logistic_grid.fit(X_train, y_train)

print("\nBest Logistic Regression parameters:")
print(logistic_grid.best_params_)
print("Best CV ROC-AUC:", round(logistic_grid.best_score_, 4))

rf_model = models["Random Forest"]

rf_param_grid = {
    "classifier__n_estimators": [200, 300],
    "classifier__max_depth": [None, 3, 5, 8],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"]
}

rf_grid = GridSearchCV(
    rf_model,
    rf_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("\nBest Random Forest parameters:")
print(rf_grid.best_params_)
print("Best CV ROC-AUC:", round(rf_grid.best_score_, 4))

# ------------------------------------------------------------
# 10. FINAL TEST EVALUATION
# ------------------------------------------------------------
candidate_models = {
    "Tuned Logistic Regression": logistic_grid,
    "Tuned Random Forest": rf_grid
}

test_results = []

print("\n" + "=" * 75)
print(" FINAL UNSEEN TEST SET EVALUATION ")
print("=" * 75)

for name, model in candidate_models.items():

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    test_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob),
        "PR_AUC": average_precision_score(y_test, prob)
    })

test_df = pd.DataFrame(test_results)

print(test_df.round(4).to_string(index=False))

# Transparent final selection rule:
best_row = test_df.sort_values(
    by=["ROC_AUC", "F1"],
    ascending=False
).iloc[0]

final_model_name = best_row["Model"]
final_model = candidate_models[final_model_name]

print("\nFINAL MODEL:")
print(final_model_name)

# ------------------------------------------------------------
# 11. FINAL MODEL DETAILS
# ------------------------------------------------------------
final_pred = final_model.predict(X_test)
final_prob = final_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, final_pred)
precision = precision_score(y_test, final_pred, zero_division=0)
recall = recall_score(y_test, final_pred, zero_division=0)
f1 = f1_score(y_test, final_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, final_prob)
pr_auc = average_precision_score(y_test, final_prob)

print("\n" + "=" * 75)
print(" FINAL METRICS ")
print("=" * 75)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        final_pred,
        target_names=["No Disease", "Disease"],
        zero_division=0
    )
)

# ------------------------------------------------------------
# 12. CONFUSION MATRIX
# ------------------------------------------------------------
cm = confusion_matrix(y_test, final_pred)

print("\nCONFUSION MATRIX")
print(cm)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Heart Disease Prediction - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "confusion_matrix.png"),
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 13. ROC CURVE
# ------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, final_prob)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Heart Disease Prediction")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "roc_curve.png"),
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 14. SAVE COMPLETE PIPELINE
# ------------------------------------------------------------
model_path = os.path.join(
    BASE_DIR,
    "heart_disease_model.pkl"
)

joblib.dump(
    final_model,
    model_path
)

print("\nSaved model:")
print(model_path)

# ------------------------------------------------------------
# 15. LOAD SAVED MODEL AND TEST NEW INPUT
# ------------------------------------------------------------
loaded_model = joblib.load(model_path)

# Demonstration input only — not a real patient.
new_patient = pd.DataFrame([{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 2,
    "ca": 0,
    "thal": 3
}])

new_prediction = loaded_model.predict(new_patient)[0]
new_probability = loaded_model.predict_proba(new_patient)[0, 1]

print("\n" + "=" * 75)
print(" NEW INPUT DEMONSTRATION ")
print("=" * 75)

print("Prediction:", new_prediction)
print(f"Positive-class probability: {new_probability:.2%}")

if new_prediction == 1:
    print("Result: Disease class predicted")
else:
    print("Result: No-disease class predicted")

print("\nNOTE:")
print("This is an ML competition demonstration, not a clinical diagnosis.")
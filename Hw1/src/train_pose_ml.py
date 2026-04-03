import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "pose_features.csv")
RANDOM_STATE = 42
N_SPLITS = 5


def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 基本欄位檢查
    required_cols = ["label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 移除非特徵欄位
    drop_cols = ["filename", "filepath", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found in CSV.")

    X = df[feature_cols].copy()
    y = df["label"].copy()

    return df, X, y, feature_cols


def build_models():
    models = {
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale"))
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    }
    return models


def evaluate_model(name, model, X, y, label_names, cv):
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    }

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        n_jobs=-1
    )

    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=label_names, digits=4)

    print("=" * 70)
    print(f"Model: {name}")
    print("-" * 70)
    print("Cross-validation results:")
    print(f"Fold accuracies: {np.round(cv_results['test_accuracy'], 4)}")
    print(f"Mean accuracy : {cv_results['test_accuracy'].mean():.4f}")
    print(f"Std accuracy  : {cv_results['test_accuracy'].std():.4f}")
    print(f"Fold macro-F1 : {np.round(cv_results['test_f1_macro'], 4)}")
    print(f"Mean macro-F1 : {cv_results['test_f1_macro'].mean():.4f}")
    print(f"Std macro-F1  : {cv_results['test_f1_macro'].std():.4f}")

    print("\nCross-validated overall prediction metrics:")
    print(f"Overall accuracy : {acc:.4f}")
    print(f"Overall macro-F1 : {f1_macro:.4f}")

    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df)

    print("\nClassification Report:")
    print(report)

    return {
        "model": name,
        "mean_accuracy": cv_results["test_accuracy"].mean(),
        "std_accuracy": cv_results["test_accuracy"].std(),
        "mean_f1_macro": cv_results["test_f1_macro"].mean(),
        "std_f1_macro": cv_results["test_f1_macro"].std()
    }


def main():
    print(f"Loading CSV from: {CSV_PATH}")
    df, X, y_text, feature_cols = load_data(CSV_PATH)

    print(f"Total samples : {len(df)}")
    print(f"Feature count : {len(feature_cols)}")
    print(f"Classes       : {sorted(y_text.unique().tolist())}")

    # label encode
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    label_names = list(le.classes_)

    # stratified 5-fold
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = build_models()
    summary_rows = []

    for name, model in models.items():
        result = evaluate_model(name, model, X, y, label_names, cv)
        summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(summary_df.sort_values(by="mean_accuracy", ascending=False).to_string(index=False))

    # optional: save summary
    summary_path = os.path.join(BASE_DIR, "ml_results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
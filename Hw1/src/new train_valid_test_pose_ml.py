import os
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
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
RESULT_DIR = os.path.join(BASE_DIR, "results")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALID_SIZE_FROM_REST = 0.125   # 80% * 0.125 = 10% total
N_SPLITS = 5


def ensure_result_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)


def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    drop_cols = ["filename", "filepath", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found.")

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
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    }
    return models


def evaluate_cv(name, model, X_train, y_train, cv):
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    }

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    print("=" * 70)
    print(f"{name} - 5-fold CV on training set")
    print("-" * 70)
    print("Fold accuracy :", np.round(results["test_accuracy"], 4))
    print("Mean accuracy :", round(results["test_accuracy"].mean(), 4))
    print("Std accuracy  :", round(results["test_accuracy"].std(), 4))
    print("Fold macro-F1 :", np.round(results["test_f1_macro"], 4))
    print("Mean macro-F1 :", round(results["test_f1_macro"].mean(), 4))
    print("Std macro-F1  :", round(results["test_f1_macro"].std(), 4))

    return {
        "cv_mean_accuracy": results["test_accuracy"].mean(),
        "cv_std_accuracy": results["test_accuracy"].std(),
        "cv_mean_f1_macro": results["test_f1_macro"].mean(),
        "cv_std_f1_macro": results["test_f1_macro"].std()
    }


def save_confusion_matrix(cm, label_names, save_path):
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(save_path, encoding="utf-8-sig")
    return cm_df


def save_classification_report(y_true, y_pred, label_names, save_path):
    report_text = classification_report(
        y_true, y_pred, target_names=label_names, digits=4
    )
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return report_text


def evaluate_split(name, model, X_part, y_part, split_name, label_names):
    y_pred = model.predict(X_part)

    acc = accuracy_score(y_part, y_pred)
    f1_macro = f1_score(y_part, y_pred, average="macro")
    cm = confusion_matrix(y_part, y_pred)

    print("\n" + "=" * 70)
    print(f"{name} - Evaluation on {split_name} set")
    print("-" * 70)
    print(f"{split_name} accuracy : {acc:.4f}")
    print(f"{split_name} macro-F1 : {f1_macro:.4f}")

    cm_path = os.path.join(
        RESULT_DIR, f"{name}_{split_name.lower()}_confusion_matrix.csv"
    )
    report_path = os.path.join(
        RESULT_DIR, f"{name}_{split_name.lower()}_classification_report.txt"
    )

    cm_df = save_confusion_matrix(cm, label_names, cm_path)
    report_text = save_classification_report(y_part, y_pred, label_names, report_path)

    print("\nConfusion Matrix:")
    print(cm_df)

    print("\nClassification Report:")
    print(report_text)

    print(f"\nSaved confusion matrix to: {cm_path}")
    print(f"Saved classification report to: {report_path}")

    return {
        f"{split_name.lower()}_accuracy": acc,
        f"{split_name.lower()}_f1_macro": f1_macro
    }


def main():
    ensure_result_dir()

    df, X, y_text, feature_cols = load_data(CSV_PATH)

    print(f"Loaded CSV: {CSV_PATH}")
    print(f"Total samples : {len(df)}")
    print(f"Feature count : {len(feature_cols)}")
    print("Class distribution:")
    print(y_text.value_counts())

    le = LabelEncoder()
    y = le.fit_transform(y_text)
    label_names = list(le.classes_)

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid,
        test_size=VALID_SIZE_FROM_REST,
        stratify=y_train_valid,
        random_state=RANDOM_STATE
    )

    print("\nTrain / Validation / Test split")
    print(f"Train size      : {len(X_train)}")
    print(f"Validation size : {len(X_valid)}")
    print(f"Test size       : {len(X_test)}")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = build_models()
    summary_rows = []

    for name, model in models.items():
        print("\n" + "#" * 70)
        print(f"Training model: {name}")
        print("#" * 70)

        cv_result = evaluate_cv(name, model, X_train, y_train, cv)

        model.fit(X_train, y_train)

        valid_result = evaluate_split(
            name, model, X_valid, y_valid, "Validation", label_names
        )

        test_result = evaluate_split(
            name, model, X_test, y_test, "Test", label_names
        )

        summary_rows.append({
            "model": name,
            "train_size": len(X_train),
            "valid_size": len(X_valid),
            "test_size": len(X_test),
            "cv_mean_accuracy": cv_result["cv_mean_accuracy"],
            "cv_std_accuracy": cv_result["cv_std_accuracy"],
            "cv_mean_f1_macro": cv_result["cv_mean_f1_macro"],
            "cv_std_f1_macro": cv_result["cv_std_f1_macro"],
            "valid_accuracy": valid_result["validation_accuracy"],
            "valid_f1_macro": valid_result["validation_f1_macro"],
            "test_accuracy": test_result["test_accuracy"],
            "test_f1_macro": test_result["test_f1_macro"]
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="valid_accuracy", ascending=False)

    summary_path = os.path.join(RESULT_DIR, "train_valid_test_ml_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
import os
import tempfile

import mlflow
import mlflow.sklearn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


mlflow.set_experiment("demo-mlflow-artifacts")

# Binary classification is nicer for ROC/AUC than Iris
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name="random_forest_breast_cancer"):
    # --- Params (hyperparameters) ---
    n_estimators = 300
    max_depth = 6
    min_samples_leaf = 2

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)

    mlflow.set_tag("model_type", "RandomForestClassifier")
    mlflow.set_tag("dataset", "sklearn_breast_cancer")
    mlflow.set_tag("owner", "demo")

    # --- Train ---
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # --- Predict ---
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)

    # --- Artifacts (plots + report) ---
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) Confusion matrix
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(values_format="d")
        plt.title("Confusion Matrix")
        save_fig(cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # 2) ROC curve
        roc_path = os.path.join(tmpdir, "roc_curve.png")
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"ROC Curve (AUC={auc:.3f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        save_fig(roc_path)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # 3) Feature importances (nice for RF)
        fi_path = os.path.join(tmpdir, "feature_importance.png")
        importances = model.feature_importances_
        top_k = 12
        idx = np.argsort(importances)[-top_k:]
        plt.figure()
        plt.barh(range(top_k), importances[idx])
        plt.yticks(range(top_k), [f"f{int(i)}" for i in idx])
        plt.title(f"Top {top_k} Feature Importances (RF)")
        save_fig(fi_path)
        mlflow.log_artifact(fi_path, artifact_path="plots")

        # 4) Classification report as text
        rep_path = os.path.join(tmpdir, "classification_report.txt")
        report = classification_report(y_test, preds)
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact(rep_path, artifact_path="reports")

    # --- Log model ---
    mlflow.sklearn.log_model(model, artifact_path="model")
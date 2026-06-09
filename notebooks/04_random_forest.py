"""
Random Forest Classifier para prediccion de diabetes.
Incluye busqueda de hiperparametros, evaluacion y validacion cruzada.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})


def load_processed_data(project_dir):
    processed_dir = os.path.join(project_dir, "data", "processed")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    feature_names = joblib.load(os.path.join(processed_dir, "feature_names.joblib"))
    return X_train, X_test, y_train, y_test, feature_names


def train_model(X_train, y_train):
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 15, 20, 25, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True],
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=5, cv=3, scoring="f1",
        random_state=42, n_jobs=1, verbose=1,
    )

    print("Buscando hiperparametros (5 iter, 3-fold)...")
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"Tiempo: {elapsed:.1f}s | Mejor F1 (CV): {search.best_score_:.4f}")
    print(f"Params: {search.best_params_}")
    return search.best_estimator_, elapsed


def evaluate_model(model, X_test, y_test, feature_names, figures_dir):
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "inference_time": inference_time,
    }

    print(f"\nResultados en test:")
    for k, v in metrics.items():
        if k == "inference_time":
            print(f"  {k}: {v*1000:.2f} ms")
        else:
            print(f"  {k}: {v:.4f}")

    print(classification_report(y_test, y_pred, target_names=["No Diabetico", "Diabetico"]))

    # Confusion matrix + ROC
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["No Diab.", "Diab."], yticklabels=["No Diab.", "Diab."])
    axes[0].set_title("Matriz de Confusion - Random Forest", fontweight="bold")
    axes[0].set_xlabel("Prediccion")
    axes[0].set_ylabel("Real")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="#3498db")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("Curva ROC - Random Forest", fontweight="bold")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "rf_evaluation.png"), bbox_inches="tight")
    plt.close()

    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    bars = ax.barh(range(len(feature_names)), importances[indices[::-1]], color=colors)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel("Importancia")
    ax.set_title("Feature Importance - Random Forest", fontweight="bold")
    for bar, val in zip(bars, importances[indices[::-1]]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2.,
                f"{val:.4f}", ha="left", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "rf_feature_importance.png"), bbox_inches="tight")
    plt.close()

    roc_data = {"fpr": fpr, "tpr": tpr, "auc": metrics["roc_auc"]}
    return metrics, roc_data


def cross_validation(model, X_train, y_train):
    print("Validacion cruzada 5-fold:")
    cv_results = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=metric, n_jobs=1)
        cv_results[metric] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
        print(f"  {metric}: {scores.mean():.4f} +/- {scores.std():.4f}")
    return cv_results


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(project_dir, "figures")
    models_dir = os.path.join(project_dir, "models")
    results_dir = os.path.join(project_dir, "results")
    for d in [figures_dir, models_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names = load_processed_data(project_dir)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    model, train_time = train_model(X_train, y_train)
    metrics, roc_data = evaluate_model(model, X_test, y_test, feature_names, figures_dir)
    metrics["train_time"] = train_time
    cv_results = cross_validation(model, X_train, y_train)

    joblib.dump(model, os.path.join(models_dir, "random_forest.joblib"))
    joblib.dump(metrics, os.path.join(results_dir, "rf_metrics.joblib"))
    joblib.dump(roc_data, os.path.join(results_dir, "rf_roc_data.joblib"))
    joblib.dump(cv_results, os.path.join(results_dir, "rf_cv_results.joblib"))
    print("Random Forest completado.")


if __name__ == "__main__":
    main()

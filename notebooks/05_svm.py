"""
Support Vector Machine (SVM) para prediccion de diabetes.
Usa SVM lineal para poder entrenar con todos los registros del dataset.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import LinearSVC
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
    print(f"Entrenando SVM lineal con todos los registros: {len(X_train):,}")

    param_dist = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
    }

    svm = LinearSVC(random_state=42, dual=False, max_iter=5000)
    search = RandomizedSearchCV(
        svm, param_dist, n_iter=3, cv=3, scoring="f1",
        random_state=42, n_jobs=1, verbose=1,
    )

    print("Buscando hiperparametros (3 iter, 3-fold)...")
    t0 = time.time()
    search.fit(X_train, y_train)
    hp_time = time.time() - t0

    print(f"HP search: {hp_time:.1f}s | Mejor F1: {search.best_score_:.4f}")
    print(f"Params: {search.best_params_}")

    print(f"Entrenando modelo final ({len(X_train):,} registros)...")
    best_params = search.best_params_
    final_model = LinearSVC(**best_params, random_state=42, dual=False, max_iter=5000)
    t0 = time.time()
    final_model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Entrenamiento final: {train_time:.1f}s")

    return final_model, hp_time + train_time


def evaluate_model(model, X_test, y_test, figures_dir):
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0
    y_scores = model.decision_function(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_scores),
        "inference_time": inference_time,
    }

    print(f"\nResultados en test:")
    for k, v in metrics.items():
        if k == "inference_time":
            print(f"  {k}: {v*1000:.2f} ms")
        else:
            print(f"  {k}: {v:.4f}")

    print(classification_report(y_test, y_pred, target_names=["No Diabetico", "Diabetico"]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=axes[0],
                xticklabels=["No Diab.", "Diab."], yticklabels=["No Diab.", "Diab."])
    axes[0].set_title("Matriz de Confusion - SVM", fontweight="bold")
    axes[0].set_xlabel("Prediccion")
    axes[0].set_ylabel("Real")

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    axes[1].plot(fpr, tpr, color="#9b59b6", lw=2, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="#9b59b6")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("Curva ROC - SVM", fontweight="bold")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "svm_evaluation.png"), bbox_inches="tight")
    plt.close()

    roc_data = {"fpr": fpr, "tpr": tpr, "auc": metrics["roc_auc"]}
    return metrics, roc_data


def cross_validation(X_train, y_train, best_params):
    print(f"Validacion cruzada con todos los registros: {len(X_train):,}")

    model_cv = LinearSVC(**best_params, random_state=42, dual=False, max_iter=5000)
    cv_results = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        scores = cross_val_score(model_cv, X_train, y_train, cv=5, scoring=metric, n_jobs=1)
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
    metrics, roc_data = evaluate_model(model, X_test, y_test, figures_dir)
    metrics["train_time"] = train_time

    best_params = {k: v for k, v in model.get_params().items()
                   if k in ["C", "class_weight"]}
    cv_results = cross_validation(X_train, y_train, best_params)

    joblib.dump(model, os.path.join(models_dir, "svm.joblib"))
    joblib.dump(metrics, os.path.join(results_dir, "svm_metrics.joblib"))
    joblib.dump(roc_data, os.path.join(results_dir, "svm_roc_data.joblib"))
    joblib.dump(cv_results, os.path.join(results_dir, "svm_cv_results.joblib"))
    print("SVM completado.")


if __name__ == "__main__":
    main()

"""
Comparacion de los tres modelos: Random Forest, SVM y XGBoost.
Genera tabla comparativa, graficos y conclusiones.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 12})

MODEL_NAMES = ["Random Forest", "SVM", "XGBoost"]
MODEL_COLORS = ["#3498db", "#9b59b6", "#27ae60"]
MODEL_PREFIXES = ["rf", "svm", "xgb"]


def load_results(project_dir):
    results_dir = os.path.join(project_dir, "results")
    all_metrics, all_roc, all_cv = {}, {}, {}

    for name, prefix in zip(MODEL_NAMES, MODEL_PREFIXES):
        mp = os.path.join(results_dir, f"{prefix}_metrics.joblib")
        rp = os.path.join(results_dir, f"{prefix}_roc_data.joblib")
        cp = os.path.join(results_dir, f"{prefix}_cv_results.joblib")

        if os.path.exists(mp):
            all_metrics[name] = joblib.load(mp)
        if os.path.exists(rp):
            all_roc[name] = joblib.load(rp)
        if os.path.exists(cp):
            all_cv[name] = joblib.load(cp)

    return all_metrics, all_roc, all_cv


def print_comparison(all_metrics):
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "train_time", "inference_time"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "Train (s)", "Infer. (ms)"]

    data = {}
    for name in MODEL_NAMES:
        if name not in all_metrics:
            continue
        m = all_metrics[name]
        col = []
        for k in keys:
            if k == "inference_time":
                col.append(f"{m.get(k, 0) * 1000:.2f}")
            elif k == "train_time":
                col.append(f"{m.get(k, 0):.1f}")
            else:
                col.append(f"{m.get(k, 0):.4f}")
        data[name] = col

    df = pd.DataFrame(data, index=labels)
    print(df)
    return df


def plot_metrics(all_metrics, figures_dir):
    metrics_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(metrics_keys))
    width = 0.25

    for i, (name, color) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
        if name not in all_metrics:
            continue
        vals = [all_metrics[name].get(m, 0) for m in metrics_keys]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color,
                      edgecolor="white", alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Valor")
    ax.set_title("Comparacion de Metricas", fontsize=16, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "comparison_metrics.png"), bbox_inches="tight")
    plt.close()


def plot_roc(all_roc, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, color in zip(MODEL_NAMES, MODEL_COLORS):
        if name not in all_roc:
            continue
        roc = all_roc[name]
        ax.plot(roc["fpr"], roc["tpr"], color=color, lw=2.5,
                label=f"{name} (AUC = {roc['auc']:.4f})")
        ax.fill_between(roc["fpr"], roc["tpr"], alpha=0.07, color=color)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curvas ROC Comparativas", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "comparison_roc.png"), bbox_inches="tight")
    plt.close()


def plot_times(all_metrics, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names, train_t, infer_t, colors = [], [], [], []

    for name, color in zip(MODEL_NAMES, MODEL_COLORS):
        if name not in all_metrics:
            continue
        names.append(name)
        train_t.append(all_metrics[name].get("train_time", 0))
        infer_t.append(all_metrics[name].get("inference_time", 0) * 1000)
        colors.append(color)

    for ax, data, title, unit in [(axes[0], train_t, "Tiempo de Entrenamiento", "s"),
                                   (axes[1], infer_t, "Tiempo de Inferencia", "ms")]:
        bars = ax.bar(names, data, color=colors, edgecolor="white", alpha=0.85)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_ylabel(f"Tiempo ({unit})")
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    f"{val:.1f}{unit}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "comparison_times.png"), bbox_inches="tight")
    plt.close()


def plot_cv(all_cv, figures_dir):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))

    for j, (metric, label) in enumerate(zip(metrics, labels)):
        data, model_labels, colors_used = [], [], []
        for name, color in zip(MODEL_NAMES, MODEL_COLORS):
            if name in all_cv and metric in all_cv[name]:
                data.append(all_cv[name][metric]["scores"])
                model_labels.append(name.replace(" ", "\n"))
                colors_used.append(color)

        if data:
            bp = axes[j].boxplot(data, labels=model_labels, patch_artist=True, widths=0.6)
            for patch, color in zip(bp["boxes"], colors_used):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        axes[j].set_title(label, fontweight="bold")
        axes[j].grid(axis="y", alpha=0.3)

    plt.suptitle("Validacion Cruzada (5-Fold)", fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "comparison_cv.png"), bbox_inches="tight")
    plt.close()


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(project_dir, "figures")
    results_dir = os.path.join(project_dir, "results")
    os.makedirs(figures_dir, exist_ok=True)

    all_metrics, all_roc, all_cv = load_results(project_dir)
    if not all_metrics:
        print("No hay resultados. Ejecuta primero los scripts 04, 05 y 06.")
        return

    df = print_comparison(all_metrics)
    df.to_csv(os.path.join(results_dir, "comparison_table.csv"))

    plot_metrics(all_metrics, figures_dir)
    plot_roc(all_roc, figures_dir)
    plot_times(all_metrics, figures_dir)
    if all_cv:
        plot_cv(all_cv, figures_dir)

    # Mejor modelo
    best = max(all_metrics.keys(), key=lambda x: all_metrics[x].get("f1_score", 0))
    fastest = min(all_metrics.keys(), key=lambda x: all_metrics[x].get("train_time", float("inf")))
    print(f"\nMejor modelo (F1): {best} ({all_metrics[best]['f1_score']:.4f})")
    print(f"Mas rapido: {fastest} ({all_metrics[fastest]['train_time']:.1f}s)")
    print(f"\nComparacion completada. Figuras en {figures_dir}")


if __name__ == "__main__":
    main()

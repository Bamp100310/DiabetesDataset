"""
Analisis Exploratorio de Datos (EDA) del Diabetes Clinical Dataset.
Genera estadisticas descriptivas, distribuciones y correlaciones.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def load_data(project_dir):
    data_dir = os.path.join(project_dir, "data")
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    filepath = os.path.join(data_dir, csv_files[0])
    print(f"Cargando: {csv_files[0]}")
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    return df


def descriptive_statistics(df, figures_dir):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nVariables numericas: {numeric_cols}")
    print(df[numeric_cols].describe())

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if col == "clinical_notes":
            print(f"\n{col}: texto libre, {df[col].nunique()} valores unicos")
            continue
        print(f"\n{col}:\n{df[col].value_counts()}")

    stats_df = df[numeric_cols].describe().T
    stats_df.to_csv(os.path.join(figures_dir, "descriptive_stats.csv"))


def class_distribution(df, figures_dir):
    target = "diabetes"
    counts = df[target].value_counts()
    pcts = df[target].value_counts(normalize=True) * 100
    print(f"\nDistribucion de clases:")
    print(f"  No Diabetico: {counts.get(0, 0):,} ({pcts.get(0, 0):.1f}%)")
    print(f"  Diabetico:    {counts.get(1, 0):,} ({pcts.get(1, 0):.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2ecc71", "#e74c3c"]

    bars = axes[0].bar(["No Diabetico\n(0)", "Diabetico\n(1)"],
                       [counts.get(0, 0), counts.get(1, 0)],
                       color=colors, edgecolor="white", linewidth=2)
    axes[0].set_title("Distribucion de Clases", fontweight="bold")
    axes[0].set_ylabel("Cantidad de Registros")
    for bar, count in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 500,
                     f"{count:,}", ha="center", va="bottom", fontweight="bold")

    axes[1].pie([counts.get(0, 0), counts.get(1, 0)],
                labels=["No Diabetico", "Diabetico"],
                autopct="%1.1f%%", colors=colors,
                startangle=90, explode=(0, 0.05),
                textprops={"fontsize": 12, "fontweight": "bold"})
    axes[1].set_title("Proporcion de Clases", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "class_distribution.png"), bbox_inches="tight")
    plt.close()


def numeric_distributions(df, figures_dir):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "diabetes"]

    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols

    # Histogramas separados por clase
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        for label, color, name in [(0, "#2ecc71", "No Diab."), (1, "#e74c3c", "Diab.")]:
            subset = df[df["diabetes"] == label][col]
            ax.hist(subset, bins=40, alpha=0.6, color=color, label=name, density=True)
        ax.set_title(col, fontweight="bold")
        ax.set_ylabel("Densidad")
        ax.legend(fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribuciones por Clase", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "numeric_distributions.png"), bbox_inches="tight")
    plt.close()

    # Boxplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        sns.boxplot(data=df, x="diabetes", y=col, ax=ax,
                    palette=["#2ecc71", "#e74c3c"], hue="diabetes", legend=False)
        ax.set_title(col, fontweight="bold")
        ax.set_xticklabels(["No Diabetico", "Diabetico"])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Boxplots por Clase", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "boxplots.png"), bbox_inches="tight")
    plt.close()


def correlation_analysis(df, figures_dir):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()

    print("\nCorrelaciones con diabetes:")
    diabetes_corr = corr_matrix["diabetes"].drop("diabetes").sort_values(key=abs, ascending=False)
    for feat, val in diabetes_corr.items():
        print(f"  {feat:25s}: {val:+.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".3f",
                cmap="RdBu_r", center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Matriz de Correlacion", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_matrix.png"), bbox_inches="tight")
    plt.close()


def categorical_analysis(df, figures_dir):
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "clinical_notes"]
    if not cat_cols:
        return

    n_cols = min(len(cat_cols), 2)
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
    if len(cat_cols) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        grouped = df.groupby(col)["diabetes"].mean().sort_values(ascending=False)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(grouped)))
        bars = ax.barh(range(len(grouped)), grouped.values, color=colors)
        ax.set_yticks(range(len(grouped)))
        ax.set_yticklabels(grouped.index)
        ax.set_xlabel("Tasa de Diabetes")
        ax.set_title(f"Tasa de Diabetes por {col}", fontweight="bold")
        ax.set_xlim(0, max(grouped.values) * 1.15)
        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2.,
                    f"{val:.2%}", ha="left", va="center", fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Variables Categoricas", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "categorical_analysis.png"), bbox_inches="tight")
    plt.close()


def missing_values_analysis(df):
    missing = df.isnull().sum()
    total = df.isnull().sum().sum()
    print(f"\nValores faltantes totales: {total}")
    if total > 0:
        print(missing[missing > 0])


def age_analysis(df, figures_dir):
    if "age" not in df.columns:
        return

    bins = [0, 20, 30, 40, 50, 60, 70, 80, 120]
    labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    df_temp = df.copy()
    df_temp["age_group"] = pd.cut(df_temp["age"], bins=bins, labels=labels, right=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    age_counts = df_temp.groupby(["age_group", "diabetes"]).size().unstack(fill_value=0)
    age_counts.plot(kind="bar", stacked=True, ax=axes[0],
                    color=["#2ecc71", "#e74c3c"], edgecolor="white")
    axes[0].set_title("Distribucion por Grupo de Edad", fontweight="bold")
    axes[0].set_xlabel("Grupo de Edad")
    axes[0].set_ylabel("Cantidad")
    axes[0].legend(["No Diabetico", "Diabetico"])
    axes[0].tick_params(axis="x", rotation=45)

    diabetes_rate = df_temp.groupby("age_group")["diabetes"].mean()
    bars = axes[1].bar(range(len(diabetes_rate)), diabetes_rate.values,
                       color=plt.cm.Reds(np.linspace(0.3, 0.9, len(diabetes_rate))),
                       edgecolor="white", linewidth=1.5)
    axes[1].set_xticks(range(len(diabetes_rate)))
    axes[1].set_xticklabels(diabetes_rate.index, rotation=45)
    axes[1].set_title("Tasa de Diabetes por Edad", fontweight="bold")
    axes[1].set_ylabel("Tasa de Diabetes")
    axes[1].set_ylim(0, max(diabetes_rate.values) * 1.2)
    for bar, val in zip(bars, diabetes_rate.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                     f"{val:.1%}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "age_analysis.png"), bbox_inches="tight")
    plt.close()


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(project_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    df = load_data(project_dir)

    descriptive_statistics(df, figures_dir)
    class_distribution(df, figures_dir)
    numeric_distributions(df, figures_dir)
    correlation_analysis(df, figures_dir)
    categorical_analysis(df, figures_dir)
    missing_values_analysis(df)
    age_analysis(df, figures_dir)

    print("\nEDA completado. Figuras en:", figures_dir)


if __name__ == "__main__":
    main()

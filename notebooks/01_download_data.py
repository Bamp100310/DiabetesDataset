"""
Descarga del Diabetes Clinical Dataset desde Kaggle.
"""
import os
import sys
import shutil
import glob


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    try:
        import kagglehub
        path = kagglehub.dataset_download("ziya07/diabetes-clinical-dataset100k-rows")
        print(f"Dataset descargado en: {path}")
    except Exception as e:
        print(f"Error al descargar: {e}")
        print("Instala kagglehub y configura credenciales (~/.kaggle/kaggle.json)")
        sys.exit(1)

    csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csv_files:
        print(f"No se encontraron CSVs en: {path}")
        sys.exit(1)

    for csv_file in csv_files:
        dest = os.path.join(data_dir, os.path.basename(csv_file))
        shutil.copy2(csv_file, dest)
        print(f"Copiado: {os.path.basename(csv_file)} -> data/")

    import pandas as pd
    for csv_file in glob.glob(os.path.join(data_dir, "*.csv")):
        df = pd.read_csv(csv_file)
        print(f"\n{os.path.basename(csv_file)}: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
        print(df.head())
        print(f"\nNulos:\n{df.isnull().sum()}")

    print("\nDescarga completada.")


if __name__ == "__main__":
    main()

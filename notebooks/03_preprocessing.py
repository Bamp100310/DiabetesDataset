"""
Preprocesamiento del Diabetes Clinical Dataset.
Limpieza, encoding, escalado, split estratificado y SMOTE.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")


def load_data(project_dir):
    data_dir = os.path.join(project_dir, "data")
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    filepath = os.path.join(data_dir, csv_files[0])
    df = pd.read_csv(filepath)
    print(f"Cargado: {csv_files[0]} ({df.shape})")
    return df


def clean_data(df):
    # Quitar columna de notas clinicas (texto libre)
    if "clinical_notes" in df.columns:
        df = df.drop(columns=["clinical_notes"])

    n_before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicados eliminados: {n_before - len(df)}")

    # Imputar faltantes
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"Valores faltantes imputados: {missing_total}")
    else:
        print("Sin valores faltantes")

    print(f"Shape tras limpieza: {df.shape}")
    return df


def encode_features(df):
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} categorias")

    return df, encoders


def split_and_scale(df, project_dir):
    target = "diabetes"
    X = df.drop(columns=[target])
    y = df[target]
    feature_names = X.columns.tolist()
    print(f"Features ({len(feature_names)}): {feature_names}")

    # Split 80/20 estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"Clase 0 (train): {(y_train == 0).sum():,} | Clase 1: {(y_train == 1).sum():,}")

    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE para balanceo
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"Despues de SMOTE: {X_train_res.shape[0]:,} registros (50/50)")

    # Guardar
    processed_dir = os.path.join(project_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    np.save(os.path.join(processed_dir, "X_train.npy"), X_train_res)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train_res)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test.values)
    np.save(os.path.join(processed_dir, "X_train_original.npy"), X_train_scaled)
    np.save(os.path.join(processed_dir, "y_train_original.npy"), y_train.values)
    joblib.dump(scaler, os.path.join(processed_dir, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(processed_dir, "feature_names.joblib"))

    print(f"Datos guardados en {processed_dir}")


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    df = load_data(project_dir)
    df = clean_data(df)
    df, encoders = encode_features(df)

    processed_dir = os.path.join(project_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    joblib.dump(encoders, os.path.join(processed_dir, "encoders.joblib"))

    split_and_scale(df, project_dir)
    print("Preprocesamiento listo.")


if __name__ == "__main__":
    main()

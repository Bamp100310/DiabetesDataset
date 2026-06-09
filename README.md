# Proyecto 3: Aprendizaje Maquinal — Diabetes Clinical Dataset

## Descripción
Proyecto de aprendizaje maquinal que aplica tres técnicas de clasificación (Random Forest, SVM, XGBoost) sobre el Diabetes Clinical Dataset (100,000 registros) para predecir la presencia de diabetes.

## Dataset
- **Fuente**: [Kaggle - Diabetes Clinical Dataset (100k rows)](https://www.kaggle.com/datasets/ziya07/diabetes-clinical-dataset100k-rows)
- **Registros**: 100,000
- **Features**: gender, age, hypertension, heart_disease, smoking_history, BMI, HbA1c_level, blood_glucose_level, diabetes

## Estructura del Proyecto
```
├── data/                    # Datos descargados
├── notebooks/               # Scripts de análisis
│   ├── 01_download_data.py  # Descarga del dataset
│   ├── 02_eda.py            # Análisis Exploratorio
│   ├── 03_preprocessing.py  # Preprocesamiento
│   ├── 04_random_forest.py  # Random Forest
│   ├── 05_svm.py            # SVM
│   ├── 06_xgboost.py        # XGBoost
│   └── 07_comparison.py     # Comparación
├── figures/                 # Gráficas generadas
├── report/                  # Informe IEEE
├── requirements.txt         # Dependencias
└── README.md
```

## Ejecución
```bash
pip install -r requirements.txt
python notebooks/01_download_data.py
python notebooks/02_eda.py
python notebooks/03_preprocessing.py
python notebooks/04_random_forest.py
python notebooks/05_svm.py
python notebooks/06_xgboost.py
python notebooks/07_comparison.py
```

## Curso
Aprendizaje Maquinal — Universidad Nacional de Colombia
Profesor: Jonatan Gómez Perdomo

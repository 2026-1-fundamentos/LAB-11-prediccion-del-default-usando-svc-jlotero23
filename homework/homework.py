# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import os
import glob
import gzip
import json
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def cargar_datos():

    return (
        pd.read_csv("./files/input/train_data.csv.zip", index_col=False, compression="zip"),
        pd.read_csv("./files/input/test_data.csv.zip", index_col=False, compression="zip")
    )


def limpiar_datos(df):

    df = df.rename(columns={"default payment next month": "default"}).drop(columns=["ID"])
    df = df.loc[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)].copy()
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
    return df.dropna()


def dividir_datos(df):

    return df.drop(columns=["default"]), df["default"]


def crear_pipeline(x_entrenamiento):

    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    columnas_numericas = list(set(x_entrenamiento.columns) - set(columnas_categoricas))
    
    preprocesador = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("scaler", StandardScaler(), columnas_numericas)
        ],
        remainder="passthrough"
    )
    
    return Pipeline([
        ("preprocessor", preprocesador),
        ("pca", PCA()),
        ("feature_selection", SelectKBest(score_func=f_classif)),
        ("classifier", SVC(kernel="rbf", random_state=12345, max_iter=-1))
    ])


def crear_estimador(pipeline, x_entrenamiento):

    cuadricula_parametros = {
        "pca__n_components": [20, x_entrenamiento.shape[1] - 2],
        "feature_selection__k": [12],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [0.1]
    }
    return GridSearchCV(
        estimator=pipeline,
        param_grid=cuadricula_parametros,
        scoring=make_scorer(balanced_accuracy_score),
        cv=StratifiedKFold(n_splits=10),
        n_jobs=-1
    )


def _guardar_modelo(ruta, estimador):

    directorio = os.path.dirname(ruta)
    if os.path.exists(directorio):
        for archivo in glob.glob(f"{directorio}/*"):
            os.remove(archivo)
        os.rmdir(directorio)
    os.makedirs(directorio, exist_ok=True)
    with gzip.open(ruta, "wb") as f:
        pickle.dump(estimador, f)


def calcular_metricas(tipo_conjunto, y_real, y_prediccion):

    matriz = confusion_matrix(y_real, y_prediccion)
    return [
        {
            "type": "metrics",
            "dataset": tipo_conjunto,
            "precision": precision_score(y_real, y_prediccion, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_real, y_prediccion),
            "recall": recall_score(y_real, y_prediccion, zero_division=0),
            "f1_score": f1_score(y_real, y_prediccion, zero_division=0)
        },
        {
            "type": "cm_matrix",
            "dataset": tipo_conjunto,
            "true_0": {
                "predicted_0": int(matriz[0][0]),
                "predicted_1": int(matriz[0][1])
            },
            "true_1": {
                "predicted_0": int(matriz[1][0]),
                "predicted_1": int(matriz[1][1])
            }
        }
    ]


def homework():

    datos_entrenamiento, datos_prueba = (limpiar_datos(df) for df in cargar_datos())
    

    x_entrenamiento, y_entrenamiento = dividir_datos(datos_entrenamiento)
    x_prueba, y_prueba = dividir_datos(datos_prueba)


    pipeline = crear_pipeline(x_entrenamiento)
    estimador = crear_estimador(pipeline, x_entrenamiento)
    estimador.fit(x_entrenamiento, y_entrenamiento)


    _guardar_modelo("files/models/model.pkl.gz", estimador)

   
    metricas_entrenamiento = calcular_metricas("train", y_entrenamiento, estimador.predict(x_entrenamiento))
    metricas_prueba = calcular_metricas("test", y_prueba, estimador.predict(x_prueba))
    

    metricas = [
        metricas_entrenamiento[0],
        metricas_prueba[0],
        metricas_entrenamiento[1],
        metricas_prueba[1]
    ]

    os.makedirs("files/output/", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as archivo:
        archivo.writelines(json.dumps(m) + "\n" for m in metricas)


if __name__ == "__main__":
    homework()
'''
Descarga y preparación del dataset para el análisis.

Se ejecuta desde main.py:  python main.py analisa
main.py pasa la ruta del dataset (DATA_DIR) como argumento.

Funciones:
    download_dataset(dataset_path)
        Lee un CSV y lo devuelve como DataFrame, mostrando cuántas filas y columnas tiene.
    parse_feature_list(feature_str, valid_columns)
        Convierte el texto del usuario (p. ej. "[ingredients, steps]") en una lista
        de nombres de columna, quitando las que no existen en el DataFrame.
    bow_distribution_colum(df, column)
        Aplica un Bag of Words (CountVectorizer) a una columna de texto y muestra
        el tamaño del vocabulario, la forma de la matriz y las primeras palabras.
    main()
        Carga los datasets y lanza un menú interactivo para elegir qué análisis hacer.
'''

import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple
import numpy as np
import os
import csv
import ast
from Edge_cases import  analise_column_edge_cases
from range_cases import analise_column_range_cases
from Csv_to_graph import generate_graph, OUT_DIR_DICTIONARY, FEATURES

DATA_DIR = Path(sys.argv[1])

DICT_COLUMN_TYPE = {
    "id":             "int",
    "minutes":        "int",
    "contributor_id": "int",
    "tags":           "list_str",
    "nutrition":      "list_num",
    "n_steps":        "int",
    "steps":          "list_str",
    "description":    "string",
    "ingredients":    "list_str",
    "n_ingredients":  "int",
}

NUTRITION_POSITIONS = {
    0: "calories",
    1: "total_fat_pdv",
    2: "sugar_pdv",
    3: "sodium_pdv",
    4: "protein_pdv",
    5: "saturated_fat_pdv",
    6: "carbohydrates_pdv",
}

def download_dataset(dataset_path):
    '''
    Carga un dataset. Si ya existe su pickle en pickles/, lo carga;
    si no, lee el CSV y guarda el pickle para la próxima vez.

    Parámetros:
        dataset_path (Path): ruta del archivo CSV.

    Devuelve:
        pd.DataFrame: el contenido del dataset.
    '''
    pickle_dir = Path(__file__).resolve().parent.parent / "pickles"
    pickle_path = pickle_dir / (dataset_path.stem + ".pkl")

    if pickle_path.exists():
        data_df = pd.read_pickle(pickle_path)
    else:
        data_df = pd.read_csv(dataset_path)
        os.makedirs(pickle_dir, exist_ok=True)
        data_df.to_pickle(pickle_path)

    print(f"{dataset_path.name}: {data_df.shape[0]} filas, {data_df.shape[1]} columnas")

    return data_df


def parse_feature_list(feature_str, valid_columns):
    '''
    Convierte el texto escrito por el usuario en una lista de features.

    Acepta formatos como "[ingredients, steps]", "ingredients, steps",
    "['ingredients', 'steps']" o una sola columna "ingredients".
    Las columnas que no existen en el DataFrame se descartan con un aviso.

    Parámetros:
        feature_str (str): texto introducido por el usuario.
        valid_columns (iterable): nombres de columna disponibles en el DataFrame.

    Devuelve:
        list[str]: lista de columnas válidas a analizar (puede estar vacía).
    '''
    # Quitar espacios y corchetes de los extremos
    cleaned = feature_str.strip().strip("[]")

    # Separar por comas y limpiar espacios y comillas de cada elemento
    candidates = [item.strip().strip("'\"").strip() for item in cleaned.split(",")]
    candidates = [item for item in candidates if item]  # eliminar vacíos

    feature_list = []
    for feature in candidates:
        if feature in valid_columns:
            if feature not in feature_list:  # evitar duplicados
                feature_list.append(feature)
        else:
            print(f"Aviso: la columna '{feature}' no existe y se ignora.")

    return feature_list


def main():
    '''
    Punto de entrada del análisis.

    Carga RAW_recipes.csv y RAW_interactions.csv y muestra un menú interactivo:
        bow_dist -> pide una lista de features y calcula el BoW de cada una.
        S        -> sale del programa.
    '''
    df_raw_recipes = download_dataset(DATA_DIR / "RAW_recipes.csv")
    df_raw_interactions = download_dataset(DATA_DIR / "RAW_interactions.csv")
    option = "T"

    while option != "S":
        option = input("What do you want to do? (edge_cases // value_range // generate_grafo // S): ").strip()

        if option == "edge_cases":
            print("Columnas disponibles:", list(df_raw_recipes.columns))
            feature_str = input("Analising edge_cases for which features?([feature1, feature2, ..., featuren]): ")

            feature_list = parse_feature_list(feature_str, df_raw_recipes.columns)

            if not feature_list:
                print("No se ha indicado ninguna columna válida.")
                continue

            for feature in feature_list:
                analise_column_edge_cases(df_raw_recipes, feature)
        elif option == "value_range":
            print("Columnas disponibles:", list(df_raw_recipes.columns))
            feature_str = input("Analising value_range for which features?([feature1, feature2, ..., featuren]): ")

            feature_list = parse_feature_list(feature_str, df_raw_recipes.columns)

            if not feature_list:
                print("No se ha indicado ninguna columna válida.")
                continue

            for feature in feature_list:
                analise_column_range_cases(df_raw_recipes, feature)

        elif option == "generate_grafo":
            print("Pruebas disponibles: edgecases, valuerange")
            prueba = input("Enter the test name: ").strip()

            print("Features disponibles: all,", ", ".join(FEATURES))
            feature = input("Enter the feature name: ").strip()

            print("Percentiles: edgecases -> 95, 90 | valuerange -> 5, 10")
            default = "95" if prueba == "edgecases" else "5"
            percentile = input(f"Enter the percentile (default {default}): ").strip() or default

            generate_graph(prueba, feature, percentile)

        elif option == "S":
            print("Saliendo...")

        else:
            print(f"Opción '{option}' no reconocida.")


if __name__ == "__main__":
    main()
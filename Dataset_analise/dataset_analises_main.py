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
    ask_features(df, prueba)
        Muestra las columnas disponibles, pide la lista de features al usuario
        y la devuelve validada (acepta "all").
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
from Frequency import analise_column_frequency
from value_range import analise_column_range_cases
from Csv_to_graph import generate_graph, OUT_DIR_DICTIONARY, FEATURES
from Get_stats import get_stats_main, STATS_TYPES
from Interactions import analise_interactions, PARTS
from Stat_tests import analise_tests, PARTS as STAT_PARTS

PRUEBAS = ["frequency", "value_range", "generate_grafo", "get_stats", "interactions", "stat_tests", "Salir"]


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

PRUEBAS = [
    "frequency",
    "value_range",
    "generate_grafo",
    "get_stats",
    "Salir"
]



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


def ask_features(df, prueba):
    '''
    Pide al usuario qué columnas analizar y devuelve la lista validada.

    Escribir "all" selecciona todas las columnas del DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame sobre el que se hará el análisis.
        prueba (str): nombre de la prueba (solo para el mensaje).

    Devuelve:
        list[str]: columnas válidas (vacía si no se indicó ninguna válida).
    '''
    print("Columnas disponibles: all,", ", ".join(df.columns))
    feature_str = input(
        f"Analising {prueba} for which features?([feature1, feature2, ..., featuren] or all): "
    )

    if feature_str.strip().strip("[]").strip().lower() == "all":
        return list(df.columns)

    feature_list = parse_feature_list(feature_str, df.columns)
    if not feature_list:
        print("No se ha indicado ninguna columna válida.")
    return feature_list


def main():
    '''
    Punto de entrada del análisis.

    Carga RAW_recipes.csv y RAW_interactions.csv y muestra un menú interactivo:
        frequency      -> pide una lista de features y analiza sus valores más frecuentes.
        value_range    -> pide una lista de features y analiza su rango de valores.
        generate_grafo -> genera gráficos a partir de los CSV de resultados.
        get_stats      -> pide una lista de features y un tipo (frequency/value)
                          y calcula las estadísticas de cada una.
        S / Salir      -> sale del programa.
    '''
    df_raw_recipes = download_dataset(DATA_DIR / "RAW_recipes.csv")
    df_raw_interactions = download_dataset(DATA_DIR / "RAW_interactions.csv")
    option = "T"

    while option not in ("S", "Salir"):
        option = input(f"What do you want to do? {PRUEBAS}: ").strip()

        if option == "frequency":
            for feature in ask_features(df_raw_recipes, option):
                analise_column_frequency(df_raw_recipes, feature)

        elif option == "value_range":
            for feature in ask_features(df_raw_recipes, option):
                analise_column_range_cases(df_raw_recipes, feature)

        elif option == "generate_grafo":
            print("Pruebas disponibles: frequency, valuerange")
            prueba = input("Enter the test name: ").strip()

            print("Features disponibles: all,", ", ".join(FEATURES))
            feature = input("Enter the feature name: ").strip()

            print("Percentiles: frequency -> 95, 90 | valuerange -> 5, 10")
            default = "95" if prueba == "frequency" else "5"
            percentile = input(f"Enter the percentile (default {default}): ").strip() or default

            generate_graph(prueba, feature, percentile)

        elif option == "get_stats":
            tipus = input(f"Enter the type of analysis {STATS_TYPES}: ").strip().lower()
            if tipus not in STATS_TYPES:
                print(f"Tipo '{tipus}' no reconocido. Opciones: {STATS_TYPES}")
                continue

            feature_list = ask_features(df_raw_recipes, f"get_stats ({tipus})")
            if not feature_list:
                continue

            for feature in feature_list:
                get_stats_main(df_raw_recipes, feature, tipus)

        elif option == "interactions":
            part = input(f"Which part? {PARTS}: ").strip() or "all"
            analise_interactions(df_raw_interactions, df_raw_recipes, part)

        elif option == "stat_tests":
            part = input(f"Which part? {STAT_PARTS}: ").strip() or "all"
            analise_tests(df_raw_interactions, df_raw_recipes, part)

        elif option in ("S", "Salir"):
            print("Saliendo...")

        else:
            print(f"Opción '{option}' no reconocida.")


if __name__ == "__main__":
    main()
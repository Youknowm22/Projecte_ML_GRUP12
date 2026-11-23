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


DATA_DIR = Path(sys.argv[1])


def download_dataset(dataset_path):
    '''
    Lee un archivo CSV y lo carga en un DataFrame.

    Parámetros:
        dataset_path (Path): ruta del archivo CSV.

    Devuelve:
        pd.DataFrame: el contenido del CSV.
    '''
    data_df = pd.read_csv(dataset_path)

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


def bow_distribution_colum(df, column):
    '''
    Aplica un Bag of Words (CountVectorizer) a una columna de texto del DataFrame
    y muestra información básica del vocabulario obtenido.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        column (str): nombre de la columna de texto a analizar.

    Devuelve:
        tuple: (bow, X_docs)
            bow (CountVectorizer): el vectorizador ya entrenado.
            X_docs (scipy.sparse matrix): matriz documento-término.
    '''
    # Rellenar los valores vacíos (NaN) para que CountVectorizer no falle
    recipe_columm = df[column].fillna("").astype(str)

    bow = CountVectorizer()
    X_docs = bow.fit_transform(recipe_columm)

    print(f"\n--- BoW de '{column}' ---")
    print("Mida del vocabulari:", len(bow.vocabulary_))
    print("Tipus de matriu:", type(X_docs), X_docs.shape)
    print(bow.get_feature_names_out()[:30])

    return bow, X_docs

def get_edge_cases(bow_tuple):
    bow_feat, bow_matrix = bow_tuple

    feat_colum = bow_feat.get_feature_names_out()
    word_freq = np.sum(bow_matrix, axis=0)
    percentile_value = np.percentile(word_freq, 95, axis=0)
    word_fre_mask = ( word_freq >= percentile_value)

    edge_cases = feat_colum[word_fre_mask]

    print(edge_cases)

    print("hola")

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
        option = input("What do you want to do? (bow_dist / S): ").strip()

        if option == "bow_dist":
            print("Columnas disponibles:", list(df_raw_recipes.columns))
            feature_str = input("Analising bow for which features?([feature1, feature2, ..., featuren]): ")

            feature_list = parse_feature_list(feature_str, df_raw_recipes.columns)

            if not feature_list:
                print("No se ha indicado ninguna columna válida.")
                continue

            for feature in feature_list:
                bow_tuple = bow_distribution_colum(df_raw_recipes, feature)
                get_edge_cases(bow_tuple)

        elif option == "S":
            print("Sortint")

        else:
            print(f"Opción '{option}' no reconocida.")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple
import numpy as np
import os
import csv
import ast

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
    str_columm = df[column].fillna("").astype(str)

    bow = CountVectorizer()
    X_docs = bow.fit_transform(str_columm)

    return bow, X_docs

def int_dist(df, column):

    counts = df[column].value_counts().sort_index()
    labels_array = counts.index.to_numpy()
    freq_array = counts.to_numpy()

    return labels_array, freq_array

def list_int_dist(df, column):

    # En el CSV las listas vienen como texto: "[51.5, 0.0, 13.0, ...]"
    lists = df[column].apply(ast.literal_eval)

    # Una columna por posición de la lista
    expanded = pd.DataFrame(lists.tolist(), index=df.index)

    result = []
    for position in expanded.columns:
        counts = expanded[position].value_counts().sort_index()

        labels_array = counts.index.to_numpy()
        freq_array = counts.to_numpy()

        result.append((labels_array, freq_array))

    return result

def bow_to_freq(bow_tuple):
    '''Convierte (CountVectorizer, matriz) en (palabras, frecuencias).'''
    bow_feat, bow_matrix = bow_tuple
    labels_array = bow_feat.get_feature_names_out()
    freq_array = np.asarray(np.sum(bow_matrix, axis=0)).ravel()
    return labels_array, freq_array

def get_edge_cases(labels_array, freq_array, filename, percentile=95, output_dir="Dataset_analise/Resultats/edgecases"):

    filename = f"{filename}_edgecases_{percentile}.csv"

    percentile_value = np.percentile(freq_array, percentile)
    freq_mask = (freq_array >= percentile_value)

    edge_cases = labels_array[freq_mask]
    edge_freqs = freq_array[freq_mask]

    order = np.argsort(-edge_freqs)

    edge_cases = edge_cases[order]
    edge_freqs = edge_freqs[order]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["value", "freq"])
        writer.writerows(zip(edge_cases, edge_freqs))

    print(f"Results saved to {output_path}")

    return edge_cases, edge_freqs


def analise_column_edge_cases(df_raw_recipe, feature, percentile=95):

    type_feat = DICT_COLUMN_TYPE.get(feature)

    if type_feat == "int":
        labels_array, freq_array = int_dist(df_raw_recipe, feature)
        get_edge_cases(labels_array, freq_array, filename=feature, percentile=percentile)

    elif type_feat in ("list_str", "string"):
        bow_tuple = bow_distribution_colum(df_raw_recipe, feature)
        labels_array, freq_array = bow_to_freq(bow_tuple)
        get_edge_cases(labels_array, freq_array, filename=feature, percentile=percentile)

    elif type_feat == "list_num":
        result = list_int_dist(df_raw_recipe, feature)
        for position, (labels_array, freq_array) in enumerate(result):
            nutrient = NUTRITION_POSITIONS.get(position, str(position))
            get_edge_cases(labels_array, freq_array, filename=f"{feature}_{nutrient}", percentile=percentile)

    else:
        print(f"Aviso: la columna '{feature}' no tiene tipo definido y se ignora.")

    return None

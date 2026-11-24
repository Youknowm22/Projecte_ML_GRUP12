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


def get_labels_and_values_int(df, feature):

    labels = df["name"].fillna("").to_numpy()
    values = df[feature].to_numpy()

    return labels, values

def get_labels_and_values_list_string(df, feature):

    labels = df["name"].fillna("").to_numpy()

    values = df[feature].apply(lambda cell: len(ast.literal_eval(cell)) if pd.notna(cell) else 0).to_numpy()

    return labels, values

def get_labels_and_values_string(df, feature):

    labels = df["name"].fillna("").to_numpy()

    values = df[feature].fillna("").astype(str).str.split().str.len().to_numpy()

    return labels, values

def get_labels_and_values_list_num(df, feature, position):

    labels = df["name"].fillna("").to_numpy()

    lists = df[feature].apply(lambda cell: ast.literal_eval(cell) if pd.notna(cell) else [])

    values = lists.apply(lambda l: l[position] if len(l) > position else np.nan).to_numpy(dtype=float)

    return labels, values

def get_range_values(labels, values, filename, percentile_range=5, output_dir="Dataset_analise/Resultats/value_range"):

    filename = f"{filename}_valuerange_{percentile_range}.csv"

    percentile_up = np.percentile(values, 100 - percentile_range)
    percentile_down = np.percentile(values, percentile_range)

    freq_mask_up = (values >= percentile_up)
    freq_mask_down = (values <= percentile_down)

    labels_up = labels[freq_mask_up]
    labels_down = labels[freq_mask_down]
    values_up = values[freq_mask_up]
    values_down = values[freq_mask_down]

    order_up = np.argsort(-values_up)
    order_down = np.argsort(values_down)

    values_up = values_up[order_up]
    labels_up = labels_up[order_up]
    values_down = values_down[order_down]
    labels_down = labels_down[order_down]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["RANGE"])
        writer.writerow(["min", "max", "range",
                         f"p{percentile_range}", f"p{100 - percentile_range}"])
        writer.writerow([np.min(values), np.max(values), np.max(values) - np.min(values),
                         percentile_down, percentile_up])
        writer.writerow([])

        writer.writerow([f"TOP (>= p{100 - percentile_range})"])
        writer.writerow(["label", "value"])
        writer.writerows(zip(labels_up, values_up))
        writer.writerow([])

        writer.writerow([f"BOTTOM (<= p{percentile_range})"])
        writer.writerow(["label", "value"])
        writer.writerows(zip(labels_down, values_down))

    print(f"Results saved to {output_path}")

    return (labels_up, values_up), (labels_down, values_down)

def analise_column_range_cases(df_raw_recipe, feature, percentile=95):

    type_feat = DICT_COLUMN_TYPE.get(feature)

    if type_feat == "int":
        labels_array, values_array = get_labels_and_values_int(df_raw_recipe, feature)
        get_range_values(labels_array, values_array, filename=feature, percentile_range=percentile)

    elif type_feat in ("list_str"):
        labels_array, values_array = get_labels_and_values_list_string(df_raw_recipe, feature)
        get_range_values(labels_array, values_array, filename=feature, percentile_range=percentile)

    elif type_feat == "string":
        labels_array, values_array = get_labels_and_values_string(df_raw_recipe, feature)
        get_range_values(labels_array, values_array, filename=feature, percentile_range=percentile)

    elif type_feat == "list_num":
        for position, nutrient in NUTRITION_POSITIONS.items():
            labels, values = get_labels_and_values_list_num(df_raw_recipe, "nutrition", position)
            get_range_values(labels, values, filename=f"nutrition_{nutrient}")

    else:
        print(f"Aviso: la columna '{feature}' no tiene tipo definido y se ignora.")

    return None
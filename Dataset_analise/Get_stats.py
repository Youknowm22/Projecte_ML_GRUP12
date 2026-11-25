import os
import csv
import numpy as np

from Frequency import (
    DICT_COLUMN_TYPE,
    NUTRITION_POSITIONS,
    int_dist,
    list_int_dist,
    bow_distribution_colum,
    bow_to_freq,
)
from value_range import (
    get_labels_and_values_int,
    get_labels_and_values_list_num,
    get_labels_and_values_list_string,
    get_labels_and_values_string,
)

OUT_DIR = "Dataset_analise/Resultats/stats"
STATS_TYPES = ["frequency", "value_range"]


def get_stats(values_array, feature, prueba):
    '''Calcula estadísticas básicas de un array de valores y las guarda en CSV.'''
    values_array = np.asarray(values_array, dtype=float)
    values_array = values_array[~np.isnan(values_array)]

    if values_array.size == 0:
        print(f"Aviso: '{feature}' no tiene valores numéricos; no se calculan estadísticas.")
        return None

    stats = {
        "count": values_array.size,
        "mean": np.mean(values_array),
        "std": np.std(values_array),
        "min": np.min(values_array),
        "25%": np.percentile(values_array, 25),
        "50%": np.percentile(values_array, 50),
        "75%": np.percentile(values_array, 75),
        "max": np.max(values_array),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    output_path = os.path.join(OUT_DIR, f"{feature}_getstats_{prueba}.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["statistic", "value"])
        for stat, value in stats.items():
            writer.writerow([stat, value])

    print(f"Stats de '{feature}' ({prueba}) guardadas en {output_path}")
    return stats


def get_stats_main(df_raw_recipe, feature, prueba):
    '''Calcula las estadísticas de una columna según su tipo y el tipo de prueba.'''
    type_feat = DICT_COLUMN_TYPE.get(feature)

    if type_feat is None:
        print(f"Aviso: la columna '{feature}' no tiene tipo definido en DICT_COLUMN_TYPE; se ignora.")
        return

    if prueba == "frequency":
        if type_feat == "int":
            _, freq_array = int_dist(df_raw_recipe, feature)
            get_stats(freq_array, feature, prueba)

        elif type_feat in ("list_str", "string"):
            bow_tuple = bow_distribution_colum(df_raw_recipe, feature)
            _, freq_array = bow_to_freq(bow_tuple)
            get_stats(freq_array, feature, prueba)

        elif type_feat == "list_num":
            for position, (_, freq_array) in enumerate(list_int_dist(df_raw_recipe, feature)):
                nutrient = NUTRITION_POSITIONS.get(position, str(position))
                get_stats(freq_array, f"{feature}_{nutrient}", prueba)

    elif prueba == "value_range":
        if type_feat == "int":
            _, values_array = get_labels_and_values_int(df_raw_recipe, feature)
            get_stats(values_array, feature, prueba)

        elif type_feat == "string":
            _, values_array = get_labels_and_values_string(df_raw_recipe, feature)
            get_stats(values_array, feature, prueba)

        elif type_feat == "list_str":
            _, values_array = get_labels_and_values_list_string(df_raw_recipe, feature)
            get_stats(values_array, feature, prueba)

        elif type_feat == "list_num":
            for position, nutrient in NUTRITION_POSITIONS.items():
                _, values_array = get_labels_and_values_list_num(df_raw_recipe, feature, position)
                get_stats(values_array, f"{feature}_{nutrient}", prueba)
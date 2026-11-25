'''
Genera gráficos a partir de los CSV de resultados (frequency y valuerange).

Funciones:
    read_frequency_csv(file_path)
        Lee un CSV de frequency (value, freq).
    read_valuerange_csv(file_path)
        Lee un CSV de valuerange (secciones RANGE / TOP / BOTTOM).
    frequency_graph_generate(df, feature, prova, percentile)
        Gráfico de barras con los valores más frecuentes.
    valuerange_graph_generate(range_dict, df_top, df_bottom, feature, prova, percentile)
        Dos gráficos de barras horizontales con las recetas de valores más altos y más bajos.
    generate_graphs_from_csv(prueba, percentile, ingredient)
        Lee el CSV correspondiente y genera su gráfico.
    generate_graph(prueba, feature, percentile)
        Genera el gráfico de una feature o de todas ("all").
'''

import os
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Carpeta donde está este archivo (Dataset_analise/)
BASE_DIR = Path(__file__).resolve().parent

OUT_DIR_DICTIONARY = {
    "frequency":  BASE_DIR / "Resultats" / "frequency",
    "valuerange": BASE_DIR / "Resultats" / "valuerange",
}

GRAPH_DIR = BASE_DIR / "grafos"

FEATURES = [
    "minutes", "n_steps", "n_ingredients",
    "tags", "steps", "ingredients", "description",
    "nutrition_calories", "nutrition_total_fat_pdv", "nutrition_sugar_pdv",
    "nutrition_sodium_pdv", "nutrition_protein_pdv",
    "nutrition_saturated_fat_pdv", "nutrition_carbohydrates_pdv",
]


def read_frequency_csv(file_path):
    '''
    Lee un CSV de frequency. Devuelve un DataFrame vacío si el fichero
    no existe o está vacío.
    '''
    try:
        df = pd.read_csv(file_path, keep_default_na=False)

        if df.empty:
            print(f"The CSV file {file_path} is empty.")
            return pd.DataFrame()

        return df

    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return pd.DataFrame()

    except pd.errors.EmptyDataError:
        print(f"The CSV file {file_path} is empty.")
        return pd.DataFrame()


def read_valuerange_csv(file_path):

    sections, current = {}, None
    with open(file_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) == 1:
                current = row[0].split()[0]
                sections[current] = []
            else:
                sections[current].append(row)

    range_dict = dict(zip(sections["RANGE"][0], map(float, sections["RANGE"][1])))
    df_top = pd.DataFrame(sections["TOP"][1:], columns=sections["TOP"][0])
    df_bottom = pd.DataFrame(sections["BOTTOM"][1:], columns=sections["BOTTOM"][0])
    df_top["value"] = df_top["value"].astype(float)
    df_bottom["value"] = df_bottom["value"].astype(float)
    return range_dict, df_top, df_bottom


def valuerange_graph_generate(range_dict, df_top, df_bottom, feature, prova, percentile, top_n=20):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, df, title in [(ax1, df_top.head(top_n), "TOP"), (ax2, df_bottom.head(top_n), "BOTTOM")]:
        positions = range(len(df))
        ax.barh(positions, df["value"], color="#4C72B0")
        ax.set_yticks(positions)
        ax.set_yticklabels(df["label"], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{title} {top_n}")
        ax.set_xlabel(feature)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{feature} — {prova} — p{percentile}  "
                 f"(min={range_dict['min']:g}, max={range_dict['max']:g})")
    fig.tight_layout()

    out_dir = GRAPH_DIR / prova
    os.makedirs(out_dir, exist_ok=True)
    output_path = out_dir / f"{feature}_{prova}_{percentile}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Graph saved to {output_path}")


def frequency_graph_generate(df, feature=None, prova=None, percentile=None, top_n=30):

    # Limitar el número de barras para que la imagen no sea enorme
    df = df.head(top_n)

    value_col, freq_col = df.columns[0], df.columns[1]

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.35), 5))
    bars = ax.bar(df[value_col].astype(str), df[freq_col], color="#4C72B0")

    ax.bar_label(bars, fontsize=8)

    ax.set_title(f"{feature} — {prova} — percentile {percentile}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    out_dir = GRAPH_DIR / prova
    os.makedirs(out_dir, exist_ok=True)
    output_path = out_dir / f"{feature}_{prova}_{percentile}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Graph saved to {output_path}")


def generate_graphs_from_csv(prueba, percentile, ingredient):

    if prueba not in OUT_DIR_DICTIONARY:
        print(f"Prueba '{prueba}' no reconocida. Opciones: {list(OUT_DIR_DICTIONARY)}")
        return

    filepath = OUT_DIR_DICTIONARY[prueba] / f"{ingredient}_{prueba}_{percentile}.csv"

    if prueba == "frequency":
        df = read_frequency_csv(filepath)
        if not df.empty:
            frequency_graph_generate(df, feature=ingredient, prova=prueba, percentile=percentile)

    elif prueba == "valuerange":
        if not filepath.exists():
            print(f"The file at {filepath} was not found.")
            return
        range_dict, df_top, df_bottom = read_valuerange_csv(filepath)
        valuerange_graph_generate(range_dict, df_top, df_bottom,
                                  feature=ingredient, prova=prueba, percentile=percentile)


def generate_graph(prueba, feature, percentile):
    if feature == "all":
        for feature in FEATURES:
            generate_graphs_from_csv(prueba, percentile, feature)
    else:
        generate_graphs_from_csv(prueba, percentile, feature)
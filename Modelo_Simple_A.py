'''
Modelo simple de recomendación de recetas con Bag of Words.

Se ejecuta con:  python Modelo_Simple.py <ruta_al_dataset>
La ruta (DATA_DIR) debe contener RAW_recipes.csv.

Idea: cada receta se convierte en un vector (una columna por ingrediente) y dos
recetas se parecen cuando comparten ingredientes. La similitud se mide con el
coseno entre sus vectores (0 = nada en común, 1 = mismos ingredientes).
No hay etiquetas ni train/test: es un modelo de similitud, no de clasificación.

Funciones:
    download_dataset(dataset_path)
        Lee un CSV (con caché en pickle) y lo devuelve como DataFrame.
    identity(lista)
        Analizador del vectorizador: cada elemento de la lista ya es un token.
    prepare_recipes(df)
        Convierte la columna "ingredients" de texto a lista y devuelve
        los nombres y las listas de ingredientes de cada receta.
    build_bow(docs, use_tfidf)
        Construye el vocabulario de ingredientes y la matriz receta x ingrediente.
    show_results(sim, nombres, docs, exclude, query_ingredients)
        Muestra las recetas más similares a partir de un vector de similitudes.
    option_ingredientes(nombres, docs, vectorizer, X)
        Recetas que mejor encajan con una lista de ingredientes escrita por el usuario.
    main()
        Carga el dataset, construye el modelo y lanza un menú interactivo.
'''

import sys
import os
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(sys.argv[1])

TOP_N = 5          # nº de recetas a mostrar
USE_TFIDF = True   # True: TF-IDF (da menos peso a ingredientes comunes como la sal)
                   # False: BoW binario (presente / no presente)

PRUEBAS = [
    "similares",
    "ingredientes",
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


def identity(lista):
    '''
    Analizador del vectorizador. Como cada receta ya es una lista de
    ingredientes, no hay que tokenizar nada: cada elemento es un "token".
    (Se define como función con nombre en vez de lambda para poder reutilizarla.)
    '''
    return lista


def prepare_recipes(df):
    '''
    Prepara las recetas para el modelo.

    En el CSV las listas se guardan como texto ("['egg', 'flour']"), así que
    se convierten de vuelta a listas con ast.literal_eval. Se descartan las
    recetas sin nombre o sin ingredientes y se normalizan los ingredientes
    (minúsculas y sin espacios sobrantes).

    Parámetros:
        df (pd.DataFrame): DataFrame de RAW_recipes.csv.

    Devuelve:
        tuple[list[str], list[list[str]]]: nombres de las recetas y sus
        listas de ingredientes, en el mismo orden.
    '''
    df = df[["name", "ingredients"]].dropna().copy()

    df["ingredients"] = df["ingredients"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["ingredients"] = df["ingredients"].apply(
        lambda lst: [ing.strip().lower() for ing in lst]
    )

    df = df[df["ingredients"].map(len) > 0].reset_index(drop=True)

    return df["name"].tolist(), df["ingredients"].tolist()


def build_bow(docs, use_tfidf=USE_TFIDF):
    '''
    Construye el Bag of Words de ingredientes.

    El vocabulario son todos los ingredientes distintos y la matriz X tiene
    una fila por receta y una columna por ingrediente (matriz dispersa).

    Parámetros:
        docs (list[list[str]]): lista de ingredientes de cada receta.
        use_tfidf (bool): True para TF-IDF, False para BoW binario.

    Devuelve:
        tuple: (vectorizador ya ajustado, matriz X de forma (n_recetas, n_ingredientes)).
    '''
    if use_tfidf:
        vectorizer = TfidfVectorizer(analyzer=lambda lista: lista)
    else:
        vectorizer = CountVectorizer(analyzer=identity, binary=True)

    X = vectorizer.fit_transform(docs)
    return vectorizer, X



def show_results(sim, nombres, docs, exclude=None, query_ingredients=None, n=TOP_N):
    '''
    Muestra las n recetas con mayor similitud.

    Parámetros:
        sim (np.ndarray): similitud de la consulta con cada receta, forma (n_recetas,).
        nombres (list[str]): nombres de las recetas.
        docs (list[list[str]]): ingredientes de cada receta.
        exclude (int | None): índice a saltar (la propia receta consultada).
        query_ingredients (set[str] | None): ingredientes de la consulta, para
            mostrar cuántos comparte cada resultado.
        n (int): cuántas recetas mostrar.
    '''
    orden = np.argsort(-sim)
    mostradas = 0

    for j in orden:
        if j == exclude:
            continue

        linea = f"{sim[j]:.3f}  {nombres[j]}"
        if query_ingredients is not None:
            comunes = query_ingredients & set(docs[j])
            linea += f"   ({len(comunes)} en común: {', '.join(sorted(comunes))})"
        print(linea)

        mostradas += 1
        if mostradas == n:
            break


def option_ingredientes(nombres, docs, vectorizer, X):
    '''
    Pide una lista de ingredientes (separados por comas) y muestra las
    recetas que mejor encajan. Los ingredientes que no están en el
    vocabulario se ignoran con un aviso.
    '''
    texto = input("Ingredientes separados por comas (en inglés, p. ej. egg, flour, sugar): ")
    pedidos = [x.strip().lower() for x in texto.split(",") if x.strip()]

    conocidos = [ing for ing in pedidos if ing in vectorizer.vocabulary_]
    for ing in pedidos:
        if ing not in vectorizer.vocabulary_:
            print(f"Aviso: el ingrediente '{ing}' no está en el vocabulario y se ignora.")

    if not conocidos:
        print("No se ha indicado ningún ingrediente conocido.")
        return

    consulta = vectorizer.transform([conocidos])
    sim = cosine_similarity(consulta, X)[0]

    print()
    show_results(sim, nombres, docs, query_ingredients=set(conocidos))


def main():
    '''
    Punto de entrada del modelo.

    Carga RAW_recipes.csv, construye el Bag of Words de ingredientes y muestra
    un menú interactivo:
        similares    -> pide el nombre de una receta y muestra las más parecidas.
        ingredientes -> pide una lista de ingredientes y muestra las recetas que mejor encajan.
        S / Salir    -> sale del programa.
    '''
    df_raw_recipes = download_dataset(DATA_DIR / "R")

    nombres, docs = prepare_recipes(df_raw_recipes)
    vectorizer, X = build_bow(docs)
    print(f"Recetas: {X.shape[0]}, ingredientes distintos (vocabulario): {X.shape[1]}")

    option = "T"

    while option not in ("S", "Salir"):
        option = input(f"\nWhat do you want to do? {PRUEBAS}: ").strip()
        if option == "ingredientes":
            option_ingredientes(nombres, docs, vectorizer, X)

        elif option in ("S", "Salir"):
            print("Saliendo...")

        else:
            print(f"Opción '{option}' no reconocida.")


if __name__ == "__main__":
    main()

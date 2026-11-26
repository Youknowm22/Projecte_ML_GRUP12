'''
Evaluación de un modelo de recomendación de recetas con Bag of Words.

Se ejecuta con:  python Modelo_Evaluacion.py <ruta_al_dataset>
La ruta (DATA_DIR) debe contener RAW_recipes.csv e interactions_train.csv.

Idea de la prueba (autoevaluación con ingredientes ocultos):
    1. Se elige una receta al azar.
    2. Se conserva solo un x % de sus ingredientes (elegidos al azar): es la consulta.
    3. Se buscan las k recetas más cercanas a esa consulta (similitud del coseno).
    4. Se mira si la receta original está entre esas k.
Repitiendo esto muchas veces se obtiene Hit@k: el % de veces que la original aparece.

Las recetas que se usan son las que aparecen en interactions_train.csv.
Ese archivo solo trae el id de cada receta (recipe_id), así que el nombre y
los ingredientes se obtienen de RAW_recipes.csv a través de su columna "id".

Qué se guarda (en carpetas junto a pickles/):
    modelo/count_vectorizer.pkl
        El CountVectorizer ya ajustado (el vocabulario de ingredientes). Se
        sobrescribe cada vez que se ejecuta el programa.
    resultados/test_<fecha>_p<porcentaje>_k<k>/
        Una carpeta nueva cada vez que se ejecuta la opción "test", con:
        bow_matrix.pkl     -> la matriz receta x ingrediente (el BoW).
        recetas_prueba.pkl -> DataFrame con las recetas escogidas y los
                              ingredientes con los que se consultó cada una.

Funciones:
    download_dataset(dataset_path)
        Lee un CSV (con caché en pickle) y lo devuelve como DataFrame.
    identity(lista)
        Analizador del vectorizador: cada elemento de la lista ya es un token.
    prepare_recipes(df_recipes, df_interactions)
        Se queda con las recetas del train y devuelve sus ids, nombres e ingredientes.
    build_bow(docs)
        Construye el vocabulario de ingredientes y la matriz receta x ingrediente (binaria).
    save_vectorizer(vectorizer)
        Guarda el CountVectorizer en modelo/count_vectorizer.pkl.
    save_run(X, df_recetas, porcentaje, k)
        Guarda la matriz BoW y las recetas de prueba en una carpeta nueva de resultados/.
    sample_ingredients(ingredients, porcentaje, rng)
        Elige al azar un porcentaje de los ingredientes de una receta.
    top_k(sim, k)
        Devuelve los índices de las k recetas con mayor similitud.
    query_recipe(idx, docs, vectorizer, X, porcentaje, k, rng)
        Hace una consulta con una receta y dice si la original aparece entre las k.
    option_test(ids, nombres, docs, vectorizer, X)
        Repite la prueba muchas veces, muestra el Hit@k y guarda los resultados.
    option_ejemplo(nombres, docs, vectorizer, X)
        Hace una sola prueba y la muestra con detalle.
    main()
        Carga los datos, construye el modelo y lanza un menú interactivo.
'''

import sys
import os
import ast
import time
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("Dataset")

PORCENTAJE = 60    # % de ingredientes que se conservan en la consulta
K = 10             # nº de recetas más cercanas que se devuelven
N_PRUEBAS = 1000   # nº de recetas aleatorias en la prueba completa
SEED = 0           # semilla del azar (para que los resultados se puedan repetir)

BASE_DIR = Path(__file__).resolve().parent.parent   # misma carpeta donde está pickles/
MODEL_DIR = BASE_DIR / "modelo"                     # aquí va el CountVectorizer
RESULTS_DIR = BASE_DIR / "resultados"               # aquí va una subcarpeta por cada "test"

PRUEBAS = [
    "test",
    "ejemplo",
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
    '''
    return lista


def prepare_recipes(df_recipes, df_interactions):
    '''
    Prepara las recetas para el modelo.

    1. Se queda solo con las recetas cuyo id aparece en las interacciones del train.
    2. Convierte la columna "ingredients" de texto ("['egg', 'flour']") a lista
       con ast.literal_eval y normaliza cada ingrediente (minúsculas, sin espacios).
    3. Descarta las recetas sin nombre o sin ingredientes.

    Parámetros:
        df_recipes (pd.DataFrame): DataFrame de RAW_recipes.csv.
        df_interactions (pd.DataFrame): DataFrame de interactions_train.csv.

    Devuelve:
        tuple[list[int], list[str], list[list[str]]]: ids, nombres y listas de
        ingredientes de las recetas, los tres en el mismo orden.
    '''
    ids_train = df_interactions["recipe_id"].unique()

    df = df_recipes[["id", "name", "ingredients"]].dropna()
    df = df[df["id"].isin(ids_train)].copy()

    df["ingredients"] = df["ingredients"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["ingredients"] = df["ingredients"].apply(
        lambda lst: [ing.strip().lower() for ing in lst]
    )

    df = df[df["ingredients"].map(len) > 0].reset_index(drop=True)

    return df["id"].tolist(), df["name"].tolist(), df["ingredients"].tolist()


def build_bow(docs):
    '''
    Construye el Bag of Words de ingredientes (binario: el ingrediente está o no está).

    El vocabulario son todos los ingredientes distintos y la matriz X tiene
    una fila por receta y una columna por ingrediente (matriz dispersa).

    Parámetros:
        docs (list[list[str]]): lista de ingredientes de cada receta.

    Devuelve:
        tuple: (vectorizador ya ajustado, matriz X de forma (n_recetas, n_ingredientes)).
    '''
    vectorizer = CountVectorizer(analyzer=identity, binary=True)
    X = vectorizer.fit_transform(docs)
    return vectorizer, X


def save_vectorizer(vectorizer):
    '''
    Guarda el CountVectorizer ya ajustado en modelo/count_vectorizer.pkl.
    La carpeta se crea si no existe y el archivo se sobrescribe cada vez.

    Para cargarlo más tarde (en un script donde esté definida la función identity):
        with open("modelo/count_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

    Parámetros:
        vectorizer: CountVectorizer ya ajustado.

    Devuelve:
        Path: ruta del archivo guardado.
    '''
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = MODEL_DIR / "count_vectorizer.pkl"

    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"CountVectorizer guardado en {path}")
    return path


def save_run(X, df_recetas, porcentaje, k):
    '''
    Guarda los resultados de una ejecución de "test" en una carpeta nueva,
    resultados/test_<fecha>_p<porcentaje>_k<k>/, con dos pickles:
        bow_matrix.pkl     -> la matriz receta x ingrediente (el BoW).
        recetas_prueba.pkl -> DataFrame con una fila por receta escogida.

    Parámetros:
        X: matriz receta x ingrediente.
        df_recetas (pd.DataFrame): recetas de prueba y sus ingredientes de consulta.
        porcentaje (float): % de ingredientes conservados (solo para el nombre de la carpeta).
        k (int): nº de recetas devueltas (solo para el nombre de la carpeta).

    Devuelve:
        Path: carpeta donde se ha guardado todo.
    '''
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = RESULTS_DIR / f"test_{marca}_p{porcentaje:g}_k{k}"
    os.makedirs(carpeta, exist_ok=True)

    with open(carpeta / "bow_matrix.pkl", "wb") as f:
        pickle.dump(X, f)
    df_recetas.to_pickle(carpeta / "recetas_prueba.pkl")

    print(f"Resultados guardados en {carpeta}")
    return carpeta


def sample_ingredients(ingredients, porcentaje, rng):
    '''
    Elige al azar un porcentaje de los ingredientes de una receta (mínimo 1).

    Parámetros:
        ingredients (list[str]): ingredientes de la receta.
        porcentaje (float): % a conservar, entre 1 y 100.
        rng (np.random.Generator): generador de números aleatorios.

    Devuelve:
        list[str]: los ingredientes elegidos.
    '''
    n = max(1, round(len(ingredients) * porcentaje / 100))
    posiciones = rng.choice(len(ingredients), size=n, replace=False)
    return [ingredients[j] for j in posiciones]


def top_k(sim, k):
    '''
    Devuelve los índices de las k recetas con mayor similitud, de mayor a menor.

    Parámetros:
        sim (np.ndarray): similitud de la consulta con cada receta.
        k (int): cuántas recetas devolver.

    Devuelve:
        np.ndarray: índices de las k recetas más cercanas.
    '''
    candidatas = np.argpartition(-sim, k)[:k]          # las k mejores (sin ordenar)
    return candidatas[np.argsort(-sim[candidatas])]    # ordenadas de mayor a menor


def query_recipe(idx, docs, vectorizer, X, porcentaje, k, rng):
    '''
    Hace una consulta con la receta idx: conserva un porcentaje de sus
    ingredientes, busca las k recetas más cercanas y mira si la original está.

    Parámetros:
        idx (int): índice de la receta original.
        docs (list[list[str]]): ingredientes de cada receta.
        vectorizer: vectorizador ya ajustado.
        X: matriz receta x ingrediente.
        porcentaje (float): % de ingredientes que se conservan.
        k (int): nº de recetas a devolver.
        rng (np.random.Generator): generador de números aleatorios.

    Devuelve:
        tuple: (ingredientes de la consulta, similitudes con todas las recetas,
        índices de las k más cercanas, True si la original está entre ellas).
    '''
    consulta = sample_ingredients(docs[idx], porcentaje, rng)
    sim = cosine_similarity(vectorizer.transform([consulta]), X)[0]
    vecinas = top_k(sim, k)
    return consulta, sim, vecinas, idx in vecinas


def ask_params(con_pruebas):
    '''
    Pide por pantalla el porcentaje, k y (opcionalmente) el nº de pruebas.
    Si se pulsa Enter se usa el valor por defecto.

    Devuelve:
        tuple: (porcentaje, k, n_pruebas); n_pruebas es None si con_pruebas es False.
    '''
    porcentaje = float(input(f"% de ingredientes a conservar (default {PORCENTAJE}): ").strip() or PORCENTAJE)
    k = int(input(f"k, nº de recetas cercanas (default {K}): ").strip() or K)
    n_pruebas = None
    if con_pruebas:
        n_pruebas = int(input(f"Nº de recetas de prueba (default {N_PRUEBAS}): ").strip() or N_PRUEBAS)
    return porcentaje, k, n_pruebas


def option_test(ids, nombres, docs, vectorizer, X):
    '''
    Repite la prueba con muchas recetas aleatorias y muestra Hit@k:
    el % de veces que la receta original aparece entre las k más cercanas.
    También muestra lo que saldría eligiendo al azar (k / nº de recetas).

    Al terminar guarda la matriz BoW y las recetas escogidas (con los
    ingredientes de cada consulta) en una carpeta nueva de resultados/.
    Columnas de recetas_prueba.pkl:
        idx                   -> fila de la receta en la matriz X.
        id                    -> id de la receta en RAW_recipes.csv.
        name                  -> nombre de la receta.
        ingredientes_consulta -> ingredientes que se conservaron en la consulta.
        acierto               -> True si la original quedó entre las k más cercanas.
    '''
    porcentaje, k, n_pruebas = ask_params(con_pruebas=True)

    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(docs), size=n_pruebas, replace=False)

    aciertos = 0
    filas = []
    inicio = time.time()
    for idx in indices:
        consulta, _, _, acierto = query_recipe(idx, docs, vectorizer, X, porcentaje, k, rng)
        aciertos += acierto
        filas.append({
            "idx": int(idx),
            "id": ids[idx],
            "name": nombres[idx],
            "ingredientes_consulta": consulta,
            "acierto": bool(acierto),
        })

    print(f"\nRecetas de prueba: {n_pruebas}  |  conserva {porcentaje:g}% de ingredientes  |  k = {k}")
    print(f"Hit@{k}: {aciertos / n_pruebas:.3f}  ({aciertos} de {n_pruebas})")
    print(f"Al azar sería: {k / len(docs):.5f}")
    print(f"Tiempo: {time.time() - inicio:.1f} s")

    save_run(X, pd.DataFrame(filas), porcentaje, k)


def option_ejemplo(nombres, docs, vectorizer, X):
    '''
    Hace una sola prueba con una receta aleatoria y la muestra con detalle:
    ingredientes originales, ingredientes de la consulta y las k más cercanas.
    '''
    porcentaje, k, _ = ask_params(con_pruebas=False)

    rng = np.random.default_rng()
    idx = int(rng.integers(len(docs)))
    consulta, sim, vecinas, acierto = query_recipe(idx, docs, vectorizer, X, porcentaje, k, rng)

    print(f"\nReceta original: {nombres[idx]}")
    print(f"Ingredientes ({len(docs[idx])}): {', '.join(docs[idx])}")
    print(f"Consulta ({len(consulta)}): {', '.join(consulta)}\n")

    for j in vecinas:
        marca = "   <-- ORIGINAL" if j == idx else ""
        print(f"{sim[j]:.3f}  {nombres[j]}{marca}")

    print("\nLa receta original SÍ está entre las k." if acierto else "\nLa receta original NO está entre las k.")


def main():
    '''
    Punto de entrada.

    Carga RAW_recipes.csv e interactions_train.csv, se queda con las recetas
    del train, construye el Bag of Words de ingredientes, guarda el CountVectorizer
    y muestra un menú:
        test    -> repite la prueba con muchas recetas, muestra Hit@k y guarda los resultados.
        ejemplo -> hace una sola prueba y la muestra con detalle.
        S / Salir -> sale del programa.
    '''
    df_recipes = download_dataset(DATA_DIR / "RAW_recipes.csv")
    df_interactions = download_dataset(DATA_DIR / "interactions_train.csv")

    ids, nombres, docs = prepare_recipes(df_recipes, df_interactions)
    vectorizer, X = build_bow(docs)
    print(f"Recetas: {X.shape[0]}, ingredientes distintos (vocabulario): {X.shape[1]}")
    save_vectorizer(vectorizer)

    option = "T"

    while option not in ("S", "Salir"):
        option = input(f"\nWhat do you want to do? {PRUEBAS}: ").strip()

        if option == "test":
            option_test(ids, nombres, docs, vectorizer, X)

        elif option == "ejemplo":
            option_ejemplo(nombres, docs, vectorizer, X)

        elif option in ("S", "Salir"):
            print("Saliendo...")

        else:
            print(f"Opción '{option}' no reconocida.")


if __name__ == "__main__":
    main()
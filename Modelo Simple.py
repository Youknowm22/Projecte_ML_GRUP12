from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas
import pandas as pd

#Esta es una herramienta muy util que no servirá para más tarde (Unificar los ingredientes que tienen distintos nombres --> Sal, sal gorda, sal rosa...)

df = pd.read_pickle("Dataset/ingr_map.pkl")
print(type(df))
print(df.info())
print(df.iloc[0])
print(df.iloc[2])
print(df.iloc[1])
print(df.iloc[3])


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



"""
nombres = list(recetas.keys())
docs = list(recetas.values())

bow = CountVectorizer(analyzer=lambda lista: lista, binary=True)
X = bow.fit_transform(docs)              # matriz receta × ingrediente

print(bow.get_feature_names_out())       # el vocabulario
print(X.toarray())                       # los vectores

S = cosine_similarity(X) 
"""
'''
Dataset download and preparation for analises

Se ejecuta desde main.py:  python main.py analisa
main.py pasa la ruta del dataset (DATA_DIR) como argumento.
'''

import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


DATA_DIR = Path(sys.argv[1])

def download_dataset(dataset_path):

    data_df = pd.read_csv(dataset_path)

    print(f"{dataset_path.name}: {data_df.shape[0]} filas, {data_df.shape[1]} columnas")

    return data_df

def bow_distribution_colum(df, column):
    recipe_columm = df[column]

    bow = CountVectorizer()
    X_docs = bow.fit_transform(recipe_columm)

    print("Mida del vocabulari:", len(bow.vocabulary_))
    print("Tipus de matriu:", type(X_docs), X_docs.shape)
    print(bow.get_feature_names_out()[:30])


def main():

    df_raw_recipes = download_dataset(DATA_DIR / "RAW_recipes.csv")
    df_raw_interactions = download_dataset(DATA_DIR / "RAW_interactions.csv")
    option = "T"

    while option != "S":
        option = input("What do you want to do?: ")
        if option == "bow_dist":
            feature_str= input("Analising bow for which features?([feature1, feature2, ..., featuren]): ")

            feature_list = ...

            feature_list = ["ingredients"]

            for feature in feature_list:
                bow_feat = bow_distribution_colum(df_raw_recipes, feature)
        elif option == "S":
            print("Sortint")


if __name__ == "__main__":
    main()
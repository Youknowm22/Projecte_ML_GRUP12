'''
Dataset download and preparation for analises
'''

import pandas as pd

def download_dataset(dataset_path):

    data_df = pd.read_csv(dataset_path)

    data_df.head()

    return data_df

def main():
    raw_recipes_path = ""

    df_raw_recipes = download_dataset(raw_recipes_path)
    pass

if __name__ == "__main__":
    main()
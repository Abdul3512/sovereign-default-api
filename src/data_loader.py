import pandas as pd

def load_and_combine_data():
    nigeria = pd.read_csv('data/nigeria.csv')
    lebanon = pd.read_csv('data/lebanon.csv')
    ecuador = pd.read_csv('data/ecuador.csv')
    df = pd.concat([nigeria, lebanon, ecuador], ignore_index=True)
    return df

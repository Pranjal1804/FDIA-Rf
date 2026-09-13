import os
import pandas as pd

def load_mendeley_data(path="data/raw/mendeley.csv"):
    """
    Load the Mendeley FDIA dataset.
    If the file is not present, instruct the user on how to download it.
    URL: https://data.mendeley.com/datasets/4gb62c72sx/1
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

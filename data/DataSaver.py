import pandas as pd

class DataSaver:
    @staticmethod
    def save_to_csv(df: pd.DataFrame, filename: str):
        df.to_csv(filename, index=False) 
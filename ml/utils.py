# -*- coding: utf-8 -*-
import pandas as pd

def format_results(results):
    return pd.DataFrame(results).T

def save_results(df, path="results.csv"):
    df.to_csv(path, index=True)

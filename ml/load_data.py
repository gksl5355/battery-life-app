# -*- coding: utf-8 -*-
import pandas as pd

def load_data():
    cells_meta = pd.read_csv("./data/cells_meta.csv")
    early = pd.read_csv("./data/early_cycle_features.csv")
    df = pd.merge(cells_meta, early, on="cell_id", how="inner")
    return df
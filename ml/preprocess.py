from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# config에서 import 해와야 함
from config import NUM_KEPT, CAT_COLS  

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_KEPT),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ],
        remainder="drop",
    )

# feature_eng.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1) 로그 변환 (분포 안정화)
    for col in ["initial_DCIR_mOhm", "deltaV_50", "deltaT_50", "early_deg_slope"]:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(out[col].astype(float).clip(lower=0))

    # 2) 에너지 밀도 기반 비율 피처
    if set(["deltaV_50","energy_density"]).issubset(out.columns):
        out["deltaV_per_en"] = out["deltaV_50"] / (out["energy_density"].replace(0, np.nan))
    if set(["deltaT_50","energy_density"]).issubset(out.columns):
        out["deltaT_per_en"] = out["deltaT_50"] / (out["energy_density"].replace(0, np.nan))
    if set(["initial_DCIR_mOhm","energy_density"]).issubset(out.columns):
        out["dcir_per_en"] = out["initial_DCIR_mOhm"] / (out["energy_density"].replace(0, np.nan))

    # 3) 상호작용항
    if set(["early_deg_slope","energy_density"]).issubset(out.columns):
        out["eds_x_en"] = out["early_deg_slope"] * out["energy_density"]

    # 4) 그룹 평균 인코딩 (범주형 × 수치형)
    if set(["formation_recipe","energy_density"]).issubset(out.columns):
        out["en_by_recipe_mean"] = (
            out.groupby("formation_recipe")["energy_density"].transform("mean")
        )
    if set(["line_id","early_deg_slope"]).issubset(out.columns):
        out["eds_by_line_mean"] = (
            out.groupby("line_id")["early_deg_slope"].transform("mean")
        )

    # 안전 처리
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

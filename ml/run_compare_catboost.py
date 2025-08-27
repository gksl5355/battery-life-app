# run_compare_catboost.py  (CatBoost 단독, log 제거, 파생 전/후 엄격 비교)
# -*- coding: utf-8 -*-
import pandas as pd
from catboost import CatBoostRegressor, Pool

from load_data import load_data
from features import get_features
from config import TARGET, GROUP_COL, CAT_COLS, NUM_KEPT
from train_eval import eval_group_cv_catboost
from feature_eng import apply_feature_engineering

# 파생 컬럼 네이밍 규칙(우리가 만든 것만 허용)
ENGINEERED_NUM_PATTERNS = (
    "log_",               # log 변환 계열
    "deltaV_per_en",
    "deltaT_per_en",
    "dcir_per_en",
    "eds_x_en",
    "en_by_recipe_mean",
    "eds_by_line_mean",
    # 필요 시 여기에 추가
)

EXCLUDE_COLS = {"cell_id"}  # 안전 차단
DATE_TIME_KEYWORDS = ("date", "time")  # 날짜/시간 컬럼 차단용 키워드(이름 기준)

def select_engineered_numeric(df, baseline_num_cols):
    """
    '우리가 만든 파생 컬럼'만 골라 숫자형으로 추가.
    예상치 못한 다른 수치 컬럼(예: mass_g, cycle_* 등)은 포함하지 않음.
    """
    extra = []
    for c in df.columns:
        if c in baseline_num_cols: 
            continue
        if c in (TARGET, GROUP_COL) or c in CAT_COLS or c in EXCLUDE_COLS:
            continue
        cl = c.lower()
        if any(k in cl for k in DATE_TIME_KEYWORDS):   # 이름에 date/time 포함 → 제외
            continue
        # 우리가 정의한 접두/정확 이름만 허용
        if isinstance(c, str) and (c.startswith(ENGINEERED_NUM_PATTERNS) or c in ENGINEERED_NUM_PATTERNS):
            # 숫자형만 허용
            if pd.api.types.is_numeric_dtype(df[c]):
                extra.append(c)
    return extra

def run_catboost_cv(df, use_engineered=False, label="original"):
    # 원본 기준 컬럼 고정
    X_base, y, groups, base_num, base_cat = get_features(df)

    if use_engineered:
        # 파생 생성
        df_eng = apply_feature_engineering(df.copy())
        # 파생 숫자 컬럼만 엄격 선별
        extra_num = select_engineered_numeric(df_eng, baseline_num_cols=base_num)
        use_num = base_num + [c for c in extra_num if c in df_eng.columns]
        use_cat = [c for c in base_cat if c in df_eng.columns]  # 범주형은 기존 CAT_COLS만
        X = df_eng[use_num + use_cat].copy()
    else:
        X = X_base.copy()
        use_cat = base_cat

    # CatBoost CV (raw만, log 타깃 제거)
    model = CatBoostRegressor(
        n_estimators=800,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=0,
        loss_function="RMSE",
    )
    res_raw = eval_group_cv_catboost(
        model, X, y, groups, use_cat, n_splits=5, log_target=False
    )

    # 전체 적합 후 중요도
    cat_idx = [X.columns.get_loc(c) for c in use_cat]
    full = CatBoostRegressor(
        n_estimators=800,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=0,
        loss_function="RMSE",
    )
    pool = Pool(X, y, cat_features=cat_idx)
    full.fit(pool, verbose=False)
    imp = full.get_feature_importance(prettified=True)
    if isinstance(imp, pd.DataFrame):
        imp = imp.rename(columns={"Feature Id": "feature", "Importances": "importance"})
    else:
        imp = pd.DataFrame({"feature": list(X.columns),
                            "importance": full.get_feature_importance()})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)

    return {"raw": res_raw, "fi": imp}

def main():
    df = load_data()

    # Case A: Original (NUM_KEPT + CAT_COLS만)
    results_original = run_catboost_cv(df, use_engineered=False, label="original")

    # Case B: Engineered (Original + 우리가 만든 파생만)
    results_engineered = run_catboost_cv(df, use_engineered=True, label="engineered")

    print("\n=== CatBoost CV Results (Original) ===")
    print(pd.Series(results_original["raw"]))
    print("\n=== CatBoost CV Results (Engineered) ===")
    print(pd.Series(results_engineered["raw"]))

    print("\n=== Feature Importance (Original, Top 20) ===")
    print(results_original["fi"].head(20))
    print("\n=== Feature Importance (Engineered, Top 20) ===")
    print(results_engineered["fi"].head(20))

    # 저장
    pd.Series(results_original["raw"]).to_csv("catboost_results_original.csv", encoding="utf-8-sig")
    pd.Series(results_engineered["raw"]).to_csv("catboost_results_engineered.csv", encoding="utf-8-sig")
    results_original["fi"].to_csv("fi_catboost_original.csv", index=False, encoding="utf-8-sig")
    results_engineered["fi"].to_csv("fi_catboost_engineered.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()

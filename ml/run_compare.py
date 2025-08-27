# run_compare.py
# -*- coding: utf-8 -*-
import pandas as pd

from load_data import load_data
from features import get_features
from preprocess import make_preprocessor
from config import CAT_COLS, NUM_KEPT

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# train_eval.py 안에 다음 두 함수가 있어야 합니다:
# - eval_group_cv (기존, OHE 파이프라인용: RF/LGBM)
# - eval_group_cv_catboost (CatBoost 네이티브 카테고리 처리용)
from train_eval import eval_group_cv, eval_group_cv_catboost


def get_ohe_feature_names(prep, num_cols, cat_cols):
    """
    ColumnTransformer:
      ("num", "passthrough", num_cols)
      ("cat", OneHotEncoder(...), cat_cols)
    구조를 가정하고, 학습된 prep 으로 최종 피처 이름을 생성.
    """
    num_names = list(num_cols)

    # OneHotEncoder 추출
    ohe = None
    for name, trans, cols in prep.transformers_:
        if name == "cat":
            ohe = trans
            break
    if ohe is None:
        raise ValueError("No 'cat' transformer found in ColumnTransformer.")

    # 카테고리 이름 전개
    cat_names = []
    for col, cats in zip(cat_cols, ohe.categories_):
        cat_names.extend([f"{col}__{c}" for c in cats])

    return num_names + cat_names


def main():
    # 0) 데이터 적재
    df = load_data()
    X, y, groups = get_features(df)

    # 1) 모델 정의
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    lgbm = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42
    )
    cat = CatBoostRegressor(
        n_estimators=1000, learning_rate=0.05, depth=6, random_seed=42, verbose=0
    )

    # 2) 성능 비교 (GroupKFold)
    results = {}

    # RF / LGBM: 기존 전처리 파이프라인 사용
    for name, est in [("RF", rf), ("LGBM", lgbm)]:
        results[(name, "raw")] = eval_group_cv(
            est, X, y, groups, n_splits=5, log_target=False
        )
        results[(name, "log")] = eval_group_cv(
            est, X, y, groups, n_splits=5, log_target=True
        )

    # CatBoost: 원핫 X, cat_features 네이티브 처리
    # ⚠️ eval_group_cv_catboost 시그니처에 fit_params가 없다면 아래처럼 간단히 호출하세요.
    results[("CatBoost", "raw")] = eval_group_cv_catboost(
        cat, X, y, groups, CAT_COLS, n_splits=5, log_target=False
    )
    results[("CatBoost", "log")] = eval_group_cv_catboost(
        cat, X, y, groups, CAT_COLS, n_splits=5, log_target=True
    )

    # 3) 결과 테이블 출력
    df_results = pd.DataFrame(results).T
    print("\n=== Cross-Validation Results (mean ± std) ===")
    print(df_results)

    # 4) 피처 중요도 계산
    # 4-1) RF / LGBM 은 OHE 이후 중요도
    print("\n=== Feature Importances (RF / LGBM) ===")
    prep = make_preprocessor()

    # RF 파이프라인 적합
    pipe_rf = Pipeline([("prep", prep), ("model", rf)])
    pipe_rf.fit(X, y)

    # OHE 이후 피처명 획득
    ohe_feature_names = get_ohe_feature_names(
        pipe_rf.named_steps["prep"], NUM_KEPT, CAT_COLS
    )

    rf_importances = pipe_rf.named_steps["model"].feature_importances_
    df_rf_imp = (
        pd.DataFrame({"feature": ohe_feature_names, "importance": rf_importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("\n[RF] Top 20")
    print(df_rf_imp.head(20))

    # LGBM 파이프라인 적합
    pipe_lgbm = Pipeline([("prep", make_preprocessor()), ("model", lgbm)])
    pipe_lgbm.fit(X, y)
    lgbm_importances = pipe_lgbm.named_steps["model"].feature_importances_
    df_lgbm_imp = (
        pd.DataFrame({"feature": ohe_feature_names, "importance": lgbm_importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("\n[LGBM] Top 20")
    print(df_lgbm_imp.head(20))

    # 4-2) CatBoost 는 네이티브 중요도
    print("\n=== Feature Importances (CatBoost native) ===")
    from catboost import Pool

    cat_full = CatBoostRegressor(
        n_estimators=1000, learning_rate=0.05, depth=6, random_seed=42, verbose=0
    )
    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS]
    pool_full = Pool(X, y, cat_features=cat_idx)
    cat_full.fit(pool_full, verbose=False)

    imp_cat = cat_full.get_feature_importance(prettified=True)
    if isinstance(imp_cat, pd.DataFrame):
        imp_cat = imp_cat.rename(
            columns={"Feature Id": "feature", "Importances": "importance"}
        )
    else:
        imp_cat = pd.DataFrame(
            {"feature": list(X.columns), "importance": cat_full.get_feature_importance()}
        )

    imp_cat = imp_cat.sort_values("importance", ascending=False).reset_index(drop=True)
    print("\n[CatBoost] Top 20")
    print(imp_cat.head(20))

    # 5) CSV 저장 (옵션)
    df_results.to_csv("model_cv_results.csv", encoding="utf-8-sig")
    df_rf_imp.to_csv("fi_rf.csv", index=False, encoding="utf-8-sig")
    df_lgbm_imp.to_csv("fi_lgbm.csv", index=False, encoding="utf-8-sig")
    imp_cat.to_csv("fi_catboost.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Optuna-based CatBoostRegressor HPO with GroupKFold (+ optional 6:2:2 split, tqdm progress, Optuna Dashboard storage).
- Uses: load_data, features.get_features, feature_eng.apply_feature_engineering, config
- Objective: mean RMSE across folds (lower is better), with early stopping + pruning
- Saves: best params (JSON), study CSV, final model, feature importance, summary

Examples:
    # Baseline (original features)
    python run_optuna_catboost.py --n-trials 80 --use-engineered false --n-splits 5 --early-stopping 100

    # Engineered features (whitelist) + same tuning budget
    python run_optuna_catboost.py --n-trials 80 --use-engineered true --n-splits 5 --early-stopping 100

    # Group-preserving 6:2:2 split; tuning on train, final training on train+valid, test eval
    python run_optuna_catboost.py --use-engineered true --split 0.6,0.2,0.2 --n-trials 80 --n-splits 5 --early-stopping 100

    # Use Optuna Dashboard (persists study to SQLite)
    python run_optuna_catboost.py --storage optuna_study.db
    # then in another terminal:
    # optuna-dashboard sqlite:///path/to/optuna_study.db
"""
import argparse
import json
import os
from math import sqrt
from typing import List, Tuple

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor, Pool
from tqdm import tqdm

# project modules
from load_data import load_data
from features import get_features
from feature_eng import apply_feature_engineering
from config import TARGET, GROUP_COL, CAT_COLS, NUM_KEPT, SEED


# ---- engineered feature selection (strict whitelist, adapted from run_compare_catboost.py) ----
ENGINEERED_NUM_PATTERNS = (
    "log_",
    "deltaV_per_en",
    "deltaT_per_en",
    "dcir_per_en",
    "eds_x_en",
    "en_by_recipe_mean",
    "eds_by_line_mean",
)
EXCLUDE_COLS = {"cell_id"}
DATE_TIME_KEYWORDS = ("date", "time")


def select_engineered_numeric(df: pd.DataFrame, baseline_num_cols: List[str]) -> List[str]:
    extra = []
    for c in df.columns:
        if c in baseline_num_cols:
            continue
        if c in (TARGET, GROUP_COL) or c in CAT_COLS or c in EXCLUDE_COLS:
            continue
        cl = c.lower() if isinstance(c, str) else ""
        if any(k in cl for k in DATE_TIME_KEYWORDS):
            continue
        if isinstance(c, str) and (c.startswith(ENGINEERED_NUM_PATTERNS) or c in ENGINEERED_NUM_PATTERNS):
            if pd.api.types.is_numeric_dtype(df[c]):
                extra.append(c)
    return extra


def make_dataset(use_engineered: bool) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    df = load_data()
    X_base, y, groups, base_num, base_cat = get_features(df)

    if not use_engineered:
        return X_base, y, groups, base_cat

    # generate engineered columns and select strictly whitelisted numeric ones
    df_eng = apply_feature_engineering(df.copy())
    extra_num = select_engineered_numeric(df_eng, baseline_num_cols=base_num)
    use_num = base_num + [c for c in extra_num if c in df_eng.columns]
    use_cat = [c for c in base_cat if c in df_eng.columns]
    X = df_eng[use_num + use_cat].copy()
    return X, y, groups, use_cat


def group_split_three(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series,
    train_ratio: float, valid_ratio: float, test_ratio: float, seed: int
):
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-8, "split 비율 합은 1이어야 합니다."
    # 1) test 분리
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    tr_idx, te_idx = next(gss1.split(X, y, groups))
    X_trv, y_trv, g_trv = X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx]
    X_te,  y_te,  g_te  = X.iloc[te_idx],  y.iloc[te_idx],  groups.iloc[te_idx]
    # 2) 남은 집합에서 valid 분리 (비율 보정)
    valid_ratio_in_trv = valid_ratio / (1.0 - test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=valid_ratio_in_trv, random_state=seed)
    tr_idx2, va_idx = next(gss2.split(X_trv, y_trv, g_trv))
    X_tr, y_tr, g_tr = X_trv.iloc[tr_idx2], y_trv.iloc[tr_idx2], g_trv.iloc[tr_idx2]
    X_va, y_va, g_va = X_trv.iloc[va_idx],  y_trv.iloc[va_idx],  g_trv.iloc[va_idx]
    return (
        X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), g_tr.reset_index(drop=True),
        X_va.reset_index(drop=True), y_va.reset_index(drop=True), g_va.reset_index(drop=True),
        X_te.reset_index(drop=True), y_te.reset_index(drop=True), g_te.reset_index(drop=True),
    )


def cv_rmse(params: dict, X: pd.DataFrame, y: pd.Series, groups: pd.Series, cat_cols: List[str],
            n_splits: int, early_stopping_rounds: int, trial: optuna.Trial) -> float:
    """GroupKFold CV returning mean RMSE; reports per-fold for pruning."""
    gkf = GroupKFold(n_splits=n_splits)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    rmses = []
    for fold_id, (tr, va) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        valid_pool = Pool(X_va, y_va, cat_features=cat_idx)

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=SEED,
            verbose=False,
            **params
        )
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )

        y_pred = model.predict(valid_pool)
        rmse = sqrt(mean_squared_error(y_va, y_pred))
        rmses.append(rmse)

        # report per-fold to enable pruning
        trial.report(float(np.mean(rmses)), step=fold_id)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(rmses))


def suggest_params(trial: optuna.Trial) -> dict:
    """Compact, fast search space with conditional bootstrap settings."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 5.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "border_count": trial.suggest_int("border_count", 32, 128),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
        "thread_count": -1,
    }
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--use-engineered", type=str, default="false", help="true/false")
    parser.add_argument("--study-name", type=str, default="optuna_catboost_rmse")
    parser.add_argument("--outdir", type=str, default="./artifacts_optuna_cat")
    parser.add_argument("--split", type=str, default="", help="예: '0.6,0.2,0.2' (train,valid,test 비율). 비우면 전체를 CV로만 사용")
    parser.add_argument("--storage", type=str, default="", help="Optuna SQLite 파일 경로(예: optuna_study.db). 비우면 메모리 사용")
    args = parser.parse_args()

    use_engineered = args.use_engineered.lower() in ("true", "1", "yes", "y")
    os.makedirs(args.outdir, exist_ok=True)

    # dataset
    X, y, groups, cat_cols = make_dataset(use_engineered=use_engineered)

    # optional group-preserving split (e.g., 0.6,0.2,0.2)
    X_tr, y_tr, g_tr = X, y, groups
    X_va = y_va = g_va = None
    X_te = y_te = g_te = None
    if args.split:
        vals = [float(v) for v in args.split.split(",")]
        assert len(vals) == 3, "--split은 'train,valid,test' 세 값이어야 합니다. 예: 0.6,0.2,0.2"
        trr, var, ter = vals
        (X_tr, y_tr, g_tr,
         X_va, y_va, g_va,
         X_te, y_te, g_te) = group_split_three(X, y, groups, trr, var, ter, seed=SEED)

    # --- train 그룹 수에 맞춰 n_splits 자동 보정 (재발 방지) ---
    n_groups_train = int(pd.Series(g_tr).nunique())
    if args.n_splits > n_groups_train:
        print(f"[WARN] n_splits={args.n_splits} > train 그룹수={n_groups_train} → n_splits={n_groups_train}로 조정")
        args.n_splits = n_groups_train
    if args.n_splits < 2:
        raise ValueError(f"GroupKFold는 최소 2개 split 필요. 현재 train 그룹수={n_groups_train}. "
                         f"--split에서 train 비율을 늘리거나 그룹 구성이 충분한지 확인하세요.")

    # 참고용: train 그룹 분포 저장
    pd.Series(g_tr).value_counts().to_csv(os.path.join(args.outdir, "train_group_counts.csv"), encoding="utf-8-sig")

    # study
    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner(n_startup_trials=min(10, max(1, args.n_trials // 4)))
    if args.storage:
        storage_url = f"sqlite:///{os.path.abspath(args.storage)}"
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=args.study_name,
            storage=storage_url,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=args.study_name
        )

    # tqdm progress bar
    pbar = tqdm(total=args.n_trials, desc="Optuna Trials", unit="trial")

    def objective(trial: optuna.Trial):
        params = suggest_params(trial)
        score = cv_rmse(
            params, X_tr, y_tr, g_tr, cat_cols,
            args.n_splits, args.early_stopping, trial
        )
        pbar.update(1)
        return score

    # optimize
    try:
        study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    finally:
        pbar.close()

    # save best params
    best_params = study.best_params
    with open(os.path.join(args.outdir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    # export study dataframe
    df_study = study.trials_dataframe()
    df_study.to_csv(os.path.join(args.outdir, "study_trials.csv"), index=False, encoding="utf-8-sig")

    # --- retrain final model ---
    cat_idx_tr = [X_tr.columns.get_loc(c) for c in cat_cols]
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=SEED,
        verbose=False,
        **best_params
    )

    if X_va is not None:
        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx_tr)
        valid_pool = Pool(X_va, y_va, cat_features=cat_idx_tr)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True,
                  early_stopping_rounds=args.early_stopping, verbose=False)
        final_X_cols = X_tr.columns
    else:
        n_valid = max(1, int(len(X_tr) * 0.1))
        train_pool = Pool(X_tr.iloc[:-n_valid], y_tr.iloc[:-n_valid], cat_features=cat_idx_tr)
        valid_pool = Pool(X_tr.iloc[-n_valid:],  y_tr.iloc[-n_valid:],  cat_features=cat_idx_tr)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True,
                  early_stopping_rounds=args.early_stopping, verbose=False)
        final_X_cols = X_tr.columns

    # save model
    model_path = os.path.join(args.outdir, "catboost_model.cbm")
    model.save_model(model_path)

    # feature importances
    imp = model.get_feature_importance(prettified=True)
    if isinstance(imp, pd.DataFrame):
        cols = {c.lower(): c for c in imp.columns}
        rename_map = {}
        if "feature id" in cols:
            rename_map[cols["feature id"]] = "feature"
        if "importances" in cols:
            rename_map[cols["importances"]] = "importance"
        imp = imp.rename(columns=rename_map)
    else:
        imp = pd.DataFrame({"feature": list(final_X_cols), "importance": model.get_feature_importance()})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(args.outdir, "feature_importance.csv"), index=False, encoding="utf-8-sig")

    # CV score on train with best params (for reproducibility)
    mean_rmse = cv_rmse(
        best_params, X_tr, y_tr, g_tr, cat_cols,
        args.n_splits, args.early_stopping,
        trial=optuna.trial.FixedTrial(best_params)
    )

    # optional test evaluation
    test_rmse = None
    if X_te is not None:
        cat_idx_te = [X_te.columns.get_loc(c) for c in cat_cols]
        test_pool = Pool(X_te, cat_features=cat_idx_te)
        y_pred = model.predict(test_pool)
        test_rmse = sqrt(mean_squared_error(y_te, y_pred))

    # summary
    with open(os.path.join(args.outdir, "cv_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"mean_RMSE(CV on train): {mean_rmse:.6f}\n")
        f.write(f"n_splits: {args.n_splits}\nuse_engineered: {use_engineered}\n")
        f.write(f"groups_in_train: {n_groups_train}\n")
        if test_rmse is not None:
            f.write(f"test_RMSE: {test_rmse:.6f}\n")

    print("[Optuna] Best params:", best_params)
    print(f"[Optuna] Mean RMSE (CV on train): {mean_rmse:.6f}")
    if test_rmse is not None:
        print(f"[Optuna] Test RMSE: {test_rmse:.6f}")
    print(f"Artifacts saved under: {args.outdir}")


if __name__ == "__main__":
    main()

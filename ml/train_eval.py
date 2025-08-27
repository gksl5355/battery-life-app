# -*- coding: utf-8 -*-
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from preprocess import make_preprocessor

def eval_group_cv(estimator, X, y, groups, n_splits=5, log_target=False):
    gkf = GroupKFold(n_splits=n_splits)
    rmses, maes, r2s = [], [], []
    for tr, va in gkf.split(X, y, groups):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        if log_target:
            y_tr_fit = np.log1p(y_tr)
        else:
            y_tr_fit = y_tr

        pipe = Pipeline([("prep", make_preprocessor()), ("model", estimator)])
        pipe.fit(X_tr, y_tr_fit)

        y_pred = pipe.predict(X_va)
        if log_target:
            y_pred = np.expm1(y_pred)

        rmses.append(sqrt(mean_squared_error(y_va, y_pred)))
        maes.append(mean_absolute_error(y_va, y_pred))
        r2s.append(r2_score(y_va, y_pred))

    def stat(xs): return f"{np.mean(xs):.2f} ± {np.std(xs):.2f}"
    return {"RMSE": stat(rmses), "MAE": stat(maes), "R2": stat(r2s)}

def eval_group_cv_catboost(estimator, X, y, groups, cat_cols, n_splits=5, log_target=False):
    from catboost import Pool
    import numpy as np
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from math import sqrt

    gkf = GroupKFold(n_splits=n_splits)
    rmses, maes, r2s = [], [], []
    
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    for tr, va in gkf.split(X, y, groups):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        if log_target:
            y_tr_fit = np.log1p(y_tr)
        else:
            y_tr_fit = y_tr

        train_pool = Pool(X_tr, y_tr_fit, cat_features=cat_idx)
        valid_pool = Pool(X_va, y_va, cat_features=cat_idx)

        estimator.fit(train_pool, eval_set=valid_pool, verbose=False)

        y_pred = estimator.predict(valid_pool)
        if log_target:
            y_pred = np.expm1(y_pred)

        rmses.append(sqrt(mean_squared_error(y_va, y_pred)))
        maes.append(mean_absolute_error(y_va, y_pred))
        r2s.append(r2_score(y_va, y_pred))

    def stat(xs): return f"{np.mean(xs):.2f} ± {np.std(xs):.2f}"
    return {"RMSE": stat(rmses), "MAE": stat(maes), "R2": stat(r2s)}

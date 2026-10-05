"""Train one LightGBM model per position and compare it against baselines."""
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

from features import FEATURES, POSITIONS, TARGET

warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from lightgbm.sklearn import LGBMDeprecationWarning
    warnings.filterwarnings("ignore", category=LGBMDeprecationWarning)
except ImportError:
    pass

BASE_PARAMS = dict(
    learning_rate=0.03, num_leaves=15, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=5.0, verbose=-1,
)
RELEVANT_MIN_R5 = 8.0  # "fantasy-relevant": averaged 8+ PPR over last 5 games


def add_baselines(df):
    """Simple predictors the model has to beat."""
    df = df.copy()
    df["base_last3"] = df[f"{TARGET}_r3"]
    df["base_season"] = df.season_to_date.fillna(df.prev_season_avg).fillna(df.base_last3)
    return df


def fit_position(train, valid):
    """Fit with early stopping on the validation season; return the model."""
    m = lgb.LGBMRegressor(n_estimators=3000, **BASE_PARAMS)
    m.fit(train[FEATURES], train[TARGET],
          eval_set=[(valid[FEATURES], valid[TARGET])], eval_metric="l1",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def weekly_spearman(df, col):
    """Average over weeks of the rank correlation between predicted and actual."""
    scores = []
    for _, wk in df.groupby(["season", "week"]):
        if len(wk) >= 10:
            scores.append(spearmanr(wk[col], wk[TARGET])[0])
    return float(np.nanmean(scores))


def evaluate(df, test_season):
    """Train on seasons before test_season-1, early-stop on test_season-1,
    score on test_season. Returns metrics and best iteration per position."""
    df = add_baselines(df)
    df = df[df.career_games >= 1]
    train = df[df.season < test_season - 1]
    valid = df[df.season == test_season - 1]
    test = df[df.season == test_season]

    metrics, best_iters = {}, {}
    for pos in POSITIONS:
        tr, va, te = (x[x.position == pos].copy() for x in (train, valid, test))
        model = fit_position(tr, va)
        best_iters[pos] = int(model.best_iteration_ or 200)
        te["pred"] = model.predict(te[FEATURES])
        rel = te[te[f"{TARGET}_r5"] >= RELEVANT_MIN_R5]
        metrics[pos] = {
            "n_test_rows": int(len(rel)),
            "mae": {c: float((rel[c] - rel[TARGET]).abs().mean())
                    for c in ["pred", "base_last3", "base_season"]},
            "spearman": {c: weekly_spearman(rel, c)
                         for c in ["pred", "base_last3", "base_season"]},
        }
    return {"test_season": int(test_season), "by_position": metrics}, best_iters


def fit_final_and_predict(df, future, best_iters):
    """Retrain on ALL completed games, then predict the upcoming week.
    Also fits 15th/85th percentile models for a floor and ceiling."""
    df = add_baselines(df)
    hist = df[df[TARGET].notna() & (df.career_games >= 1)]
    out = []
    for pos in POSITIONS:
        tr = hist[hist.position == pos]
        fu = future[future.position == pos].copy()
        n = max(int(best_iters[pos] * 1.1), 50)
        mean_model = lgb.LGBMRegressor(n_estimators=n, **BASE_PARAMS).fit(tr[FEATURES], tr[TARGET])
        fu["proj"] = mean_model.predict(fu[FEATURES])
        for name, alpha in [("floor", 0.15), ("ceiling", 0.85)]:
            q = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=n, **BASE_PARAMS)
            fu[name] = q.fit(tr[FEATURES], tr[TARGET]).predict(fu[FEATURES])
        fu["floor"] = np.minimum(fu.floor, fu.proj)
        fu["ceiling"] = np.maximum(fu.ceiling, fu.proj)
        out.append(fu)
    return pd.concat(out)


import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================================================
# Config
# =========================================================
TRAIN_PATH = "train.csv"
TEST_PATH = "test_for_participants.csv"  # adjust if your platform uses a different name
SUBMISSION_PATH = "my_submission.csv"

MARKET_COL = "market"
TIME_COL = "delivery_start"
TARGET_COL = "target"
ID_COL = "id"

# Validation: last ~3 months (matches test horizon)
HOLDOUT_DAYS = 90

# CV (fast tuning on pre-holdout data)
CV_SPLITS = 3

# Blend schedule: AR fades out over time in test/holdout
# tau in hours: smaller -> faster fade to exogenous model
TAU_HOURS = 96  # ~4 days; AR strong early, exog dominates long-run

RANDOM_STATE = 42

# =========================================================
# Helpers
# =========================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[TIME_COL])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Core fundamentals
    df["renewable_index"] = df["wind_forecast"] + df["solar_forecast"]
    df["net_load"] = df["load_forecast"] - df["renewable_index"]
    df["wind_share"] = df["wind_forecast"] / (df["load_forecast"] + 1.0)

    # Proxies
    if "wind_speed_80m" in df.columns:
        df["wind_cubic_proxy"] = df["wind_speed_80m"] ** 3
    else:
        df["wind_cubic_proxy"] = np.nan

    if "apparent_temperature_2m" in df.columns:
        df["heating_demand"] = np.maximum(0, 18 - df["apparent_temperature_2m"])
        df["cooling_demand"] = np.maximum(0, df["apparent_temperature_2m"] - 18)
    else:
        df["heating_demand"] = np.nan
        df["cooling_demand"] = np.nan

    if "wind_gust_speed_10m" in df.columns and "wind_speed_10m" in df.columns:
        df["gust_factor"] = df["wind_gust_speed_10m"] / (df["wind_speed_10m"] + 1.0)
    else:
        df["gust_factor"] = np.nan

    # Exogenous ramps (per market)
    df["net_load_ramp"] = df.groupby(MARKET_COL)["net_load"].diff()
    df["wind_ramp"] = df.groupby(MARKET_COL)["wind_forecast"].diff()
    df["solar_ramp"] = df.groupby(MARKET_COL)["solar_forecast"].diff()

    # Fill ramp NaNs (first row per market)
    for c in ["net_load_ramp", "wind_ramp", "solar_ramp"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    return df


def add_target_lag_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Adds lag/rolling features of target. Assumes df is sorted by market,time and target exists (or has preds)."""
    df = df.copy()

    for lag in [1, 2, 24, 48]:
        df[f"lag_{lag}"] = df.groupby(MARKET_COL)[target_col].shift(lag)

    shifted = df.groupby(MARKET_COL)[target_col].shift(1)
    df["rolling_mean_24"] = (
        shifted.groupby(df[MARKET_COL])
        .rolling(24, min_periods=12)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rolling_std_24"] = (
        shifted.groupby(df[MARKET_COL])
        .rolling(24, min_periods=12)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Momentum-style features
    df["mom_1_2"] = df["lag_1"] - df["lag_2"]
    df["mom_1_24"] = df["lag_1"] - df["lag_24"]

    return df


@dataclass
class MarketModels:
    ar_boost: HistGradientBoostingRegressor
    exog_boost: HistGradientBoostingRegressor
    ridge: Pipeline
    fill_values: Dict[str, float]
    ar_features: List[str]
    exog_features: List[str]


def build_feature_lists(df_cols: List[str]) -> Tuple[List[str], List[str]]:
    # Exogenous + calendar (stable over long horizon)
    exog = [
        "hour_sin", "hour_cos", "dayofweek", "month",
        "load_forecast", "wind_forecast", "solar_forecast",
        "renewable_index", "net_load", "wind_share",
        "global_horizontal_irradiance",
        "wind_speed_80m", "wind_direction_80m",
        "air_temperature_2m", "apparent_temperature_2m",
        "gust_factor",
        "net_load_ramp", "wind_ramp", "solar_ramp",
    ]
    exog = [c for c in exog if c in df_cols]

    ar = exog + [
        "lag_1", "lag_2", "lag_24", "lag_48",
        "rolling_mean_24", "rolling_std_24",
        "mom_1_2", "mom_1_24",
    ]
    ar = [c for c in ar if c in df_cols]

    return ar, exog


def compute_fill_values(train_df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    fills = {}
    for c in cols:
        if c not in train_df.columns:
            continue
        v = train_df[c]
        if v.isna().all():
            fills[c] = 0.0
        else:
            fills[c] = float(v.median())
    return fills


def apply_fills(df: pd.DataFrame, fill_values: Dict[str, float], cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].fillna(fill_values.get(c, 0.0))
    return out


def make_hgbr(params: Dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        max_iter=params["max_iter"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        random_state=RANDOM_STATE,
    )


def tune_hgbr_one_step(df_sup: pd.DataFrame, features: List[str], y_col: str) -> Dict:
    """Fast tuning using one-step RMSE with TimeSeriesSplit (pre-holdout only)."""
    grid = []
    for lr in [0.03, 0.05]:
        for md in [4, 6]:
            for leaf in [40, 80]:
                for l2 in [0.0, 1.0]:
                    grid.append({
                        "learning_rate": lr,
                        "max_depth": md,
                        "max_iter": 600,
                        "min_samples_leaf": leaf,
                        "l2_regularization": l2,
                    })

    X = df_sup[features].to_numpy()
    y = df_sup[y_col].to_numpy()

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    best_params = None
    best_score = float("inf")

    for params in grid:
        fold_scores = []
        for tr_idx, va_idx in tscv.split(X):
            m = make_hgbr(params)
            m.fit(X[tr_idx], y[tr_idx])
            pred = m.predict(X[va_idx])
            fold_scores.append(rmse(y[va_idx], pred))
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_params = params

    return best_params


def tune_ridge_one_step(df_sup: pd.DataFrame, features: List[str], y_col: str) -> Pipeline:
    X = df_sup[features].to_numpy()
    y = df_sup[y_col].to_numpy()

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    alphas = [1.0, 10.0, 100.0]

    best_alpha = None
    best_score = float("inf")

    for a in alphas:
        fold_scores = []
        for tr_idx, va_idx in tscv.split(X):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=a, random_state=RANDOM_STATE)),
            ])
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx])
            fold_scores.append(rmse(y[va_idx], pred))
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_alpha = a

    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha, random_state=RANDOM_STATE)),
    ])


def blend_weight(step: int, tau_hours: float = TAU_HOURS) -> float:
    # step is number of hours ahead from the start of the forecast block (0-based)
    return float(np.exp(-step / tau_hours))


def recursive_forecast_block(
    df_hist_and_block: pd.DataFrame,
    block_mask: pd.Series,
    models: MarketModels,
) -> np.ndarray:
    """Recursive forecast over block rows (block_mask True). df contains history + block, sorted by time."""

    df = df_hist_and_block.copy()

    # We'll write predictions into a working column, used for future lag computation
    work_target = "_work_target"
    df[work_target] = df[TARGET_COL] if TARGET_COL in df.columns else np.nan

    # Ensure that in the forecast block, the true target is not present/used
    df.loc[block_mask, work_target] = np.nan

    # Precompute exog features once (they don't depend on target)
    ar_feats = models.ar_features
    exog_feats = models.exog_features

    # Indices of rows to predict (in chronological order)
    block_idx = np.where(block_mask.to_numpy())[0]

    preds = np.zeros(len(block_idx), dtype=float)

    for j, idx in enumerate(block_idx):
        # For current idx, compute lag features using work_target up to idx-1
        # We only need lags for current row, but easiest is to compute on prefix
        prefix = df.iloc[: idx + 1].copy()
        prefix = add_target_lag_features(prefix, target_col=work_target)

        row = prefix.iloc[-1:].copy()

        # Fill missing (lags early / rolling)
        row = apply_fills(row, models.fill_values, list(set(ar_feats + exog_feats)))

        # Predict
        ar_pred = float(models.ar_boost.predict(row[ar_feats])[0])
        exog_pred = float(models.exog_boost.predict(row[exog_feats])[0])
        ridge_pred = float(models.ridge.predict(row[ar_feats])[0])

        w = blend_weight(step=j)
        short_term = 0.6 * ar_pred + 0.4 * ridge_pred
        final_pred = w * short_term + (1.0 - w) * exog_pred

        preds[j] = final_pred

        # Write back for future lags
        df.iloc[idx, df.columns.get_loc(work_target)] = final_pred

    return preds


def train_models_for_market(df_market: pd.DataFrame) -> MarketModels:
    dm = df_market.sort_values(TIME_COL).reset_index(drop=True)

    # Build supervised frame for AR features (requires target)
    dm_sup = add_target_lag_features(dm, target_col=TARGET_COL)

    ar_features, exog_features = build_feature_lists(list(dm_sup.columns))

    # Fill values computed from training portion (will be applied in forecasting)
    fill_values = compute_fill_values(dm_sup, list(set(ar_features + exog_features)))

    # Drop rows with missing required data
    dm_ar = apply_fills(dm_sup, fill_values, ar_features)
    dm_ex = apply_fills(dm_sup, fill_values, exog_features)

    # For AR supervised training, drop rows where lags are still NaN (very early window)
    dm_ar = dm_ar.dropna(subset=[TARGET_COL] + [c for c in ["lag_1", "lag_2", "lag_24"] if c in dm_ar.columns]).copy()

    # Exog model can use more rows
    dm_ex = dm_ex.dropna(subset=[TARGET_COL]).copy()

    # Tune (fast CV on one-step)
    best_ar_params = tune_hgbr_one_step(dm_ar, ar_features, TARGET_COL)
    best_ex_params = tune_hgbr_one_step(dm_ex, exog_features, TARGET_COL)
    ridge_model = tune_ridge_one_step(dm_ar, ar_features, TARGET_COL)

    ar_boost = make_hgbr(best_ar_params)
    exog_boost = make_hgbr(best_ex_params)

    # Fit full on all available supervised data
    ar_boost.fit(dm_ar[ar_features], dm_ar[TARGET_COL])
    exog_boost.fit(dm_ex[exog_features], dm_ex[TARGET_COL])
    ridge_model.fit(dm_ar[ar_features], dm_ar[TARGET_COL])

    return MarketModels(
        ar_boost=ar_boost,
        exog_boost=exog_boost,
        ridge=ridge_model,
        fill_values=fill_values,
        ar_features=ar_features,
        exog_features=exog_features,
    )


def main():
    # -------------------------
    # Load
    # -------------------------
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train[TIME_COL] = pd.to_datetime(train[TIME_COL])
    test[TIME_COL] = pd.to_datetime(test[TIME_COL])

    # Concatenate only for consistent feature engineering (NOT for training labels)
    all_df = pd.concat([
        train.assign(_is_train=1),
        test.assign(_is_train=0, **{TARGET_COL: np.nan}),
    ], ignore_index=True)

    all_df = all_df.sort_values([MARKET_COL, TIME_COL]).reset_index(drop=True)

    # Base features
    all_df = add_time_features(all_df)
    all_df = add_domain_features(all_df)

    # Split back
    train_fe = all_df[all_df["_is_train"] == 1].drop(columns=["_is_train"]).copy()
    test_fe = all_df[all_df["_is_train"] == 0].drop(columns=["_is_train"]).copy()

    # -------------------------
    # Validation (last 3 months per market) + Training
    # -------------------------
    markets = sorted(train_fe[MARKET_COL].unique())

    oof_records = []
    market_models: Dict[str, MarketModels] = {}

    print("[1/3] Training per-market models + holdout validation (recursive, last 90 days)...")

    for i, m in enumerate(markets, start=1):
        dm = train_fe[train_fe[MARKET_COL] == m].sort_values(TIME_COL).reset_index(drop=True)

        cutoff = dm[TIME_COL].max() - pd.Timedelta(days=HOLDOUT_DAYS)
        hist = dm[dm[TIME_COL] < cutoff].copy()
        hold = dm[dm[TIME_COL] >= cutoff].copy()

        if len(hist) < 2000 or len(hold) < 200:
            # Too small; still train but skip validation reliability
            print(f"  - {m}: not enough data for robust holdout (hist={len(hist)}, hold={len(hold)}). Training anyway.")
            models = train_models_for_market(dm)
            market_models[m] = models
            continue

        # Train models on history only
        models = train_models_for_market(hist)
        market_models[m] = models

        # Recursive forecast on holdout
        block_df = pd.concat([hist, hold], ignore_index=True).sort_values(TIME_COL).reset_index(drop=True)
        block_mask = block_df[TIME_COL] >= cutoff

        preds = recursive_forecast_block(block_df, block_mask, models)
        y_true = hold[TARGET_COL].to_numpy()
        r = rmse(y_true, preds)

        print(f"  - {m} ({i}/{len(markets)}): holdout RMSE={r:.2f} (n={len(hold)})")

        oof = hold[[ID_COL, MARKET_COL, TIME_COL, TARGET_COL]].copy()
        oof["pred"] = preds
        oof_records.append(oof)

    if oof_records:
        oof_all = pd.concat(oof_records, ignore_index=True)
        overall = rmse(oof_all[TARGET_COL].to_numpy(), oof_all["pred"].to_numpy())
        print(f"\nTOTAL HOLDOUT RMSE (global, last {HOLDOUT_DAYS} days): {overall:.2f}")
    else:
        print("\nNo holdout validation computed (insufficient data per market).")

    # -------------------------
    # Fit final models on full train (per market)
    # -------------------------
    print("\n[2/3] Fitting final per-market models on full training data...")
    for m in markets:
        dm = train_fe[train_fe[MARKET_COL] == m].sort_values(TIME_COL).reset_index(drop=True)
        market_models[m] = train_models_for_market(dm)

    # -------------------------
    # Predict test (recursive per market)
    # -------------------------
    print("\n[3/3] Predicting test (recursive) + writing submission...")

    preds_all = []

    for m in markets:
        dm_train = train_fe[train_fe[MARKET_COL] == m].sort_values(TIME_COL).reset_index(drop=True)
        dm_test = test_fe[test_fe[MARKET_COL] == m].sort_values(TIME_COL).reset_index(drop=True)

        if len(dm_test) == 0:
            continue

        models = market_models[m]

        block_df = pd.concat([dm_train, dm_test.assign(**{TARGET_COL: np.nan})], ignore_index=True)
        block_df = block_df.sort_values(TIME_COL).reset_index(drop=True)

        # The block is exactly the test part
        block_mask = block_df[ID_COL].isin(dm_test[ID_COL]).to_numpy()

        preds = recursive_forecast_block(block_df, pd.Series(block_mask), models)

        out = dm_test[[ID_COL]].copy()
        out[TARGET_COL] = preds
        preds_all.append(out)

    sub = pd.concat(preds_all, ignore_index=True)
    sub = sub.sort_values(ID_COL)
    sub.to_csv(SUBMISSION_PATH, index=False)

    print(f"Saved submission: {SUBMISSION_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()

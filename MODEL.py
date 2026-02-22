import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================================================
# Config
# =========================================================
TRAIN_PATH = "train.csv"
TEST_PATH = "test_for_participants.csv"
SUBMISSION_PATH = "my_submission.csv"

MARKET_COL = "market"
TIME_COL = "delivery_start"
TARGET_COL = "target"
ID_COL = "id"

HOLDOUT_DAYS = 90
CV_SPLITS = 3
TAU_HOURS = 168
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
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Richer cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Peak/off-peak flag (common in energy markets)
    df["is_peak_hour"] = ((df["hour"] >= 8) & (df["hour"] <= 20)).astype(int)

    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # === Core fundamentals (unchanged) ===
    df["renewable_index"] = df["wind_forecast"] + df["solar_forecast"]
    df["net_load"] = df["load_forecast"] - df["renewable_index"]
    df["wind_share"] = df["wind_forecast"] / (df["load_forecast"] + 1.0)
    df["solar_share"] = df["solar_forecast"] / (df["load_forecast"] + 1.0)
    df["renewable_share"] = df["renewable_index"] / (df["load_forecast"] + 1.0)

    # === Wind features ===
    if "wind_speed_80m" in df.columns:
        df["wind_cubic_proxy"] = df["wind_speed_80m"] ** 3
        df["wind_sq"] = df["wind_speed_80m"] ** 2
    else:
        df["wind_cubic_proxy"] = np.nan
        df["wind_sq"] = np.nan

    # === Temperature / demand ===
    if "apparent_temperature_2m" in df.columns:
        df["heating_demand"] = np.maximum(0, 18 - df["apparent_temperature_2m"])
        df["cooling_demand"] = np.maximum(0, df["apparent_temperature_2m"] - 18)
        df["temp_sq"] = df["apparent_temperature_2m"] ** 2  # non-linear demand response
    else:
        df["heating_demand"] = np.nan
        df["cooling_demand"] = np.nan
        df["temp_sq"] = np.nan

    # === Gust factor ===
    if "wind_gust_speed_10m" in df.columns and "wind_speed_10m" in df.columns:
        df["gust_factor"] = df["wind_gust_speed_10m"] / (df["wind_speed_10m"] + 1.0)
    else:
        df["gust_factor"] = np.nan

    # Cloud cover aggregations
    if "cloud_cover_total" in df.columns:
        df["cloud_impact"] = df["cloud_cover_total"] * df["solar_forecast"] / (df["load_forecast"] + 1.0)
    if all(c in df.columns for c in ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]):
        df["cloud_cover_weighted"] = (
            0.5 * df["cloud_cover_low"] +
            0.3 * df["cloud_cover_mid"] +
            0.2 * df["cloud_cover_high"]
        )

    # Humidity / dew point spread (proxy for fog, condensation)
    if "dew_point_temperature_2m" in df.columns and "air_temperature_2m" in df.columns:
        df["dew_point_spread"] = df["air_temperature_2m"] - df["dew_point_temperature_2m"]

    # Pressure (weather regime indicator)
    if "surface_pressure" in df.columns:
        df["surface_pressure_norm"] = df["surface_pressure"] / 1013.25  # normalize to std atm

    # Atmospheric stability (CAPE * LI interaction)
    if "convective_available_potential_energy" in df.columns and "lifted_index" in df.columns:
        df["instability_index"] = df["convective_available_potential_energy"] / (df["lifted_index"].abs() + 1.0)

    # Solar irradiance components
    if "direct_normal_irradiance" in df.columns and "diffuse_horizontal_irradiance" in df.columns:
        df["total_irradiance"] = df["direct_normal_irradiance"] + df["diffuse_horizontal_irradiance"]
        df["diffuse_ratio"] = df["diffuse_horizontal_irradiance"] / (df["total_irradiance"] + 1.0)

    # Relative humidity interacts with temperature for demand
    if "relative_humidity_2m" in df.columns and "apparent_temperature_2m" in df.columns:
        df["heat_index_proxy"] = df["apparent_temperature_2m"] * df["relative_humidity_2m"] / 100.0

    # === Ramps (per market) ===
    df["net_load_ramp"] = df.groupby(MARKET_COL)["net_load"].diff()
    df["wind_ramp"] = df.groupby(MARKET_COL)["wind_forecast"].diff()
    df["solar_ramp"] = df.groupby(MARKET_COL)["solar_forecast"].diff()
    df["load_ramp"] = df.groupby(MARKET_COL)["load_forecast"].diff()

    # Second-order ramps (acceleration)
    df["net_load_accel"] = df.groupby(MARKET_COL)["net_load_ramp"].diff()

    for c in ["net_load_ramp", "wind_ramp", "solar_ramp", "load_ramp", "net_load_accel"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    return df


def add_target_lag_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Adds lag/rolling features of target."""
    df = df.copy()

    # More lags: capture daily (24h), 2-day (48h), weekly (168h) seasonality
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        col = f"lag_{lag}"
        df[col] = df.groupby(MARKET_COL)[target_col].shift(lag)

    shifted = df.groupby(MARKET_COL)[target_col].shift(1)

    # Rolling windows at multiple horizons
    for window in [6, 12, 24, 48, 168]:
        min_p = max(3, window // 4)
        df[f"rolling_mean_{window}"] = (
            shifted.groupby(df[MARKET_COL])
            .rolling(window, min_periods=min_p)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"rolling_std_{window}"] = (
            shifted.groupby(df[MARKET_COL])
            .rolling(window, min_periods=min_p)
            .std()
            .reset_index(level=0, drop=True)
        )

    # Min/max over 24h window (price extremes are meaningful in intraday)
    df["rolling_min_24"] = (
        shifted.groupby(df[MARKET_COL])
        .rolling(24, min_periods=6)
        .min()
        .reset_index(level=0, drop=True)
    )
    df["rolling_max_24"] = (
        shifted.groupby(df[MARKET_COL])
        .rolling(24, min_periods=6)
        .max()
        .reset_index(level=0, drop=True)
    )
    df["rolling_range_24"] = df["rolling_max_24"] - df["rolling_min_24"]

    # Momentum
    df["mom_1_2"] = df["lag_1"] - df["lag_2"]
    df["mom_1_24"] = df["lag_1"] - df["lag_24"]
    df["mom_24_48"] = df["lag_24"] - df["lag_48"]

    # Z-score of lag_1 relative to rolling window (price spike detector)
    df["lag1_zscore_24"] = (df["lag_1"] - df["rolling_mean_24"]) / (df["rolling_std_24"] + 1e-6)

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
    exog = [
        # Calendar
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "dayofweek", "month", "is_weekend", "is_peak_hour", "weekofyear",
        # Core energy
        "load_forecast", "wind_forecast", "solar_forecast",
        "renewable_index", "net_load", "wind_share", "solar_share", "renewable_share",
        # Wind
        "wind_speed_80m", "wind_direction_80m", "wind_cubic_proxy", "wind_sq",
        "wind_gust_speed_10m", "wind_speed_10m", "gust_factor",
        # Solar/irradiance
        "global_horizontal_irradiance", "direct_normal_irradiance",
        "diffuse_horizontal_irradiance", "total_irradiance", "diffuse_ratio",
        # Temperature/demand
        "air_temperature_2m", "apparent_temperature_2m",
        "heating_demand", "cooling_demand", "temp_sq",
        # Humidity/weather
        "dew_point_temperature_2m", "wet_bulb_temperature_2m",
        "relative_humidity_2m", "dew_point_spread", "heat_index_proxy",
        # Cloud
        "cloud_cover_total", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
        "cloud_cover_weighted", "cloud_impact",
        # Pressure / stability
        "surface_pressure", "surface_pressure_norm",
        "convective_available_potential_energy", "lifted_index", "convective_inhibition",
        "instability_index",
        # Other
        "precipitation_amount", "visibility", "freezing_level_height",
        # Ramps
        "net_load_ramp", "wind_ramp", "solar_ramp", "load_ramp", "net_load_accel",
    ]
    exog = [c for c in exog if c in df_cols]

    ar = exog + [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "lag_24", "lag_48", "lag_168",
        "rolling_mean_6", "rolling_std_6",
        "rolling_mean_12", "rolling_std_12",
        "rolling_mean_24", "rolling_std_24",
        "rolling_mean_48", "rolling_std_48",
        "rolling_mean_168", "rolling_std_168",
        "rolling_min_24", "rolling_max_24", "rolling_range_24",
        "mom_1_2", "mom_1_24", "mom_24_48",
        "lag1_zscore_24",
    ]
    ar = [c for c in ar if c in df_cols]

    return ar, exog

def compute_fill_values(train_df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    fills = {}
    for c in cols:
        if c not in train_df.columns:
            continue
        v = train_df[c]
        fills[c] = 0.0 if v.isna().all() else float(v.median())
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
    """Fast tuning with TimeSeriesSplit."""
    grid = []
    for lr in [0.03, 0.05, 0.08]:
        for md in [4, 6, 8]:
            for leaf in [20, 40, 80]:
                for l2 in [0.0, 0.5, 2.0]:
                    grid.append({
                        "learning_rate": lr,
                        "max_depth": md,
                        "max_iter": 700,
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
    alphas = [0.1, 1.0, 10.0, 100.0, 500.0]

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
    return float(np.exp(-step / tau_hours))


def recursive_forecast_block(
    df_hist_and_block: pd.DataFrame,
    block_mask: pd.Series,
    models: MarketModels,
) -> np.ndarray:
    """Recursive forecast over block rows."""
    df = df_hist_and_block.copy()

    work_target = "_work_target"
    df[work_target] = df[TARGET_COL] if TARGET_COL in df.columns else np.nan
    df.loc[block_mask, work_target] = np.nan

    ar_feats = models.ar_features
    exog_feats = models.exog_features

    block_idx = np.where(block_mask.to_numpy())[0]
    preds = np.zeros(len(block_idx), dtype=float)

    for j, idx in enumerate(block_idx):
        prefix = df.iloc[: idx + 1].copy()
        prefix = add_target_lag_features(prefix, target_col=work_target)

        row = prefix.iloc[-1:].copy()
        row = apply_fills(row, models.fill_values, list(set(ar_feats + exog_feats)))

        ar_pred = float(models.ar_boost.predict(row[ar_feats])[0])
        exog_pred = float(models.exog_boost.predict(row[exog_feats])[0])
        ridge_pred = float(models.ridge.predict(row[ar_feats])[0])

        w = blend_weight(step=j)
        short_term = 0.6 * ar_pred + 0.4 * ridge_pred
        final_pred = w * short_term + (1.0 - w) * exog_pred

        preds[j] = final_pred
        df.iloc[idx, df.columns.get_loc(work_target)] = final_pred

    return preds


def train_models_for_market(df_market: pd.DataFrame) -> MarketModels:
    dm = df_market.sort_values(TIME_COL).reset_index(drop=True)
    dm_sup = add_target_lag_features(dm, target_col=TARGET_COL)

    ar_features, exog_features = build_feature_lists(list(dm_sup.columns))
    fill_values = compute_fill_values(dm_sup, list(set(ar_features + exog_features)))

    dm_ar = apply_fills(dm_sup, fill_values, ar_features)
    dm_ex = apply_fills(dm_sup, fill_values, exog_features)

    lag_cols = [c for c in ["lag_1", "lag_2", "lag_24"] if c in dm_ar.columns]
    dm_ar = dm_ar.dropna(subset=[TARGET_COL] + lag_cols).copy()
    dm_ex = dm_ex.dropna(subset=[TARGET_COL]).copy()

    best_ar_params = tune_hgbr_one_step(dm_ar, ar_features, TARGET_COL)
    best_ex_params = tune_hgbr_one_step(dm_ex, exog_features, TARGET_COL)
    ridge_model = tune_ridge_one_step(dm_ar, ar_features, TARGET_COL)

    ar_boost = make_hgbr(best_ar_params)
    exog_boost = make_hgbr(best_ex_params)

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

    all_df = pd.concat([
        train.assign(_is_train=1),
        test.assign(_is_train=0, **{TARGET_COL: np.nan}),
    ], ignore_index=True)

    all_df = all_df.sort_values([MARKET_COL, TIME_COL]).reset_index(drop=True)

    all_df = add_time_features(all_df)
    all_df = add_domain_features(all_df)

    train_fe = all_df[all_df["_is_train"] == 1].drop(columns=["_is_train"]).copy()
    test_fe = all_df[all_df["_is_train"] == 0].drop(columns=["_is_train"]).copy()

    # -------------------------
    # Validation
    # -------------------------
    markets = sorted(train_fe[MARKET_COL].unique())
    oof_records = []
    market_models: Dict[str, MarketModels] = {}

    print("[1/3] Training per-market models + holdout validation...")

    for i, m in enumerate(markets, start=1):
        dm = train_fe[train_fe[MARKET_COL] == m].sort_values(TIME_COL).reset_index(drop=True)

        cutoff = dm[TIME_COL].max() - pd.Timedelta(days=HOLDOUT_DAYS)
        hist = dm[dm[TIME_COL] < cutoff].copy()
        hold = dm[dm[TIME_COL] >= cutoff].copy()

        if len(hist) < 2000 or len(hold) < 200:
            print(f"  - {m}: insufficient data for holdout. Training anyway.")
            models = train_models_for_market(dm)
            market_models[m] = models
            continue

        models = train_models_for_market(hist)
        market_models[m] = models

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

    # -------------------------
    # Fit final models on full train
    # -------------------------
    print("\n[2/3] Fitting final per-market models on full training data...")
    for m in markets:
        dm = train_fe[train_fe[MARKET_COL] == m].sort_values(TIME_COL).reset_index(drop=True)
        market_models[m] = train_models_for_market(dm)

    # -------------------------
    # Predict test
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

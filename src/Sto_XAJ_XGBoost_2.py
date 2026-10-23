import numpy as np
import pandas as pd
import warnings
import optuna
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pywt
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data" / "basins"
TRAJ_DIR = PROJECT_ROOT / "results" / "sto_xaj_W"
OUTPUT_DIR = PROJECT_ROOT / "results" / "hybrid_xgb_full"
RESULTS_SUMMARY = OUTPUT_DIR / "xgb_hybrid_full_batch_summary.csv"
ERRORS_FILE = OUTPUT_DIR / "errors.csv"
LAG = 3
CALIBRATION_RATIO = 0.7
N_TRIALS = 80
WAVELET_FAMILIES = [
    "haar",
    "db1", "db2", "db3", "db4",
    "sym2", "sym3", "sym4",
    "coif1", "coif2", "coif3",
]
def nse(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    if len(o) < 2:
        return -9999
    denominator = np.sum((o - np.mean(o)) ** 2)
    if denominator == 0:
        return -9999
    return 1 - np.sum((o - s) ** 2) / denominator
def calc_rmse(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    return np.sqrt(np.mean((o - s) ** 2)) if len(o) > 0 else np.nan
def calc_kge(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    if len(o) < 2:
        return -9999
    std_o = np.std(o)
    mean_o = np.mean(o)
    if std_o == 0 or mean_o == 0:
        return -9999
    r = np.corrcoef(o, s)[0, 1]
    if not np.isfinite(r):
        return -9999
    alpha = np.std(s) / std_o
    beta = np.mean(s) / mean_o
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
def calc_pbias(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    return 100 * np.sum(s - o) / np.sum(o) if len(o) > 0 and np.sum(o) != 0 else np.nan
def wavelet_decompose(series, wavelet="coif1", level=2):
    coeffs = pywt.wavedec(series, wavelet, level=level)
    return np.concatenate([c.flatten() for c in coeffs])
def create_features(mean_traj, min_traj, max_traj, range_traj, discharge,
                    lag=LAG, wavelet="coif1", level=2):
    X_list = []
    y_list = []
    length = min(len(mean_traj), len(discharge))
    for i in range(lag, length):
        feat_mean = mean_traj[i - lag:i]
        feat_min = min_traj[i - lag:i]
        feat_max = max_traj[i - lag:i]
        feat_range = range_traj[i - lag:i]
        wc_mean = wavelet_decompose(mean_traj[i - lag:i + 1], wavelet=wavelet, level=level)
        wc_min = wavelet_decompose(min_traj[i - lag:i + 1], wavelet=wavelet, level=level)
        wc_max = wavelet_decompose(max_traj[i - lag:i + 1], wavelet=wavelet, level=level)
        wc_range = wavelet_decompose(range_traj[i - lag:i + 1], wavelet=wavelet, level=level)
        features = np.concatenate([
            feat_mean, feat_min, feat_max, feat_range,
            wc_mean, wc_min, wc_max, wc_range
        ])
        X_list.append(features)
        y_list.append(discharge[i])
    return np.array(X_list), np.array(y_list)
def objective(trial, X_calib, y_calib):
    n_estimators = trial.suggest_int("n_estimators", 50, 600)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 12)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 5.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 5.0)
    split_idx = int(len(X_calib) * 0.7)
    X_train = X_calib[:split_idx]
    y_train = y_calib[:split_idx]
    X_inner_valid = X_calib[split_idx:]
    y_inner_valid = y_calib[split_idx:]
    if len(X_train) == 0 or len(X_inner_valid) < 10:
        return -9999
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_inner_valid_scaled = scaler_X.transform(X_inner_valid)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train_scaled, y_train_scaled)
    y_inner_pred_scaled = model.predict(X_inner_valid_scaled)
    y_inner_pred = scaler_y.inverse_transform(y_inner_pred_scaled.reshape(-1, 1)).flatten()
    return nse(y_inner_valid, y_inner_pred)
def process_single_gauge(mean_file):
    gauge_id = mean_file.split("_sto_xaj_mean_traj.xlsx")[0]
    mean_path = TRAJ_DIR / mean_file
    quant_path = TRAJ_DIR / f"{gauge_id}_sto_xaj_quantile_traj.xlsx"
    orig_path = DATA_DIR / f"{gauge_id}.csv"
    try:
        if not orig_path.is_file():
            raise FileNotFoundError(f"Basin data file not found: {orig_path}")
        if not quant_path.is_file():
            raise FileNotFoundError(f"Quantile trajectory file not found: {quant_path}")
        mean_df = pd.read_excel(mean_path)
        quant_df = pd.read_excel(quant_path)
        if "mean_traj" not in mean_df.columns:
            raise ValueError("Mean trajectory file must contain a 'mean_traj' column")
        if "min_traj" not in quant_df.columns or "max_traj" not in quant_df.columns:
            raise ValueError("Quantile trajectory file must contain 'min_traj' and 'max_traj' columns")
        mean_traj = mean_df["mean_traj"].to_numpy(dtype=float)
        min_traj = quant_df["min_traj"].to_numpy(dtype=float)
        max_traj = quant_df["max_traj"].to_numpy(dtype=float)
        range_traj = max_traj - min_traj
        df = pd.read_csv(orig_path)
        df.columns = [c.strip().lower() for c in df.columns]
        required_columns = {"date", "discharge"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Basin data is missing columns: {sorted(missing_columns)}")
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        discharge = np.clip(df["discharge"].to_numpy(dtype=float), 0.01, None)
        X_temp, y_temp = create_features(
            mean_traj, min_traj, max_traj, range_traj, discharge,
            lag=LAG, wavelet="coif1", level=2
        )
        n_total = len(y_temp)
        n_calib = int(n_total * CALIBRATION_RATIO)
        if n_total < 20 or n_calib < 2:
            raise ValueError("Insufficient samples for model calibration and validation")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        def optuna_objective(trial):
            wavelet = trial.suggest_categorical("wavelet", WAVELET_FAMILIES)
            level = trial.suggest_int("level", 1, 4)
            X_full, y_full = create_features(
                mean_traj, min_traj, max_traj, range_traj, discharge,
                lag=LAG, wavelet=wavelet, level=level
            )
            X_calib = X_full[:n_calib]
            y_calib = y_full[:n_calib]
            return objective(trial, X_calib, y_calib)
        study.optimize(optuna_objective, n_trials=N_TRIALS, timeout=3600)
        if study.best_value <= -9999:
            raise RuntimeError("Optuna did not find a valid model configuration")
        best_params = study.best_params.copy()
        best_wavelet = best_params.pop("wavelet")
        best_level = best_params.pop("level")
        X, y = create_features(
            mean_traj, min_traj, max_traj, range_traj, discharge,
            lag=LAG, wavelet=best_wavelet, level=best_level
        )
        n_total = len(y)
        n_calib = int(n_total * CALIBRATION_RATIO)
        X_calib = X[:n_calib]
        y_calib = y[:n_calib]
        X_valid = X[n_calib:]
        y_valid = y[n_calib:]
        if len(X_calib) == 0 or len(X_valid) == 0:
            raise ValueError("Calibration or validation set is empty")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_calib_scaled = scaler_X.fit_transform(X_calib)
        X_valid_scaled = scaler_X.transform(X_valid)
        y_calib_scaled = scaler_y.fit_transform(y_calib.reshape(-1, 1)).flatten()
        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **best_params,
        )
        xgb_model.fit(X_calib_scaled, y_calib_scaled)
        y_valid_pred = scaler_y.inverse_transform(
            xgb_model.predict(X_valid_scaled).reshape(-1, 1)
        ).flatten()
        prediction_dates = df.index.to_numpy()[LAG:][n_calib:]
        if len(prediction_dates) != len(y_valid_pred):
            prediction_dates = np.arange(len(y_valid_pred))
        pd.DataFrame({"date": prediction_dates, "y_pred": y_valid_pred}).to_excel(
            OUTPUT_DIR / f"{gauge_id}_xgb_hybrid_full_pred.xlsx", index=False
        )
        result = {
            "gauge_id": gauge_id,
            "NSE_val": round(nse(y_valid, y_valid_pred), 4),
            "KGE_val": round(calc_kge(y_valid, y_valid_pred), 4),
            "RMSE_val": round(calc_rmse(y_valid, y_valid_pred), 4),
            "PBIAS_val": round(calc_pbias(y_valid, y_valid_pred), 4),
            "Best_Wavelet": best_wavelet,
            "Best_Level": best_level,
            "Best_NSE": round(study.best_value, 4),
        }
        return result
    except Exception as exc:
        return {"gauge_id": gauge_id, "error": str(exc)}
def batch_run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Basin data directory not found: {DATA_DIR}")
    if not TRAJ_DIR.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {TRAJ_DIR}")
    mean_files = sorted(
        f.name for f in TRAJ_DIR.iterdir()
        if f.name.endswith("_sto_xaj_mean_traj.xlsx")
    )
    results = []
    errors = []
    for mean_file in mean_files:
        result = process_single_gauge(mean_file)
        if result and "error" in result:
            errors.append(result)
        elif result:
            results.append(result)
    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="NSE_val", ascending=False)
    summary_df.to_csv(RESULTS_SUMMARY, index=False)
    if errors:
        pd.DataFrame(errors, columns=["gauge_id", "error"]).to_csv(ERRORS_FILE, index=False)

if __name__ == "__main__":
    batch_run()
import numpy as np
import pandas as pd
import warnings
import os
import gc
import optuna
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import pywt
import tensorflow as tf
import concurrent.futures
import multiprocessing
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'src' else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / 'data' / 'basins'
MEAN_TRAJ_DIR = PROJECT_ROOT / 'results' / 'sto_xaj_W'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'hybrid_gru_mean'
RESULTS_SUMMARY = OUTPUT_DIR / 'gru_hybrid_mean_batch_summary.csv'
ERRORS_FILE = OUTPUT_DIR / 'errors.csv'
LAG = 3
CALIBRATION_RATIO = 0.7
N_TRIALS = 60
OPTUNA_TIMEOUT = 3600
MAX_WORKERS = min(multiprocessing.cpu_count(), 8)
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
def wavelet_decompose(series, wavelet='db2', level=2):
    coeffs = pywt.wavedec(series, wavelet, level=level)
    return np.concatenate([c.flatten() for c in coeffs])
WAVELET_FAMILIES = [
    'haar',
    'db1', 'db2', 'db3', 'db4',
    'sym2', 'sym3', 'sym4',
    'coif1', 'coif2', 'coif3',
]
def create_features(mean_traj, discharge, lag=LAG, wavelet='db2', level=2):
    X_list = []
    y_list = []
    length = min(len(mean_traj), len(discharge))
    for i in range(lag, length):
        feat_mean = mean_traj[i - lag:i]
        wc_mean = wavelet_decompose(mean_traj[i - lag:i + 1], wavelet=wavelet, level=level)
        features = np.concatenate([feat_mean, wc_mean])
        X_list.append(features)
        y_list.append(discharge[i])
    return np.array(X_list), np.array(y_list)
def objective(trial, X_calib, y_calib):
    split_idx = int(len(X_calib) * 0.7)
    X_train = X_calib[:split_idx]
    y_train = y_calib[:split_idx]
    X_inner_valid = X_calib[split_idx:]
    y_inner_valid = y_calib[split_idx:]
    if len(X_train) == 0 or len(X_inner_valid) < 10:
        return -9999
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_inner_valid_scaled = scaler_X.transform(X_inner_valid)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_inner_valid_scaled = scaler_y.transform(y_inner_valid.reshape(-1, 1))
    X_train_gru = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_inner_valid_gru = X_inner_valid_scaled.reshape(
        X_inner_valid_scaled.shape[0], 1, X_inner_valid_scaled.shape[1]
    )
    gru_units = trial.suggest_int('gru_units', 32, 256, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    model = Sequential([
        Input(shape=(1, X_train_gru.shape[2])),
        GRU(gru_units, activation='tanh', kernel_regularizer=l2(l2_reg), return_sequences=False),
        Dropout(dropout_rate),
        Dense(1),
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    try:
        model.fit(
            X_train_gru,
            y_train_scaled,
            epochs=150,
            batch_size=batch_size,
            validation_data=(X_inner_valid_gru, y_inner_valid_scaled),
            callbacks=[early_stop],
            verbose=0,
            shuffle=True,
        )
        y_inner_pred_scaled = model.predict(X_inner_valid_gru, verbose=0)
        y_inner_pred = scaler_y.inverse_transform(y_inner_pred_scaled).flatten()
        return nse(y_inner_valid, y_inner_pred)
    finally:
        K.clear_session()
        gc.collect()
def process_single_gauge(mean_file, DATA_DIR, MEAN_TRAJ_DIR, OUTPUT_DIR, LAG, WAVELET_FAMILIES):
    tf.config.set_visible_devices([], 'GPU')
    K.clear_session()
    gc.collect()
    gauge_id = mean_file.split('_sto_xaj_mean_traj.xlsx')[0]
    mean_path = MEAN_TRAJ_DIR / mean_file
    orig_path = DATA_DIR / f'{gauge_id}.csv'
    try:
        if not orig_path.is_file():
            raise FileNotFoundError(f'Basin data file not found: {orig_path}')
        mean_df = pd.read_excel(mean_path)
        if 'mean_traj' not in mean_df.columns:
            raise ValueError("Mean trajectory file must contain a 'mean_traj' column")
        mean_traj = mean_df['mean_traj'].to_numpy(dtype=float)
        df = pd.read_csv(orig_path)
        df.columns = [column.strip().lower() for column in df.columns]
        required_columns = {'date', 'discharge'}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f'Basin data is missing columns: {sorted(missing_columns)}')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        discharge = np.clip(df['discharge'].to_numpy(dtype=float), 0.01, None)
        X_temp, y_temp = create_features(mean_traj, discharge, lag=LAG, wavelet='db2', level=2)
        n_total = len(y_temp)
        n_calib = int(n_total * CALIBRATION_RATIO)
        if n_total < 20 or n_calib < 2:
            raise ValueError('Insufficient samples for model calibration and validation')
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        def optuna_objective(trial):
            wavelet = trial.suggest_categorical('wavelet', WAVELET_FAMILIES)
            level = trial.suggest_int('level', 1, 4)
            X_full, y_full = create_features(
                mean_traj, discharge, lag=LAG, wavelet=wavelet, level=level
            )
            X_calib = X_full[:n_calib]
            y_calib = y_full[:n_calib]
            return objective(trial, X_calib, y_calib)
        study.optimize(optuna_objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT)
        if study.best_value <= -9999:
            raise RuntimeError('Optuna did not find a valid model configuration')
        best_params = study.best_params.copy()
        best_wavelet = best_params.pop('wavelet')
        best_level = best_params.pop('level')
        X, y = create_features(mean_traj, discharge, lag=LAG, wavelet=best_wavelet, level=best_level)
        n_total = len(y)
        n_calib = int(n_total * CALIBRATION_RATIO)
        X_calib = X[:n_calib]
        y_calib = y[:n_calib]
        X_valid = X[n_calib:]
        y_valid = y[n_calib:]
        if len(X_calib) == 0 or len(X_valid) == 0:
            raise ValueError('Calibration or validation set is empty')
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_calib_scaled = scaler_X.fit_transform(X_calib)
        X_valid_scaled = scaler_X.transform(X_valid)
        y_calib_scaled = scaler_y.fit_transform(y_calib.reshape(-1, 1))
        y_valid_scaled = scaler_y.transform(y_valid.reshape(-1, 1))
        X_calib_gru = X_calib_scaled.reshape(X_calib_scaled.shape[0], 1, X_calib_scaled.shape[1])
        X_valid_gru = X_valid_scaled.reshape(X_valid_scaled.shape[0], 1, X_valid_scaled.shape[1])
        final_model = Sequential([
            Input(shape=(1, X_calib_gru.shape[2])),
            GRU(
                best_params['gru_units'],
                activation='tanh',
                kernel_regularizer=l2(best_params['l2_reg']),
                return_sequences=False,
            ),
            Dropout(best_params['dropout_rate']),
            Dense(1),
        ])
        optimizer = Adam(learning_rate=best_params['learning_rate'])
        final_model.compile(optimizer=optimizer, loss='mse')
        early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        final_model.fit(
            X_calib_gru,
            y_calib_scaled,
            epochs=200,
            batch_size=best_params['batch_size'],
            validation_data=(X_valid_gru, y_valid_scaled),
            callbacks=[early_stop],
            verbose=0,
            shuffle=True,
        )
        y_valid_pred_scaled = final_model.predict(X_valid_gru, verbose=0)
        y_valid_pred = scaler_y.inverse_transform(y_valid_pred_scaled).flatten()

        prediction_dates = df.index.to_numpy()[LAG:][n_calib:]
        if len(prediction_dates) != len(y_valid_pred):
            prediction_dates = np.arange(len(y_valid_pred))
        pd.DataFrame({'date': prediction_dates, 'y_pred': y_valid_pred}).to_excel(
            OUTPUT_DIR / f'{gauge_id}_gru_hybrid_pred.xlsx',
            index=False,
        )
        result = {
            'gauge_id': gauge_id,
            'NSE_val': round(nse(y_valid, y_valid_pred), 4),
            'KGE_val': round(calc_kge(y_valid, y_valid_pred), 4),
            'RMSE_val': round(calc_rmse(y_valid, y_valid_pred), 4),
            'PBIAS_val': round(calc_pbias(y_valid, y_valid_pred), 4),
            'Best_Wavelet': best_wavelet,
            'Best_Level': best_level,
            'Best_NSE': round(study.best_value, 4),
        }
        return result
    except Exception as exc:
        return {'gauge_id': gauge_id, 'error': str(exc)}
    finally:
        K.clear_session()
        gc.collect()
def batch_run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f'Basin data directory not found: {DATA_DIR}')
    if not MEAN_TRAJ_DIR.is_dir():
        raise FileNotFoundError(f'Mean trajectory directory not found: {MEAN_TRAJ_DIR}')
    mean_files = sorted(
        file_name for file_name in os.listdir(MEAN_TRAJ_DIR)
        if file_name.endswith('_sto_xaj_mean_traj.xlsx')
    )
    results = []
    errors = []
    max_workers = min(multiprocessing.cpu_count(), 8)
    valid_files = []
    for mean_file in mean_files:
        gauge_id = mean_file.split('_sto_xaj_mean_traj.xlsx')[0]
        orig_path = DATA_DIR / f'{gauge_id}.csv'
        if orig_path.is_file():
            valid_files.append(mean_file)
        else:
            errors.append({'gauge_id': gauge_id, 'error': f'Basin data file not found: {orig_path}'})
    chunk_size = max_workers
    chunks = [valid_files[i:i + chunk_size] for i in range(0, len(valid_files), chunk_size)]
    for chunk_idx, chunk in enumerate(chunks, start=1):
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    process_single_gauge,
                    mean_file, DATA_DIR, MEAN_TRAJ_DIR, OUTPUT_DIR, LAG, WAVELET_FAMILIES,
                ): mean_file
                for mean_file in chunk
            }
            for future in concurrent.futures.as_completed(future_to_file):
                mean_file = future_to_file[future]
                try:
                    result = future.result()
                    if result and 'error' in result:
                        errors.append(result)
                    elif result:
                        results.append(result)
                except Exception as exc:
                    gauge_id = mean_file.split('_sto_xaj_mean_traj.xlsx')[0]
                    errors.append({'gauge_id': gauge_id, 'error': str(exc)})
    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by='NSE_val', ascending=False)
    summary_df.to_csv(RESULTS_SUMMARY, index=False)
    pd.DataFrame(errors, columns=['gauge_id', 'error']).to_csv(ERRORS_FILE, index=False)


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')
    batch_run()

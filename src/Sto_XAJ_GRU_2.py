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
LAG = 3
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'src' else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / 'data' / 'basins'
MEAN_TRAJ_DIR = PROJECT_ROOT / 'results' / 'sto_xaj_W'
QUANTILE_TRAJ_DIR = PROJECT_ROOT / 'results' / 'sto_xaj_W'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'hybrid_gru_full'
RESULTS_SUMMARY = OUTPUT_DIR / 'gru_hybrid_full_batch_summary.csv'
ERRORS_FILE = OUTPUT_DIR / 'errors.csv'
def nse(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 2:
        return -9999
    denominator = np.sum((o - np.mean(o)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - np.sum((o - s) ** 2) / denominator
def calc_rmse(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o, s = obs[mask], sim[mask]
    return np.sqrt(np.mean((o - s) ** 2)) if len(o) > 0 else np.nan
def calc_kge(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 2:
        return -9999
    obs_std, obs_mean = np.std(o), np.mean(o)
    if obs_std == 0 or obs_mean == 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    alpha, beta = np.std(s) / obs_std, np.mean(s) / obs_mean
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
def calc_pbias(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o, s = obs[mask], sim[mask]
    denominator = np.sum(o)
    return 100 * np.sum(s - o) / denominator if denominator != 0 else np.nan
def wavelet_decompose(series, wavelet='db2', level=2):
    coeffs = pywt.wavedec(series, wavelet, level=level)
    return np.concatenate([c.flatten() for c in coeffs])
WAVELET_FAMILIES = [
    'haar',
    'db1', 'db2', 'db3', 'db4',
    'sym2', 'sym3', 'sym4',
    'coif1', 'coif2', 'coif3',
]
def create_features(mean_traj, min_traj, max_traj, range_traj, discharge,
                    lag=LAG, wavelet='db2', level=2):
    X_list, y_list = [], []
    length = min(len(mean_traj), len(min_traj), len(max_traj), len(range_traj), len(discharge))
    for i in range(lag, length):
        feat_mean = mean_traj[i - lag:i]
        feat_min = min_traj[i - lag:i]
        feat_max = max_traj[i - lag:i]
        feat_range = range_traj[i - lag:i]
        wc_mean = wavelet_decompose(mean_traj[i - lag:i + 1], wavelet=wavelet, level=level)
        wc_min = wavelet_decompose(min_traj[i - lag:i + 1], wavelet=wavelet, level=level)
        wc_max = wavelet_decompose(max_traj[i - lag:i + 1], wavelet=wavelet, level=level)
        wc_range = wavelet_decompose(range_traj[i - lag:i + 1], wavelet=wavelet, level=level)
        features = np.concatenate([feat_mean, feat_min, feat_max, feat_range,
                                   wc_mean, wc_min, wc_max, wc_range])
        X_list.append(features)
        y_list.append(discharge[i])
    return np.array(X_list), np.array(y_list)
def objective(trial, X_calib, y_calib):
    split_idx = int(len(X_calib) * 0.7)
    X_train, y_train = X_calib[:split_idx], y_calib[:split_idx]
    X_inner_valid, y_inner_valid = X_calib[split_idx:], y_calib[split_idx:]
    if len(X_inner_valid) < 10:
        return -9999
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_inner_valid_scaled = scaler_X.transform(X_inner_valid)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_inner_valid_scaled = scaler_y.transform(y_inner_valid.reshape(-1, 1))
    X_train_gru = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_inner_valid_gru = X_inner_valid_scaled.reshape(
        X_inner_valid_scaled.shape[0], 1, X_inner_valid_scaled.shape[1])
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
        model.fit(X_train_gru, y_train_scaled, epochs=150, batch_size=batch_size,
                  validation_data=(X_inner_valid_gru, y_inner_valid_scaled),
                  callbacks=[early_stop], verbose=0, shuffle=True)
        y_inner_pred_scaled = model.predict(X_inner_valid_gru, verbose=0)
        y_inner_pred = scaler_y.inverse_transform(y_inner_pred_scaled).flatten()
        return nse(y_inner_valid, y_inner_pred)
    finally:
        K.clear_session()
        gc.collect()
def process_single_gauge(mean_file, DATA_DIR, MEAN_TRAJ_DIR, QUANTILE_TRAJ_DIR,
                         OUTPUT_DIR, LAG, WAVELET_FAMILIES):
    tf.config.set_visible_devices([], 'GPU')
    K.clear_session()
    gc.collect()
    gauge_id = mean_file.split('_sto_xaj_mean_traj.xlsx')[0]
    mean_path = MEAN_TRAJ_DIR / mean_file
    quantile_path = QUANTILE_TRAJ_DIR / f'{gauge_id}_sto_xaj_quantile_traj.xlsx'
    orig_path = DATA_DIR / f'{gauge_id}.csv'
    if not orig_path.exists():
        print(f'[{gauge_id}] Original data file not found: {orig_path}')
        return None
    if not quantile_path.exists():
        print(f'[{gauge_id}] Quantile trajectory file not found: {quantile_path}')
        return None
    try:
        print(f'Processing basin: {gauge_id}')
        mean_df, quant_df = pd.read_excel(mean_path), pd.read_excel(quantile_path)
        mean_traj = mean_df['mean_traj'].to_numpy()
        min_traj, max_traj = quant_df['min_traj'].to_numpy(), quant_df['max_traj'].to_numpy()
        range_traj = max_traj - min_traj
        df = pd.read_csv(orig_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        discharge = np.clip(df['discharge'].to_numpy(), 0.01, None)
        X_temp, y_temp = create_features(mean_traj, min_traj, max_traj, range_traj, discharge,
                                         lag=LAG, wavelet='db2', level=2)
        n_total = len(y_temp)
        n_calib = int(n_total * 0.7)
        if n_calib < 2:
            raise ValueError(f'Insufficient samples for calibration: {n_calib}')
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        def optuna_objective(trial):
            wavelet = trial.suggest_categorical('wavelet', WAVELET_FAMILIES)
            level = trial.suggest_int('level', 1, 4)
            X_full, y_full = create_features(mean_traj, min_traj, max_traj, range_traj, discharge,
                                             lag=LAG, wavelet=wavelet, level=level)
            return objective(trial, X_full[:n_calib], y_full[:n_calib])
        study.optimize(optuna_objective, n_trials=60, timeout=3600)
        if study.best_trial is None:
            raise RuntimeError('Optuna did not produce a valid trial.')
        best_params = study.best_params.copy()
        best_wavelet, best_level = best_params.pop('wavelet'), best_params.pop('level')
        X, y = create_features(mean_traj, min_traj, max_traj, range_traj, discharge,
                               lag=LAG, wavelet=best_wavelet, level=best_level)
        n_calib = int(len(y) * 0.7)
        X_calib, y_calib = X[:n_calib], y[:n_calib]
        X_valid, y_valid = X[n_calib:], y[n_calib:]
        if len(X_calib) == 0 or len(X_valid) == 0:
            raise ValueError('Calibration or validation set is empty.')
        scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_calib_scaled = scaler_X.fit_transform(X_calib)
        X_valid_scaled = scaler_X.transform(X_valid)
        y_calib_scaled = scaler_y.fit_transform(y_calib.reshape(-1, 1))
        y_valid_scaled = scaler_y.transform(y_valid.reshape(-1, 1))
        X_calib_gru = X_calib_scaled.reshape(X_calib_scaled.shape[0], 1, X_calib_scaled.shape[1])
        X_valid_gru = X_valid_scaled.reshape(X_valid_scaled.shape[0], 1, X_valid_scaled.shape[1])
        final_model = Sequential([
            Input(shape=(1, X_calib_gru.shape[2])),
            GRU(best_params['gru_units'], activation='tanh',
                kernel_regularizer=l2(best_params['l2_reg']), return_sequences=False),
            Dropout(best_params['dropout_rate']),
            Dense(1),
        ])
        optimizer = Adam(learning_rate=best_params['learning_rate'])
        final_model.compile(optimizer=optimizer, loss='mse')
        early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        print('  Training final GRU-Full model...')
        final_model.fit(X_calib_gru, y_calib_scaled, epochs=200,
                        batch_size=best_params['batch_size'],
                        validation_data=(X_valid_gru, y_valid_scaled),
                        callbacks=[early_stop], verbose=0, shuffle=True)
        y_valid_pred_scaled = final_model.predict(X_valid_gru, verbose=0)
        y_valid_pred = scaler_y.inverse_transform(y_valid_pred_scaled).flatten()
        pd.DataFrame({'y_pred': y_valid_pred}).to_excel(
            OUTPUT_DIR / f'{gauge_id}_gru_full_hybrid_pred.xlsx', index=False)
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
        print(f"  Completed | NSE_val: {result['NSE_val']:.4f} | KGE_val: {result['KGE_val']:.4f} | "
              f"RMSE: {result['RMSE_val']:.2f} | PBIAS: {result['PBIAS_val']:.2f}%")
        return result
    except Exception as e:
        print(f'[{gauge_id}] Processing failed: {e}')
        return None
    finally:
        K.clear_session()
        gc.collect()
def batch_run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MEAN_TRAJ_DIR.exists():
        raise FileNotFoundError(f'Mean trajectory directory not found: {MEAN_TRAJ_DIR}')
    if not DATA_DIR.exists():
        raise FileNotFoundError(f'Basin data directory not found: {DATA_DIR}')
    mean_files = sorted(f.name for f in MEAN_TRAJ_DIR.iterdir()
                        if f.is_file() and f.name.endswith('_sto_xaj_mean_traj.xlsx'))
    print(f'Found {len(mean_files)} mean trajectory files. Starting batch processing...')
    results, errors = [], []
    max_workers = min(multiprocessing.cpu_count(), 8)
    print(f'Using up to {max_workers} worker processes.')
    valid_files = []
    for mean_file in mean_files:
        gauge_id = mean_file.split('_sto_xaj_mean_traj.xlsx')[0]
        orig_path = DATA_DIR / f'{gauge_id}.csv'
        quantile_path = QUANTILE_TRAJ_DIR / f'{gauge_id}_sto_xaj_quantile_traj.xlsx'
        if not orig_path.exists():
            message = f'[{gauge_id}] Original data file not found: {orig_path}'
            print(message)
            errors.append({'gauge_id': gauge_id, 'error': message})
            continue
        if not quantile_path.exists():
            message = f'[{gauge_id}] Quantile trajectory file not found: {quantile_path}'
            print(message)
            errors.append({'gauge_id': gauge_id, 'error': message})
            continue
        valid_files.append(mean_file)
    chunks = [valid_files[i:i + max_workers] for i in range(0, len(valid_files), max_workers)]
    for chunk_idx, chunk in enumerate(chunks, start=1):
        print(f'\n=== Processing batch {chunk_idx}/{len(chunks)} ({len(chunk)} basins) ===')
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_single_gauge, mean_file, DATA_DIR, MEAN_TRAJ_DIR,
                                QUANTILE_TRAJ_DIR, OUTPUT_DIR, LAG, WAVELET_FAMILIES): mean_file
                for mean_file in chunk
            }
            for future in concurrent.futures.as_completed(future_to_file):
                gauge_id = future_to_file[future].split('_sto_xaj_mean_traj.xlsx')[0]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    else:
                        errors.append({'gauge_id': gauge_id, 'error': 'Processing returned no result; see console output.'})
                except Exception as e:
                    print(f'[{gauge_id}] Processing failed: {e}')
                    errors.append({'gauge_id': gauge_id, 'error': str(e)})
    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by='NSE_val', ascending=False)
    summary_df.to_csv(RESULTS_SUMMARY, index=False)
    pd.DataFrame(errors, columns=['gauge_id', 'error']).to_csv(ERRORS_FILE, index=False)
    print(f'\nBatch processing completed. Successful basins: {len(summary_df)}')
    print(f'Summary saved to: {RESULTS_SUMMARY}')
    print(f'Error log saved to: {ERRORS_FILE}')

if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')
    print('=== Running on CPU only ===')
    batch_run()

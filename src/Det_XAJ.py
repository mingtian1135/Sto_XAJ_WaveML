import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import warnings
import os
from pathlib import Path
from numba import njit

warnings.filterwarnings("ignore")

# Project paths. Place input files under the data directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data" / "basins"
STATIC_FILE = PROJECT_ROOT / "data" / "static_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_SUMMARY = RESULTS_DIR / "xaj_batch_results.csv"
CALIB_DIR = RESULTS_DIR / "calibration"
VALID_DIR = RESULTS_DIR / "validation"
ERRORS_FILE = RESULTS_DIR / "errors.csv"

CALIBRATION_RATIO = 0.7
LAG = 3



@njit(fastmath=True, cache=True)
def run_xaj_model_numba(params, p_arr, e_arr, area):
    K, WM, B, C, SM, EX, KG, KI, CS, CI, CG, L = params
    L = int(round(L))
    n_steps = len(p_arr)
    W = WM * 0.7
    S = SM * 0.2
    Q_surface_res = 0.0
    Q_inter_res = 0.0
    Q_ground_res = 0.0
    q_sim_total = np.zeros(n_steps)
    q_sim_surf = np.zeros(n_steps)
    q_sim_inter = np.zeros(n_steps)
    q_sim_ground = np.zeros(n_steps)

    WMM = WM * (1 + B) / (1 - (1 - W / WM) ** (1 / (1 + B))) if W < WM else WM

    for t in range(n_steps):
        P = p_arr[t]
        E = e_arr[t] * K
        PE = P - E

        if PE > 0:
            R = 0.0
            a = WMM * (1 - (1 - W / WM) ** (1 / (1 + B)))
            if a + PE < WMM:
                tmp = (PE + a) / WMM
                R = PE - (WM - W) + WM * (1 - tmp) ** (1 + B)
                if R < 0: R = 0.0
            else:
                R = PE - (WM - W)
            W += (PE - R)
        else:
            R = 0.0
            W += PE
            if W < C * WM:
                W = C * WM

        if W > WM: W = WM
        if W < 0: W = 0.0

        S += R
        RS = RI = RG = 0.0
        if S > SM:
            RS = S - SM
            S = SM
        RI = S * KI
        RG = S * KG
        S = S - RI - RG
        if S < 0: S = 0.0

        U = area / 86.4
        Q_surface_res = Q_surface_res * CS + RS * (1 - CS)
        qs = Q_surface_res * U
        Q_inter_res = Q_inter_res * CI + RI * (1 - CI)
        qi = Q_inter_res * U
        Q_ground_res = Q_ground_res * CG + RG * (1 - CG)
        qg = Q_ground_res * U

        q_sim_surf[t] = qs
        q_sim_inter[t] = qi
        q_sim_ground[t] = qg
        q_sim_total[t] = qs + qi + qg

    if L > 0:
        pad = np.zeros(L)
        q_sim_total = np.concatenate((pad, q_sim_total[:-L]))
        q_sim_surf = np.concatenate((pad, q_sim_surf[:-L]))
        q_sim_inter = np.concatenate((pad, q_sim_inter[:-L]))
        q_sim_ground = np.concatenate((pad, q_sim_ground[:-L]))

    return q_sim_total, q_sim_surf, q_sim_inter, q_sim_ground



def run_xaj_model(params, p_arr, e_arr, area):
    return run_xaj_model_numba(params, p_arr, e_arr, area)


def calc_nse(obs, sim):
    if len(obs) == 0: return -9999
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    if len(o) == 0: return -9999
    numerator = np.sum((o - s) ** 2)
    denominator = np.sum((o - np.mean(o)) ** 2)
    if denominator == 0: return -9999
    return 1 - numerator / denominator


def calc_kge(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    if len(o) < 2: return -9999
    r = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s) / np.std(o)
    beta = np.mean(s) / np.mean(o)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge


def calc_rmse(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    if len(o) == 0: return np.nan
    return np.sqrt(np.mean((o - s) ** 2))


def calc_pbias(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    if len(o) == 0 or np.sum(o) == 0: return np.nan
    return 100 * np.sum(s - o) / np.sum(o)


def objective_function(params, p_cal, e_cal, q_cal, area):
    K, WM, B, C, SM, EX, KG, KI, CS, CI, CG, L = params
    if KG + KI >= 1.0:
        return 1e9
    q_sim, q_surf, q_inter, q_ground = run_xaj_model(params, p_cal, e_cal, area)
    nse = calc_nse(q_cal, q_sim)
    loss = 1 - nse if nse > -9999 else 1e9

    total_vol = np.sum(q_sim)
    surf_vol = np.sum(q_surf)
    if total_vol > 1:
        surf_ratio = surf_vol / total_vol
        if surf_ratio < 0.03:
            loss += (0.03 - surf_ratio) * 150

    if WM < 50 or WM > 300:
        loss += abs(WM - 200) * 0.02
    if B < 0.05 or B > 0.9:
        loss += 8
    if KG < 0.001 or KI < 0.005:
        loss += 15
    if CS < 0.05 or CS > 0.98:
        loss += 5

    return loss



def batch_run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    VALID_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Input basin directory not found: {DATA_DIR}")
    if not STATIC_FILE.is_file():
        raise FileNotFoundError(f"Static data file not found: {STATIC_FILE}")

    static = pd.read_csv(STATIC_FILE)
    static['gauge_id'] = static['gauge_id'].astype(str).str.zfill(8)

    results = []
    gauge_files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv'))

    errors = []

    for i, filename in enumerate(gauge_files, 1):
        gauge_id = filename.replace('.csv', '')
        file_path = DATA_DIR / filename

        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            area_row = static[static['gauge_id'] == gauge_id]
            if area_row.empty:
                errors.append({"gauge_id": gauge_id, "error": "Gauge ID not found in static_data.csv"})
                continue
            area_km2 = float(area_row['area_gages2'].values[0])

            total_length = len(df)
            calib_length = int(total_length * CALIBRATION_RATIO)
            cal_df = df.iloc[:calib_length]
            val_df = df.iloc[calib_length:]

            p_cal = cal_df['plui'].values.astype(np.float64)
            e_cal = cal_df['etp'].values.astype(np.float64)
            q_cal = cal_df['discharge'].values.astype(np.float64)
            p_val = val_df['plui'].values.astype(np.float64)
            e_val = val_df['etp'].values.astype(np.float64)
            q_val = val_df['discharge'].values.astype(np.float64)

            bounds = [
                (0.70, 1.50), (60, 300), (0.05, 0.90), (0.01, 0.25),
                (10, 90), (1.0, 2.0), (0.001, 0.12), (0.005, 0.35),
                (0.10, 0.95), (0.20, 0.90), (0.90, 0.999), (0, 6)
            ]

            result = differential_evolution(
                objective_function,
                bounds,
                args=(p_cal, e_cal, q_cal, area_km2),
                strategy='best1bin',
                maxiter=200,
                popsize=40,
                tol=0.005,
                mutation=(0.5, 1),
                recombination=0.8,
                workers=-1
            )
            best_params = result.x

            q_sim_cal, _, _, _ = run_xaj_model(best_params, p_cal, e_cal, area_km2)
            q_sim_cal = q_sim_cal[LAG:]
            q_cal_aligned = q_cal[LAG:]

            q_sim_val, _, _, _ = run_xaj_model(best_params, p_val, e_val, area_km2)


            nse_cal = calc_nse(q_cal_aligned, q_sim_cal)
            kge_cal = calc_kge(q_cal_aligned, q_sim_cal)
            rmse_cal = calc_rmse(q_cal_aligned, q_sim_cal)
            pbias_cal = calc_pbias(q_cal_aligned, q_sim_cal)

            nse_val = calc_nse(q_val, q_sim_val)
            kge_val = calc_kge(q_val, q_sim_val)
            rmse_val = calc_rmse(q_val, q_sim_val)
            pbias_val = calc_pbias(q_val, q_sim_val)

            calib_file = CALIB_DIR / f"{gauge_id}_calib_pred.xlsx"
            valid_file = VALID_DIR / f"{gauge_id}_valid_pred.xlsx"

            pd.DataFrame({'y_calib_pred': q_sim_cal}).to_excel(calib_file, index=False)
            pd.DataFrame({'y_pred': q_sim_val}).to_excel(valid_file, index=False)

            results.append({
                'gauge_id': gauge_id,
                'area_km2': area_km2,
                'NSE_cal': round(nse_cal, 4),
                'KGE_cal': round(kge_cal, 4),
                'RMSE_cal': round(rmse_cal, 4),
                'PBIAS_cal': round(pbias_cal, 4),
                'NSE_val': round(nse_val, 4),
                'KGE_val': round(kge_val, 4),
                'RMSE_val': round(rmse_val, 4),
                'PBIAS_val': round(pbias_val, 4)
            })


        except Exception as exc:
            errors.append({"gauge_id": gauge_id, "error": str(exc)})

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(by="NSE_val", ascending=False)
    result_df.to_csv(RESULTS_SUMMARY, index=False)


    pd.DataFrame(errors, columns=["gauge_id", "error"]).to_csv(ERRORS_FILE, index=False)



if __name__ == '__main__':
    batch_run()

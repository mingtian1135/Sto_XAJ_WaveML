import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing
from scipy.optimize import differential_evolution
from scipy.stats import levy_stable
import warnings
from numba import njit
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data" / "basins"
STATIC_FILE = PROJECT_ROOT / "data" / "static_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "sto_xaj_qs"
ERRORS_FILE = OUTPUT_DIR / "errors.csv"
CALIBRATION_RATIO = 0.7
N_TRAJ = 1000
LAG = 3
ALPHA_RANGE = (0.8, 2.0)
BETA_RANGE = (-1.0, 1.0)
@njit(fastmath=True, cache=True)
def run_xaj_deterministic(params, p_arr, e_arr, area):
    K, WM, B, C, SM, EX, KG, KI, CS, CI, CG, L = params
    L = int(round(L))
    n_steps = len(p_arr)
    W = WM * 0.7
    S = SM * 0.2
    Q_surface_res = 0.0
    Q_inter_res = 0.0
    Q_ground_res = 0.0
    q_sim_total = np.zeros(n_steps)
    WMM = WM * (1 + B) / (1 - (1 - W / WM) ** (1 / (1 + B))) if W < WM else WM
    for t in range(n_steps):
        P = p_arr[t]
        E = e_arr[t] * K
        PE = P - E
        R = 0.0
        if PE > 0:
            if W >= WM:
                R = PE
            else:
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
        q_sim_total[t] = qs + qi + qg
    if L > 0:
        pad = np.zeros(L)
        q_sim_total = np.concatenate((pad, q_sim_total[:-L]))
    return q_sim_total

def run_xaj_model_stochastic(params, p_arr, e_arr, area, is_stochastic=False, sigma=0.0, noise_series=None):
    K, WM, B, C, SM, EX, KG, KI, CS, CI, CG, L = params
    L = int(round(L))
    n_steps = len(p_arr)
    W = WM * 0.7
    S = SM * 0.2
    Q_surface_res = 0.0
    Q_inter_res = 0.0
    Q_ground_res = 0.0
    q_sim_total = np.zeros(n_steps)
    try:
        WMM = WM * (1 + B) / (1 - (1 - W / WM) ** (1 / (1 + B))) if W < WM else WM
    except:
        WMM = WM
    for t in range(n_steps):
        P = p_arr[t]
        E = e_arr[t] * K
        PE = P - E
        R = 0
        if PE > 0:
            if W >= WM:
                R = PE
            else:
                a = WMM * (1 - (1 - W / WM) ** (1 / (1 + B)))
                if a + PE < WMM:
                    R = PE - (WM - W) + WM * (1 - (PE + a) / WMM) ** (1 + B)
                    if R < 0: R = 0
                else:
                    R = PE - (WM - W)
            W += (PE - R)
        else:
            R = 0
            W += PE
            if W < C * WM:
                W = C * WM
        if W > WM: W = WM
        if W < 0: W = 0
        S += R
        RS = RI = RG = 0
        if S > SM:
            RS = S - SM
            S = SM
        RI = S * KI
        RG = S * KG
        S = S - RI - RG
        if S < 0: S = 0
        U = area / 86.4
        Q_surface_res = Q_surface_res * CS + RS * (1 - CS)
        qs = Q_surface_res * U
        if is_stochastic and noise_series is not None:
            qs = qs + sigma * noise_series[t]
            if qs < 0:
                qs = 0.0
        Q_inter_res = Q_inter_res * CI + RI * (1 - CI)
        qi = Q_inter_res * U
        Q_ground_res = Q_ground_res * CG + RG * (1 - CG)
        qg = Q_ground_res * U
        q_sim_total[t] = qs + qi + qg
    if L > 0:
        q_sim_total = np.concatenate([np.zeros(L), q_sim_total[:-L]])
    return q_sim_total
def run_sto_xaj_mean(params, p_arr, e_arr, n_traj, area):
    xaj_params = params[:12]
    sigma = params[12]
    alpha_levy = params[13]
    beta_levy = params[14]
    n_steps = len(p_arr)
    levy_dist = levy_stable(alpha=alpha_levy, beta=beta_levy, loc=0.0)
    noise_traj = levy_dist.rvs(size=n_steps * n_traj).reshape(n_steps, n_traj)
    traj_sims = np.zeros((n_steps, n_traj))
    for j in range(n_traj):
        q_total = run_xaj_model_stochastic(
            xaj_params, p_arr, e_arr, area,
            is_stochastic=True, sigma=sigma, noise_series=noise_traj[:, j]
        )
        traj_sims[:, j] = q_total
    return np.mean(traj_sims, axis=1), traj_sims
def calc_nse(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    if len(o) == 0: return -9999
    return 1 - np.sum((o - s) ** 2) / np.sum((o - np.mean(o)) ** 2)
def calc_kge(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    if len(o) < 2: return -9999
    r = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s) / np.std(o)
    beta = np.mean(s) / np.mean(o)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
def calc_rmse(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    return np.sqrt(np.mean((o - s) ** 2)) if len(o) > 0 else np.nan
def calc_pbias(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    return 100 * np.sum(s - o) / np.sum(o) if np.sum(o) != 0 else np.nan
def objective_function(params, p_cal, e_cal, q_cal, area):
    kg = params[6]
    ki = params[7]
    if kg + ki >= 1.0:
        return 1e9
    mean_traj, _ = run_sto_xaj_mean(params, p_cal, e_cal, n_traj=20, area=area)
    nse_val = calc_nse(q_cal, mean_traj)
    return 1 - nse_val if nse_val > -9999 else 1e9
def process_single_gauge(gauge_id, file_path, static):
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.strip().lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        area_row = static[static['gauge_id'] == gauge_id]
        if area_row.empty:
            return gauge_id, None, 'Gauge area not found in static data'
        area_km2 = float(area_row['area_gages2'].values[0])
        total_length = len(df)
        calib_length = int(total_length * CALIBRATION_RATIO)
        full_p = df['plui'].values
        full_e = df['etp'].values
        full_q = np.clip(df['discharge'].values, 0.01, None)
        full_date = df.index
        p_cal = full_p[:calib_length]
        e_cal = full_e[:calib_length]
        q_cal = full_q[:calib_length]
        bounds = [
            (0.70, 1.50), (60, 300), (0.05, 0.90), (0.01, 0.25),
            (10, 90), (1.0, 2.0), (0.001, 0.12), (0.005, 0.35),
            (0.10, 0.95), (0.20, 0.90), (0.90, 0.999), (0, 6),
            (0.001, 20.0), ALPHA_RANGE, BETA_RANGE
        ]
        result = differential_evolution(
            objective_function,
            bounds,
            args=(p_cal, e_cal, q_cal, area_km2),
            strategy='best1bin',
            maxiter=50,
            popsize=20,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.8,
            workers=1,
            disp=False
        )
        best_params = result.x
        full_mean, full_sims = run_sto_xaj_mean(best_params, full_p, full_e, N_TRAJ, area_km2)
        val_start = calib_length
        q_sim_val = full_mean[val_start:]
        q_val = full_q[val_start:]
        nse_val = calc_nse(q_val, q_sim_val)
        kge_val = calc_kge(q_val, q_sim_val)
        rmse_val = calc_rmse(q_val, q_sim_val)
        pbias_val = calc_pbias(q_val, q_sim_val)
        prefix = f"{gauge_id}_"
        pd.DataFrame({'date': full_date, 'mean_traj': full_mean}).to_excel(
            OUTPUT_DIR / f"{prefix}sto_xaj_mean_traj.xlsx", index=False)
        pd.DataFrame({
            'date': full_date,
            'min_traj': np.percentile(full_sims, 2.5, axis=1),
            'max_traj': np.percentile(full_sims, 97.5, axis=1)
        }).to_excel(OUTPUT_DIR / f"{prefix}sto_xaj_quantile_traj.xlsx", index=False)
        return gauge_id, {
            'gauge_id': gauge_id,
            'area_km2': area_km2,
            'NSE_cal': round(1.0 - result.fun, 4),
            'NSE_val': round(nse_val, 4),
            'KGE_val': round(kge_val, 4),
            'RMSE_val': round(rmse_val, 4),
            'PBIAS_val': round(pbias_val, 4)
        }, None
    except Exception as e:
        return gauge_id, None, str(e)
def batch_run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Input data directory not found: {DATA_DIR}")
    if not STATIC_FILE.is_file():
        raise FileNotFoundError(f"Static data file not found: {STATIC_FILE}")

    static = pd.read_csv(STATIC_FILE, dtype={"gauge_id": str})
    static['gauge_id'] = static['gauge_id'].str.replace('camels_', '', regex=False).str.strip().str.zfill(8)
    gauge_files = [f for f in [path.name for path in DATA_DIR.iterdir()] if f.endswith('.csv')]
    tasks = [(f.replace('.csv', '').strip(), DATA_DIR / f, static) for f in gauge_files]
    n_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=n_cores, backend='loky', verbose=5)(
        delayed(process_single_gauge)(gid, path, static) for gid, path, static in tasks
    )
    summary_data = [res[1] for res in results if res[1] is not None]
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="NSE_val", ascending=False)
    summary_df.to_csv(OUTPUT_DIR / "sto_xaj_qs_batch_summary.csv", index=False)

    errors = [
        {"gauge_id": res[0], "error": res[2]}
        for res in results
        if len(res) > 2 and res[2] is not None
    ]
    if errors:
        pd.DataFrame(errors).to_csv(ERRORS_FILE, index=False)
if __name__ == '__main__':
    batch_run()

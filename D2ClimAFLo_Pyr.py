# ================================================================
# Projet : D2ClimAFLo-Pyr
# Auteur : Amadou FOFANA
# Stage  : UPVD / CEFREM (2026)
# Script : Script unifie — Pipeline + Application Streamlit
#
# Usage :
#   MODE = "pipeline" → exécute le pipeline complet
#                        (CHELSA, dataset PR, downscaling, validation)
#   MODE = "app"      → lance l'application Streamlit
#                        commande : streamlit run ce_fichier.py
# ================================================================

import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Mode d'exécution (modifier ici) ──────────────────────────
MODE = "pipeline"   # "pipeline" ou "app"
# ─────────────────────────────────────────────────────────────

DOSSIER = "/content/drive/MyDrive/Colab Notebooks/dataStage/"
DOSSIER_MF = os.path.join(
    DOSSIER,
    "InSituStations/MeteoFrance_reseauProfessionnel/"
    "raw_Data_MeteoFrance_stations/"
)
MNT_PATH = os.path.join(DOSSIER, "MNT/MNT_Pyrenees_30m.tif")
OUT_DIR  = os.path.join(DOSSIER, "downscaling_25m")

mois_lbls = ['Jan','Fev','Mar','Avr','Mai','Jun',
             'Jul','Aou','Sep','Oct','Nov','Dec']


# ================================================================
# SECTION 0 — FONCTIONS COMMUNES
# ================================================================

def monter_drive():
    from google.colab import drive
    drive.mount('/content/drive')
    os.makedirs(OUT_DIR, exist_ok=True)


def calculer_pente_exposition(mnt_path):
    """
    Calcule pente (degrés) et exposition (0-360°, N=0)
    depuis un MNT rasterio.
    Retourne : pente_arr, expo_arr, transform, crs
    """
    import rasterio
    with rasterio.open(mnt_path) as src:
        elev      = src.read(1).astype(np.float32)
        transform = src.transform
        crs       = src.crs
        res_x     = abs(transform.a)
        res_y     = abs(transform.e)

    elev[elev < -9000] = np.nan
    dz_dy, dz_dx = np.gradient(elev, res_y, res_x)

    pente_arr = np.degrees(
        np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    )
    expo_arr = np.degrees(np.arctan2(-dz_dx, dz_dy))
    expo_arr = (expo_arr + 360) % 360

    print(f"  Pente : moy={np.nanmean(pente_arr):.1f} deg  "
          f"max={np.nanmax(pente_arr):.1f} deg")
    return pente_arr, expo_arr, transform, crs


def extraire_topo(lon_arr, lat_arr,
                  pente_arr, expo_arr, transform):
    """Extrait pente et exposition pour des points (lon, lat)."""
    from rasterio.transform import rowcol
    pente_pts, expo_pts = [], []
    for lon, lat in zip(lon_arr, lat_arr):
        try:
            row, col = rowcol(transform, lon, lat)
            row = int(np.clip(row, 0, pente_arr.shape[0] - 1))
            col = int(np.clip(col, 0, pente_arr.shape[1] - 1))
            pente_pts.append(float(pente_arr[row, col]))
            expo_pts.append(float(expo_arr[row, col]))
        except Exception:
            pente_pts.append(np.nan)
            expo_pts.append(np.nan)
    return np.array(pente_pts), np.array(expo_pts)


def encoder_exposition(expo_deg):
    """Encodage cyclique sin + cos de l'exposition."""
    rad = np.deg2rad(expo_deg)
    return np.sin(rad), np.cos(rad)


def extraire_alt_mnt(lon_arr, lat_arr, mnt_path):
    """Extrait l'altitude MNT pour une liste de points."""
    import rasterio
    with rasterio.open(mnt_path) as src:
        coords = list(zip(lon_arr, lat_arr))
        vals   = [v[0] for v in src.sample(coords)]
    return np.array(vals, dtype=float)


def entrainer_rf_cv(X, y, groups, n_splits=5,
                    n_estimators=200, random_state=42):
    """
    GroupKFold CV puis entraînement final.
    Retourne : rf_final, cv_results, y_pred_cv
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import (mean_absolute_error,
                                 mean_squared_error, r2_score)

    gkf        = GroupKFold(n_splits=n_splits)
    cv_results = {'mae': [], 'rmse': [], 'r2': []}
    y_pred_cv  = np.full(len(y), np.nan)

    for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(X, y, groups), 1):
        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=20,
            min_samples_leaf=3, n_jobs=-1,
            random_state=random_state
        )
        rf.fit(X[train_idx], y[train_idx])
        pred = rf.predict(X[test_idx])
        y_pred_cv[test_idx] = pred

        mae  = mean_absolute_error(y[test_idx], pred)
        rmse = np.sqrt(mean_squared_error(y[test_idx], pred))
        r2   = r2_score(y[test_idx], pred)
        cv_results['mae'].append(mae)
        cv_results['rmse'].append(rmse)
        cv_results['r2'].append(r2)

        n_st = len(np.unique(groups[test_idx]))
        print(f"    Fold {fold_idx} : MAE={mae:.2f}  "
              f"RMSE={rmse:.2f}  R2={r2:.3f}  "
              f"({n_st} stations)")

    rf_final = RandomForestRegressor(
        n_estimators=300, max_depth=20,
        min_samples_leaf=3, n_jobs=-1, random_state=42
    )
    rf_final.fit(X, y)
    return rf_final, cv_results, y_pred_cv


def afficher_importance(rf, features, titre=""):
    """Affiche l'importance des features."""
    imp = pd.DataFrame({
        'feature'   : features,
        'importance': rf.feature_importances_,
    }).sort_values('importance', ascending=False)
    label = f" — {titre}" if titre else ""
    print(f"\n  Importance features{label} :")
    for _, row in imp.iterrows():
        bar = "=" * int(row['importance'] * 40)
        print(f"    {row['feature']:<22} {bar} "
              f"{row['importance']:.1%}")
    return imp


def charger_chelsa_sites(dossier):
    """Charge le fichier CHELSA et retourne les sites."""
    df = pd.read_csv(
        os.path.join(dossier, "chelsa_par_point_complet.csv")
    )
    df['date'] = pd.to_datetime(df['date'])
    sites = df[df['type'] == 'SITE'].copy()
    sites['altitude'] = sites['altitude_meta']
    return df, sites


# ================================================================
# SECTION 1 — PIPELINE CHELSA (exploration)
# ================================================================

def run_chelsa_exploration():
    """Exploration et correction des données CHELSA."""
    print("\nSection 1 : Exploration CHELSA")

    SITES_ALT = {
        'NOHEDES': 1790, 'CADI_POP1': 2357,
        'CADI_POP2': 2306, 'CADI_POP3': 1990,
        'CADI_POP4': 2394, 'EYNE_POP1': 2655,
        'EYNE_POP2': 2215, 'EYNE_POP3': 2078,
        'VALLTER': 2148
    }

    # Affichage altitudes
    print("\n  Altitudes officielles :")
    for site, alt in sorted(SITES_ALT.items(),
                            key=lambda x: x[1], reverse=True):
        marker = " <- NOHEDES" if site == "NOHEDES" else ""
        print(f"    {site:<12} : {alt} m{marker}")

    # Chargement CHELSA
    chemin = os.path.join(
        DOSSIER, "chelsa_fusion_T_PR_2000_2020.csv"
    )
    if not os.path.exists(chemin):
        print(f"  Fichier non trouve : {chemin}")
        return

    df_chelsa = pd.read_csv(chemin)
    print(f"\n  CHELSA — shape : {df_chelsa.shape}")

    # Stats T° par site
    cols_T = [c for c in df_chelsa.columns if c.startswith('T_')]
    print(f"\n  Temperature moyenne par site (deg C) :")
    print(f"  {'Site':<15} {'Moy':>8} {'Min':>8} {'Max':>8}")
    for col in cols_T:
        site = col.replace('T_', '')
        v = df_chelsa[col].dropna()
        print(f"  {site:<15} {v.mean():>8.2f} "
              f"{v.min():>8.2f} {v.max():>8.2f}")

    print("\n  CHELSA exploration terminee")


# ================================================================
# SECTION 2 — PRÉPARATION DATASET PR
# ================================================================

def run_preparation_dataset_pr():
    """Construit le dataset training RF précipitations."""
    import rasterio
    from tqdm import tqdm

    print("\nSection 2 : Preparation dataset PR (MF + AEMET)")

    # Catalogue
    cat    = pd.read_csv(
        os.path.join(DOSSIER, "catalogue_stations_meteo.csv")
    )
    cat_pr = cat[cat['has_precip'] == True].copy()

    mask_niv    = cat_pr['station_name'].str.contains(
        'NIVOSE|NIVO|_CLIM', case=False, regex=True, na=False
    )
    nivose_list = cat_pr[mask_niv]['station_name'].tolist()
    cat_pr      = cat_pr[~mask_niv].copy()
    print(f"  Stations PR (sans NIVOSE) : {len(cat_pr)}")

    # Agrégation mensuelle MF
    dfs_mensuel = []
    n_ok, n_skip = 0, 0

    for _, row in tqdm(cat_pr.iterrows(),
                       total=len(cat_pr), desc="MF-PR"):
        fic = os.path.join(DOSSIER_MF, row['fichier'])
        if not os.path.exists(fic):
            n_skip += 1
            continue
        try:
            df_j = pd.read_csv(fic, usecols=[
                'station_id', 'date', 'precipitation_mm',
                'quality_precipitation',
                'station_name', 'latitude', 'longitude',
                'altitude'
            ])
        except Exception:
            try:
                df_j = pd.read_csv(fic)
            except Exception:
                n_skip += 1
                continue

        df_j['date'] = pd.to_datetime(df_j['date'],
                                       errors='coerce')
        df_j = df_j.dropna(subset=['date'])
        df_j = df_j[
            (df_j['date'] >= '2000-01-01') &
            (df_j['date'] <= '2021-12-31')
        ].dropna(subset=['precipitation_mm'])

        if len(df_j) == 0:
            n_skip += 1
            continue

        if 'quality_precipitation' in df_j.columns:
            df_j = df_j[
                df_j['quality_precipitation'].isin([1,2,5,6])
            ].copy()
        if len(df_j) == 0:
            n_skip += 1
            continue

        df_j['annee'] = df_j['date'].dt.year
        df_j['mois']  = df_j['date'].dt.month
        df_m = df_j.groupby(['annee','mois']).agg(
            pr_obs_mm = ('precipitation_mm', 'sum'),
            n_jours   = ('precipitation_mm', 'count'),
        ).reset_index()
        df_m = df_m[df_m['n_jours'] >= 25].copy()
        if len(df_m) == 0:
            n_skip += 1
            continue

        df_m['nom']        = row['station_name']
        df_m['station_id'] = str(row['station_id'])
        df_m['lat']        = row['latitude']
        df_m['lon']        = row['longitude']
        df_m['altitude']   = row['altitude']
        df_m['date']       = pd.to_datetime(
            df_m['annee'].astype(str) + '-' +
            df_m['mois'].astype(str).str.zfill(2) + '-15'
        )
        dfs_mensuel.append(df_m)
        n_ok += 1

    df_mf = pd.concat(dfs_mensuel, ignore_index=True)
    print(f"\n  MF : {n_ok} stations | {len(df_mf):,} obs")

    # Exclusions supplémentaires
    stats_st = (df_mf.groupby('nom')['pr_obs_mm']
                .agg(['mean','count']).reset_index())
    stats_st.columns = ['nom','pr_moy','n_obs']
    st_pr_faible = stats_st[stats_st['pr_moy'] < 5]['nom'].tolist()

    df_chelsa_ref = pd.read_csv(
        os.path.join(DOSSIER, "chelsa_par_point_complet.csv")
    )
    df_chelsa_ref['date'] = pd.to_datetime(df_chelsa_ref['date'])
    df_chelsa_st = df_chelsa_ref[
        df_chelsa_ref['type'] == 'STATION'
    ].copy()

    df_check = df_mf.merge(
        df_chelsa_st[['nom','date','pr_chelsa_mm']],
        on=['nom','date'], how='inner'
    )
    ratio = df_check.groupby('nom').apply(
        lambda x: x['pr_obs_mm'].mean() /
        x['pr_chelsa_mm'].mean()
        if x['pr_chelsa_mm'].mean() > 0 else 0
    ).reset_index()
    ratio.columns = ['nom','ratio']
    st_ratio_bas = ratio[ratio['ratio'] < 0.10]['nom'].tolist()

    excl = set(st_pr_faible) | set(st_ratio_bas)
    df_mf_final = df_mf[~df_mf['nom'].isin(excl)].copy()

    # AEMET
    aemet_meta = {
        'AEMET_MARTINET':   {'fname': 'aemet_martinet_2000_2020.csv',
                             'lat': 42.362,'lon': 1.693,'altitude': 1038},
        'AEMET_CAP_DE_REC': {'fname': 'aemet_cap_de_rec_2000_2020.csv',
                             'lat': 42.430,'lon': 1.667,'altitude': 1940},
        'AEMET_PLANOLES':   {'fname': 'aemet_planoles_2000_2020.csv',
                             'lat': 42.317,'lon': 2.103,'altitude': 1151},
    }
    dfs_aemet = []
    for nom_st, info in aemet_meta.items():
        chemin = os.path.join(DOSSIER, info['fname'])
        if not os.path.exists(chemin):
            continue
        df_j = pd.read_csv(chemin)
        df_j['DATE'] = pd.to_datetime(df_j['DATE'])
        col_pr = next(
            (c for c in ['PRECIP_MM','PR_MM','PRECIPITATION',
                         'precipitation','RR','precip_mm','PRCP']
             if c in df_j.columns), None
        )
        if col_pr is None:
            continue
        df_j = df_j.rename(columns={col_pr: 'PRECIP_MM'}).dropna(
            subset=['PRECIP_MM']
        )
        df_j['annee'] = df_j['DATE'].dt.year
        df_j['mois']  = df_j['DATE'].dt.month
        df_m = df_j.groupby(['annee','mois']).agg(
            pr_obs_mm=('PRECIP_MM','sum'),
            n_jours  =('PRECIP_MM','count'),
        ).reset_index()
        df_m = df_m[df_m['n_jours'] >= 25].copy()
        if len(df_m) == 0:
            continue
        df_m['nom']        = nom_st
        df_m['station_id'] = nom_st
        df_m['lat']        = info['lat']
        df_m['lon']        = info['lon']
        df_m['altitude']   = info['altitude']
        df_m['date']       = pd.to_datetime(
            df_m['annee'].astype(str) + '-' +
            df_m['mois'].astype(str).str.zfill(2) + '-15'
        )
        dfs_aemet.append(df_m)

    cols = ['nom','station_id','date','annee','mois',
            'pr_obs_mm','lat','lon','altitude']
    df_mf_std          = df_mf_final[cols].copy()
    df_mf_std['source'] = 'METEOFRANCE'

    if dfs_aemet:
        df_aemet           = pd.concat(dfs_aemet, ignore_index=True)
        df_aemet_std       = df_aemet[cols].copy()
        df_aemet_std['source'] = 'AEMET'
        df_obs = pd.concat([df_mf_std, df_aemet_std],
                           ignore_index=True)
    else:
        df_obs = df_mf_std

    # Join CHELSA
    df_join = df_obs.merge(
        df_chelsa_st[['nom','date','tas_chelsa_C','pr_chelsa_mm']],
        on=['nom','date'], how='inner'
    )

    # Alt MNT
    with rasterio.open(MNT_PATH) as src:
        coords = list(zip(df_join['lon'].values,
                          df_join['lat'].values))
        df_join['alt_mnt'] = [v[0] for v in src.sample(coords)]

    df_join['delta_alt']     = df_join['altitude'] - df_join['alt_mnt']
    df_join['mois_sin']      = np.sin(2*np.pi*df_join['mois']/12)
    df_join['mois_cos']      = np.cos(2*np.pi*df_join['mois']/12)
    df_join['pr_chelsa_log'] = np.log1p(
        df_join['pr_chelsa_mm'].clip(lower=0)
    )
    df_join['pr_obs_log'] = np.log1p(
        df_join['pr_obs_mm'].clip(lower=0)
    )
    df_join = df_join.dropna(subset=[
        'pr_obs_mm','pr_chelsa_mm','altitude','alt_mnt'
    ])

    df_join.to_csv(
        os.path.join(DOSSIER, "training_RF_PR_chelsa_obs.csv"),
        index=False
    )
    print(f"  Dataset PR sauvegarde : "
          f"{len(df_join):,} obs | "
          f"{df_join['nom'].nunique()} stations")


# ================================================================
# SECTION 3 — DOWNSCALING RF (T° + PR + Neige)
# ================================================================

def run_downscaling():
    """Pipeline complet downscaling RF pour T°, PR et Neige."""
    import rasterio
    import xarray as xr
    from tqdm import tqdm
    from sklearn.metrics import mean_absolute_error, r2_score

    print("\nSection 3 : Downscaling RF (T + PR + Neige)")

    # Calcul pente + exposition
    print("\n  Calcul pente et exposition depuis le MNT...")
    pente_arr, expo_arr, mnt_tr, _ = calculer_pente_exposition(
        MNT_PATH
    )

    # Chargement CHELSA sites
    df_chelsa, sites_ref = charger_chelsa_sites(DOSSIER)

    # Prépare les sites avec topo
    def prep_sites_topo(sites):
        s = sites.copy()
        s['alt_mnt'] = extraire_alt_mnt(
            s['lon'].values, s['lat'].values, MNT_PATH
        )
        p, e = extraire_topo(
            s['lon'].values, s['lat'].values,
            pente_arr, expo_arr, mnt_tr
        )
        s['pente']   = p
        es, ec = encoder_exposition(e)
        s['expo_sin'] = es
        s['expo_cos'] = ec
        s['delta_alt'] = s['altitude'] - s['alt_mnt']
        s['mois_sin']  = np.sin(2*np.pi*s['mois']/12)
        s['mois_cos']  = np.cos(2*np.pi*s['mois']/12)
        return s

    # ── 3A : TEMPÉRATURE ─────────────────────────────────────
    print("\n  [T] Downscaling temperature...")

    df_T = pd.read_csv(
        os.path.join(DOSSIER, "training_RF_T_chelsa_obs.csv")
    )
    df_T['date'] = pd.to_datetime(df_T['date'])

    pt, et = extraire_topo(
        df_T['lon'].values, df_T['lat'].values,
        pente_arr, expo_arr, mnt_tr
    )
    df_T['pente']   = pt
    es, ec = encoder_exposition(et)
    df_T['expo_sin'] = es
    df_T['expo_cos'] = ec
    if 'mois_sin' not in df_T.columns:
        df_T['mois_sin'] = np.sin(2*np.pi*df_T['mois']/12)
        df_T['mois_cos'] = np.cos(2*np.pi*df_T['mois']/12)
    if 'delta_alt' not in df_T.columns:
        df_T['delta_alt'] = df_T['altitude'] - df_T['alt_mnt']

    FEAT_T = ['tas_chelsa_C',
              'altitude','alt_mnt','delta_alt',
              'pente','expo_sin','expo_cos',
              'lat','lon','mois_sin','mois_cos']

    df_T = df_T.dropna(subset=FEAT_T + ['temp_moy_c'])
    X_T, y_T = df_T[FEAT_T].values, df_T['temp_moy_c'].values

    print(f"    {len(df_T):,} obs | {df_T['nom'].nunique()} stations")
    rf_T, cv_T, _ = entrainer_rf_cv(X_T, y_T, df_T['nom'].values)
    print(f"\n    MAE={np.mean(cv_T['mae']):.2f}C  "
          f"R2={np.mean(cv_T['r2']):.3f}")
    afficher_importance(rf_T, FEAT_T, "Temperature")

    joblib.dump({'model': rf_T, 'features': FEAT_T,
                 'cv_results': cv_T},
                os.path.join(DOSSIER, "model_RF_T_chelsa.pkl"))

    # Prédiction sites
    s_T = prep_sites_topo(sites_ref.copy())
    s_T = s_T.dropna(subset=FEAT_T)
    s_T['temp_RF'] = rf_T.predict(s_T[FEAT_T])
    s_T[['nom','lat','lon','altitude','alt_mnt',
          'date','annee','mois','tas_chelsa_C','temp_RF']
         ].to_csv(
        os.path.join(DOSSIER, "predictions_RF_T_sites.csv"),
        index=False
    )
    print(f"    predictions_RF_T_sites.csv ({len(s_T):,} obs)")

    # ── 3B : PRÉCIPITATIONS ──────────────────────────────────
    print("\n  [PR] Downscaling precipitations...")

    df_PR = pd.read_csv(
        os.path.join(DOSSIER, "training_RF_PR_chelsa_obs.csv")
    )
    df_PR['date'] = pd.to_datetime(df_PR['date'])

    pp, ep = extraire_topo(
        df_PR['lon'].values, df_PR['lat'].values,
        pente_arr, expo_arr, mnt_tr
    )
    df_PR['pente'] = pp
    es, ec = encoder_exposition(ep)
    df_PR['expo_sin'] = es
    df_PR['expo_cos'] = ec
    if 'mois_sin' not in df_PR.columns:
        df_PR['mois_sin'] = np.sin(2*np.pi*df_PR['mois']/12)
        df_PR['mois_cos'] = np.cos(2*np.pi*df_PR['mois']/12)
    if 'delta_alt' not in df_PR.columns:
        df_PR['delta_alt'] = df_PR['altitude'] - df_PR['alt_mnt']
    if 'pr_chelsa_log' not in df_PR.columns:
        df_PR['pr_chelsa_log'] = np.log1p(
            df_PR['pr_chelsa_mm'].clip(lower=0)
        )

    FEAT_PR = ['pr_chelsa_mm','pr_chelsa_log','tas_chelsa_C',
               'altitude','alt_mnt','delta_alt',
               'pente','expo_sin','expo_cos',
               'lat','lon','mois_sin','mois_cos']

    df_PR = df_PR.dropna(subset=FEAT_PR + ['pr_obs_mm'])
    X_PR, y_PR = df_PR[FEAT_PR].values, df_PR['pr_obs_mm'].values

    print(f"    {len(df_PR):,} obs | {df_PR['nom'].nunique()} stations")
    rf_PR, cv_PR, _ = entrainer_rf_cv(X_PR, y_PR, df_PR['nom'].values)
    print(f"\n    MAE={np.mean(cv_PR['mae']):.1f}mm  "
          f"R2={np.mean(cv_PR['r2']):.3f}")
    afficher_importance(rf_PR, FEAT_PR, "Precipitation")

    joblib.dump({'model': rf_PR, 'features': FEAT_PR,
                 'cv_results': cv_PR},
                os.path.join(DOSSIER, "model_RF_PR_chelsa.pkl"))

    # Prédiction sites
    s_PR = prep_sites_topo(sites_ref.copy())
    s_PR['pr_chelsa_log'] = np.log1p(
        s_PR['pr_chelsa_mm'].clip(lower=0)
    )
    s_PR = s_PR.dropna(subset=FEAT_PR)
    s_PR['pr_RF'] = rf_PR.predict(s_PR[FEAT_PR])
    s_PR[['nom','lat','lon','altitude','alt_mnt',
           'date','annee','mois',
           'tas_chelsa_C','pr_chelsa_mm','pr_RF']
          ].to_csv(
        os.path.join(DOSSIER, "predictions_RF_PR_sites.csv"),
        index=False
    )
    print(f"    predictions_RF_PR_sites.csv ({len(s_PR):,} obs)")

    # ── 3C : NEIGE ───────────────────────────────────────────
    print("\n  [Neige] Downscaling neige...")

    ds_snow = xr.open_dataset(
        os.path.join(DOSSIER,
                     "ERA5_Land/2000_2025/"
                     "era5_land_snow_depth_FULL_2000_2025.nc"),
        engine='h5netcdf', phony_dims='access'
    )
    sde_max = float(ds_snow['sde'].max().values)
    facteur = 100 if sde_max < 50 else 1

    cat_niv = pd.read_csv(
        os.path.join(DOSSIER, "catalogue_stations_meteo.csv")
    )
    mask_niv = cat_niv['station_name'].str.contains(
        'NIVOSE|NIVO|_CLIM', case=False, regex=True, na=False
    )
    cat_niv = cat_niv[~mask_niv].copy()

    res_niv = []
    for _, row in tqdm(cat_niv.iterrows(),
                       total=len(cat_niv), desc="Neige"):
        fic = os.path.join(DOSSIER_MF, row['fichier'])
        if not os.path.exists(fic):
            continue
        try:
            dh = pd.read_csv(fic, nrows=1)
            if 'total_snow_depth_6am_cm' not in dh.columns:
                continue
            dj = pd.read_csv(fic, usecols=[
                'date','total_snow_depth_6am_cm',
                'quality_total_snow_depth_6am'
            ])
        except Exception:
            continue

        dj['date'] = pd.to_datetime(dj['date'], errors='coerce')
        dj = dj[(dj['date'] >= '2000-01-01') &
                (dj['date'] <= '2021-12-31')
                ].dropna(subset=['total_snow_depth_6am_cm'])
        if len(dj) < 365:
            continue
        if 'quality_total_snow_depth_6am' in dj.columns:
            dj = dj[dj['quality_total_snow_depth_6am'].isin(
                [1,2,5,6]
            )]
        if len(dj) < 365:
            continue

        dj['annee'] = dj['date'].dt.year
        dj['mois']  = dj['date'].dt.month
        dm = dj.groupby(['annee','mois']).agg(
            OBS_sde_cm_moy=('total_snow_depth_6am_cm','mean'),
            n_jours=('total_snow_depth_6am_cm','count'),
        ).reset_index()
        dm = dm[dm['n_jours'] >= 20].copy()
        if len(dm) == 0:
            continue

        try:
            sde_pt = ds_snow['sde'].sel(
                latitude=row['latitude'],
                longitude=row['longitude'], method='nearest'
            )
            de = sde_pt.to_dataframe(name='sde').reset_index()
            de['date']        = pd.to_datetime(de['valid_time'])
            de['ERA5_sde_cm'] = de['sde'] * facteur
            de['annee']       = de['date'].dt.year
            de['mois']        = de['date'].dt.month
            de_m = de.groupby(['annee','mois']).agg(
                ERA5_sde_cm=('ERA5_sde_cm','mean')
            ).reset_index()
        except Exception:
            continue

        dm2 = dm.merge(de_m, on=['annee','mois'], how='inner')
        if len(dm2) < 6:
            continue
        dm2['nom']      = row['station_name']
        dm2['lat']      = row['latitude']
        dm2['lon']      = row['longitude']
        dm2['altitude'] = row['altitude']
        res_niv.append(dm2)

    df_N = pd.concat(res_niv, ignore_index=True)
    df_N['alt_mnt'] = extraire_alt_mnt(
        df_N['lon'].values, df_N['lat'].values, MNT_PATH
    )
    pn, en = extraire_topo(
        df_N['lon'].values, df_N['lat'].values,
        pente_arr, expo_arr, mnt_tr
    )
    df_N['pente']    = pn
    es, ec = encoder_exposition(en)
    df_N['expo_sin'] = es
    df_N['expo_cos'] = ec
    df_N['delta_alt'] = df_N['altitude'] - df_N['alt_mnt']
    df_N['mois_sin']  = np.sin(2*np.pi*df_N['mois']/12)
    df_N['mois_cos']  = np.cos(2*np.pi*df_N['mois']/12)

    FEAT_N = ['ERA5_sde_cm',
              'altitude','alt_mnt','delta_alt',
              'pente','expo_sin','expo_cos',
              'lat','lon','mois_sin','mois_cos']

    df_N = df_N.dropna(subset=FEAT_N + ['OBS_sde_cm_moy'])
    X_N, y_N = df_N[FEAT_N].values, df_N['OBS_sde_cm_moy'].values

    print(f"    {len(df_N):,} obs | {df_N['nom'].nunique()} stations")
    rf_N, cv_N, _ = entrainer_rf_cv(X_N, y_N, df_N['nom'].values)
    print(f"\n    MAE={np.mean(cv_N['mae']):.2f}cm  "
          f"R2={np.mean(cv_N['r2']):.3f}")
    afficher_importance(rf_N, FEAT_N, "Neige")

    joblib.dump({'model': rf_N, 'features': FEAT_N,
                 'cv_results': cv_N},
                os.path.join(DOSSIER, "model_RF_neige_mensuel.pkl"))

    # Prédiction sites neige
    s_N = prep_sites_topo(sites_ref.copy())
    s_N['mois_sin'] = np.sin(2*np.pi*s_N['mois']/12)
    s_N['mois_cos'] = np.cos(2*np.pi*s_N['mois']/12)

    era5_sites = []
    for nom_s in s_N['nom'].unique():
        sub = s_N[s_N['nom'] == nom_s].iloc[0]
        sde_pt = ds_snow['sde'].sel(
            latitude=sub['lat'], longitude=sub['lon'],
            method='nearest'
        )
        de = sde_pt.to_dataframe(name='sde').reset_index()
        de['date']        = pd.to_datetime(de['valid_time'])
        de['ERA5_sde_cm'] = de['sde'] * facteur
        de['annee']       = de['date'].dt.year
        de['mois']        = de['date'].dt.month
        de_m = de.groupby(['annee','mois']).agg(
            ERA5_sde_cm=('ERA5_sde_cm','mean')
        ).reset_index()
        de_m['nom'] = nom_s
        era5_sites.append(de_m)

    df_era5_s = pd.concat(era5_sites, ignore_index=True)
    sp_N = s_N[['nom','lat','lon','altitude','alt_mnt',
                'pente','expo_sin','expo_cos',
                'delta_alt','mois_sin','mois_cos',
                'annee','mois']].drop_duplicates()
    sp_N = sp_N.merge(df_era5_s, on=['nom','annee','mois'],
                      how='inner')
    sp_N = sp_N.dropna(subset=FEAT_N)
    sp_N['neige_RF_cm'] = rf_N.predict(sp_N[FEAT_N])
    sp_N['date'] = pd.to_datetime(
        sp_N['annee'].astype(str) + '-' +
        sp_N['mois'].astype(str).str.zfill(2) + '-15'
    )
    sp_N[['nom','date','annee','mois','lat','lon','altitude',
           'ERA5_sde_cm','neige_RF_cm']].to_csv(
        os.path.join(DOSSIER, "predictions_RF_neige_sites.csv"),
        index=False
    )
    print(f"    predictions_RF_neige_sites.csv ({len(sp_N):,} obs)")

    print("\n  Downscaling RF termine !")
    print(f"    T   : MAE={np.mean(cv_T['mae']):.2f}C  "
          f"R2={np.mean(cv_T['r2']):.3f}")
    print(f"    PR  : MAE={np.mean(cv_PR['mae']):.1f}mm  "
          f"R2={np.mean(cv_PR['r2']):.3f}")
    print(f"    Nge : MAE={np.mean(cv_N['mae']):.2f}cm  "
          f"R2={np.mean(cv_N['r2']):.3f}")


# ================================================================
# SECTION 4 — VALIDATION 9 SITES
# ================================================================

def run_validation():
    """Validation mensuelle T° aux 9 sites + carte."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score

    print("\nSection 4 : Validation 9 sites")

    pred = pd.read_csv(
        os.path.join(DOSSIER, "predictions_RF_T_sites.csv")
    )
    pred['date'] = pd.to_datetime(pred['date'])
    print(f"  {len(pred):,} obs | "
          f"{pred['nom'].nunique()} sites | "
          f"{pred['mois'].nunique()} mois")

    # Tableau mensuel
    recap = pred.groupby(['nom','mois']).agg(
        alt       = ('altitude', 'first'),
        T_CHELSA  = ('tas_chelsa_C', 'mean'),
        T_RF      = ('temp_RF', 'mean'),
    ).reset_index()
    recap['correction'] = recap['T_RF'] - recap['T_CHELSA']
    recap['mois_nom']   = recap['mois'].apply(
        lambda m: mois_lbls[m-1]
    )
    recap = recap.round(2)

    # Pivot T_RF
    pivot_RF = recap.pivot_table(
        values='T_RF', index='nom', columns='mois_nom'
    )[mois_lbls].round(2)

    order_alt = (recap[['nom','alt']].drop_duplicates()
                 .sort_values('alt')['nom'].values)
    pivot_RF = pivot_RF.reindex(order_alt)
    print("\n  T_RF (°C) par site × mois :")
    print(pivot_RF.to_string())

    # Heatmap
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    pivot_CHELSA = recap.pivot_table(
        values='T_CHELSA', index='nom', columns='mois_nom'
    )[mois_lbls].round(2).reindex(order_alt)
    pivot_corr = recap.pivot_table(
        values='correction', index='nom', columns='mois_nom'
    )[mois_lbls].round(2).reindex(order_alt)

    alts = (recap[['nom','alt']].drop_duplicates()
            .set_index('nom')['alt'])

    for ax, data, titre, cmap, sym in [
        (axes[0], pivot_CHELSA, 'T CHELSA (C)',
         'RdYlBu_r', False),
        (axes[1], pivot_RF,     'T RF (C)',
         'RdYlBu_r', False),
        (axes[2], pivot_corr,   'Correction (C)',
         'RdBu_r',   True),
    ]:
        vmin = -3 if sym else data.values.min()
        vmax =  3 if sym else data.values.max()
        im = ax.imshow(data.values, cmap=cmap, aspect='auto',
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(12))
        ax.set_xticklabels(mois_lbls, fontsize=8)
        ax.set_yticks(range(len(order_alt)))
        ax.set_yticklabels(
            [f"{n} ({int(alts[n])}m)" for n in order_alt],
            fontsize=9
        )
        ax.set_title(titre, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(order_alt)):
            for j in range(12):
                val = data.values[i, j]
                if not np.isnan(val):
                    txt = f"{val:+.1f}" if sym else f"{val:.1f}"
                    ax.text(j, i, txt, ha='center',
                            va='center', fontsize=7,
                            color='white'
                            if abs(val) > 1.5 and sym
                            else 'black')

    plt.suptitle(
        "Validation 9 sites Delphinium — T RF vs CHELSA\n"
        "Moyenne 2000-2021",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR,
                             "validation_9sites_heatmap.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"\n  Heatmap sauvegardee : {os.path.basename(fig_path)}")

    # Sauvegardes CSV
    recap.to_csv(
        os.path.join(OUT_DIR, "validation_9sites_mensuel.csv"),
        index=False
    )
    pivot_RF.to_csv(
        os.path.join(OUT_DIR, "validation_9sites_pivot_T_RF.csv")
    )
    print("  CSV sauvegardes")

    print("\n  Validation terminee")


# ================================================================
# SECTION 5 — APPLICATION STREAMLIT
# ================================================================

def run_app():
    """Lance l'application Streamlit D2ClimAFLo-Pyr."""
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.stats import mannwhitneyu
    import matplotlib.pyplot as plt
    import streamlit.components.v1 as components

    # ── Config ────────────────────────────────────────────────
    st.set_page_config(
        page_title="D2ClimAFLo-Pyr",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .titre-page {
        font-size: 2rem; font-weight: 700; color: #1B2D26;
        border-left: 5px solid #1D9E75;
        padding-left: 12px; margin-bottom: 20px;
    }
    .carte-site {
        background: #EEEDFE; border-radius: 10px;
        padding: 15px; margin: 5px;
        border-left: 4px solid #534AB7;
    }
    .nohedes-box {
        background: #E8F5E9; border-radius: 10px;
        padding: 15px; margin: 5px;
        border-left: 4px solid #1D9E75;
    }
    .histoire-box {
        background: linear-gradient(135deg,#1B2D26,#2D6A4F);
        color: white; border-radius: 15px; padding: 30px;
        font-size: 1.1rem; line-height: 1.8; margin: 10px 0;
    }
    .question-box {
        background: #EEEDFE; border-radius: 10px;
        padding: 20px; text-align: center;
        font-size: 1.3rem; font-weight: 700; color: #3C3489;
        border: 2px solid #7F77DD; margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Constantes ────────────────────────────────────────────
    SITES = pd.DataFrame({
        "nom"      : ["CADI_POP1","CADI_POP2","CADI_POP3",
                      "CADI_POP4","EYNE_POP1","EYNE_POP2",
                      "EYNE_POP3","NOHEDES","VALLTER"],
        "lat"      : [42.237,42.277,42.289,42.283,
                      42.450,42.446,42.443,42.615,42.426],
        "lon"      : [1.704,1.688,1.688,1.574,
                      2.130,2.122,2.116,2.263,2.264],
        "altitude" : [2365,2283,1979,2358,
                      2586,2181,2164,1790,2147],
        "versant"  : ["Espagne","Espagne","Espagne","Espagne",
                      "France","France","France",
                      "France","Espagne"],
        "floraison": [True,True,True,True,
                      True,True,True,False,True],
    })

    VARS = {
        "temp_RF"    : "Temperature (C)",
        "pr_RF"      : "Precipitations (mm/mois)",
        "neige_RF_cm": "Hauteur neige (cm)",
        "pct_neige"  : "% Couverture neigeuse"
    }

    COULEURS = {"NOHEDES": "#1D9E75", "Autres sites": "#7F77DD"}

    # ── Chargement données ────────────────────────────────────
    @st.cache_data
    def charger(chemin):
        df = pd.read_csv(chemin, sep=";")
        df["groupe"] = df["nom"].apply(
            lambda x: "NOHEDES"
            if x == "NOHEDES" else "Autres sites"
        )
        df["saison"] = df["mois"].apply(lambda m:
            "Hiver"     if m in [12,1,2] else
            "Printemps" if m in [3,4,5]  else
            "Ete"       if m in [6,7,8]  else "Automne"
        )
        df["periode"] = df["annee"].apply(
            lambda a: "Avant 2010"
            if a < 2010 else "Apres 2010"
        )
        return df

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🌸 D2ClimAFLo-Pyr")
        st.markdown("**Stage BUT3 Science des Donnees**")
        st.markdown("*CEFREM · UPVD · 2026*")
        st.divider()

        page = st.radio(
            "Navigation",
            ["Introduction", "Zone d'etude", "Downscaling",
             "Statistiques descriptives", "Evolution temporelle",
             "ACP + CAH", "Test d'hypothese", "Conclusion"],
            label_visibility="collapsed"
        )
        st.divider()

        chemin = st.text_input(
            "Chemin CSV",
            value="df_complete_2000_2020.csv"
        )
        try:
            df = charger(chemin)
            st.success(f"✅ {len(df):,} observations")
            data_ok = True
        except Exception:
            st.warning("Donnees simulees")
            np.random.seed(42)
            noms = (["CADI_POP1"]*12 + ["CADI_POP2"]*12 +
                    ["CADI_POP3"]*12 + ["CADI_POP4"]*12 +
                    ["EYNE_POP1"]*12 + ["EYNE_POP2"]*12 +
                    ["EYNE_POP3"]*12 + ["NOHEDES"]*12 +
                    ["VALLTER"]*12) * 21
            annees = sorted(list(range(2000,2021)) * (9*12))
            mois_s = list(range(1,13)) * (9*21)
            n = len(annees)
            s = pd.Series(noms[:n])
            df = pd.DataFrame({
                "nom"        : noms[:n],
                "annee"      : annees,
                "mois"       : mois_s[:n],
                "temp_RF"    : np.where(
                    s=="NOHEDES",
                    np.random.normal(5.5,2,n),
                    np.random.normal(4.3,2,n)),
                "pr_RF"      : np.where(
                    s=="NOHEDES",
                    np.random.normal(78,20,n),
                    np.random.normal(70,20,n)),
                "neige_RF_cm": np.where(
                    s=="NOHEDES",
                    np.random.exponential(3,n),
                    np.random.exponential(15,n)),
                "pct_neige"  : np.where(
                    s=="NOHEDES",
                    np.random.uniform(5,30,n),
                    np.random.uniform(15,60,n)),
                "altitude"   : [
                    SITES.set_index("nom").loc[x,"altitude"]
                    for x in noms[:n]
                ]
            })
            df["groupe"]  = df["nom"].apply(
                lambda x: "NOHEDES"
                if x == "NOHEDES" else "Autres sites"
            )
            df["saison"]  = df["mois"].apply(lambda m:
                "Hiver" if m in [12,1,2] else
                "Printemps" if m in [3,4,5] else
                "Ete" if m in [6,7,8] else "Automne"
            )
            df["periode"] = df["annee"].apply(
                lambda a: "Avant 2010"
                if a < 2010 else "Apres 2010"
            )
            data_ok = True

        st.divider()
        st.markdown("Amadou FOFANA")
        st.markdown("S. Pinel · N. Collette")

    # ── PAGE : Introduction ───────────────────────────────────
    if page == "Introduction":
        st.markdown(
            '<div class="titre-page">'
            '🌸 Defaut de floraison de Delphinium montanum'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
        <div class="question-box">
        Pourquoi le site de Nohedes presente-t-il un defaut de
        floraison recurrent a partir de ~2010, alors que les
        8 autres populations restent florissantes ?
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        col1,col2,col3,col4,col5 = st.columns(5)
        col1.metric("Sites etudies", "9")
        col2.metric("Periode", "2000-2020")
        col3.metric("Variables", "4")
        col4.metric("Sites France", "5")
        col5.metric("Sites Espagne", "4")

        st.divider()
        st.markdown("### Tableau des 9 sites")
        df_sites_aff = pd.DataFrame({
            "Site"        : ["NOHEDES","EYNE_POP1","EYNE_POP2",
                             "EYNE_POP3","VALLTER","CADI_POP1",
                             "CADI_POP2","CADI_POP3","CADI_POP4"],
            "Versant"     : ["France","France","France","France",
                             "Espagne","Espagne","Espagne",
                             "Espagne","Espagne"],
            "Altitude (m)": [1790,2586,2181,2164,2147,
                             2365,2283,1979,2358],
            "Statut"      : ["Defaut","Floraison","Floraison",
                             "Floraison","Floraison","Floraison",
                             "Floraison","Floraison","Floraison"]
        })
        st.dataframe(df_sites_aff, use_container_width=True,
                     hide_index=True)

    # ── PAGE : Zone d'etude ───────────────────────────────────
    elif page == "Zone d'etude":
        st.markdown(
            '<div class="titre-page">'
            "🗺️ Zone d'etude — 9 sites Pyrenees"
            '</div>',
            unsafe_allow_html=True
        )
        SITES["statut"] = SITES["floraison"].apply(
            lambda x: "Floraison" if x else "Defaut"
        )
        fig = go.Figure()

        sites_fl = SITES[SITES["floraison"]]
        fig.add_trace(go.Scattermapbox(
            lat=sites_fl["lat"], lon=sites_fl["lon"],
            mode="markers+text",
            marker=dict(size=14, color="#7F77DD"),
            text=sites_fl["nom"],
            textposition="top right",
            customdata=sites_fl[["altitude","versant"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Altitude : %{customdata[0]} m<br>"
                "Versant  : %{customdata[1]}<extra></extra>"
            ),
            name="Floraison (8 sites)"
        ))
        noh = SITES[~SITES["floraison"]]
        fig.add_trace(go.Scattermapbox(
            lat=noh["lat"], lon=noh["lon"],
            mode="markers+text",
            marker=dict(size=18, color="#1D9E75"),
            text=noh["nom"], textposition="top right",
            hovertemplate=(
                "<b>%{text}</b><br>Defaut floraison"
                "<extra></extra>"
            ),
            name="NOHEDES — Defaut"
        ))
        fig.update_layout(
            mapbox=dict(style="open-street-map",
                        center=dict(lat=42.42, lon=1.95),
                        zoom=8.5),
            height=460,
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── PAGE : Downscaling ────────────────────────────────────
    elif page == "Downscaling":
        st.markdown(
            '<div class="titre-page">'
            "🛰️ Downscaling — De 1 km a 30 m"
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
        <div class="histoire-box">
        Le pipeline de <strong>descente d'echelle (downscaling)</strong>
        par Random Forest permet de passer de la resolution
        <strong>CHELSA (1 km)</strong> a une resolution de
        <strong>30 m</strong>, integrant la topographie locale
        (altitude, pente, exposition).<br><br>
        Trois modeles RF sont entraïnes : <strong>Temperature</strong>,
        <strong>Precipitations</strong> et <strong>Neige</strong>.
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Source T/PR", "CHELSA 1km")
        col2.metric("Source Neige", "ERA5-Land")
        col3.metric("Resolution finale", "30 m")
        col4.metric("Prédicteurs topo", "Altitude, Pente, Expo")

        st.divider()
        st.markdown("### Features par variable")
        tab1, tab2, tab3 = st.tabs(["Temperature","Precipitation","Neige"])

        with tab1:
            st.markdown("""
            **Features T° (11) :**
            - `tas_chelsa_C` — Temperature CHELSA brute
            - `altitude`, `alt_mnt`, `delta_alt` — Topographie
            - `pente`, `expo_sin`, `expo_cos` — Relief MNT 30m
            - `lat`, `lon` — Position geographique
            - `mois_sin`, `mois_cos` — Saisonnalite
            """)
        with tab2:
            st.markdown("""
            **Features PR (13) :**
            - `pr_chelsa_mm`, `pr_chelsa_log` — PR CHELSA
            - `tas_chelsa_C` — Co-variable temperature
            - `altitude`, `alt_mnt`, `delta_alt` — Topographie
            - `pente`, `expo_sin`, `expo_cos` — Relief MNT 30m
            - `lat`, `lon` — Position geographique
            - `mois_sin`, `mois_cos` — Saisonnalite
            """)
        with tab3:
            st.markdown("""
            **Features Neige (11) :**
            - `ERA5_sde_cm` — Hauteur neige ERA5-Land
            - `altitude`, `alt_mnt`, `delta_alt` — Topographie
            - `pente`, `expo_sin`, `expo_cos` — Relief MNT 30m
            - `lat`, `lon` — Position geographique
            - `mois_sin`, `mois_cos` — Saisonnalite
            """)

    # ── PAGE : Statistiques descriptives ─────────────────────
    elif page == "Statistiques descriptives":
        st.markdown(
            '<div class="titre-page">'
            "📊 Statistiques descriptives"
            '</div>',
            unsafe_allow_html=True
        )
        col1,col2,col3 = st.columns(3)
        with col1:
            var_ch = st.selectbox(
                "Variable", list(VARS.keys()),
                format_func=lambda x: VARS[x]
            )
        with col2:
            ann_ch = st.selectbox(
                "Annee",
                ["Toutes"] + sorted(df["annee"].unique().tolist())
            )
        with col3:
            sai_ch = st.selectbox(
                "Saison",
                ["Toutes","Hiver","Printemps","Ete","Automne"]
            )

        df_f = df.copy()
        if ann_ch != "Toutes":
            df_f = df_f[df_f["annee"] == ann_ch]
        if sai_ch != "Toutes":
            df_f = df_f[df_f["saison"] == sai_ch]

        col_s, col_b = st.columns([1,2])
        with col_s:
            st.markdown("#### Tableau comparatif")
            stats = df_f.groupby("groupe")[var_ch].agg([
                ("Moy",    lambda x: round(x.mean(),2)),
                ("Med",    lambda x: round(x.median(),2)),
                ("Std",    lambda x: round(x.std(),2)),
                ("Min",    lambda x: round(x.min(),2)),
                ("Max",    lambda x: round(x.max(),2)),
            ]).reset_index()
            stats.columns = ["Groupe","Moy","Med","Std","Min","Max"]
            st.dataframe(stats, hide_index=True,
                         use_container_width=True)
            if len(stats) == 2:
                nv = stats[stats["Groupe"]=="NOHEDES"]["Moy"].values[0]
                av = stats[stats["Groupe"]=="Autres sites"]["Moy"].values[0]
                st.metric("NOHEDES - Autres", f"{nv-av:+.2f}")

        with col_b:
            st.markdown("#### Boxplot")
            fig = go.Figure()
            for g, c in COULEURS.items():
                fig.add_trace(go.Box(
                    y=df_f[df_f["groupe"]==g][var_ch],
                    name=g, marker_color=c,
                    boxpoints="all", jitter=0.4, pointpos=0,
                    marker=dict(size=5, opacity=0.5),
                    line=dict(width=2)
                ))
            fig.update_layout(
                title=f"{VARS[var_ch]} — NOHEDES vs Autres",
                plot_bgcolor="white", paper_bgcolor="white",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Heatmap mensuelle")
        df_heat = df_f.groupby(["nom","mois"])[var_ch].mean(
        ).reset_index()
        pivot   = df_heat.pivot(index="nom", columns="mois",
                                values=var_ch)
        pivot.columns = [mois_lbls[c-1] for c in pivot.columns]
        fig_h = px.imshow(
            pivot, color_continuous_scale="RdYlBu_r",
            title=f"{VARS[var_ch]} — Moyenne mensuelle",
            aspect="auto"
        )
        st.plotly_chart(fig_h, use_container_width=True)

    # ── PAGE : Evolution temporelle ───────────────────────────
    elif page == "Evolution temporelle":
        st.markdown(
            '<div class="titre-page">'
            "📈 Evolution temporelle 2000-2020"
            '</div>',
            unsafe_allow_html=True
        )
        col1,col2 = st.columns(2)
        with col1:
            var_ev = st.selectbox(
                "Variable", list(VARS.keys()),
                format_func=lambda x: VARS[x], key="ev_var"
            )
        with col2:
            sai_ev = st.selectbox(
                "Saison",
                ["Toutes","Hiver","Printemps","Ete","Automne"],
                key="ev_sai"
            )

        df_ev = df.copy()
        if sai_ev != "Toutes":
            df_ev = df_ev[df_ev["saison"] == sai_ev]

        df_ann = df_ev.groupby(["annee","groupe"])[var_ev].mean(
        ).reset_index()
        df_env = (df_ev[df_ev["groupe"]=="Autres sites"]
                  .groupby(["annee","nom"])[var_ev].mean()
                  .reset_index()
                  .groupby("annee")[var_ev].agg(["min","max"])
                  .reset_index())

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([df_env["annee"],
                         df_env["annee"][::-1]]),
            y=pd.concat([df_env["max"],
                         df_env["min"][::-1]]),
            fill="toself",
            fillcolor="rgba(127,119,221,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Enveloppe autres"
        ))
        for g, c in COULEURS.items():
            dg = df_ann[df_ann["groupe"]==g]
            fig.add_trace(go.Scatter(
                x=dg["annee"], y=dg[var_ev],
                mode="lines+markers", name=g,
                line=dict(color=c, width=2.5),
                marker=dict(size=7)
            ))
        fig.add_vline(x=2010, line_dash="dash",
                      line_color="grey",
                      annotation_text="~2010")
        fig.update_layout(
            title=f"{VARS[var_ev]} — {sai_ev}",
            xaxis_title="Annee",
            yaxis_title=VARS[var_ev],
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── PAGE : ACP + CAH ──────────────────────────────────────
    elif page == "ACP + CAH":
        st.markdown(
            '<div class="titre-page">'
            "🔬 ACP + CAH — Annee 2010"
            '</div>',
            unsafe_allow_html=True
        )
        annee_acp = 2010
        df_acp = (df[df["annee"]==annee_acp]
                  .groupby("nom")[list(VARS.keys())]
                  .mean().reset_index())
        df_acp["floraison"] = df_acp["nom"].apply(
            lambda x: "Defaut" if x=="NOHEDES" else "Floraison"
        )

        X_acp    = df_acp[list(VARS.keys())].values
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_acp)
        pca      = PCA(n_components=min(4, X_acp.shape[1]))
        coords   = pca.fit_transform(X_scaled)
        var_exp  = pca.explained_variance_ratio_ * 100

        df_pca = pd.DataFrame({
            "PC1": coords[:,0], "PC2": coords[:,1],
            "nom": df_acp["nom"].values,
            "floraison": df_acp["floraison"].values
        })

        col1,col2 = st.columns(2)
        with col1:
            st.markdown("#### Valeurs propres")
            dv = pd.DataFrame({
                "Axe"       : [f"PC{i+1}" for i in
                               range(len(var_exp))],
                "Var (%)"   : np.round(var_exp, 1),
                "Cumule (%)": np.round(np.cumsum(var_exp), 1)
            })
            st.dataframe(dv, hide_index=True,
                         use_container_width=True)

        with col2:
            fig_ind = px.scatter(
                df_pca, x="PC1", y="PC2",
                color="floraison", text="nom",
                color_discrete_map={
                    "Defaut": "#1D9E75",
                    "Floraison": "#7F77DD"
                },
                title=f"ACP — {annee_acp}",
                labels={
                    "PC1": f"PC1 ({var_exp[0]:.1f}%)",
                    "PC2": f"PC2 ({var_exp[1]:.1f}%)"
                }
            )
            fig_ind.update_traces(textposition="top center",
                                  marker=dict(size=12))
            fig_ind.update_layout(plot_bgcolor="white",
                                  paper_bgcolor="white")
            st.plotly_chart(fig_ind, use_container_width=True)

        st.divider()
        st.markdown("#### CAH — Dendrogramme")
        Z        = linkage(X_scaled, method="ward")
        clusters = fcluster(Z, t=2, criterion="maxclust")
        df_pca["cluster"] = [f"C{c}" for c in clusters]

        fig_d, ax = plt.subplots(figsize=(8,5))
        ax.set_facecolor("#F8F9FA")
        seuil = (Z[-2,2] + Z[-1,2]) / 2
        dendrogram(Z, labels=df_acp["nom"].values,
                   color_threshold=seuil, ax=ax,
                   above_threshold_color="grey")
        for lbl in ax.get_xticklabels():
            txt = lbl.get_text()
            lbl.set_color("#1D9E75"
                          if txt=="NOHEDES" else "#7F77DD")
            lbl.set_fontweight("bold")
        ax.axhline(y=seuil, color="red",
                   linestyle="--", linewidth=1.2)
        ax.set_title("CAH (Ward) — 2010", fontweight="bold")
        ax.set_ylabel("Distance")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_d)

    # ── PAGE : Test d'hypothese ───────────────────────────────
    elif page == "Test d'hypothese":
        st.markdown(
            '<div class="titre-page">'
            "🧪 Test de Wilcoxon — Annee 2010"
            '</div>',
            unsafe_allow_html=True
        )
        annee_t = 2010
        sai_t   = st.selectbox(
            "Saison",
            ["Toutes","Hiver","Printemps","Ete","Automne"]
        )
        df_t = df[df["annee"]==annee_t].copy()
        if sai_t != "Toutes":
            df_t = df_t[df_t["saison"]==sai_t]

        res = []
        for var, nom_v in VARS.items():
            v_noh = df_t[df_t["groupe"]=="NOHEDES"][var].dropna()
            v_aut = df_t[df_t["groupe"]=="Autres sites"][var].dropna()
            if len(v_noh) > 0 and len(v_aut) > 0:
                _, p = mannwhitneyu(v_noh, v_aut,
                                    alternative="two-sided")
                sig = ("***" if p<0.001 else
                       "**" if p<0.01 else
                       "*" if p<0.05 else "ns")
                res.append({
                    "Variable"   : nom_v,
                    "Moy NOHEDES": round(v_noh.mean(),2),
                    "Moy Autres" : round(v_aut.mean(),2),
                    "Diff"       : round(v_noh.mean()-v_aut.mean(),2),
                    "p-value"    : round(p,4),
                    "Sig."       : sig,
                    "Conclusion" : ("H0 rejetee"
                                   if p<0.05 else "H0 acceptee")
                })

        df_res = pd.DataFrame(res)
        st.dataframe(df_res, hide_index=True,
                     use_container_width=True)

        st.divider()
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=list(VARS.values()))
        for i, (var, nom_v) in enumerate(VARS.items()):
            r, c = i//2+1, i%2+1
            for g, col in COULEURS.items():
                fig.add_trace(
                    go.Box(
                        y=df_t[df_t["groupe"]==g][var],
                        name=g, marker_color=col,
                        showlegend=(i==0),
                        boxpoints="all", jitter=0.3, pointpos=0,
                        marker=dict(size=6, opacity=0.6)
                    ),
                    row=r, col=c
                )
        fig.update_layout(
            height=550, title="Wilcoxon 2010",
            plot_bgcolor="white", paper_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── PAGE : Conclusion ─────────────────────────────────────
    elif page == "Conclusion":
        st.markdown(
            '<div class="titre-page">🏁 Conclusion</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
        <div class="histoire-box">
        <strong>Synthese des resultats</strong><br><br>
        Les analyses montrent que NOHEDES se distingue
        climatiquement des sites florissants :<br><br>
        Temperatures : +1.2 C au-dessus des autres sites<br>
        Neige : deficit significatif (~17x moins de neige)<br>
        Precipitations : differences non structurantes
        </div>
        """, unsafe_allow_html=True)

        col1,col2 = st.columns(2)
        with col1:
            st.success("""
            **Variable discriminante : la NEIGE**
            NOHEDES presente ~3 cm vs ~54 cm pour les autres.
            Hauteur de neige hors de la gamme observee.
            """)
        with col2:
            st.metric("Difference T", "+1.2 C")
            st.metric("Ratio neige", "~17x moins")
            st.metric("Test Wilcoxon neige", "p < 0.001 ***")

        st.divider()
        col1,col2,col3 = st.columns(3)
        col1.info("**Sebastien PINEL**\nCEFREM — UPVD")
        col2.info("**Noemie COLLETTE**\nLGDP — UPVD")
        col3.info("**IUT Carcassonne**\nBUT3 Sci. Donnees")


# ================================================================
# POINT D'ENTRÉE
# ================================================================

if __name__ == "__main__" or \
   (len(sys.argv) > 0 and "streamlit" in sys.argv[0]):

    # Détection automatique si lancé par streamlit
    if "streamlit" in sys.modules or \
       (len(sys.argv) > 0 and "streamlit" in sys.argv[0]):
        run_app()

    elif MODE == "pipeline":
        monter_drive()

        print("\nD2ClimAFLo-Pyr — Pipeline complet")
        print("=" * 65)

        run_chelsa_exploration()
        run_preparation_dataset_pr()
        run_downscaling()
        run_validation()

        print("\n" + "=" * 65)
        print("Pipeline termine !")
        print("Fichiers generes dans : " + DOSSIER)

    elif MODE == "app":
        # Pour lancer l'app, utilise :
        # streamlit run ce_fichier.py
        run_app()

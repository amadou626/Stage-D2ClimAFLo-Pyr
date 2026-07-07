"""
════════════════════════════════════════════════════════════════════
DOWNSCALING DES PRECIPITATIONS PAR RANDOM FOREST
════════════════════════════════════════════════════════════════════

Projet : D2ClimAFLo-Pyr (Delphinium montanum a NOHEDES)
Auteur : Amadou Fofana
Date   : 2026
Version : 1.0

OBJECTIF DU SCRIPT
──────────────────
Ce script downscale les precipitations CHELSA (resolution 1 km) 
vers les positions des 9 sites Delphinium des Pyrenees Orientales,
en utilisant les stations Meteo-France comme reference.

METHODE
───────
1. On dispose de :
   - CHELSA PR a 1 km (source basse resolution)
   - Stations Meteo-France (vraies mesures = TARGET)
   - MNT SRTM 30m (topographie fine)

2. On entraine un modele Random Forest :
   - X (features) = valeurs CHELSA + topographie
   - y (target)   = vraies mesures Meteo-France (pr_obs_mm)

3. On applique le modele aux 9 sites Delphinium
   pour obtenir leurs precipitations downscalees.

FICHIERS UTILISES
─────────────────
- data/training_RF_PR_chelsa_obs.csv : jeu d'entrainement
- data/chelsa_par_point_complet.csv : CHELSA aux 9 sites
- MNT/MNT_Pyrenees_30m.tif : MNT 30m (sur Google Drive)

FEATURES UTILISEES
──────────────────
- tas_chelsa_C : Temperature CHELSA au point (deg C)
- pr_chelsa_mm : Precipitations CHELSA (mm)
- alt_mnt      : Altitude fine (SRTM 30m, en metres)
- mois_sin     : Composante sinus du mois (saisonnalite)
- mois_cos     : Composante cosinus du mois (saisonnalite)

VALIDATION
──────────
Cross-Validation 5 folds sur le jeu d'entrainement

Metriques :
- R2    : coefficient de determination (0-1, plus haut = mieux)
- RMSE  : erreur quadratique moyenne (mm, plus bas = mieux)
- MAE   : erreur absolue moyenne (mm, plus bas = mieux)
- Biais : erreur systematique (proche de 0 = mieux)

════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import rasterio
from rasterio.warp import reproject, Resampling


# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Racine du projet
RACINE_PROJET = Path(__file__).parent.parent

# Donnees GitHub
DOSSIER_DATA = RACINE_PROJET / "data"
FICHIER_TRAIN = DOSSIER_DATA / "training_RF_PR_chelsa_obs.csv"
FICHIER_CHELSA_POINTS = DOSSIER_DATA / "chelsa_par_point_complet.csv"

# Donnees Drive
DOSSIER_DRIVE = Path("/content/drive/MyDrive/Colab Notebooks/dataStage")
DOSSIER_CHELSA_1KM = DOSSIER_DRIVE / "CHELSA" / "monthly" / "pr"
DOSSIER_CHELSA_30M = DOSSIER_DRIVE / "downscaling_25m"
FICHIER_MNT = DOSSIER_DRIVE / "MNT" / "MNT_Pyrenees_30m.tif"

# Sorties
DOSSIER_OUTPUT = RACINE_PROJET / "outputs"
DOSSIER_RASTERS = DOSSIER_OUTPUT / "rasters_30m"
DOSSIER_OUTPUT.mkdir(exist_ok=True)
DOSSIER_RASTERS.mkdir(exist_ok=True)

# Parametres Random Forest
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 20,
    'min_samples_leaf': 3,
    'random_state': 42,
    'n_jobs': -1
}

N_FOLDS = 5

# Features utilisees (memes que pour T)
COLONNES_FEATURES = [
    'tas_chelsa_C',
    'pr_chelsa_mm',
    'alt_mnt',
    'mois_sin',
    'mois_cos'
]

COLONNE_TARGET = 'pr_obs_mm'


# ═══════════════════════════════════════════════════════════════
# 3. CHARGEMENT DES DONNEES
# ═══════════════════════════════════════════════════════════════

def charger_donnees_train(fichier_csv):
    """
    Charge les donnees d'entrainement depuis un fichier CSV.
    
    Parameters
    ----------
    fichier_csv : str ou Path
        Chemin vers le fichier CSV
    
    Returns
    -------
    df : pd.DataFrame
        Donnees pretes a l'emploi
    """
    print(f"Chargement des donnees d'entrainement : {fichier_csv}")
    df = pd.read_csv(fichier_csv)
    
    colonnes_requises = COLONNES_FEATURES + [COLONNE_TARGET]
    for col in colonnes_requises:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")
    
    n_avant = len(df)
    df = df.dropna(subset=colonnes_requises)
    n_apres = len(df)
    
    print(f"   {n_apres} lignes chargees ({n_avant - n_apres} lignes retirees)")
    return df


# ═══════════════════════════════════════════════════════════════
# 4. PREPARATION FEATURES / TARGET
# ═══════════════════════════════════════════════════════════════

def preparer_features_target(df, colonnes_features, colonne_target):
    """
    Separe les features (X) et la target (y).
    
    Parameters
    ----------
    df : pd.DataFrame
        Donnees completes
    colonnes_features : list
        Noms des colonnes a utiliser comme features
    colonne_target : str
        Nom de la colonne target
    
    Returns
    -------
    X : np.ndarray
        Matrice features
    y : np.ndarray
        Vecteur target
    """
    X = df[colonnes_features].values
    y = df[colonne_target].values
    
    print(f"Features utilisees : {colonnes_features}")
    print(f"   X shape : {X.shape}")
    print(f"   y shape : {y.shape}")
    print(f"   y range : [{y.min():.2f}, {y.max():.2f}] mm")
    
    return X, y


# ═══════════════════════════════════════════════════════════════
# 5. VALIDATION - CROSS-VALIDATION 5 FOLDS
# ═══════════════════════════════════════════════════════════════

def cross_validation(X, y, params, n_folds=5):
    """
    Effectue une cross-validation k-fold sur le modele Random Forest.
    
    Parameters
    ----------
    X, y : np.ndarray
        Features et target
    params : dict
        Parametres du modele
    n_folds : int
        Nombre de folds
    
    Returns
    -------
    y_pred_cv : np.ndarray
        Predictions cross-validees
    metriques : dict
        R2, RMSE, MAE, biais
    """
    print(f"\nCross-validation {n_folds} folds")
    print("-" * 60)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    modele = RandomForestRegressor(**params)
    y_pred_cv = cross_val_predict(modele, X, y, cv=kf, n_jobs=-1)
    
    r2 = r2_score(y, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
    mae = mean_absolute_error(y, y_pred_cv)
    biais = np.mean(y_pred_cv - y)
    
    print(f"   R2    = {r2:.3f}")
    print(f"   RMSE  = {rmse:.2f} mm")
    print(f"   MAE   = {mae:.2f} mm")
    print(f"   Biais = {biais:+.2f} mm")
    
    return y_pred_cv, {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'biais': biais
    }


# ═══════════════════════════════════════════════════════════════
# 6. ENTRAINEMENT DU MODELE FINAL
# ═══════════════════════════════════════════════════════════════

def entrainer_modele_final(X, y, params):
    """
    Entraine le modele final sur TOUTES les donnees.
    
    Parameters
    ----------
    X, y : np.ndarray
        Features et target
    params : dict
        Parametres du modele
    
    Returns
    -------
    modele : RandomForestRegressor
        Modele entraine
    """
    print(f"\nEntrainement du modele final")
    print("-" * 60)
    
    modele = RandomForestRegressor(**params)
    modele.fit(X, y)
    
    y_pred_train = modele.predict(X)
    mae_train = mean_absolute_error(y, y_pred_train)
    r2_train = r2_score(y, y_pred_train)
    
    print(f"   Modele entraine sur {len(y)} echantillons")
    print(f"   MAE training : {mae_train:.2f} mm")
    print(f"   R2 training  : {r2_train:.3f}")
    
    return modele


# ═══════════════════════════════════════════════════════════════
# 7. IMPORTANCE DES FEATURES
# ═══════════════════════════════════════════════════════════════

def calculer_importance(modele, colonnes_features):
    """
    Calcule et affiche l'importance de chaque feature.
    
    Parameters
    ----------
    modele : RandomForestRegressor
        Modele entraine
    colonnes_features : list
        Noms des features
    
    Returns
    -------
    df_importance : pd.DataFrame
    """
    print(f"\nImportance des features")
    print("-" * 60)
    
    df_importance = pd.DataFrame({
        'feature': colonnes_features,
        'importance_pct': modele.feature_importances_ * 100
    }).sort_values('importance_pct', ascending=False)
    
    print(df_importance.to_string(index=False))
    return df_importance


# ═══════════════════════════════════════════════════════════════
# 8. VISUALISATION DE LA VALIDATION
# ═══════════════════════════════════════════════════════════════

def plot_validation(y_reel, y_pred, metriques, titre="Validation"):
    """
    Genere un graphique scatter predit vs reel avec metriques.
    
    Parameters
    ----------
    y_reel : np.ndarray
        Valeurs reelles
    y_pred : np.ndarray
        Valeurs predites
    metriques : dict
        R2, RMSE, biais
    titre : str
        Titre du graphique
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    
    ax.scatter(y_reel, y_pred, alpha=0.5, s=30, color='#1976D2', 
               edgecolor='white', linewidth=0.5)
    
    lim_min = min(y_reel.min(), y_pred.min()) - 5
    lim_max = max(y_reel.max(), y_pred.max()) + 5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 
            'r--', linewidth=2, label='Y = X (parfait)')
    
    titre_complet = (f"{titre}\n"
                     f"R2 = {metriques['R2']:.3f}  |  "
                     f"RMSE = {metriques['RMSE']:.2f} mm  |  "
                     f"Biais = {metriques['biais']:+.2f} mm")
    ax.set_title(titre_complet, fontsize=13, fontweight='bold')
    ax.set_xlabel('Precipitations reelles (mm)', fontsize=12)
    ax.set_ylabel('Precipitations predites (mm)', fontsize=12)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# 9. PREDICTION SUR LES 9 SITES DELPHINIUM
# ═══════════════════════════════════════════════════════════════

def extraire_altitude_mnt(fichier_mnt, coords_lon_lat):
    """
    Extrait l'altitude du MNT aux positions donnees.
    
    Parameters
    ----------
    fichier_mnt : str ou Path
        Chemin vers le MNT
    coords_lon_lat : list de tuples
        Liste de (longitude, latitude)
    
    Returns
    -------
    altitudes : list
        Altitudes extraites
    """
    with rasterio.open(fichier_mnt) as src:
        vals = list(src.sample(coords_lon_lat))
        altitudes = []
        for v in vals:
            if src.nodata is None or v[0] != src.nodata:
                altitudes.append(v[0])
            else:
                altitudes.append(np.nan)
    return altitudes


def predire_sites_delphinium(modele, fichier_chelsa, fichier_mnt):
    """
    Applique le modele aux 9 sites Delphinium.
    
    Parameters
    ----------
    modele : RandomForestRegressor
        Modele entraine
    fichier_chelsa : Path
        CSV chelsa_par_point_complet.csv
    fichier_mnt : Path
        MNT 30m
    
    Returns
    -------
    df_sites : pd.DataFrame
        Predictions pour les 9 sites
    """
    print(f"\nPrediction sur les 9 sites Delphinium")
    print("-" * 60)
    
    df_chelsa = pd.read_csv(fichier_chelsa)
    df_chelsa['date'] = pd.to_datetime(df_chelsa['date'])
    
    sites = df_chelsa[df_chelsa['type'] == 'SITE'].copy()
    sites['altitude'] = sites['altitude_meta']
    
    pts_uniques = sites[['nom', 'lon', 'lat']].drop_duplicates()
    coords = list(zip(pts_uniques['lon'], pts_uniques['lat']))
    pts_uniques['alt_mnt'] = extraire_altitude_mnt(fichier_mnt, coords)
    
    sites = sites.merge(pts_uniques[['nom', 'alt_mnt']], on='nom', how='left')
    
    sites['mois_sin'] = np.sin(2 * np.pi * sites['mois'] / 12)
    sites['mois_cos'] = np.cos(2 * np.pi * sites['mois'] / 12)
    
    sites = sites.dropna(subset=['tas_chelsa_C', 'pr_chelsa_mm', 'alt_mnt'])
    
    sites['pr_RF'] = modele.predict(sites[COLONNES_FEATURES])
    
    print(f"   {len(sites)} observations predites")
    print(f"   {sites['nom'].nunique()} sites")
    
    return sites


def afficher_stats_sites(df_sites):
    """
    Affiche les statistiques par site.
    
    Parameters
    ----------
    df_sites : pd.DataFrame
        Predictions par site
    
    Returns
    -------
    recap : pd.DataFrame
    saison_stats : pd.DataFrame
    """
    print(f"\nStats PR annuelle par site")
    print("-" * 60)
    
    recap = df_sites.groupby('nom').agg(
        altitude=('altitude', 'first'),
        n_obs=('pr_chelsa_mm', 'count'),
        PR_chelsa=('pr_chelsa_mm', 'mean'),
        PR_RF=('pr_RF', 'mean'),
    ).round(2).sort_values('altitude')
    recap['correction'] = (recap['PR_RF'] - recap['PR_chelsa']).round(2)
    
    print(recap.to_string())
    
    print(f"\nPR saisonniere RF par site")
    print("-" * 60)
    
    saisons = {
        'hiver': [12, 1, 2],
        'printemps': [3, 4, 5],
        'ete': [6, 7, 8],
        'automne': [9, 10, 11]
    }
    
    saison_stats = pd.DataFrame({'altitude': recap['altitude']})
    for sais, mois_list in saisons.items():
        sub = df_sites[df_sites['mois'].isin(mois_list)].groupby('nom')['pr_RF'].mean()
        saison_stats[f'PR_{sais}'] = sub.round(2)
    saison_stats = saison_stats.sort_values('altitude')
    
    print(saison_stats.to_string())
    
    return recap, saison_stats


# ═══════════════════════════════════════════════════════════════
# 10. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def pipeline_downscaling_pr():
    """
    Pipeline complet de downscaling PR avec validation et prediction.
    """
    print("=" * 70)
    print("  DOWNSCALING PRECIPITATIONS CHELSA -> 30m")
    print("=" * 70)
    
    # Etape 1 : Chargement donnees
    df = charger_donnees_train(FICHIER_TRAIN)
    
    # Etape 2 : Preparation
    X, y = preparer_features_target(df, COLONNES_FEATURES, COLONNE_TARGET)
    
    # Etape 3 : Cross-validation
    y_pred_cv, metriques_cv = cross_validation(X, y, RF_PARAMS, N_FOLDS)
    
    # Etape 4 : Graphique validation
    fig_cv = plot_validation(y, y_pred_cv, metriques_cv,
                              titre="Cross-validation 5 folds - PR")
    fig_cv.savefig(DOSSIER_OUTPUT / "validation_CV_PR.png",
                   dpi=150, bbox_inches='tight')
    print(f"\nGraphique CV sauvegarde")
    
    # Etape 5 : Modele final
    modele_final = entrainer_modele_final(X, y, RF_PARAMS)
    
    # Etape 6 : Importance
    df_importance = calculer_importance(modele_final, COLONNES_FEATURES)
    
    # Etape 7 : Prediction 9 sites
    df_sites = predire_sites_delphinium(modele_final,
                                          FICHIER_CHELSA_POINTS,
                                          FICHIER_MNT)
    
    # Etape 8 : Stats
    recap, saison_stats = afficher_stats_sites(df_sites)
    
    # Resume
    print("\n" + "=" * 70)
    print("  RESUME FINAL")
    print("=" * 70)
    print(f"  R2 CV      : {metriques_cv['R2']:.3f}")
    print(f"  RMSE CV    : {metriques_cv['RMSE']:.2f} mm")
    print(f"  Biais CV   : {metriques_cv['biais']:+.2f} mm")
    print(f"  Nombre stations : {df['nom'].nunique() if 'nom' in df.columns else len(y)}")
    print(f"  Features utilisees : {len(COLONNES_FEATURES)}")
    print(f"  Sites predits      : {df_sites['nom'].nunique()}")
    print("=" * 70)
    
    return modele_final, metriques_cv, df_importance, df_sites


# ═══════════════════════════════════════════════════════════════
# 11. RESAMPLING CHELSA PR 1km VERS 30m
# ═══════════════════════════════════════════════════════════════

def resampler_chelsa_pr_30m(annee=2020, mois=4):
    """
    Resample un raster CHELSA PR 1km vers 30m.
    
    Parameters
    ----------
    annee : int
        Annee (defaut 2020)
    mois : int
        Mois (defaut 4)
    
    Returns
    -------
    fichier_sortie : Path
        Chemin du raster 30m
    """
    print(f"\nResampling CHELSA PR 1km -> 30m ({mois:02d}/{annee})")
    print("-" * 60)
    
    fichier_chelsa = DOSSIER_CHELSA_1KM / f"CHELSA_pr_{mois:02d}_{annee}_V.2.1.tif"
    fichier_sortie = DOSSIER_CHELSA_30M / f"PR_CHELSA_30m_Delphinium_{annee}{mois:02d}.tif"
    
    if not fichier_chelsa.exists():
        raise FileNotFoundError(f"CHELSA PR introuvable : {fichier_chelsa}")
    
    # Charger MNT reference
    with rasterio.open(FICHIER_MNT) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
    
    # Charger et resampler CHELSA
    with rasterio.open(fichier_chelsa) as src:
        destination = np.zeros((ref_height, ref_width), dtype=np.float32)
        
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
    
    # CHELSA PR : appliquer facteur d'echelle si necessaire
    # V2.1 : PR en kg m-2 * mois
    # Souvent le facteur d'echelle est 0.1
    val_max = np.nanmax(destination)
    if val_max > 5000:
        destination = destination * 0.01
    
    # Sauvegarder
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': -9999,
        'width': ref_width,
        'height': ref_height,
        'count': 1,
        'crs': ref_crs,
        'transform': ref_transform,
        'compress': 'lzw'
    }
    
    with rasterio.open(fichier_sortie, 'w', **profile) as dst:
        dst.write(destination.astype('float32'), 1)
    
    pr_moy = np.nanmean(destination)
    pr_min = np.nanmin(destination)
    pr_max = np.nanmax(destination)
    
    print(f"   PR moyenne : {pr_moy:.2f} mm")
    print(f"   PR range : [{pr_min:.2f}, {pr_max:.2f}] mm")
    print(f"   Sauvegarde : {fichier_sortie.name}")
    
    return fichier_sortie


# ═══════════════════════════════════════════════════════════════
# 12. VISUALISATION AVANT / APRES - AVRIL 2020
# ═══════════════════════════════════════════════════════════════

def visualisation_avril_2020_pr(modele):
    """
    Visualise le downscaling PR pour avril 2020.
    
    Parameters
    ----------
    modele : RandomForestRegressor
        Modele entraine
    """
    ANNEE = 2020
    MOIS = 4
    CHUNK_SIZE = 500
    
    print("=" * 70)
    print(f"  VISUALISATION PR AVANT/APRES - AVRIL {ANNEE}")
    print("=" * 70)
    
    # Etape 1 : Resampling CHELSA PR
    fichier_chelsa_30m = resampler_chelsa_pr_30m(ANNEE, MOIS)
    
    # Etape 2 : Charger MNT et CHELSA PR 30m
    print("\nEtape 2 : Chargement rasters")
    print("-" * 60)
    
    with rasterio.open(FICHIER_MNT) as src:
        mnt_data = src.read(1)
    
    # Charger aussi CHELSA T 30m (pour feature)
    fichier_chelsa_t = DOSSIER_CHELSA_30M / f"T_CHELSA_30m_Delphinium_{ANNEE}{MOIS:02d}.tif"
    if not fichier_chelsa_t.exists():
        print(f"   ATTENTION : {fichier_chelsa_t.name} manquant")
        print(f"   Utilisation valeur constante pour tas_chelsa_C")
        chelsa_t_data = None
    else:
        with rasterio.open(fichier_chelsa_t) as src:
            chelsa_t_data = src.read(1)
    
    with rasterio.open(fichier_chelsa_30m) as src:
        chelsa_pr_data = src.read(1)
        profile = src.profile.copy()
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    print(f"   MNT shape : {mnt_data.shape}")
    print(f"   CHELSA PR shape : {chelsa_pr_data.shape}")
    
    # Etape 3 : Prediction par blocs
    print(f"\nEtape 3 : Prediction par blocs (chunk={CHUNK_SIZE})")
    print("-" * 60)
    
    h, w = chelsa_pr_data.shape
    rf_pr = np.full((h, w), np.nan, dtype=np.float32)
    
    mois_sin = np.sin(2 * np.pi * MOIS / 12)
    mois_cos = np.cos(2 * np.pi * MOIS / 12)
    
    n_blocks_total = ((h + CHUNK_SIZE - 1) // CHUNK_SIZE) * \
                     ((w + CHUNK_SIZE - 1) // CHUNK_SIZE)
    n_done = 0
    
    for i in range(0, h, CHUNK_SIZE):
        for j in range(0, w, CHUNK_SIZE):
            n_done += 1
            i_end = min(i + CHUNK_SIZE, h)
            j_end = min(j + CHUNK_SIZE, w)
            
            chelsa_pr_chunk = chelsa_pr_data[i:i_end, j:j_end].ravel()
            mnt_chunk = mnt_data[i:i_end, j:j_end].ravel()
            
            if chelsa_t_data is not None:
                chelsa_t_chunk = chelsa_t_data[i:i_end, j:j_end].ravel()
            else:
                chelsa_t_chunk = np.full_like(chelsa_pr_chunk, 10.0)
            
            mask = (~np.isnan(chelsa_pr_chunk)) & (~np.isnan(mnt_chunk))
            
            if mask.sum() == 0:
                continue
            
            n_valid = mask.sum()
            features = np.zeros((n_valid, 5), dtype=np.float32)
            features[:, 0] = chelsa_t_chunk[mask]     # tas_chelsa_C
            features[:, 1] = chelsa_pr_chunk[mask]    # pr_chelsa_mm
            features[:, 2] = mnt_chunk[mask]          # alt_mnt
            features[:, 3] = mois_sin
            features[:, 4] = mois_cos
            
            predictions = modele.predict(features)
            
            chunk_pred = np.full_like(chelsa_pr_chunk, np.nan)
            chunk_pred[mask] = predictions
            rf_pr[i:i_end, j:j_end] = chunk_pred.reshape((i_end - i, j_end - j))
            
            if n_done % 50 == 0:
                print(f"   Bloc {n_done}/{n_blocks_total}")
    
    print(f"   Termine : {n_done} blocs traites")
    
    # Etape 4 : Sauvegarde raster
    print("\nEtape 4 : Sauvegarde raster")
    print("-" * 60)
    
    fichier_sortie = DOSSIER_RASTERS / f"PR_RF_30m_Delphinium_{ANNEE}{MOIS:02d}.tif"
    profile.update(dtype='float32', count=1, compress='lzw')
    
    with rasterio.open(fichier_sortie, 'w', **profile) as dst:
        dst.write(rf_pr.astype('float32'), 1)
    
    n_valid = (~np.isnan(rf_pr)).sum()
    pr_moy = np.nanmean(rf_pr)
    pr_min = np.nanmin(rf_pr)
    pr_max = np.nanmax(rf_pr)
    
    print(f"   PR moyenne : {pr_moy:.2f} mm")
    print(f"   PR range : [{pr_min:.2f}, {pr_max:.2f}] mm")
    print(f"   Pixels valides : {n_valid:,}")
    print(f"   Sauvegarde : {fichier_sortie.name}")
    
    # Etape 5 : Visualisation
    print("\nEtape 5 : Visualisation avant/apres")
    print("-" * 60)
    
    df_chelsa = pd.read_csv(FICHIER_CHELSA_POINTS)
    sites = df_chelsa[df_chelsa['type'] == 'SITE'][
        ['nom', 'lat', 'lon']
    ].drop_duplicates()
    
    # Downsample pour visualisation
    factor = 10
    chelsa_viz = chelsa_pr_data[::factor, ::factor]
    rf_viz = rf_pr[::factor, ::factor]
    print(f"   Downsampling x{factor} pour visualisation")
    
    all_vals = np.concatenate([
        chelsa_viz[~np.isnan(chelsa_viz)].ravel(),
        rf_viz[~np.isnan(rf_viz)].ravel()
    ])
    vmin = np.percentile(all_vals, 2)
    vmax = np.percentile(all_vals, 98)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad(color='lightgray', alpha=0.5)
    
    ax1 = axes[0]
    im1 = ax1.imshow(np.ma.masked_invalid(chelsa_viz), cmap=cmap,
                      extent=extent, origin='upper',
                      vmin=vmin, vmax=vmax, interpolation='nearest')
    ax1.set_title(f'AVANT - CHELSA PR 1km resample a 30m\nAvril {ANNEE}',
                    fontsize=14, fontweight='bold')
    
    ax2 = axes[1]
    im2 = ax2.imshow(np.ma.masked_invalid(rf_viz), cmap=cmap,
                      extent=extent, origin='upper',
                      vmin=vmin, vmax=vmax, interpolation='nearest')
    ax2.set_title(f'APRES - Downscaling Random Forest 30m\nAvril {ANNEE}',
                    fontsize=14, fontweight='bold')
    
    for ax in [ax1, ax2]:
        for _, row in sites.iterrows():
            couleur = '#27AE60' if row['nom'] == 'NOHEDES' else '#9B59B6'
            ax.scatter(row['lon'], row['lat'], s=150, marker='*',
                        c=couleur, edgecolor='black', linewidth=1.5, zorder=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(alpha=0.3, linestyle=':')
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax, label='Precipitations (mm)')
    
    plt.suptitle(f'Downscaling Precipitations - Comparaison avant/apres',
                  fontsize=16, fontweight='bold')
    
    chemin_fig = DOSSIER_OUTPUT / f"comparaison_avant_apres_PR_{ANNEE}{MOIS:02d}.png"
    fig.savefig(chemin_fig, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"   Figure sauvegardee : {chemin_fig.name}")
    print("\n" + "=" * 70)
    print("  VISUALISATION TERMINEE")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# 13. LANCEMENT DU SCRIPT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    modele, metriques, importance, sites = pipeline_downscaling_pr()
    
    # Visualisation avant/apres pour avril 2020
    visualisation_avril_2020_pr(modele)
    
    print("\nPipeline PR termine avec succes")

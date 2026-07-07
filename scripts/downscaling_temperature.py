"""
════════════════════════════════════════════════════════════════════
DOWNSCALING DE LA TEMPÉRATURE PAR RANDOM FOREST
════════════════════════════════════════════════════════════════════

Projet : D2ClimAFLo-Pyr (Delphinium montanum à NOHEDES)
Auteur : Amadou Fofana
Date   : 2026
Version : 1.0

OBJECTIF DU SCRIPT
──────────────────
Ce script downscale la température CHELSA (résolution 1 km) 
vers les positions des 9 sites Delphinium des Pyrénées Orientales,
en utilisant les stations Météo-France comme référence.

MÉTHODE
───────
1. On dispose de :
   - CHELSA à 1 km (source basse résolution)
   - Stations Météo-France (vraies mesures = TARGET)
   - MNT SRTM 30m (topographie fine)

2. On entraîne un modèle Random Forest :
   - X (features) = valeurs CHELSA + topographie
   - y (target)   = vraies mesures Météo-France

3. On applique le modèle aux 9 sites Delphinium
   pour obtenir leur température downscalée.

FICHIERS UTILISÉS
─────────────────
- data/training_RF_T_chelsa_obs.csv : jeu d'entraînement 
  (stations M-F avec CHELSA au même point)
- data/chelsa_par_point_complet.csv : CHELSA aux 9 sites 
  + stations M-F (pour prédiction)
- MNT/MNT_Pyrenees_30m.tif : MNT 30m (sur Google Drive)

FEATURES UTILISÉES
──────────────────
- tas_chelsa_C : Température CHELSA au point (°C)
- pr_chelsa_mm : Précipitations CHELSA (mm)
- alt_mnt      : Altitude fine (SRTM 30m, en mètres)
- mois_sin     : Composante sinus du mois (saisonnalité)
- mois_cos     : Composante cosinus du mois (saisonnalité)

VALIDATION
──────────
Cross-Validation 5 folds sur le jeu d'entraînement

Métriques :
- R²    : coefficient de détermination (0-1, plus haut = mieux)
- RMSE  : erreur quadratique moyenne (°C, plus bas = mieux)
- MAE   : erreur absolue moyenne (°C, plus bas = mieux)
- Biais : erreur systématique (proche de 0 = mieux)

════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Geospatial (pour extraire altitude MNT)
import rasterio


# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Racine du projet (un niveau au-dessus du dossier scripts/)
RACINE_PROJET = Path(__file__).parent.parent

# Chemins des donnees sur GitHub (relatifs)
DOSSIER_DATA = RACINE_PROJET / "data"
FICHIER_TRAIN = DOSSIER_DATA / "training_RF_T_chelsa_obs.csv"
FICHIER_CHELSA_POINTS = DOSSIER_DATA / "chelsa_par_point_complet.csv"

# Chemin du MNT sur Google Drive (absolu, trop volumineux pour GitHub)
FICHIER_MNT = Path("/content/drive/MyDrive/Colab Notebooks/dataStage/MNT/MNT_Pyrenees_30m.tif")

# Dossier de sortie
DOSSIER_OUTPUT = RACINE_PROJET / "outputs"
DOSSIER_OUTPUT.mkdir(exist_ok=True)

# Parametres Random Forest
RF_PARAMS = {
    'n_estimators': 300,        # Nombre d'arbres
    'max_depth': 20,            # Profondeur max des arbres
    'min_samples_leaf': 3,      # Nb min echantillons par feuille
    'random_state': 42,         # Reproductibilite
    'n_jobs': -1                # Utiliser tous les CPUs
}

# Nombre de folds pour cross-validation
N_FOLDS = 5

# Features utilisees pour le modele
COLONNES_FEATURES = [
    'tas_chelsa_C',
    'pr_chelsa_mm',
    'alt_mnt',
    'mois_sin',
    'mois_cos'
]

# Colonne cible (target)
COLONNE_TARGET = 'temp_moy_c'


# ═══════════════════════════════════════════════════════════════
# 3. CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════

def charger_donnees_train(fichier_csv):
    """
    Charge les donnees d'entrainement depuis un fichier CSV.
    
    Le CSV doit contenir les colonnes :
    - temp_moy_c (target : vraie temperature des stations M-F)
    - tas_chelsa_C, pr_chelsa_mm (features CHELSA)
    - alt_mnt (altitude fine)
    - mois_sin, mois_cos (saisonnalite)
    
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
    
    # Verifier les colonnes essentielles
    colonnes_requises = COLONNES_FEATURES + [COLONNE_TARGET]
    for col in colonnes_requises:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")
    
    # Retirer les lignes avec valeurs manquantes
    n_avant = len(df)
    df = df.dropna(subset=colonnes_requises)
    n_apres = len(df)
    
    print(f"   {n_apres} lignes chargees ({n_avant - n_apres} lignes retirees)")
    return df


# ═══════════════════════════════════════════════════════════════
# 4. PRÉPARATION FEATURES / TARGET
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
    print(f"   y range : [{y.min():.2f}, {y.max():.2f}] deg C")
    
    return X, y


# ═══════════════════════════════════════════════════════════════
# 5. VALIDATION — CROSS-VALIDATION 5 FOLDS
# ═══════════════════════════════════════════════════════════════

def cross_validation(X, y, params, n_folds=5):
    """
    Effectue une cross-validation k-fold sur le modele Random Forest.
    
    Principe :
    - On divise les donnees en k groupes (folds)
    - Pour chaque fold :
        * On entraine sur k-1 folds
        * On teste sur le fold restant
    - On combine toutes les predictions
    
    Cela donne une estimation ROBUSTE de la performance du modele.
    
    Parameters
    ----------
    X, y : np.ndarray
        Features et target
    params : dict
        Parametres du modele
    n_folds : int
        Nombre de folds (defaut : 5)
    
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
    print(f"   RMSE  = {rmse:.2f} deg C")
    print(f"   MAE   = {mae:.2f} deg C")
    print(f"   Biais = {biais:+.2f} deg C")
    
    metriques = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'biais': biais
    }
    
    return y_pred_cv, metriques


# ═══════════════════════════════════════════════════════════════
# 6. ENTRAÎNEMENT DU MODÈLE FINAL
# ═══════════════════════════════════════════════════════════════

def entrainer_modele_final(X, y, params):
    """
    Entraine le modele final sur TOUTES les donnees.
    
    Ce modele sera utilise pour predire aux 9 sites Delphinium.
    
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
    
    # Metriques sur donnees d'entrainement (verification)
    y_pred_train = modele.predict(X)
    mae_train = mean_absolute_error(y, y_pred_train)
    r2_train = r2_score(y, y_pred_train)
    
    print(f"   Modele entraine sur {len(y)} echantillons")
    print(f"   MAE training : {mae_train:.2f} deg C")
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
        Tableau trie par importance decroissante
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
    
    lim_min = min(y_reel.min(), y_pred.min()) - 1
    lim_max = max(y_reel.max(), y_pred.max()) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 
            'r--', linewidth=2, label='Y = X (parfait)')
    
    titre_complet = (f"{titre}\n"
                     f"R2 = {metriques['R2']:.3f}  |  "
                     f"RMSE = {metriques['RMSE']:.2f} deg C  |  "
                     f"Biais = {metriques['biais']:+.2f} deg C")
    ax.set_title(titre_complet, fontsize=13, fontweight='bold')
    ax.set_xlabel('Temperature reelle (deg C)', fontsize=12)
    ax.set_ylabel('Temperature predite (deg C)', fontsize=12)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# 9. PRÉDICTION SUR LES 9 SITES DELPHINIUM
# ═══════════════════════════════════════════════════════════════

def extraire_altitude_mnt(fichier_mnt, coords_lon_lat):
    """
    Extrait l'altitude du MNT aux positions donnees.
    
    Parameters
    ----------
    fichier_mnt : str ou Path
        Chemin vers le MNT (.tif)
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
    fichier_chelsa : str ou Path
        CSV avec CHELSA aux sites (chelsa_par_point_complet.csv)
    fichier_mnt : str ou Path
        MNT 30m
    
    Returns
    -------
    df_sites : pd.DataFrame
        Predictions pour les 9 sites
    """
    print(f"\nPrediction sur les 9 sites Delphinium")
    print("-" * 60)
    
    # Charger CHELSA aux points
    df_chelsa = pd.read_csv(fichier_chelsa)
    df_chelsa['date'] = pd.to_datetime(df_chelsa['date'])
    
    # Filtrer les 9 sites Delphinium
    sites = df_chelsa[df_chelsa['type'] == 'SITE'].copy()
    sites['altitude'] = sites['altitude_meta']
    
    # Extraire alt_mnt depuis le MNT
    pts_uniques = sites[['nom', 'lon', 'lat']].drop_duplicates()
    coords = list(zip(pts_uniques['lon'], pts_uniques['lat']))
    pts_uniques['alt_mnt'] = extraire_altitude_mnt(fichier_mnt, coords)
    
    # Merger alt_mnt avec les donnees mensuelles
    sites = sites.merge(pts_uniques[['nom', 'alt_mnt']], on='nom', how='left')
    
    # Calculer features saisonnieres
    sites['mois_sin'] = np.sin(2 * np.pi * sites['mois'] / 12)
    sites['mois_cos'] = np.cos(2 * np.pi * sites['mois'] / 12)
    
    # Retirer NaN
    sites = sites.dropna(subset=['tas_chelsa_C', 'alt_mnt'])
    
    # Predire
    sites['temp_RF'] = modele.predict(sites[COLONNES_FEATURES])
    
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
    """
    print(f"\nStats T annuelle par site")
    print("-" * 60)
    
    recap = df_sites.groupby('nom').agg(
        altitude=('altitude', 'first'),
        n_obs=('tas_chelsa_C', 'count'),
        T_chelsa=('tas_chelsa_C', 'mean'),
        T_RF=('temp_RF', 'mean'),
    ).round(2).sort_values('altitude')
    recap['correction'] = (recap['T_RF'] - recap['T_chelsa']).round(2)
    
    print(recap.to_string())
    
    # Stats saisonnieres
    print(f"\nT saisonniere RF par site")
    print("-" * 60)
    
    saisons = {
        'hiver': [12, 1, 2],
        'printemps': [3, 4, 5],
        'ete': [6, 7, 8],
        'automne': [9, 10, 11]
    }
    
    saison_stats = pd.DataFrame({'altitude': recap['altitude']})
    for sais, mois_list in saisons.items():
        sub = df_sites[df_sites['mois'].isin(mois_list)].groupby('nom')['temp_RF'].mean()
        saison_stats[f'T_{sais}'] = sub.round(2)
    saison_stats = saison_stats.sort_values('altitude')
    
    print(saison_stats.to_string())
    
    return recap, saison_stats


# ═══════════════════════════════════════════════════════════════
# 10. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def pipeline_downscaling():
    """
    Pipeline complet de downscaling avec validation et prediction.
    
    Etapes :
    1. Chargement des donnees d'entrainement
    2. Preparation features et target
    3. Cross-validation 5 folds
    4. Graphique de validation
    5. Entrainement du modele final
    6. Importance des features
    7. Prediction sur les 9 sites Delphinium
    8. Statistiques par site
    """
    print("=" * 70)
    print("  DOWNSCALING TEMPERATURE CHELSA -> 30m")
    print("=" * 70)
    
    # --- ETAPE 1 : Chargement donnees ---
    df = charger_donnees_train(FICHIER_TRAIN)
    
    # --- ETAPE 2 : Definir features et target ---
    X, y = preparer_features_target(df, COLONNES_FEATURES, COLONNE_TARGET)
    
    # --- ETAPE 3 : Cross-validation ---
    y_pred_cv, metriques_cv = cross_validation(X, y, RF_PARAMS, N_FOLDS)
    
    # --- ETAPE 4 : Graphique validation CV ---
    fig_cv = plot_validation(y, y_pred_cv, metriques_cv, 
                             titre="Cross-validation 5 folds")
    fig_cv.savefig(DOSSIER_OUTPUT / "validation_CV.png", 
                   dpi=150, bbox_inches='tight')
    print(f"\nGraphique CV sauvegarde")
    
    # --- ETAPE 5 : Entrainement modele final ---
    modele_final = entrainer_modele_final(X, y, RF_PARAMS)
    
    # --- ETAPE 6 : Importance des features ---
    df_importance = calculer_importance(modele_final, COLONNES_FEATURES)
    
    # --- ETAPE 7 : Prediction sur 9 sites Delphinium ---
    df_sites = predire_sites_delphinium(modele_final, 
                                          FICHIER_CHELSA_POINTS, 
                                          FICHIER_MNT)
    
    # --- ETAPE 8 : Stats par site ---
    recap, saison_stats = afficher_stats_sites(df_sites)
    
    # --- Resume final ---
    print("\n" + "=" * 70)
    print("  RESUME FINAL")
    print("=" * 70)
    print(f"  R2 CV      : {metriques_cv['R2']:.3f}")
    print(f"  RMSE CV    : {metriques_cv['RMSE']:.2f} deg C")
    print(f"  Biais CV   : {metriques_cv['biais']:+.2f} deg C")
    print(f"  Nombre de stations : {df['nom'].nunique() if 'nom' in df.columns else len(y)}")
    print(f"  Features utilisees : {len(COLONNES_FEATURES)}")
    print(f"  Sites predits      : {df_sites['nom'].nunique()}")
    print("=" * 70)
    
    return modele_final, metriques_cv, df_importance, df_sites


# ═══════════════════════════════════════════════════════════════
# 11. VISUALISATION AVANT / APRÈS - AVRIL 2020
# ═══════════════════════════════════════════════════════════════

def visualisation_avril_2020(modele):
    """
    Visualise le downscaling pour avril 2020.
    
    Affiche 2 cartes cote a cote :
    - AVANT : CHELSA 1km resample a 30m (interpolation)
    - APRES : Downscaling Random Forest 30m
    
    Sauvegarde egalement le raster T_RF_30m et la figure.
    
    Parameters
    ----------
    modele : RandomForestRegressor
        Modele entraine
    """
    import matplotlib.cm as cm
    
    ANNEE = 2020
    MOIS = 4
    CHUNK_SIZE = 500
    
    # Dossiers
    DOSSIER_CHELSA_RASTERS = Path("/content/drive/MyDrive/Colab Notebooks/dataStage/downscaling_25m")
    DOSSIER_RASTERS = DOSSIER_OUTPUT / "rasters_30m"
    DOSSIER_RASTERS.mkdir(exist_ok=True)
    
    print("=" * 70)
    print(f"  VISUALISATION AVANT/APRES - AVRIL {ANNEE}")
    print("=" * 70)
    
    # --- Etape 1 : Chargement rasters ---
    print("\nEtape 1 : Chargement rasters")
    print("-" * 60)
    
    with rasterio.open(FICHIER_MNT) as src:
        mnt_data = src.read(1)
    
    fichier_chelsa = DOSSIER_CHELSA_RASTERS / f"T_CHELSA_30m_Delphinium_{ANNEE}{MOIS:02d}.tif"
    with rasterio.open(fichier_chelsa) as src:
        chelsa_data = src.read(1)
        profile = src.profile.copy()
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    print(f"   MNT shape : {mnt_data.shape}")
    print(f"   CHELSA shape : {chelsa_data.shape}")
    
    # --- Etape 2 : Prediction par blocs ---
    print(f"\nEtape 2 : Prediction par blocs (chunk={CHUNK_SIZE})")
    print("-" * 60)
    
    h, w = chelsa_data.shape
    rf_t = np.full((h, w), np.nan, dtype=np.float32)
    
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
            
            chelsa_chunk = chelsa_data[i:i_end, j:j_end].ravel()
            mnt_chunk = mnt_data[i:i_end, j:j_end].ravel()
            
            mask = (~np.isnan(chelsa_chunk)) & (~np.isnan(mnt_chunk))
            
            if mask.sum() == 0:
                continue
            
            n_valid = mask.sum()
            features = np.zeros((n_valid, 5), dtype=np.float32)
            features[:, 0] = chelsa_chunk[mask]
            features[:, 1] = 0
            features[:, 2] = mnt_chunk[mask]
            features[:, 3] = mois_sin
            features[:, 4] = mois_cos
            
            predictions = modele.predict(features)
            
            chunk_pred = np.full_like(chelsa_chunk, np.nan)
            chunk_pred[mask] = predictions
            rf_t[i:i_end, j:j_end] = chunk_pred.reshape((i_end - i, j_end - j))
            
            if n_done % 50 == 0:
                print(f"   Bloc {n_done}/{n_blocks_total}")
    
    print(f"   Termine : {n_done} blocs traites")
    
    # --- Etape 3 : Sauvegarde raster ---
    print("\nEtape 3 : Sauvegarde raster")
    print("-" * 60)
    
    fichier_sortie = DOSSIER_RASTERS / f"T_RF_30m_Delphinium_{ANNEE}{MOIS:02d}.tif"
    profile.update(dtype='float32', count=1, compress='lzw')
    
    with rasterio.open(fichier_sortie, 'w', **profile) as dst:
        dst.write(rf_t.astype('float32'), 1)
    
    n_valid = (~np.isnan(rf_t)).sum()
    t_moy = np.nanmean(rf_t)
    t_min = np.nanmin(rf_t)
    t_max = np.nanmax(rf_t)
    
    print(f"   T moyenne : {t_moy:+.2f} deg C")
    print(f"   T range : [{t_min:.1f}, {t_max:.1f}] deg C")
    print(f"   Pixels valides : {n_valid:,}")
    print(f"   Sauvegarde : {fichier_sortie.name}")
    
    # --- Etape 4 : Visualisation ---
    print("\nEtape 4 : Visualisation avant/apres")
    print("-" * 60)
    
    # Sites Delphinium
    df_chelsa = pd.read_csv(FICHIER_CHELSA_POINTS)
    sites = df_chelsa[df_chelsa['type'] == 'SITE'][
        ['nom', 'lat', 'lon']
    ].drop_duplicates()
    
    # Reduire la taille pour visualisation (downsample x10)
    # Le raster 30m est trop gros pour matplotlib
    factor = 10
    chelsa_viz = chelsa_data[::factor, ::factor]
    rf_viz = rf_t[::factor, ::factor]
    print(f"   Downsampling x{factor} pour visualisation")
    print(f"   Nouvelle shape : {rf_viz.shape}")
    
    # Echelle commune
    all_vals = np.concatenate([
        chelsa_viz[~np.isnan(chelsa_viz)].ravel(),
        rf_viz[~np.isnan(rf_viz)].ravel()
    ])
    vmin = np.percentile(all_vals, 2)
    vmax = np.percentile(all_vals, 98)
    
    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    
    cmap = plt.get_cmap('RdYlBu_r').copy()
    cmap.set_bad(color='lightgray', alpha=0.5)
    
    # AVANT - CHELSA 30m
    ax1 = axes[0]
    im1 = ax1.imshow(np.ma.masked_invalid(chelsa_viz), cmap=cmap,
                      extent=extent, origin='upper',
                      vmin=vmin, vmax=vmax, interpolation='nearest')
    ax1.set_title(f'AVANT - CHELSA 1km resample a 30m\nAvril {ANNEE}',
                    fontsize=14, fontweight='bold')
    
    # APRES - RF 30m
    ax2 = axes[1]
    im2 = ax2.imshow(np.ma.masked_invalid(rf_viz), cmap=cmap,
                      extent=extent, origin='upper',
                      vmin=vmin, vmax=vmax, interpolation='nearest')
    ax2.set_title(f'APRES - Downscaling Random Forest 30m\nAvril {ANNEE}',
                    fontsize=14, fontweight='bold')
    
    # Sites Delphinium
    for ax in [ax1, ax2]:
        for _, row in sites.iterrows():
            couleur = '#27AE60' if row['nom'] == 'NOHEDES' else '#9B59B6'
            ax.scatter(row['lon'], row['lat'], s=150, marker='*',
                        c=couleur, edgecolor='black', linewidth=1.5, zorder=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(alpha=0.3, linestyle=':')
    
    # Colorbar unique
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax, label='Temperature (deg C)')
    
    plt.suptitle(f'Downscaling Temperature - Comparaison avant/apres',
                  fontsize=16, fontweight='bold')
    
    chemin_fig = DOSSIER_OUTPUT / f"comparaison_avant_apres_{ANNEE}{MOIS:02d}.png"
    fig.savefig(chemin_fig, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"   Figure sauvegardee : {chemin_fig.name}")
    print("\n" + "=" * 70)
    print("  VISUALISATION TERMINEE")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# 12. LANCEMENT DU SCRIPT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    modele, metriques, importance, sites = pipeline_downscaling()
    
    # Visualisation avant/apres pour avril 2020
    visualisation_avril_2020(modele)
    
    print("\nPipeline termine avec succes")

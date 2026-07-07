"""
════════════════════════════════════════════════════════════════════
DOWNSCALING DE LA HAUTEUR DE NEIGE PAR RANDOM FOREST
════════════════════════════════════════════════════════════════════

Projet : D2ClimAFLo-Pyr (Delphinium montanum a NOHEDES)
Auteur : Amadou Fofana
Date   : 2026
Version : 1.0

OBJECTIF DU SCRIPT
──────────────────
Ce script downscale la hauteur de neige ERA5 (resolution ~10 km) 
vers les positions des 9 sites Delphinium des Pyrenees Orientales,
en utilisant les stations Meteo-France comme reference.

METHODE
───────
1. On dispose de :
   - ERA5 hauteur neige (source basse resolution)
   - Stations Meteo-France (vraies mesures = TARGET : OBS_sde_cm_moy)
   - MNT SRTM 30m (topographie fine)

2. On entraine un modele Random Forest :
   - X (features) = ERA5 + topographie + saison
   - y (target)   = vraies mesures Meteo-France (OBS_sde_cm_moy)

3. On applique le modele aux 9 sites Delphinium
   pour obtenir leur hauteur de neige downscalee.

FICHIERS UTILISES
─────────────────
- data/training_RF_neige_mensuel.csv : jeu d'entrainement
- data/chelsa_par_point_complet.csv : positions des 9 sites
- MNT/MNT_Pyrenees_30m.tif : MNT 30m (sur Google Drive)

Pour la visualisation avant/apres (avril 2020) :
- downscaling_25m/NEIGE_ERA5_25m_Delphinium_202004.tif (AVANT)
- downscaling_25m/NEIGE_RF_30m_Delphinium_202004.tif (APRES)

FEATURES UTILISEES
──────────────────
- ERA5_sde_cm  : Hauteur neige ERA5 (cm)
- alt_mnt      : Altitude fine (SRTM 30m, en metres)
- delta_alt    : Ecart altitude station vs MNT
- mois_sin     : Composante sinus du mois (saisonnalite)
- mois_cos     : Composante cosinus du mois (saisonnalite)

VALIDATION
──────────
Cross-Validation 5 folds sur le jeu d'entrainement

Metriques :
- R2    : coefficient de determination (0-1, plus haut = mieux)
- RMSE  : erreur quadratique moyenne (cm, plus bas = mieux)
- MAE   : erreur absolue moyenne (cm, plus bas = mieux)
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


# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Racine du projet
RACINE_PROJET = Path(__file__).parent.parent

# Donnees GitHub
DOSSIER_DATA = RACINE_PROJET / "data"
FICHIER_TRAIN = DOSSIER_DATA / "training_RF_neige_mensuel.csv"
FICHIER_CHELSA_POINTS = DOSSIER_DATA / "chelsa_par_point_complet.csv"

# Donnees Drive
DOSSIER_DRIVE = Path("/content/drive/MyDrive/Colab Notebooks/dataStage")
DOSSIER_RASTERS_30M = DOSSIER_DRIVE / "downscaling_25m"
FICHIER_MNT = DOSSIER_DRIVE / "MNT" / "MNT_Pyrenees_30m.tif"

# Sorties
DOSSIER_OUTPUT = RACINE_PROJET / "outputs"
DOSSIER_OUTPUT.mkdir(exist_ok=True)

# Parametres Random Forest
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 20,
    'min_samples_leaf': 3,
    'random_state': 42,
    'n_jobs': -1
}

N_FOLDS = 5

# Features utilisees
COLONNES_FEATURES = [
    'ERA5_sde_cm',
    'alt_mnt',
    'delta_alt',
    'mois_sin',
    'mois_cos'
]

COLONNE_TARGET = 'OBS_sde_cm_moy'


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
    print(f"   y range : [{y.min():.2f}, {y.max():.2f}] cm")
    
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
    print(f"   RMSE  = {rmse:.2f} cm")
    print(f"   MAE   = {mae:.2f} cm")
    print(f"   Biais = {biais:+.2f} cm")
    
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
    print(f"   MAE training : {mae_train:.2f} cm")
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
                     f"RMSE = {metriques['RMSE']:.2f} cm  |  "
                     f"Biais = {metriques['biais']:+.2f} cm")
    ax.set_title(titre_complet, fontsize=13, fontweight='bold')
    ax.set_xlabel('Hauteur neige reelle (cm)', fontsize=12)
    ax.set_ylabel('Hauteur neige predite (cm)', fontsize=12)
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
    fichier_mnt : Path
        Chemin vers le MNT
    coords_lon_lat : list de tuples
        Liste de (longitude, latitude)
    
    Returns
    -------
    altitudes : list
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


def predire_sites_delphinium(modele, fichier_chelsa, fichier_mnt, df_train):
    """
    Applique le modele aux 9 sites Delphinium.
    
    Pour la neige, on utilise les moyennes d'ERA5 par site+mois
    (calculees depuis le training) pour predire.
    
    Parameters
    ----------
    modele : RandomForestRegressor
        Modele entraine
    fichier_chelsa : Path
        CSV chelsa_par_point_complet.csv (pour positions sites)
    fichier_mnt : Path
        MNT 30m
    df_train : pd.DataFrame
        Donnees d'entrainement (pour recuperer ERA5)
    
    Returns
    -------
    df_sites : pd.DataFrame
        Predictions pour les 9 sites
    """
    print(f"\nPrediction sur les 9 sites Delphinium")
    print("-" * 60)
    
    # Charger les positions des 9 sites
    df_chelsa = pd.read_csv(fichier_chelsa)
    sites_pos = df_chelsa[df_chelsa['type'] == 'SITE'][
        ['nom', 'lat', 'lon', 'altitude_meta']
    ].drop_duplicates()
    
    # Extraire alt_mnt aux positions des sites
    coords = list(zip(sites_pos['lon'], sites_pos['lat']))
    sites_pos['alt_mnt'] = extraire_altitude_mnt(fichier_mnt, coords)
    sites_pos['delta_alt'] = sites_pos['altitude_meta'] - sites_pos['alt_mnt']
    
    # Pour la neige : on utilise la moyenne ERA5 par mois (de tous les sites train)
    era5_moyennes_mois = df_train.groupby('mois')['ERA5_sde_cm'].mean()
    
    # Creer un dataset : chaque site x chaque mois de 2020
    predictions = []
    for _, site in sites_pos.iterrows():
        for mois in range(1, 13):
            predictions.append({
                'nom': site['nom'],
                'altitude': site['altitude_meta'],
                'lat': site['lat'],
                'lon': site['lon'],
                'alt_mnt': site['alt_mnt'],
                'delta_alt': site['delta_alt'],
                'mois': mois,
                'ERA5_sde_cm': era5_moyennes_mois.get(mois, 0),
                'mois_sin': np.sin(2 * np.pi * mois / 12),
                'mois_cos': np.cos(2 * np.pi * mois / 12)
            })
    
    df_sites = pd.DataFrame(predictions)
    df_sites = df_sites.dropna(subset=['alt_mnt'])
    
    # Predire
    df_sites['sde_RF_cm'] = modele.predict(df_sites[COLONNES_FEATURES])
    
    # Empecher valeurs negatives (physiquement impossible)
    df_sites['sde_RF_cm'] = df_sites['sde_RF_cm'].clip(lower=0)
    
    print(f"   {len(df_sites)} observations predites")
    print(f"   {df_sites['nom'].nunique()} sites")
    
    return df_sites


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
    print(f"\nStats hauteur neige annuelle par site")
    print("-" * 60)
    
    recap = df_sites.groupby('nom').agg(
        altitude=('altitude', 'first'),
        n_obs=('sde_RF_cm', 'count'),
        SDE_ERA5=('ERA5_sde_cm', 'mean'),
        SDE_RF=('sde_RF_cm', 'mean'),
    ).round(2).sort_values('altitude')
    recap['correction'] = (recap['SDE_RF'] - recap['SDE_ERA5']).round(2)
    
    print(recap.to_string())
    
    print(f"\nHauteur neige saisonniere RF par site")
    print("-" * 60)
    
    saisons = {
        'hiver': [12, 1, 2],
        'printemps': [3, 4, 5],
        'ete': [6, 7, 8],
        'automne': [9, 10, 11]
    }
    
    saison_stats = pd.DataFrame({'altitude': recap['altitude']})
    for sais, mois_list in saisons.items():
        sub = df_sites[df_sites['mois'].isin(mois_list)].groupby('nom')['sde_RF_cm'].mean()
        saison_stats[f'SDE_{sais}'] = sub.round(2)
    saison_stats = saison_stats.sort_values('altitude')
    
    print(saison_stats.to_string())
    
    return recap, saison_stats


# ═══════════════════════════════════════════════════════════════
# 10. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def pipeline_downscaling_neige():
    """
    Pipeline complet de downscaling neige avec validation et prediction.
    """
    print("=" * 70)
    print("  DOWNSCALING HAUTEUR DE NEIGE ERA5 -> 30m")
    print("=" * 70)
    
    # Etape 1
    df = charger_donnees_train(FICHIER_TRAIN)
    
    # Etape 2
    X, y = preparer_features_target(df, COLONNES_FEATURES, COLONNE_TARGET)
    
    # Etape 3
    y_pred_cv, metriques_cv = cross_validation(X, y, RF_PARAMS, N_FOLDS)
    
    # Etape 4
    fig_cv = plot_validation(y, y_pred_cv, metriques_cv,
                              titre="Cross-validation 5 folds - Neige")
    fig_cv.savefig(DOSSIER_OUTPUT / "validation_CV_neige.png",
                   dpi=150, bbox_inches='tight')
    print(f"\nGraphique CV sauvegarde")
    
    # Etape 5
    modele_final = entrainer_modele_final(X, y, RF_PARAMS)
    
    # Etape 6
    df_importance = calculer_importance(modele_final, COLONNES_FEATURES)
    
    # Etape 7
    df_sites = predire_sites_delphinium(modele_final,
                                          FICHIER_CHELSA_POINTS,
                                          FICHIER_MNT,
                                          df)
    
    # Etape 8
    recap, saison_stats = afficher_stats_sites(df_sites)
    
    # Resume
    print("\n" + "=" * 70)
    print("  RESUME FINAL")
    print("=" * 70)
    print(f"  R2 CV      : {metriques_cv['R2']:.3f}")
    print(f"  RMSE CV    : {metriques_cv['RMSE']:.2f} cm")
    print(f"  Biais CV   : {metriques_cv['biais']:+.2f} cm")
    print(f"  Nombre stations : {df['nom'].nunique() if 'nom' in df.columns else len(y)}")
    print(f"  Features utilisees : {len(COLONNES_FEATURES)}")
    print(f"  Sites predits      : {df_sites['nom'].nunique()}")
    print("=" * 70)
    
    return modele_final, metriques_cv, df_importance, df_sites


# ═══════════════════════════════════════════════════════════════
# 11. VISUALISATION AVANT / APRES - AVRIL 2020
# ═══════════════════════════════════════════════════════════════

def visualisation_avril_2020_neige():
    """
    Visualise le downscaling neige pour avril 2020.
    
    Utilise les rasters DEJA GENERES :
    - NEIGE_ERA5_25m_Delphinium_202004.tif (AVANT)
    - NEIGE_RF_30m_Delphinium_202004.tif (APRES)
    """
    ANNEE = 2020
    MOIS = 4
    
    print("=" * 70)
    print(f"  VISUALISATION NEIGE AVANT/APRES - AVRIL {ANNEE}")
    print("=" * 70)
    
    # Charger les 2 rasters existants
    fichier_era5 = DOSSIER_RASTERS_30M / f"NEIGE_ERA5_25m_Delphinium_{ANNEE}{MOIS:02d}.tif"
    fichier_rf = DOSSIER_RASTERS_30M / f"NEIGE_RF_30m_Delphinium_{ANNEE}{MOIS:02d}.tif"
    
    if not fichier_era5.exists():
        print(f"   ATTENTION : {fichier_era5.name} manquant")
        return
    if not fichier_rf.exists():
        print(f"   ATTENTION : {fichier_rf.name} manquant")
        return
    
    print(f"\nChargement rasters")
    print("-" * 60)
    
    with rasterio.open(fichier_era5) as src:
        era5_data = src.read(1)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    with rasterio.open(fichier_rf) as src:
        rf_data = src.read(1)
    
    print(f"   ERA5 shape : {era5_data.shape}")
    print(f"   RF shape : {rf_data.shape}")
    
    # Sites Delphinium
    df_chelsa = pd.read_csv(FICHIER_CHELSA_POINTS)
    sites = df_chelsa[df_chelsa['type'] == 'SITE'][
        ['nom', 'lat', 'lon']
    ].drop_duplicates()
    
    # Downsample pour visualisation
    factor = 10
    era5_viz = era5_data[::factor, ::factor]
    rf_viz = rf_data[::factor, ::factor]
    print(f"   Downsampling x{factor} pour visualisation")
    
    # Echelle commune
    all_vals = np.concatenate([
        era5_viz[~np.isnan(era5_viz)].ravel(),
        rf_viz[~np.isnan(rf_viz)].ravel()
    ])
    vmin = 0
    vmax = np.percentile(all_vals, 98)
    
    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad(color='lightgray', alpha=0.5)
    
    ax1 = axes[0]
    im1 = ax1.imshow(np.ma.masked_invalid(era5_viz), cmap=cmap,
                      extent=extent, origin='upper',
                      vmin=vmin, vmax=vmax, interpolation='nearest')
    ax1.set_title(f'AVANT - ERA5 hauteur neige\nAvril {ANNEE}',
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
    cbar = fig.colorbar(im2, cax=cbar_ax, label='Hauteur neige (cm)')
    
    plt.suptitle(f'Downscaling Hauteur de Neige - Comparaison avant/apres',
                  fontsize=16, fontweight='bold')
    
    chemin_fig = DOSSIER_OUTPUT / f"comparaison_avant_apres_NEIGE_{ANNEE}{MOIS:02d}.png"
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
    modele, metriques, importance, sites = pipeline_downscaling_neige()
    
    # Visualisation avant/apres pour avril 2020
    visualisation_avril_2020_neige()
    
    print("\nPipeline NEIGE termine avec succes")

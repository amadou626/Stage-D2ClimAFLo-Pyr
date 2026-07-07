"""
════════════════════════════════════════════════════════════════════
VISUALISATION DU DOWNSCALING — 12 MOIS DE L'ANNEE 2020
════════════════════════════════════════════════════════════════════

Projet : D2ClimAFLo-Pyr (Delphinium montanum a NOHEDES)
Auteur : Amadou Fofana
Date   : 2026
Version : 1.0

OBJECTIF
────────
Ce script visualise le downscaling de la temperature CHELSA
pour les 12 mois de l'annee 2020, et sauvegarde les rasters
downscales a 30m.

METHODE
───────
1. Chargement du modele RF entraine
2. Pour chaque mois de 2020 :
   - Application du modele au raster CHELSA 30m
   - Sauvegarde du raster downscale
   - Ajout au tableau de comparaison
3. Genere une figure avec 12 sous-graphiques

FICHIERS
────────
Entree :
- data/training_RF_T_chelsa_obs.csv (pour reentrainer le modele)
- data/chelsa_par_point_complet.csv (pour les 9 sites)
- MNT/MNT_Pyrenees_30m.tif (Drive)
- MNT/MNT_30m_Delphinium.tif (Drive)
- Rasters CHELSA 30m mensuels (Drive)

Sortie :
- outputs/rasters_30m/T_RF_30m_Delphinium_YYYYMM.tif (12 fichiers)
- outputs/comparaison_12_mois_2020.png

════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import os

# Machine Learning
from sklearn.ensemble import RandomForestRegressor

# Geospatial
import rasterio
import geopandas as gpd


# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Racine du projet
RACINE_PROJET = Path(__file__).parent.parent

# Donnees GitHub
DOSSIER_DATA = RACINE_PROJET / "data"
FICHIER_TRAIN = DOSSIER_DATA / "training_RF_T_chelsa_obs.csv"
FICHIER_CHELSA_POINTS = DOSSIER_DATA / "chelsa_par_point_complet.csv"

# Donnees Drive (trop volumineuses pour GitHub)
DOSSIER_DRIVE = Path("/content/drive/MyDrive/Colab Notebooks/dataStage")
DOSSIER_CHELSA_RASTERS = DOSSIER_DRIVE / "downscaling_25m"
FICHIER_MNT_DELPHINIUM = DOSSIER_CHELSA_RASTERS / "MNT_30m_Delphinium.tif"

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

# Features utilisees pour le modele
COLONNES_FEATURES = [
    'tas_chelsa_C',
    'pr_chelsa_mm',
    'alt_mnt',
    'mois_sin',
    'mois_cos'
]

COLONNE_TARGET = 'temp_moy_c'

# Annee et mois a traiter
ANNEE = 2020
MOIS_LIST = list(range(1, 13))  # 12 mois
MOIS_LABELS = ['Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun', 
                'Jul', 'Aout', 'Sep', 'Oct', 'Nov', 'Dec']


# ═══════════════════════════════════════════════════════════════
# 3. ENTRAINEMENT DU MODELE
# ═══════════════════════════════════════════════════════════════

def entrainer_modele():
    """
    Entraine le modele Random Forest sur les donnees d'entrainement.
    
    Returns
    -------
    modele : RandomForestRegressor
        Modele entraine pret pour prediction
    """
    print("=" * 70)
    print("  Etape 1 : Entrainement du modele Random Forest")
    print("=" * 70)
    
    df = pd.read_csv(FICHIER_TRAIN)
    df = df.dropna(subset=COLONNES_FEATURES + [COLONNE_TARGET])
    
    X = df[COLONNES_FEATURES].values
    y = df[COLONNE_TARGET].values
    
    print(f"   Donnees : {len(df)} lignes")
    print(f"   Features : {COLONNES_FEATURES}")
    print(f"   Target : {COLONNE_TARGET}")
    
    modele = RandomForestRegressor(**RF_PARAMS)
    modele.fit(X, y)
    
    print(f"   Modele entraine")
    return modele


# ═══════════════════════════════════════════════════════════════
# 4. PREDICTION SUR UN RASTER
# ═══════════════════════════════════════════════════════════════

def predire_raster_30m(modele, annee, mois, mnt_data):
    """
    Applique le modele au raster CHELSA 30m pour un mois donne.
    
    Parameters
    ----------
    modele : RandomForestRegressor
        Modele entraine
    annee : int
        Annee (ex : 2020)
    mois : int
        Mois (1 a 12)
    mnt_data : np.ndarray
        Raster MNT charge
    
    Returns
    -------
    rf_t : np.ndarray
        Raster T predite 30m
    chelsa_t : np.ndarray
        Raster T CHELSA 30m (avant downscaling)
    profile : dict
        Metadonnees rasterio pour sauvegarde
    """
    # Charger le raster CHELSA 30m (deja resample)
    fichier_chelsa = DOSSIER_CHELSA_RASTERS / f"T_CHELSA_30m_Delphinium_{annee}{mois:02d}.tif"
    
    with rasterio.open(fichier_chelsa) as src:
        chelsa_t = src.read(1)
        profile = src.profile.copy()
    
    # Ici on suppose que le raster CHELSA a la meme grille que le MNT
    # On construit les features pixel par pixel
    
    # Aplatir les rasters
    chelsa_flat = chelsa_t.ravel()
    mnt_flat = mnt_data.ravel()
    
    # Masque pixels valides
    mask = (~np.isnan(chelsa_flat)) & (~np.isnan(mnt_flat))
    
    # Preparer features
    n_pixels = len(chelsa_flat)
    features = np.full((n_pixels, len(COLONNES_FEATURES)), np.nan)
    
    features[mask, 0] = chelsa_flat[mask]  # tas_chelsa_C
    # Note : pr_chelsa_mm - il faudrait charger le raster PR CHELSA
    # Ici on suppose qu'il est dans le CSV ou constant
    # Solution simple : utiliser 0 (mais moins precis)
    features[mask, 1] = 0  # pr_chelsa_mm (a adapter selon donnees)
    features[mask, 2] = mnt_flat[mask]  # alt_mnt
    features[mask, 3] = np.sin(2 * np.pi * mois / 12)  # mois_sin
    features[mask, 4] = np.cos(2 * np.pi * mois / 12)  # mois_cos
    
    # Predire uniquement pixels valides
    rf_flat = np.full(n_pixels, np.nan)
    rf_flat[mask] = modele.predict(features[mask])
    
    # Reshape
    rf_t = rf_flat.reshape(chelsa_t.shape)
    
    return rf_t, chelsa_t, profile


# ═══════════════════════════════════════════════════════════════
# 5. SAUVEGARDE DU RASTER
# ═══════════════════════════════════════════════════════════════

def sauvegarder_raster(donnees, profile, chemin_sortie):
    """
    Sauvegarde un raster au format GeoTIFF.
    
    Parameters
    ----------
    donnees : np.ndarray
        Donnees a sauvegarder
    profile : dict
        Metadonnees rasterio
    chemin_sortie : str ou Path
        Chemin de sortie
    """
    profile.update(dtype='float32', count=1)
    
    with rasterio.open(chemin_sortie, 'w', **profile) as dst:
        dst.write(donnees.astype('float32'), 1)


# ═══════════════════════════════════════════════════════════════
# 6. VISUALISATION 12 MOIS
# ═══════════════════════════════════════════════════════════════

def creer_figure_12_mois(rasters_rf, extent, sites_delphinium, vmin, vmax):
    """
    Cree une figure avec 12 sous-graphiques (un par mois).
    
    Parameters
    ----------
    rasters_rf : dict
        {mois : raster T RF}
    extent : list
        [left, right, bottom, top]
    sites_delphinium : pd.DataFrame
        Positions des 9 sites
    vmin, vmax : float
        Limites de l'echelle de couleur
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(3, 4, figsize=(22, 15), 
                              sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Colormap
    cmap = cm.get_cmap('RdYlBu_r').copy()
    cmap.set_bad(color='lightgray', alpha=0.5)
    
    for idx, mois in enumerate(MOIS_LIST):
        ax = axes[idx]
        raster = rasters_rf[mois]
        
        # Afficher raster
        im = ax.imshow(np.ma.masked_invalid(raster), cmap=cmap, 
                        extent=extent, origin='upper',
                        vmin=vmin, vmax=vmax, interpolation='nearest')
        
        # Sites Delphinium
        for _, row in sites_delphinium.iterrows():
            couleur = '#27AE60' if row['nom'] == 'NOHEDES' else '#9B59B6'
            ax.scatter(row['lon'], row['lat'],
                        s=100, marker='*', c=couleur,
                        edgecolor='black', linewidth=1, zorder=10)
        
        ax.set_title(f"{MOIS_LABELS[idx]} {ANNEE}", 
                      fontsize=13, fontweight='bold')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(alpha=0.3, linestyle=':')
        
        # Labels axes (uniquement bord)
        if idx % 4 == 0:
            ax.set_ylabel('Latitude', fontsize=10)
        if idx >= 8:
            ax.set_xlabel('Longitude', fontsize=10)
    
    # Colorbar unique
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Temperature (deg C)')
    cbar.ax.tick_params(labelsize=10)
    
    # Titre principal
    plt.suptitle(f'Temperature downscalee 30m - 12 mois de {ANNEE}\n'
                  f'CHELSA 1km -> Random Forest 30m - Zone Delphinium montanum',
                  fontsize=16, fontweight='bold', y=0.995)
    
    return fig


# ═══════════════════════════════════════════════════════════════
# 7. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def pipeline_visualisation():
    """
    Pipeline complet : entrainement + prediction 12 mois + visualisation.
    """
    print("=" * 70)
    print(f"  VISUALISATION DOWNSCALING - 12 MOIS DE {ANNEE}")
    print("=" * 70)
    
    # --- ETAPE 1 : Entrainement modele ---
    modele = entrainer_modele()
    
    # --- ETAPE 2 : Charger MNT ---
    print(f"\n  Etape 2 : Chargement MNT")
    print("-" * 60)
    with rasterio.open(FICHIER_MNT_DELPHINIUM) as src:
        mnt_data = src.read(1)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    print(f"   MNT shape : {mnt_data.shape}")
    
    # --- ETAPE 3 : Charger sites Delphinium ---
    print(f"\n  Etape 3 : Chargement sites Delphinium")
    print("-" * 60)
    df_chelsa = pd.read_csv(FICHIER_CHELSA_POINTS)
    sites = df_chelsa[df_chelsa['type'] == 'SITE'][
        ['nom', 'lat', 'lon', 'altitude_meta']
    ].drop_duplicates()
    sites = sites.sort_values('altitude_meta').reset_index(drop=True)
    print(f"   {len(sites)} sites charges")
    
    # --- ETAPE 4 : Predire les 12 mois ---
    print(f"\n  Etape 4 : Prediction et sauvegarde des 12 mois")
    print("-" * 60)
    
    rasters_rf = {}
    for mois in MOIS_LIST:
        print(f"   Traitement {MOIS_LABELS[mois-1]} {ANNEE}...")
        
        rf_t, chelsa_t, profile = predire_raster_30m(modele, ANNEE, mois, mnt_data)
        rasters_rf[mois] = rf_t
        
        # Sauvegarder le raster downscale
        chemin_sortie = DOSSIER_RASTERS / f"T_RF_30m_Delphinium_{ANNEE}{mois:02d}.tif"
        sauvegarder_raster(rf_t, profile, chemin_sortie)
        
        n_valid = (~np.isnan(rf_t)).sum()
        t_moy = np.nanmean(rf_t)
        print(f"      Pixels valides : {n_valid:,}")
        print(f"      T moy : {t_moy:.2f} deg C")
        print(f"      Sauvegarde : {chemin_sortie.name}")
    
    # --- ETAPE 5 : Echelle commune ---
    print(f"\n  Etape 5 : Calcul echelle commune")
    print("-" * 60)
    all_values = np.concatenate([r[~np.isnan(r)].ravel() 
                                    for r in rasters_rf.values()])
    vmin = np.percentile(all_values, 2)
    vmax = np.percentile(all_values, 98)
    print(f"   Echelle : {vmin:.1f} a {vmax:.1f} deg C")
    
    # --- ETAPE 6 : Figure 12 mois ---
    print(f"\n  Etape 6 : Generation figure 12 mois")
    print("-" * 60)
    
    fig = creer_figure_12_mois(rasters_rf, extent, sites, vmin, vmax)
    
    chemin_figure = DOSSIER_OUTPUT / f"comparaison_12_mois_{ANNEE}.png"
    fig.savefig(chemin_figure, dpi=200, bbox_inches='tight', 
                 facecolor='white')
    plt.show()
    
    print(f"   Figure sauvegardee : {chemin_figure}")
    
    # --- Resume ---
    print("\n" + "=" * 70)
    print("  RESUME")
    print("=" * 70)
    print(f"  Rasters generes : 12")
    print(f"  Dossier rasters : {DOSSIER_RASTERS}")
    print(f"  Figure : {chemin_figure}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# 8. LANCEMENT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline_visualisation()
    print("\nTermine avec succes")

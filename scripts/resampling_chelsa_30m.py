"""
════════════════════════════════════════════════════════════════════
RESAMPLING CHELSA 1km VERS 30m — 12 MOIS
════════════════════════════════════════════════════════════════════

Projet : D2ClimAFLo-Pyr (Delphinium montanum a NOHEDES)
Auteur : Amadou Fofana
Date   : 2026
Version : 1.0

OBJECTIF
────────
Ce script convertit les rasters CHELSA (temperature air, 1km) 
vers la resolution 30m sur la zone Delphinium.

Il utilise le MNT_Pyrenees_30m.tif comme reference geometrique.

METHODE
───────
Pour chaque mois de l'annee choisie :
1. Charger le raster CHELSA 1km
2. Le resampler a 30m via bilinear interpolation
3. Aligner sur la grille du MNT 30m
4. Convertir Kelvin -> Celsius (CHELSA V2.1)
5. Sauvegarder le nouveau raster

FICHIERS
────────
Entree (Drive) :
- CHELSA/monthly/tas/CHELSA_tas_MM_YYYY_V.2.1.tif (raster 1km)
- MNT/MNT_Pyrenees_30m.tif (reference geometrique)

Sortie (Drive) :
- downscaling_25m/T_CHELSA_30m_Delphinium_YYYYMM.tif (12 rasters)

════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════════════════════════════

import numpy as np
from pathlib import Path

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum


# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Chemins sur Drive
DOSSIER_DRIVE = Path("/content/drive/MyDrive/Colab Notebooks/dataStage")
DOSSIER_CHELSA_1KM = DOSSIER_DRIVE / "CHELSA" / "monthly" / "tas"
FICHIER_MNT = DOSSIER_DRIVE / "MNT" / "MNT_Pyrenees_30m.tif"
DOSSIER_SORTIE = DOSSIER_DRIVE / "downscaling_25m"

# Creer le dossier de sortie s'il n'existe pas
DOSSIER_SORTIE.mkdir(exist_ok=True, parents=True)

# Annee et mois a traiter
ANNEE = 2020
MOIS_LIST = list(range(1, 13))  # 12 mois
MOIS_LABELS = ['Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun', 
                'Jul', 'Aout', 'Sep', 'Oct', 'Nov', 'Dec']


# ═══════════════════════════════════════════════════════════════
# 3. FONCTION DE RESAMPLING
# ═══════════════════════════════════════════════════════════════

def resampler_chelsa_vers_mnt(fichier_chelsa, fichier_mnt, fichier_sortie):
    """
    Resample un raster CHELSA 1km vers la grille du MNT 30m.
    
    Convertit aussi les temperatures de Kelvin vers Celsius.
    
    Parameters
    ----------
    fichier_chelsa : Path
        Raster CHELSA 1km (temperature en Kelvin)
    fichier_mnt : Path
        Raster MNT 30m (reference geometrique)
    fichier_sortie : Path
        Chemin du raster de sortie 30m
    
    Returns
    -------
    dict avec statistiques
    """
    # Charger le MNT comme reference
    with rasterio.open(fichier_mnt) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
        ref_bounds = ref.bounds
    
    # Charger CHELSA source
    with rasterio.open(fichier_chelsa) as src:
        # Preparer le raster de destination (vide)
        destination = np.zeros((ref_height, ref_width), dtype=np.float32)
        
        # Reprojeter et resampler
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
        
        # CHELSA V2.1 est en Kelvin * 10 
        # (verifier avec metadata : factor = 0.1, offset = -273.15)
        # Conversion : (valeur * 0.1) - 273.15
        # Ou si deja convertit : verifier
        
        # Detecter si CHELSA est en K ou C
        val_moyenne = np.nanmean(destination)
        if val_moyenne > 100:
            # C'est en Kelvin ou Kelvin*10
            if val_moyenne > 1000:
                # Kelvin * 10
                destination_c = (destination * 0.1) - 273.15
            else:
                # Kelvin normal
                destination_c = destination - 273.15
        else:
            # Deja en Celsius
            destination_c = destination
        
        # Remplacer les valeurs invalides par NaN
        destination_c = np.where(np.isnan(destination_c), np.nan, destination_c)
        
        # Preparer le profil de sortie
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': np.nan,
            'width': ref_width,
            'height': ref_height,
            'count': 1,
            'crs': ref_crs,
            'transform': ref_transform,
            'compress': 'lzw'
        }
        
        # Sauvegarder
        with rasterio.open(fichier_sortie, 'w', **profile) as dst:
            dst.write(destination_c.astype('float32'), 1)
    
    # Statistiques
    stats = {
        't_min': float(np.nanmin(destination_c)),
        't_max': float(np.nanmax(destination_c)),
        't_moy': float(np.nanmean(destination_c)),
        'n_valid': int((~np.isnan(destination_c)).sum())
    }
    
    return stats


# ═══════════════════════════════════════════════════════════════
# 4. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def pipeline_resampling():
    """
    Traite les 12 mois de l'annee choisie.
    """
    print("=" * 70)
    print(f"  RESAMPLING CHELSA 1km -> 30m")
    print(f"  Annee : {ANNEE} (12 mois)")
    print("=" * 70)
    
    # Verifier que le MNT existe
    if not FICHIER_MNT.exists():
        raise FileNotFoundError(f"MNT introuvable : {FICHIER_MNT}")
    print(f"\nMNT : {FICHIER_MNT.name}")
    
    with rasterio.open(FICHIER_MNT) as src:
        print(f"   Shape : {src.shape}")
        print(f"   CRS : {src.crs}")
        print(f"   Bounds : {src.bounds}")
    
    # Traiter chaque mois
    print(f"\nTraitement des 12 mois")
    print("-" * 70)
    
    for mois in MOIS_LIST:
        # Chemin CHELSA source
        nom_chelsa = f"CHELSA_tas_{mois:02d}_{ANNEE}_V.2.1.tif"
        fichier_chelsa = DOSSIER_CHELSA_1KM / nom_chelsa
        
        # Chemin sortie
        nom_sortie = f"T_CHELSA_30m_Delphinium_{ANNEE}{mois:02d}.tif"
        fichier_sortie = DOSSIER_SORTIE / nom_sortie
        
        # Verifier existence
        if not fichier_chelsa.exists():
            print(f"   [MANQUANT] {MOIS_LABELS[mois-1]} : {nom_chelsa}")
            continue
        
        # Resampler
        stats = resampler_chelsa_vers_mnt(fichier_chelsa, FICHIER_MNT, 
                                             fichier_sortie)
        
        print(f"   {MOIS_LABELS[mois-1]} {ANNEE} : "
              f"T={stats['t_moy']:+.1f}C "
              f"[{stats['t_min']:.1f}, {stats['t_max']:.1f}] "
              f"n={stats['n_valid']:,}")
    
    # Resume
    print("\n" + "=" * 70)
    print("  RESUME")
    print("=" * 70)
    
    fichiers_generes = list(DOSSIER_SORTIE.glob(
        f"T_CHELSA_30m_Delphinium_{ANNEE}*.tif"
    ))
    print(f"  Fichiers generes : {len(fichiers_generes)}")
    print(f"  Dossier sortie : {DOSSIER_SORTIE}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# 5. LANCEMENT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline_resampling()
    print("\nTermine avec succes")

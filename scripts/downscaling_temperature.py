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
vers une résolution de 30 mètres pour la zone d'étude 
(Pyrénées Orientales).

MÉTHODE
───────
1. On dispose de :
   - CHELSA à 1 km (source basse résolution)
   - Stations Météo-France (vraies mesures = TARGET)
   - MNT SRTM 30m (topographie fine)

2. On entraîne un modèle Random Forest :
   - X (features) = topographie fine + valeurs CHELSA au point
   - y (target)   = vraies mesures Météo-France
   
3. On applique le modèle à toute la zone à 30m 
   pour obtenir une carte de température fine.

FEATURES UTILISÉES
──────────────────
- tas_chelsa_C     : Température CHELSA au point (°C)
- pr_chelsa_mm     : Précipitations CHELSA (mm)
- alt_mnt          : Altitude fine (SRTM 30m, en mètres)
- mois_sin         : Composante sinus du mois (saisonnalité)
- mois_cos         : Composante cosinus du mois (saisonnalité)

VALIDATION
──────────
Cross-Validation 5 folds (qualité du modèle)

Métriques :
- R² : coefficient de détermination (entre 0 et 1, plus haut = mieux)
- RMSE : erreur quadratique moyenne (en °C, plus bas = mieux)  
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


# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Chemins RELATIFS pour compatibilite GitHub
# Le code suppose la structure suivante :
#   projet/
#     scripts/downscaling_temperature.py  (ce fichier)
#     data/donnees_train.csv
#     data/MNT_30m_Delphinium.tif
#     outputs/  (cree automatiquement)

# Racine du projet (un niveau au-dessus du dossier scripts/)
RACINE_PROJET = Path(__file__).parent.parent

DOSSIER_DATA = RACINE_PROJET / "data"
FICHIER_TRAIN = DOSSIER_DATA / "training_RF_T_chelsa_obs.csv"
FICHIER_MNT = DOSSIER_DATA / "MNT_30m_Delphinium.tif"
DOSSIER_OUTPUT = RACINE_PROJET / "outputs"
DOSSIER_OUTPUT.mkdir(exist_ok=True)

# Parametres Random Forest
RF_PARAMS = {
    'n_estimators': 500,      # Nombre d'arbres
    'max_depth': 20,          # Profondeur max des arbres
    'min_samples_split': 5,   # Nb min echantillons pour split
    'random_state': 42,       # Reproductibilite
    'n_jobs': -1              # Utiliser tous les CPUs
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
    - nom, date, annee, mois (identifiants temporels)
    - temp_moy_c (target : vraie temperature)
    - tas_chelsa_C, pr_chelsa_mm (features CHELSA)
    - lat, lon, altitude, alt_mnt (localisation)
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
    print(f"Chargement des donnees : {fichier_csv}")
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
        Matrice features (n_echantillons, n_features)
    y : np.ndarray
        Vecteur target (n_echantillons,)
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
    
    Cela donne une estimation ROBUSTE de la performance.
    
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
        R2, RMSE, biais, MAE
    """
    print(f"\nCross-validation {n_folds} folds")
    print("-" * 60)
    
    # Creer le splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Creer le modele
    modele = RandomForestRegressor(**params)
    
    # Predictions cross-validees
    y_pred_cv = cross_val_predict(modele, X, y, cv=kf, n_jobs=-1)
    
    # Calcul des metriques
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
    
    Ce modele sera utilise pour predire a 30m sur toute la zone.
    
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
    importance : np.ndarray
        Importance de chaque feature
    """
    print(f"\nEntrainement du modele final")
    print("-" * 60)
    
    modele = RandomForestRegressor(**params)
    modele.fit(X, y)
    
    print(f"   Modele entraine sur {len(y)} echantillons")
    
    # Importance des features
    importance = modele.feature_importances_
    print(f"\nImportance des features (%):")
    
    return modele, importance


# ═══════════════════════════════════════════════════════════════
# 7. VISUALISATION DES RÉSULTATS DE VALIDATION
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
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Scatter plot
    ax.scatter(y_reel, y_pred, alpha=0.5, s=30, color='#1976D2', 
                 edgecolor='white', linewidth=0.5)
    
    # Droite Y = X (reference parfaite)
    lim_min = min(y_reel.min(), y_pred.min()) - 1
    lim_max = max(y_reel.max(), y_pred.max()) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 
              'r--', linewidth=2, label='Y = X (parfait)')
    
    # Titre avec metriques
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
# 8. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def pipeline_downscaling():
    """
    Pipeline complet de downscaling avec validation.
    
    Etapes :
    1. Chargement des donnees d'entrainement
    2. Preparation features et target
    3. Cross-validation 5 folds
    4. Graphique de validation
    5. Entrainement du modele final
    6. Resume des resultats
    
    Returns
    -------
    modele : RandomForestRegressor
        Modele final entraine
    metriques : dict
        Metriques de validation
    df_importance : pd.DataFrame
        Importance des features
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
    modele_final, importance = entrainer_modele_final(X, y, RF_PARAMS)
    
    # Tableau importance
    df_importance = pd.DataFrame({
        'feature': COLONNES_FEATURES,
        'importance_pct': importance * 100
    }).sort_values('importance_pct', ascending=False)
    
    print(df_importance.to_string(index=False))
    
    # --- ETAPE 6 : Resume final ---
    print("\n" + "=" * 70)
    print("  RESUME FINAL")
    print("=" * 70)
    print(f"  R2 CV      : {metriques_cv['R2']:.3f}")
    print(f"  RMSE CV    : {metriques_cv['RMSE']:.2f} deg C")
    print(f"  Biais CV   : {metriques_cv['biais']:+.2f} deg C")
    print(f"  Nombre de stations : {len(y)}")
    print(f"  Features utilisees : {len(COLONNES_FEATURES)}")
    print("=" * 70)
    
    return modele_final, metriques_cv, df_importance


# ═══════════════════════════════════════════════════════════════
# 9. LANCEMENT DU SCRIPT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    modele, metriques, importance = pipeline_downscaling()
    
    print("\nPipeline termine avec succes")

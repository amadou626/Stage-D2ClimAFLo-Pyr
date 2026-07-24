"""
════════════════════════════════════════════════════════════════════
CALCUL DU NMI (Niche Margin Index) - APPROCHE CLASSIQUE
════════════════════════════════════════════════════════════════════

Projet : D2ClimAFLo-Pyr (Delphinium montanum a NOHEDES)
Auteur : Amadou Fofana
Date   : 2026
Version : 1.0

OBJECTIF DU SCRIPT
──────────────────
Ce script calcule le NMI pour chaque annee de NOHEDES.

Le NMI (Niche Margin Index) mesure la position d'un site par rapport
a la niche climatique de reference construite autour des 8 autres 
sites Delphinium fleurissants.

METHODE
───────
1. Charger les donnees climatiques pour tous les sites/annees
2. Calculer la moyenne des 8 variables climatiques par site (sur 21 ans)
3. Reduire a 2 dimensions par ACP
4. Construire la niche autour des 8 sites fleurissants (KDE)
5. Pour chaque annee de NOHEDES, calculer le NMI :
   - Si le point est DANS la niche : NMI positif
   - Si le point est HORS de la niche : NMI negatif

INTERPRETATION
──────────────
- NMI > 0  : NOHEDES est climatiquement similaire aux autres sites
- NMI < 0  : NOHEDES est climatiquement different
- NMI ~ 0  : NOHEDES est en limite de la niche

FICHIERS UTILISES
─────────────────
- data/df_enrichi_v4.csv : donnees climatiques par site/annee

════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde


# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Racine du projet
RACINE_PROJET = Path(__file__).parent.parent

# Donnees GitHub
DOSSIER_DATA = RACINE_PROJET / "data"
FICHIER_ENRICHI = DOSSIER_DATA / "df_enrichi_v4.csv"

# Sorties
DOSSIER_OUTPUT = RACINE_PROJET / "outputs"
DOSSIER_OUTPUT.mkdir(exist_ok=True)

# Periode d'analyse
ANNEE_MIN = 2000
ANNEE_MAX = 2020

# Les 8 variables climatiques utilisees
VARIABLES_X = [
    'SMOD_modis',           # Fin d'enneigement (satellite)
    'continuite',           # Continuite du manteau neigeux
    'temp_RF_moy',          # Temperature air moyenne
    'SCD',                  # Duree de la neige
    'humidity_moy',         # Humidite moyenne
    'neige_pct',            # Pourcentage de precipitations en neige
    'soil_temp_upper_moy',  # Temperature du sol (10 cm)
    'LFD_nival'             # Dernier jour de gel
]

# Seuil de densite pour la marge de la niche (5%)
NIVEAU_NICHE = 0.05


# ═══════════════════════════════════════════════════════════════
# 3. CHARGEMENT DES DONNEES
# ═══════════════════════════════════════════════════════════════

def charger_donnees():
    """
    Charge le fichier des donnees climatiques enrichies.
    
    Returns
    -------
    df : pd.DataFrame
        Donnees climatiques par site/annee
    """
    print(f"Chargement des donnees : {FICHIER_ENRICHI}")
    df = pd.read_csv(FICHIER_ENRICHI)
    df = df[(df['annee'] >= ANNEE_MIN) & (df['annee'] <= ANNEE_MAX)]
    
    n_sites = df['nom'].nunique()
    n_annees = df['annee'].nunique()
    print(f"   {len(df)} lignes chargees")
    print(f"   {n_sites} sites, {n_annees} annees")
    
    return df


# ═══════════════════════════════════════════════════════════════
# 4. CONSTRUCTION DE LA NICHE
# ═══════════════════════════════════════════════════════════════

def construire_niche(df, variables):
    """
    Construit la niche climatique autour des 8 sites fleurissants.
    
    Etapes :
    1. Calculer la moyenne 21 ans par site pour chaque variable
    2. Standardiser (Z-score)
    3. ACP pour reduction a 2 dimensions
    4. KDE pour estimer la densite
    5. Definir la marge au seuil de densite 5%
    
    Parameters
    ----------
    df : pd.DataFrame
        Donnees completes
    variables : list
        Liste des variables climatiques
    
    Returns
    -------
    dict avec tous les elements de la niche
    """
    print("\nConstruction de la niche")
    print("-" * 60)
    
    # Etape 1 : moyennes 21 ans par site
    df_moyen = df.groupby('nom')[variables].mean().dropna()
    print(f"   Nb sites avec donnees completes : {len(df_moyen)}")
    
    # Etape 2 : standardisation
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df_moyen.values)
    
    # Etape 3 : ACP
    pca = PCA(n_components=2)
    pca.fit(X_std)
    
    print(f"   Variance expliquee PC1 : {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"   Variance expliquee PC2 : {pca.explained_variance_ratio_[1]*100:.1f}%")
    
    # Etape 4 : coordonnees des 8 autres sites (hors NOHEDES)
    df_autres = df[df['nom'] != 'NOHEDES'][['nom', 'annee'] + variables].dropna()
    X_autres = df_autres[variables].values
    X_autres_std = scaler.transform(X_autres)
    coords_autres = pca.transform(X_autres_std)
    
    # Etape 5 : KDE (Kernel Density Estimation)
    kde = gaussian_kde(coords_autres.T)
    densites = kde(coords_autres.T)
    seuil_densite = np.quantile(densites, NIVEAU_NICHE)
    
    print(f"   Seuil densite (5%) : {seuil_densite:.4f}")
    
    # Etape 6 : construire la marge (points ou densite = seuil)
    x_min, x_max = coords_autres[:, 0].min(), coords_autres[:, 0].max()
    y_min, y_max = coords_autres[:, 1].min(), coords_autres[:, 1].max()
    x_ext = (x_max - x_min) * 0.5
    y_ext = (y_max - y_min) * 0.5
    
    gx = np.linspace(x_min - x_ext, x_max + x_ext, 200)
    gy = np.linspace(y_min - y_ext, y_max + y_ext, 200)
    XX, YY = np.meshgrid(gx, gy)
    pts_grid = np.vstack([XX.ravel(), YY.ravel()])
    densites_grid = kde(pts_grid).reshape(XX.shape)
    
    # Points sur la marge
    proche_marge = np.abs(densites_grid - seuil_densite) < 0.001
    points_marge = np.column_stack([XX[proche_marge], YY[proche_marge]])
    
    # dmax : distance max entre points pour normaliser NMI
    idx_noh = np.where(df_moyen.index == 'NOHEDES')[0]
    if len(idx_noh) > 0:
        coord_noh_moy = pca.transform(X_std[idx_noh])
        all_coords = np.vstack([coords_autres, coord_noh_moy])
    else:
        all_coords = coords_autres
    
    dmax = np.max([
        np.linalg.norm(p1 - p2) 
        for i, p1 in enumerate(all_coords[:50]) 
        for p2 in all_coords[i+1:]
    ]) if len(all_coords) > 1 else 1.0
    
    return {
        'scaler': scaler,
        'pca': pca,
        'kde': kde,
        'seuil_densite': seuil_densite,
        'points_marge': points_marge,
        'coords_autres': coords_autres,
        'dmax': dmax,
        'grid_x': gx,
        'grid_y': gy,
        'densites_grid': densites_grid
    }


# ═══════════════════════════════════════════════════════════════
# 5. CALCUL DU NMI
# ═══════════════════════════════════════════════════════════════

def calculer_nmi(valeurs, niche):
    """
    Calcule le NMI pour un point donne.
    
    Parameters
    ----------
    valeurs : array-like
        Valeurs des 8 variables climatiques
    niche : dict
        Elements de la niche (retour de construire_niche)
    
    Returns
    -------
    dict avec NMI, statut (DANS/HORS), coordonnees
    """
    X = np.array(valeurs).reshape(1, -1)
    X_std = niche['scaler'].transform(X)
    coord = niche['pca'].transform(X_std)[0]
    
    # Densite au point
    d_point = niche['kde'](coord.reshape(-1, 1))[0]
    
    # Distance a la marge la plus proche
    dist_marge = np.min(np.linalg.norm(niche['points_marge'] - coord, axis=1))
    
    # NMI normalise
    nmi_normalise = dist_marge / niche['dmax']
    
    if d_point >= niche['seuil_densite']:
        nmi = nmi_normalise
        statut = 'DANS'
    else:
        nmi = -nmi_normalise
        statut = 'HORS'
    
    return {
        'NMI': nmi,
        'statut': statut,
        'PC1': coord[0],
        'PC2': coord[1],
        'densite': d_point
    }


# ═══════════════════════════════════════════════════════════════
# 6. CALCUL DU NMI POUR NOHEDES (21 ANS)
# ═══════════════════════════════════════════════════════════════

def calculer_nmi_nohedes(df, niche, variables):
    """
    Calcule le NMI pour chaque annee de NOHEDES.
    
    Parameters
    ----------
    df : pd.DataFrame
        Donnees completes
    niche : dict
        Elements de la niche
    variables : list
        Variables climatiques
    
    Returns
    -------
    df_nmi : pd.DataFrame
        NMI par annee pour NOHEDES
    """
    print("\nCalcul du NMI pour NOHEDES (21 annees)")
    print("-" * 60)
    
    df_noh = df[df['nom'] == 'NOHEDES'][['annee'] + variables].dropna()
    
    resultats = []
    for _, row in df_noh.iterrows():
        res = calculer_nmi(row[variables].values, niche)
        res['annee'] = int(row['annee'])
        resultats.append(res)
    
    df_nmi = pd.DataFrame(resultats)
    df_nmi = df_nmi[['annee', 'NMI', 'statut', 'PC1', 'PC2', 'densite']]
    df_nmi = df_nmi.sort_values('annee').reset_index(drop=True)
    
    n_dans = (df_nmi['statut'] == 'DANS').sum()
    n_hors = (df_nmi['statut'] == 'HORS').sum()
    print(f"   Annees DANS niche : {n_dans}/{len(df_nmi)}")
    print(f"   Annees HORS niche : {n_hors}/{len(df_nmi)}")
    print(f"   Annees HORS : {df_nmi[df_nmi['statut'] == 'HORS']['annee'].tolist()}")
    
    return df_nmi


# ═══════════════════════════════════════════════════════════════
# 7. VISUALISATION
# ═══════════════════════════════════════════════════════════════

def visualiser_nmi(df_nmi, niche):
    """
    Cree 2 graphiques :
    1. Evolution du NMI dans le temps (2000-2020)
    2. Position de NOHEDES dans l'espace PCA
    
    Parameters
    ----------
    df_nmi : pd.DataFrame
        NMI par annee
    niche : dict
        Elements de la niche
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # ─── GRAPH 1 : Evolution NMI ───
    ax1 = axes[0]
    couleurs = ['#2E7D32' if s == 'DANS' else '#C62828' 
                 for s in df_nmi['statut']]
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.plot(df_nmi['annee'], df_nmi['NMI'], color='#1976D2', 
              linewidth=2, alpha=0.6, zorder=1)
    ax1.scatter(df_nmi['annee'], df_nmi['NMI'], c=couleurs, 
                 s=100, zorder=2, edgecolor='white', linewidth=1)
    
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.fill_between(df_nmi['annee'], 0, df_nmi['NMI'].max()*1.1, 
                       color='#2E7D32', alpha=0.05, label='DANS niche')
    ax1.fill_between(df_nmi['annee'], df_nmi['NMI'].min()*1.1, 0, 
                       color='#C62828', alpha=0.05, label='HORS niche')
    
    ax1.set_xlabel('Annee', fontsize=12)
    ax1.set_ylabel('NMI (Niche Margin Index)', fontsize=12)
    ax1.set_title('Evolution du NMI de NOHEDES (2000-2020)', 
                    fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(range(ANNEE_MIN, ANNEE_MAX+1, 2))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # ─── GRAPH 2 : Position PCA ───
    ax2 = axes[1]
    
    # Fond : zone de la niche
    mask_dans = niche['densites_grid'] >= niche['seuil_densite']
    Z_niche = np.where(mask_dans, niche['densites_grid'], np.nan)
    ax2.contourf(niche['grid_x'], niche['grid_y'], Z_niche,
                    levels=10, cmap='Blues', alpha=0.4)
    ax2.contour(niche['grid_x'], niche['grid_y'], niche['densites_grid'],
                  levels=[niche['seuil_densite']], colors='#1976D2', 
                  linewidths=2)
    
    # Points 8 sites
    ax2.scatter(niche['coords_autres'][:, 0], niche['coords_autres'][:, 1],
                 c='#9B59B6', s=40, alpha=0.5, label='8 sites fleurissants',
                 edgecolor='white', linewidth=0.5)
    
    # Points NOHEDES (colorer par statut)
    for _, row in df_nmi.iterrows():
        couleur = '#2E7D32' if row['statut'] == 'DANS' else '#C62828'
        ax2.scatter(row['PC1'], row['PC2'], c=couleur, s=120, 
                     marker='^', edgecolor='black', linewidth=1.2, 
                     zorder=10)
        ax2.annotate(str(row['annee']), (row['PC1'], row['PC2']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('PC1', fontsize=12)
    ax2.set_ylabel('PC2', fontsize=12)
    ax2.set_title('Position des 21 annees de NOHEDES\ndans l\'espace de la niche',
                    fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9B59B6',
                 markersize=10, label='8 sites fleurissants'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#2E7D32',
                 markersize=12, markeredgecolor='black', label='NOHEDES DANS'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#C62828',
                 markersize=12, markeredgecolor='black', label='NOHEDES HORS')
    ]
    ax2.legend(handles=legend_elements, fontsize=10, loc='best')
    
    plt.tight_layout()
    
    chemin_fig = DOSSIER_OUTPUT / "NMI_NOHEDES.png"
    fig.savefig(chemin_fig, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\nFigure sauvegardee : {chemin_fig.name}")


# ═══════════════════════════════════════════════════════════════
# 8. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def pipeline_nmi():
    """
    Pipeline complet du calcul NMI.
    """
    print("=" * 70)
    print("  CALCUL DU NMI (Niche Margin Index)")
    print("  Position de NOHEDES vs 8 sites fleurissants")
    print("=" * 70)
    
    # Etape 1 : Charger les donnees
    df = charger_donnees()
    
    # Etape 2 : Construire la niche
    niche = construire_niche(df, VARIABLES_X)
    
    # Etape 3 : Calculer NMI pour NOHEDES
    df_nmi = calculer_nmi_nohedes(df, niche, VARIABLES_X)
    
    # Etape 4 : Sauvegarder les resultats
    chemin_csv = DOSSIER_OUTPUT / "NMI_NOHEDES.csv"
    df_nmi.to_csv(chemin_csv, index=False)
    print(f"\nResultats sauvegardes : {chemin_csv.name}")
    
    # Etape 5 : Visualiser
    visualiser_nmi(df_nmi, niche)
    
    # Etape 6 : Resume
    print("\n" + "=" * 70)
    print("  RESUME DES RESULTATS")
    print("=" * 70)
    print(f"  Annees DANS niche : {(df_nmi['statut'] == 'DANS').sum()}/21")
    print(f"  Annees HORS niche : {(df_nmi['statut'] == 'HORS').sum()}/21")
    print(f"  NMI moyen : {df_nmi['NMI'].mean():.3f}")
    print(f"  NMI min : {df_nmi['NMI'].min():.3f} "
          f"(annee {df_nmi.loc[df_nmi['NMI'].idxmin(), 'annee']})")
    print(f"  NMI max : {df_nmi['NMI'].max():.3f} "
          f"(annee {df_nmi.loc[df_nmi['NMI'].idxmax(), 'annee']})")
    print("=" * 70)
    
    return df_nmi, niche


# ═══════════════════════════════════════════════════════════════
# 9. LANCEMENT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df_nmi, niche = pipeline_nmi()
    print("\nPipeline NMI termine avec succes")

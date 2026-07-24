"""
════════════════════════════════════════════════════════════════════
CALCUL DU NMI INVERSE - RESSEMBLANCE A NOHEDES
════════════════════════════════════════════════════════════════════

Projet : D2ClimAFLo-Pyr (Delphinium montanum a NOHEDES)
Auteur : Amadou Fofana
Date   : 2026
Version : 1.0

OBJECTIF DU SCRIPT
──────────────────
Ce script calcule le NMI inverse : au lieu de construire la niche
autour des 8 sites fleurissants et de mesurer si NOHEDES est dedans,
on construit la niche AUTOUR DE NOHEDES et on regarde quels autres
sites y ressemblent.

Cela permet d'identifier quels sites pourraient devenir "le prochain 
NOHEDES" en cas de derive climatique continue.

METHODE
───────
1. Charger les donnees climatiques pour tous les sites/annees
2. Construire la niche autour des 21 annees de NOHEDES
3. Pour chaque site/annee des 8 autres sites :
   - Calculer si le point est DANS ou HORS la niche NOHEDES
   - Calculer un score de ressemblance
4. Classer les sites par ordre decroissant de ressemblance

INTERPRETATION
──────────────
- Site avec beaucoup d'annees DANS la niche NOHEDES :
  → Climat proche de NOHEDES
  → Risque potentiel de suivre la meme trajectoire
  → Candidat pour "prochain NOHEDES"

- Site avec peu d'annees DANS la niche NOHEDES :
  → Climat different
  → Moins de risque

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

RACINE_PROJET = Path(__file__).parent.parent

DOSSIER_DATA = RACINE_PROJET / "data"
FICHIER_ENRICHI = DOSSIER_DATA / "df_enrichi_v4.csv"

DOSSIER_OUTPUT = RACINE_PROJET / "outputs"
DOSSIER_OUTPUT.mkdir(exist_ok=True)

ANNEE_MIN = 2000
ANNEE_MAX = 2020

# Les 8 variables climatiques
VARIABLES_X = [
    'SMOD_modis',
    'continuite',
    'temp_RF_moy',
    'SCD',
    'humidity_moy',
    'neige_pct',
    'soil_temp_upper_moy',
    'LFD_nival'
]

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
# 4. CONSTRUCTION DE LA NICHE AUTOUR DE NOHEDES
# ═══════════════════════════════════════════════════════════════

def construire_niche_nohedes(df, variables):
    """
    Construit la niche climatique autour des 21 annees de NOHEDES.
    
    Etapes :
    1. Calculer la moyenne 21 ans par site (pour l'ACP de reference)
    2. Standardiser
    3. ACP pour reduction a 2 dimensions
    4. Projeter les 21 annees de NOHEDES
    5. KDE sur les 21 annees de NOHEDES
    
    Parameters
    ----------
    df : pd.DataFrame
    variables : list
    
    Returns
    -------
    dict avec elements de la niche NOHEDES
    """
    print("\nConstruction de la niche autour de NOHEDES")
    print("-" * 60)
    
    # Etape 1 : moyennes 21 ans par site (base pour l'ACP)
    df_moyen = df.groupby('nom')[variables].mean().dropna()
    
    # Etape 2 : standardisation
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df_moyen.values)
    
    # Etape 3 : ACP
    pca = PCA(n_components=2)
    pca.fit(X_std)
    
    print(f"   Variance PC1 : {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"   Variance PC2 : {pca.explained_variance_ratio_[1]*100:.1f}%")
    
    # Etape 4 : coordonnees des 21 annees NOHEDES
    df_noh = df[df['nom'] == 'NOHEDES'][['annee'] + variables].dropna()
    X_noh = df_noh[variables].values
    X_noh_std = scaler.transform(X_noh)
    coords_noh = pca.transform(X_noh_std)
    
    print(f"   Nb annees NOHEDES : {len(coords_noh)}")
    
    # Etape 5 : KDE sur NOHEDES
    kde_noh = gaussian_kde(coords_noh.T)
    densites = kde_noh(coords_noh.T)
    seuil_densite = np.quantile(densites, NIVEAU_NICHE)
    
    print(f"   Seuil densite (5%) : {seuil_densite:.4f}")
    
    # Etape 6 : grille pour visualisation
    x_min, x_max = coords_noh[:, 0].min(), coords_noh[:, 0].max()
    y_min, y_max = coords_noh[:, 1].min(), coords_noh[:, 1].max()
    x_ext = (x_max - x_min) * 0.5
    y_ext = (y_max - y_min) * 0.5
    
    gx = np.linspace(x_min - x_ext, x_max + x_ext, 200)
    gy = np.linspace(y_min - y_ext, y_max + y_ext, 200)
    XX, YY = np.meshgrid(gx, gy)
    pts_grid = np.vstack([XX.ravel(), YY.ravel()])
    densites_grid = kde_noh(pts_grid).reshape(XX.shape)
    
    return {
        'scaler': scaler,
        'pca': pca,
        'kde_noh': kde_noh,
        'seuil_densite': seuil_densite,
        'coords_noh': coords_noh,
        'grid_x': gx,
        'grid_y': gy,
        'densites_grid': densites_grid
    }


# ═══════════════════════════════════════════════════════════════
# 5. CALCUL DE LA RESSEMBLANCE POUR CHAQUE SITE
# ═══════════════════════════════════════════════════════════════

def calculer_ressemblance_sites(df, niche_noh, variables):
    """
    Pour chaque site/annee des 8 autres sites, calcule si le point
    est DANS ou HORS la niche NOHEDES.
    
    Parameters
    ----------
    df : pd.DataFrame
    niche_noh : dict
    variables : list
    
    Returns
    -------
    df_result : pd.DataFrame
        Statut de chaque site/annee
    """
    print("\nCalcul de la ressemblance des 8 sites a NOHEDES")
    print("-" * 60)
    
    df_autres = df[df['nom'] != 'NOHEDES'][['nom', 'annee'] + variables].dropna()
    
    # Projeter tous les sites dans l'espace ACP
    X = df_autres[variables].values
    X_std = niche_noh['scaler'].transform(X)
    coords = niche_noh['pca'].transform(X_std)
    
    # Calculer densite pour chaque point
    densites = niche_noh['kde_noh'](coords.T)
    
    # Statut : DANS si densite >= seuil, sinon HORS
    statuts = ['DANS' if d >= niche_noh['seuil_densite'] else 'HORS' 
                for d in densites]
    
    df_result = df_autres[['nom', 'annee']].copy()
    df_result['PC1'] = coords[:, 0]
    df_result['PC2'] = coords[:, 1]
    df_result['densite'] = densites
    df_result['statut'] = statuts
    
    return df_result


# ═══════════════════════════════════════════════════════════════
# 6. CLASSEMENT DES SITES
# ═══════════════════════════════════════════════════════════════

def classer_sites(df_result):
    """
    Classe les sites par nombre d'annees DANS la niche NOHEDES 
    (ordre decroissant).
    
    Parameters
    ----------
    df_result : pd.DataFrame
    
    Returns
    -------
    df_classement : pd.DataFrame
    """
    print("\nClassement des sites (ordre decroissant)")
    print("-" * 60)
    
    df_classement = df_result.groupby('nom').agg(
        n_annees=('annee', 'count'),
        n_DANS=('statut', lambda x: (x == 'DANS').sum())
    ).reset_index()
    
    df_classement['n_HORS'] = df_classement['n_annees'] - df_classement['n_DANS']
    df_classement['pct_DANS'] = (100 * df_classement['n_DANS'] 
                                    / df_classement['n_annees']).round(1)
    df_classement = df_classement.sort_values('n_DANS', ascending=False).reset_index(drop=True)
    
    print(df_classement.to_string(index=False))
    
    return df_classement


# ═══════════════════════════════════════════════════════════════
# 7. VISUALISATION
# ═══════════════════════════════════════════════════════════════

def visualiser_nmi_inverse(df_result, df_classement, niche_noh):
    """
    Cree 2 graphiques :
    1. Niche NOHEDES + 8 sites (DANS = vert, HORS = rouge)
    2. Classement des sites (bar chart)
    
    Parameters
    ----------
    df_result : pd.DataFrame
    df_classement : pd.DataFrame
    niche_noh : dict
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # ─── GRAPH 1 : Niche NOHEDES + 8 sites ───
    ax1 = axes[0]
    
    # Fond : zone de la niche NOHEDES
    mask_dans = niche_noh['densites_grid'] >= niche_noh['seuil_densite']
    Z_niche = np.where(mask_dans, niche_noh['densites_grid'], np.nan)
    ax1.contourf(niche_noh['grid_x'], niche_noh['grid_y'], Z_niche,
                    levels=10, cmap='Blues', alpha=0.4)
    ax1.contour(niche_noh['grid_x'], niche_noh['grid_y'], 
                   niche_noh['densites_grid'],
                   levels=[niche_noh['seuil_densite']], 
                   colors='#1976D2', linewidths=2)
    
    # Points 8 sites : vert si DANS, rouge si HORS
    for statut, couleur in [('DANS', '#2E7D32'), ('HORS', '#C62828')]:
        mask = df_result['statut'] == statut
        ax1.scatter(df_result[mask]['PC1'], df_result[mask]['PC2'],
                     c=couleur, s=60, alpha=0.7, 
                     edgecolor='white', linewidth=0.5,
                     label=f'{statut} niche NOHEDES ({mask.sum()} obs)')
    
    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.set_title('Niche construite autour de NOHEDES\n'
                    'Position des 8 autres sites',
                    fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # ─── GRAPH 2 : Classement des sites ───
    ax2 = axes[1]
    
    # Trier pour affichage horizontal
    df_plot = df_classement.sort_values('n_DANS', ascending=True)
    
    couleurs_bars = plt.cm.Blues(df_plot['pct_DANS'] / 100)
    bars = ax2.barh(df_plot['nom'], df_plot['n_DANS'], color=couleurs_bars,
                      edgecolor='#1976D2', linewidth=1)
    
    # Ajouter les valeurs
    for i, (bar, n, pct) in enumerate(zip(bars, df_plot['n_DANS'], 
                                             df_plot['pct_DANS'])):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                   f"{n}/21 ({pct}%)", va='center', fontsize=11, 
                   fontweight='bold')
    
    ax2.set_xlabel("Nombre d'annees DANS la niche NOHEDES (sur 21)", 
                     fontsize=12)
    ax2.set_ylabel('Site', fontsize=12)
    ax2.set_title('Classement des sites\n(ordre decroissant de ressemblance)',
                    fontsize=14, fontweight='bold')
    ax2.set_xlim(0, max(df_plot['n_DANS']) * 1.3 + 3)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    chemin_fig = DOSSIER_OUTPUT / "NMI_inverse.png"
    fig.savefig(chemin_fig, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\nFigure sauvegardee : {chemin_fig.name}")


# ═══════════════════════════════════════════════════════════════
# 8. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def pipeline_nmi_inverse():
    """
    Pipeline complet du calcul NMI inverse.
    """
    print("=" * 70)
    print("  CALCUL DU NMI INVERSE")
    print("  Niche autour de NOHEDES - Quels sites lui ressemblent ?")
    print("=" * 70)
    
    # Etape 1 : Charger les donnees
    df = charger_donnees()
    
    # Etape 2 : Construire la niche autour de NOHEDES
    niche_noh = construire_niche_nohedes(df, VARIABLES_X)
    
    # Etape 3 : Calculer la ressemblance de chaque site
    df_result = calculer_ressemblance_sites(df, niche_noh, VARIABLES_X)
    
    # Etape 4 : Classer les sites
    df_classement = classer_sites(df_result)
    
    # Etape 5 : Sauvegarder les resultats
    chemin_result = DOSSIER_OUTPUT / "NMI_inverse_details.csv"
    df_result.to_csv(chemin_result, index=False)
    
    chemin_classement = DOSSIER_OUTPUT / "NMI_inverse_classement.csv"
    df_classement.to_csv(chemin_classement, index=False)
    
    print(f"\nDetails sauvegardes : {chemin_result.name}")
    print(f"Classement sauvegarde : {chemin_classement.name}")
    
    # Etape 6 : Visualiser
    visualiser_nmi_inverse(df_result, df_classement, niche_noh)
    
    # Etape 7 : Resume
    print("\n" + "=" * 70)
    print("  RESUME DES RESULTATS")
    print("=" * 70)
    
    top_site = df_classement.iloc[0]
    print(f"  Site le plus proche de NOHEDES : {top_site['nom']}")
    print(f"     -> {top_site['n_DANS']}/21 annees dans niche ({top_site['pct_DANS']}%)")
    print(f"     -> Candidat pour 'prochain NOHEDES'")
    
    bot_site = df_classement.iloc[-1]
    print(f"\n  Site le plus different : {bot_site['nom']}")
    print(f"     -> {bot_site['n_DANS']}/21 annees dans niche ({bot_site['pct_DANS']}%)")
    
    print(f"\n  Total observations DANS niche NOHEDES : "
          f"{(df_result['statut'] == 'DANS').sum()}/{len(df_result)}")
    print("=" * 70)
    
    return df_result, df_classement, niche_noh


# ═══════════════════════════════════════════════════════════════
# 9. LANCEMENT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df_result, df_classement, niche_noh = pipeline_nmi_inverse()
    print("\nPipeline NMI inverse termine avec succes")

# ----------------------------------------------------------------
# Projet : D2ClimAFLo-Pyr
# Auteur : Amadou FOFANA
# Stage  : UPVD / CEFREM (2026)
# ----------------------------------------------------------------

# Chargement des bibliothèques
library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(ggrepel)
library(patchwork)
library(FactoMineR)
library(factoextra)

# Chemin vers les données
DOSSIER <- "/content/drive/MyDrive/Colab Notebooks/dataStage/"

# Chargement des données
df_selected <- read.csv(
  paste0(DOSSIER, "df_complete_2000_2020.csv"),
  sep = ";"
)

# Variables d'intérêt
variables <- c("temp_RF", "pr_RF",
               "neige_RF_cm", "pct_neige")

noms_vars <- c(
  temp_RF     = "Température (°C)",
  pr_RF       = "Précipitations (mm/mois)",
  neige_RF_cm = "Hauteur neige (cm)",
  pct_neige   = "% Couverture neigeuse"
)

# Couleurs
COL_NOH <- "#2A9D8F"
COL_AUT <- "#7E3AC8"


# ----------------------------------------------------------------
# 1. STATISTIQUES DESCRIPTIVES
# ----------------------------------------------------------------

# Stats globales
df_stats_global <- map_dfr(variables, function(var) {
  vals <- df_selected[[var]]
  vals <- vals[!is.na(vals)]
  tibble(
    variable   = noms_vars[var],
    n          = length(vals),
    moyenne    = round(mean(vals), 2),
    mediane    = round(median(vals), 2),
    ecart_type = round(sd(vals), 2),
    min        = round(min(vals), 2),
    Q1         = round(quantile(vals, 0.25), 2),
    Q3         = round(quantile(vals, 0.75), 2),
    max        = round(max(vals), 2),
    CV         = round(sd(vals) / mean(vals) * 100, 1)
  )
})

cat("Stats globales :\n")
print(df_stats_global, n = Inf)

# Stats par site
df_stats_site <- df_selected %>%
  group_by(nom) %>%
  summarise(
    n         = n(),
    temp_moy  = round(mean(temp_RF,     na.rm = TRUE), 2),
    temp_sd   = round(sd(temp_RF,       na.rm = TRUE), 2),
    pr_moy    = round(mean(pr_RF,       na.rm = TRUE), 2),
    pr_sd     = round(sd(pr_RF,         na.rm = TRUE), 2),
    neige_moy = round(mean(neige_RF_cm, na.rm = TRUE), 2),
    neige_sd  = round(sd(neige_RF_cm,   na.rm = TRUE), 2),
    pct_moy   = round(mean(pct_neige,   na.rm = TRUE), 2),
    pct_sd    = round(sd(pct_neige,     na.rm = TRUE), 2),
    .groups   = "drop"
  )

cat("\nStats par site :\n")
print(df_stats_site, n = Inf)

# Stats NOHEDES vs Autres sites
df_stats_groupe <- df_selected %>%
  mutate(groupe = ifelse(nom == "NOHEDES",
                         "NOHEDES", "Autres sites")) %>%
  group_by(groupe) %>%
  summarise(
    n         = n(),
    temp_moy  = round(mean(temp_RF,     na.rm = TRUE), 2),
    temp_sd   = round(sd(temp_RF,       na.rm = TRUE), 2),
    pr_moy    = round(mean(pr_RF,       na.rm = TRUE), 2),
    pr_sd     = round(sd(pr_RF,         na.rm = TRUE), 2),
    neige_moy = round(mean(neige_RF_cm, na.rm = TRUE), 2),
    neige_sd  = round(sd(neige_RF_cm,   na.rm = TRUE), 2),
    pct_moy   = round(mean(pct_neige,   na.rm = TRUE), 2),
    pct_sd    = round(sd(pct_neige,     na.rm = TRUE), 2),
    .groups   = "drop"
  )

cat("\nNOHEDES vs Autres sites :\n")
print(df_stats_groupe)

# Stats par période avant/après 2010
df_stats_periode <- df_selected %>%
  mutate(
    groupe  = ifelse(nom == "NOHEDES",
                     "NOHEDES", "Autres sites"),
    periode = ifelse(annee < 2010,
                     "Avant 2010", "Après 2010")
  ) %>%
  group_by(groupe, periode) %>%
  summarise(
    n         = n(),
    temp_moy  = round(mean(temp_RF,     na.rm = TRUE), 2),
    pr_moy    = round(mean(pr_RF,       na.rm = TRUE), 2),
    neige_moy = round(mean(neige_RF_cm, na.rm = TRUE), 2),
    pct_moy   = round(mean(pct_neige,   na.rm = TRUE), 2),
    .groups   = "drop"
  ) %>%
  arrange(groupe, periode)

cat("\nStats par période :\n")
print(df_stats_periode, n = Inf)

# Sauvegarde des tableaux
write.csv(df_stats_global,
          paste0(DOSSIER, "stats_descriptives_global.csv"),
          row.names = FALSE)

write.csv(df_stats_site,
          paste0(DOSSIER, "stats_descriptives_site.csv"),
          row.names = FALSE)

write.csv(df_stats_periode,
          paste0(DOSSIER, "stats_descriptives_periode.csv"),
          row.names = FALSE)


# ----------------------------------------------------------------
# 2. ÉVOLUTION TEMPORELLE (Mann-Kendall)
# ----------------------------------------------------------------

# Moyennes annuelles par groupe
df_annuel <- df_selected %>%
  mutate(groupe = ifelse(nom == "NOHEDES",
                         "NOHEDES", "Autres sites")) %>%
  group_by(groupe, annee) %>%
  summarise(
    temp_RF     = mean(temp_RF,     na.rm = TRUE),
    pr_RF       = mean(pr_RF,       na.rm = TRUE),
    neige_RF_cm = mean(neige_RF_cm, na.rm = TRUE),
    pct_neige   = mean(pct_neige,   na.rm = TRUE),
    .groups     = "drop"
  )

# Enveloppe min-max des autres sites
df_env <- df_selected %>%
  filter(nom != "NOHEDES") %>%
  group_by(nom, annee) %>%
  summarise(
    temp_moy  = mean(temp_RF,     na.rm = TRUE),
    pr_moy    = mean(pr_RF,       na.rm = TRUE),
    neige_moy = mean(neige_RF_cm, na.rm = TRUE),
    pct_moy   = mean(pct_neige,   na.rm = TRUE),
    .groups   = "drop"
  ) %>%
  group_by(annee) %>%
  summarise(
    temp_min  = min(temp_moy),  temp_max  = max(temp_moy),
    pr_min    = min(pr_moy),    pr_max    = max(pr_moy),
    neige_min = min(neige_moy), neige_max = max(neige_moy),
    pct_min   = min(pct_moy),   pct_max   = max(pct_moy),
    .groups   = "drop"
  )

df_noh_an <- df_annuel %>% filter(groupe == "NOHEDES")
df_aut_an <- df_annuel %>% filter(groupe == "Autres sites")

# Fonction graphique série temporelle
plot_serie <- function(var, var_min, var_max, titre, unite) {
  ggplot() +
    geom_ribbon(
      data = df_env,
      aes(x = annee, ymin = .data[[var_min]],
          ymax = .data[[var_max]]),
      fill = COL_AUT, alpha = 0.15
    ) +
    geom_line(
      data = df_aut_an,
      aes(x = annee, y = .data[[var]]),
      color = COL_AUT, linewidth = 1.2, alpha = 0.8
    ) +
    geom_point(
      data = df_aut_an,
      aes(x = annee, y = .data[[var]]),
      color = COL_AUT, size = 2.5, alpha = 0.8
    ) +
    geom_line(
      data = df_noh_an,
      aes(x = annee, y = .data[[var]]),
      color = COL_NOH, linewidth = 1.8, alpha = 0.9
    ) +
    geom_point(
      data = df_noh_an,
      aes(x = annee, y = .data[[var]]),
      color = COL_NOH, size = 3.5, shape = 18, alpha = 0.9
    ) +
    scale_x_continuous(breaks = seq(2000, 2021, by = 2)) +
    labs(title = titre, x = "Année", y = unite) +
    theme_bw(base_size = 11) +
    theme(
      plot.title       = element_text(face = "bold", size = 12,
                                      hjust = 0.5, color = "#1B2D26"),
      axis.text.x      = element_text(angle = 45, hjust = 1, size = 8),
      panel.grid.minor = element_blank()
    )
}

p1 <- plot_serie("temp_RF",     "temp_min",  "temp_max",
                 "Température",       "°C")
p2 <- plot_serie("pr_RF",       "pr_min",    "pr_max",
                 "Précipitations",    "mm/mois")
p3 <- plot_serie("neige_RF_cm", "neige_min", "neige_max",
                 "Hauteur de neige",  "cm")
p4 <- plot_serie("pct_neige",   "pct_min",   "pct_max",
                 "% Couverture neigeuse", "%")

fig_evolution <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title    = "Évolution temporelle — NOHEDES vs Autres sites (2000–2020)",
    subtitle = paste0(
      "Vert = NOHEDES (sans floraison) | ",
      "Violet = Sites avec floraison\n",
      "Zone colorée = enveloppe min-max"
    ),
    theme = theme(
      plot.title    = element_text(face = "bold", size = 14,
                                   hjust = 0.5, color = "#1B2D26"),
      plot.subtitle = element_text(size = 9, face = "italic",
                                   hjust = 0.5, color = "grey40")
    )
  )

print(fig_evolution)

ggsave(paste0(DOSSIER, "evolution_temporelle_4var.png"),
       plot = fig_evolution, width = 16, height = 12,
       dpi = 200, bg = "white")


# ----------------------------------------------------------------
# 3. GRADIENT ALTITUDINAL (Corrélation de Spearman)
# ----------------------------------------------------------------

df_alt <- df_selected %>%
  group_by(nom) %>%
  summarise(
    altitude  = mean(altitude,    na.rm = TRUE),
    temp_moy  = mean(temp_RF,     na.rm = TRUE),
    pr_moy    = mean(pr_RF,       na.rm = TRUE),
    neige_moy = mean(neige_RF_cm, na.rm = TRUE),
    pct_moy   = mean(pct_neige,   na.rm = TRUE),
    .groups   = "drop"
  ) %>%
  mutate(is_nohedes = nom == "NOHEDES")

# Corrélations de Spearman
cat("Corrélations de Spearman (altitude ~ variables) :\n")
for (var in c("temp_moy","pr_moy","neige_moy","pct_moy")) {
  r_all  <- cor.test(df_alt$altitude, df_alt[[var]],
                     method = "spearman")
  r_sans <- cor.test(
    df_alt$altitude[!df_alt$is_nohedes],
    df_alt[[var]][!df_alt$is_nohedes],
    method = "spearman"
  )
  cat(sprintf(
    "  %s : tous=%.2f (p=%.3f) | sans NOHEDES=%.2f (p=%.3f)\n",
    var, r_all$estimate, r_all$p.value,
    r_sans$estimate, r_sans$p.value
  ))
}

# Fonction graphique gradient altitudinal
plot_altitude <- function(var, titre, unite) {
  cor_global <- cor.test(df_alt$altitude, df_alt[[var]],
                         method = "spearman")
  cor_sans   <- cor.test(
    df_alt$altitude[!df_alt$is_nohedes],
    df_alt[[var]][!df_alt$is_nohedes],
    method = "spearman"
  )

  sig_g <- ifelse(cor_global$p.value < 0.001, "***",
           ifelse(cor_global$p.value < 0.01,  "**",
           ifelse(cor_global$p.value < 0.05,  "*", "ns")))
  sig_s <- ifelse(cor_sans$p.value  < 0.001, "***",
           ifelse(cor_sans$p.value  < 0.01,  "**",
           ifelse(cor_sans$p.value  < 0.05,  "*", "ns")))

  ggplot(df_alt, aes(x = altitude, y = .data[[var]])) +
    geom_smooth(method = "lm", se = FALSE,
                color = "grey50", linetype = "dashed",
                linewidth = 1, alpha = 0.7) +
    geom_smooth(data = df_alt %>% filter(!is_nohedes),
                method = "lm", se = FALSE,
                color = COL_AUT, linewidth = 1, alpha = 0.7) +
    geom_point(data = df_alt %>% filter(!is_nohedes),
               color = COL_AUT, size = 5, shape = 16, alpha = 0.85) +
    geom_text(data = df_alt %>% filter(!is_nohedes),
              aes(label = nom), color = COL_AUT,
              size = 2.8, vjust = -1, fontface = "bold") +
    geom_point(data = df_alt %>% filter(is_nohedes),
               color = COL_NOH, size = 8, shape = 18, alpha = 0.9) +
    geom_label(data = df_alt %>% filter(is_nohedes),
               aes(label = sprintf("NOHEDES\n%s=%.1f",
                                   unite, .data[[var]])),
               color = COL_NOH, fill = "white", size = 3.2,
               fontface = "bold", vjust = -0.8) +
    annotate("label",
             x = min(df_alt$altitude),
             y = max(df_alt[[var]], na.rm = TRUE),
             label = sprintf("Tous sites : ρ=%.2f %s",
                             cor_global$estimate, sig_g),
             color = "grey40", fill = "white",
             fontface = "bold", size = 3.2, hjust = 0) +
    annotate("label",
             x = min(df_alt$altitude),
             y = max(df_alt[[var]], na.rm = TRUE) * 0.88,
             label = sprintf("Sans NOHEDES : ρ=%.2f %s",
                             cor_sans$estimate, sig_s),
             color = COL_AUT, fill = "white",
             fontface = "bold", size = 3.2, hjust = 0) +
    labs(title = sprintf("%s ~ Altitude", titre),
         x = "Altitude (m)", y = unite) +
    theme_bw(base_size = 11) +
    theme(
      plot.title      = element_text(face = "bold", size = 12,
                                     hjust = 0.5, color = "#1B2D26"),
      legend.position = "none",
      panel.grid.minor = element_blank()
    )
}

p1 <- plot_altitude("temp_moy",  "Température",         "°C")
p2 <- plot_altitude("pr_moy",    "Précipitations",      "mm/mois")
p3 <- plot_altitude("neige_moy", "Hauteur de neige",    "cm")
p4 <- plot_altitude("pct_moy",   "% Couverture neigeuse", "%")

fig_gradient <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title    = "Gradient altitudinal — 9 sites Delphinium montanum",
    subtitle = paste0(
      "Vert = NOHEDES (sans floraison) | Violet = Sites avec floraison\n",
      "Corrélation de Spearman | *** p<0.001 · ** p<0.01 · * p<0.05 · ns"
    ),
    theme = theme(
      plot.title    = element_text(face = "bold", size = 14,
                                   hjust = 0.5, color = "#1B2D26"),
      plot.subtitle = element_text(size = 9, face = "italic",
                                   hjust = 0.5, color = "grey40")
    )
  )

print(fig_gradient)

ggsave(paste0(DOSSIER, "gradient_altitudinal_spearman.png"),
       plot = fig_gradient, width = 16, height = 12,
       dpi = 200, bg = "white")


# ----------------------------------------------------------------
# 4. MATRICE DE CORRÉLATION DE SPEARMAN
# ----------------------------------------------------------------

calc_cor_matrix <- function(df_g, groupe) {
  paires <- expand_grid(var1 = variables, var2 = variables)
  map_dfr(1:nrow(paires), function(i) {
    v1 <- paires$var1[i]
    v2 <- paires$var2[i]
    if (v1 == v2) {
      tibble(var1 = v1, var2 = v2, r = 1,
             p_value = 0, sig = "", groupe = groupe)
    } else {
      x   <- df_g[[v1]]
      y   <- df_g[[v2]]
      idx <- !is.na(x) & !is.na(y)
      test <- cor.test(x[idx], y[idx], method = "spearman")
      tibble(
        var1    = v1, var2 = v2,
        r       = round(test$estimate, 2),
        p_value = round(test$p.value, 4),
        sig     = ifelse(test$p.value < 0.001, "***",
                  ifelse(test$p.value < 0.01,  "**",
                  ifelse(test$p.value < 0.05,  "*", "ns"))),
        groupe  = groupe
      )
    }
  })
}

noms_labels <- c(
  temp_RF     = "Température",
  pr_RF       = "Précipitations",
  neige_RF_cm = "Hauteur neige",
  pct_neige   = "% Neige"
)

df_noh_cor    <- df_selected %>% filter(nom == "NOHEDES")
df_autres_cor <- df_selected %>% filter(nom != "NOHEDES")

mat_noh    <- calc_cor_matrix(df_noh_cor,    "NOHEDES\n(sans floraison)")
mat_autres <- calc_cor_matrix(df_autres_cor, "Autres sites\n(avec floraison)")

df_mat_all <- bind_rows(mat_noh, mat_autres) %>%
  mutate(
    var1  = factor(noms_labels[var1], levels = rev(noms_labels)),
    var2  = factor(noms_labels[var2], levels = noms_labels),
    label = ifelse(var1 == var2, "1",
                   sprintf("%.2f\n%s", r, sig))
  )

p_mat <- ggplot(df_mat_all, aes(x = var2, y = var1, fill = r)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = label, color = abs(r) > 0.4),
            size = 4, fontface = "bold", lineheight = 0.9) +
  scale_fill_gradient2(low = "#E63946", mid = "white",
                       high = "#2A9D8F", midpoint = 0,
                       limits = c(-1, 1), name = "ρ de Spearman") +
  scale_color_manual(values = c("TRUE" = "white",
                                "FALSE" = "grey30"),
                     guide = "none") +
  facet_wrap(~ groupe, ncol = 2) +
  labs(
    title    = "Matrice de corrélation de Spearman — NOHEDES vs Autres sites",
    subtitle = "ρ = coefficient de Spearman | *** p<0.001 | ** p<0.01 | * p<0.05 | ns",
    x = NULL, y = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle    = element_text(size = 9, face = "italic",
                                    hjust = 0.5, color = "grey40"),
    axis.text.x      = element_text(angle = 45, hjust = 1,
                                    face = "bold", size = 10),
    axis.text.y      = element_text(face = "bold", size = 10),
    strip.text       = element_text(face = "bold", size = 12),
    strip.background = element_rect(fill = "#1B2D26"),
    strip.text.x     = element_text(color = "white"),
    legend.position  = "right",
    panel.grid       = element_blank()
  )

print(p_mat)

ggsave(paste0(DOSSIER, "matrice_correlation_spearman.png"),
       plot = p_mat, width = 14, height = 7,
       dpi = 200, bg = "white")


# ----------------------------------------------------------------
# 5. TEST DE WILCOXON — NOHEDES vs Autres sites
# ----------------------------------------------------------------

cat("Test de Wilcoxon — NOHEDES vs Autres sites :\n\n")
cat(sprintf("%-25s %10s %10s %10s %10s\n",
            "Variable", "W", "p-value", "Sig", "Mediane NOH"))
cat(strrep("-", 65), "\n")

for (var in variables) {
  g_noh <- df_selected %>%
    filter(nom == "NOHEDES") %>%
    pull(.data[[var]])
  g_aut <- df_selected %>%
    filter(nom != "NOHEDES") %>%
    pull(.data[[var]])

  test <- wilcox.test(g_noh, g_aut, exact = FALSE)

  sig <- ifelse(test$p.value < 0.001, "***",
         ifelse(test$p.value < 0.01,  "**",
         ifelse(test$p.value < 0.05,  "*", "ns")))

  cat(sprintf("%-25s %10.1f %10.4f %10s %10.2f\n",
              noms_vars[var], test$statistic,
              test$p.value, sig,
              median(g_noh, na.rm = TRUE)))
}


# ----------------------------------------------------------------
# 6. BOOTSTRAP — IC 95% de NOHEDES
# ----------------------------------------------------------------

ORDRE_SITES <- c(
  "CADI_POP3", "CADI_POP1", "CADI_POP2", "CADI_POP4",
  "EYNE_POP3", "EYNE_POP2", "NOHEDES",
  "EYNE_POP1", "VALLTER"
)

df_site <- df_selected %>%
  group_by(nom) %>%
  summarise(
    temp_RF     = mean(temp_RF,     na.rm = TRUE),
    pr_RF       = mean(pr_RF,       na.rm = TRUE),
    neige_RF_cm = mean(neige_RF_cm, na.rm = TRUE),
    pct_neige   = mean(pct_neige,   na.rm = TRUE),
    .groups     = "drop"
  )

df_noh_boot <- df_selected %>% filter(nom == "NOHEDES")

bootstrap_ic <- function(data, n_boot = 10000,
                          alpha = 0.05, seed = 42) {
  set.seed(seed)
  n          <- length(data)
  boot_means <- replicate(n_boot, {
    mean(sample(data, size = n, replace = TRUE))
  })
  list(
    mean  = mean(boot_means),
    lower = quantile(boot_means, alpha / 2),
    upper = quantile(boot_means, 1 - alpha / 2)
  )
}

var_labels_boot <- c(
  pct_neige   = "% Couverture neigeuse",
  neige_RF_cm = "Manteau neigeux (cm)",
  pr_RF       = "Précipitations (mm/mois)",
  temp_RF     = "Température (°C)"
)

vars_boot <- c("pct_neige", "neige_RF_cm", "pr_RF", "temp_RF")

COL_IN  <- "#E67E22"
COL_OUT <- "#1ABC9C"
COL_IC  <- "#E74C3C"

results_boot <- list()

for (var in vars_boot) {
  data_noh <- df_noh_boot[[var]]
  data_noh <- data_noh[!is.na(data_noh)]
  ic       <- bootstrap_ic(data_noh)

  autres <- df_site %>%
    filter(nom != "NOHEDES") %>%
    mutate(
      valeur = .data[[var]],
      in_ic  = valeur >= ic$lower & valeur <= ic$upper,
      statut = ifelse(in_ic, "Dans IC NOHEDES", "Hors IC NOHEDES")
    )

  results_boot[[var]] <- list(
    ic      = ic,
    autres  = autres,
    noh_val = df_site %>% filter(nom == "NOHEDES") %>%
      pull(.data[[var]])
  )
}

# Fonction graphique bootstrap
make_plot_boot <- function(var, res) {
  ic      <- res$ic
  autres  <- res$autres
  noh_val <- res$noh_val

  df_plot <- bind_rows(
    autres %>% select(nom, valeur, statut),
    data.frame(nom = "NOHEDES", valeur = noh_val,
               statut = "NOHEDES")
  )
  df_plot$nom <- factor(df_plot$nom, levels = ORDRE_SITES)

  ggplot(df_plot, aes(x = nom, y = valeur)) +
    annotate("rect", xmin = -Inf, xmax = Inf,
             ymin = ic$lower, ymax = ic$upper,
             fill = COL_IC, alpha = 0.1) +
    geom_hline(yintercept = ic$mean,
               color = COL_IC, linewidth = 1.5) +
    geom_hline(yintercept = ic$lower,
               color = COL_IC, linewidth = 1,
               linetype = "dashed", alpha = 0.8) +
    geom_hline(yintercept = ic$upper,
               color = COL_IC, linewidth = 1,
               linetype = "dashed", alpha = 0.8) +
    geom_point(aes(color = statut, shape = statut,
                   size = statut)) +
    geom_label(aes(label = round(valeur, 1), color = statut),
               vjust = -0.8, size = 3.2, fontface = "bold",
               fill = "white", label.size = 0.3) +
    annotate("label", x = 1, y = ic$mean,
             label = sprintf("NOH : %.1f\n[%.1f-%.1f]",
                             ic$mean, ic$lower, ic$upper),
             color = COL_IC, fill = "white", fontface = "bold",
             size = 3, hjust = 0.5, vjust = -0.3) +
    scale_color_manual(values = c(
      "Dans IC NOHEDES" = COL_IN,
      "Hors IC NOHEDES" = COL_OUT,
      "NOHEDES"         = COL_IC)) +
    scale_shape_manual(values = c(
      "Dans IC NOHEDES" = 16,
      "Hors IC NOHEDES" = 16,
      "NOHEDES"         = 18)) +
    scale_size_manual(values = c(
      "Dans IC NOHEDES" = 4,
      "Hors IC NOHEDES" = 4,
      "NOHEDES"         = 6)) +
    scale_y_continuous(expand = expansion(mult = c(0.1, 0.2))) +
    labs(title = var_labels_boot[var], x = NULL,
         y = "Valeur moyenne 2000–2020") +
    theme_minimal(base_size = 11) +
    theme(
      plot.title       = element_text(face = "bold", size = 12,
                                      color = "#1B2D26", hjust = 0.5),
      axis.text.x      = element_text(angle = 45, hjust = 1,
                                      face = "bold", size = 9),
      legend.position  = "none",
      panel.background = element_rect(fill = "#FAFAFA", color = NA),
      panel.grid.minor = element_blank(),
      plot.background  = element_rect(fill = "#FFF9E6",
                                      color = "#CCCCCC", linewidth = 0.5)
    )
}

plot_list_boot <- lapply(vars_boot,
                         function(v) make_plot_boot(v, results_boot[[v]]))

fig_bootstrap <- (plot_list_boot[[1]] | plot_list_boot[[2]]) /
                 (plot_list_boot[[3]] | plot_list_boot[[4]]) +
  plot_annotation(
    title    = "IC Bootstrap 95% de NOHEDES — Quels sites sont similaires ?",
    subtitle = paste0(
      "Zone rouge = IC 95% | Ligne rouge = moyenne NOHEDES\n",
      "Orange = dans l'IC (similaire à NOHEDES) | ",
      "Vert = hors IC (différent de NOHEDES)"
    ),
    theme = theme(
      plot.title    = element_text(face = "bold", size = 14,
                                   hjust = 0.5, color = "#1B2D26"),
      plot.subtitle = element_text(size = 9, face = "italic",
                                   hjust = 0.5, color = "grey40")
    )
  )

print(fig_bootstrap)

ggsave(paste0(DOSSIER, "bootstrap_ic_nohedes_inverse.png"),
       plot = fig_bootstrap, width = 16, height = 12,
       dpi = 200, bg = "white")


# ----------------------------------------------------------------
# 7. ACP
# ----------------------------------------------------------------

X_clim <- df_selected %>%
  group_by(nom) %>%
  summarise(
    temp_RF     = mean(temp_RF,     na.rm = TRUE),
    pr_RF       = mean(pr_RF,       na.rm = TRUE),
    neige_RF_cm = mean(neige_RF_cm, na.rm = TRUE),
    pct_neige   = mean(pct_neige,   na.rm = TRUE),
    floraison   = first(ifelse(nom == "NOHEDES", "NON", "OUI")),
    .groups     = "drop"
  )

X_mat <- X_clim %>%
  select(temp_RF, pr_RF, neige_RF_cm, pct_neige) %>%
  as.data.frame()

rownames(X_mat) <- X_clim$nom

resultat_acp    <- PCA(X_mat, graph = FALSE)
valeurs_propres <- resultat_acp$eig

cat("\nValeurs propres (ACP) :\n")
print(round(valeurs_propres, 2))

n_axes   <- sum(valeurs_propres[, 1] > 1)
var_dim1 <- round(valeurs_propres[1, 2], 1)
var_dim2 <- round(valeurs_propres[2, 2], 1)

cat(sprintf("Axes retenus (Kaiser λ>1) : %d\n", n_axes))

# Cercle de corrélation
p_var_acp <- fviz_pca_var(
  resultat_acp,
  col.var       = "contrib",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel         = TRUE,
  title         = "Cercle de corrélation des variables"
)
print(p_var_acp)

# Projection des individus
pourplot <- data.frame(
  jeu       = rownames(resultat_acp$ind$coord),
  Dim.1     = resultat_acp$ind$coord[, 1],
  Dim.2     = resultat_acp$ind$coord[, 2],
  floraison = X_clim$floraison
)

p_ind_acp <- ggplot(pourplot,
                    aes(x = Dim.1, y = Dim.2,
                        color = floraison)) +
  geom_point(alpha = 0.8, size = 3) +
  geom_text_repel(aes(label = jeu), size = 3.5,
                  max.overlaps = 20, fontface = "bold") +
  geom_hline(yintercept = 0, color = "gray40",
             linetype = "dashed") +
  geom_vline(xintercept = 0, color = "gray40",
             linetype = "dashed") +
  scale_color_manual(
    values = c("OUI" = COL_AUT, "NON" = COL_NOH),
    labels = c("OUI" = "Floraison",
               "NON" = "Défaut floraison"),
    name   = NULL
  ) +
  labs(
    title = "Projection des individus sur le plan factoriel (ACP)",
    x     = paste0("Dim 1 (", var_dim1, "%)"),
    y     = paste0("Dim 2 (", var_dim2, "%)")
  ) +
  theme_minimal() +
  theme(
    legend.position = "top",
    plot.title      = element_text(hjust = 0.5, size = 14,
                                   face = "bold"),
    legend.text     = element_text(face = "bold")
  )

print(p_ind_acp)

# Sauvegarde ACP
ggsave(paste0(DOSSIER, "acp_variables.png"),
       plot = p_var_acp, width = 8, height = 7,
       dpi = 200, bg = "white")

ggsave(paste0(DOSSIER, "acp_individus.png"),
       plot = p_ind_acp, width = 10, height = 8,
       dpi = 200, bg = "white")


# ----------------------------------------------------------------
# 8. CAH — Classification Ascendante Hiérarchique
# ----------------------------------------------------------------

cp <- resultat_acp$ind$coord[, 1:n_axes]

# Méthode silhouette pour K optimal
p_silh <- fviz_nbclust(
  cp, FUN = hcut, method = "silhouette",
  k.max = 8, hc_method = "ward.D2"
) +
  labs(
    title    = "Largeur moyenne de silhouette",
    subtitle = "k optimal = valeur maximisant la silhouette",
    x        = "Nombre de clusters (k)",
    y        = "Largeur moyenne de silhouette"
  ) +
  theme_minimal() +
  theme(
    plot.title    = element_text(face = "bold", color = "#1B2D26",
                                 size = 14),
    plot.subtitle = element_text(color = "#4A5568", size = 11),
    axis.title    = element_text(face = "bold", color = "#1B2D26")
  )

print(p_silh)

# CAH Ward.D2
res_hclust <- hclust(dist(cp), method = "ward.D2")
cah        <- cutree(res_hclust, k = 2)

cat("\nClusters par site :\n")
for (i in seq_along(cah)) {
  cat(sprintf("  %-15s → Cluster %d\n",
              names(cah)[i], cah[i]))
}

# Dendrogramme
p_dendro <- fviz_dend(
  res_hclust,
  k                 = 2,
  cex               = 0.7,
  k_colors          = c("#2D6A4F", "#7E3AC8"),
  color_labels_by_k = TRUE,
  rect              = TRUE,
  rect_fill         = TRUE,
  rect_border       = c("#2D6A4F", "#7E3AC8"),
  main              = "Dendrogramme — CAH (Ward.D2)",
  xlab              = "",
  ylab              = "Distance (Ward.D2)",
  ggtheme           = theme_minimal()
) +
  theme(
    plot.title       = element_text(face = "bold", color = "#1B2D26",
                                    hjust = 0.5, size = 14),
    panel.background = element_rect(fill = "#F0F4F8", color = NA),
    plot.background  = element_rect(fill = "#FFFFFF", color = NA),
    panel.grid.major = element_line(color = "white"),
    panel.grid.minor = element_blank()
  )

print(p_dendro)

# Moyennes par cluster
mparclasse <- function(donnees, classes) {
  donnees %>%
    mutate(cluster = classes) %>%
    group_by(cluster) %>%
    summarise(across(where(is.numeric),
                     ~ round(mean(.x, na.rm = TRUE), 2)),
              .groups = "drop")
}

cat("\nMoyennes par cluster :\n")
print(mparclasse(as.data.frame(X_mat), cah))

# Sauvegarde CAH
ggsave(paste0(DOSSIER, "silhouette.png"),
       plot = p_silh, width = 8, height = 6,
       dpi = 200, bg = "white")

ggsave(paste0(DOSSIER, "dendrogramme.png"),
       plot = p_dendro, width = 12, height = 7,
       dpi = 200, bg = "white")

cat("\nToutes les figures ont été sauvegardées !\n")

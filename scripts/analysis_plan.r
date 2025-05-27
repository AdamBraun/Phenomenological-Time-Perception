# =======================================================================
#  analysis_plan.R
#  Statistical models and visualisation pipeline for
#  the multiscale ε–δ framework manuscript.
#
#  • Psychophysics  (human or simulated)
#  • Agent-learning simulation (Grid-world)
#
#  Author: Adam Braun
#  OSF location:  Statistical Models/analysis_plan.R
# =======================================================================

## ---------------------------------------------------------------------
## 0 ‒ Libraries --------------------------------------------------------
## ---------------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse)        # ggplot2, dplyr, readr, etc.
  library(lme4)             # GLMMs and LMMs
  library(broom.mixed)      # tidy() for lme4 objects
  library(afex)             # Anova() wrapper, type-III tests
  library(emmeans)          # post-hoc contrasts / Tukey HSD
})

## Utility for clean, reproducible output
theme_set(theme_bw(base_size = 11))

## ---------------------------------------------------------------------
## 1 ‒ Psychophysical dataset spec
## ---------------------------------------------------------------------
# Expected CSV columns:
#  id           : participant or simulation seed         (factor)
#  displacement : RMS dot displacement (deg)             (numeric)
#  duration     : physical duration (s)                  (numeric)
#  resp_long    : 0 / 1 “long” judgement                 (integer)
#  reproduced   : reproduced duration (s) — optional     (numeric)

# Adjust path as needed
psy_path <- "../data/psychophys.csv"

stopifnot(file.exists(psy_path))

psy <- read_csv(psy_path,
                col_types = cols(
                  id           = col_factor(),
                  displacement = col_double(),
                  duration     = col_double(),
                  resp_long    = col_integer(),
                  reproduced   = col_double()
                ))

## ---------------------------------------------------------------------
## 2 ‒ Temporal bisection analysis
## ---------------------------------------------------------------------
# GLMM with log(duration) and displacement; random intercepts by id
psy$log_dur <- log(psy$duration)

fit_bisect <- glmer(resp_long ~ log_dur * displacement +
                      (1 | id),
                    data = psy,
                    family = binomial)

## Save model summary
write_lines(capture.output(summary(fit_bisect)),
            "bisect_glmm_summary.txt")

## Wald χ² tests (Type-III)
bisect_anova <- Anova(fit_bisect, type = 3)
write_lines(capture.output(bisect_anova),
            "bisect_glmm_typeIII.txt")

## Predicted curve for manuscript figure
newdat <- expand_grid(
  displacement = sort(unique(psy$displacement)),
  duration     = seq(min(psy$duration),
                     max(psy$duration),
                     length.out = 200)
) %>%
  mutate(log_dur = log(duration),
         pred    = predict(fit_bisect, newdata = ., type = "response",
                           re.form = NA))

ggplot(newdat, aes(duration, pred,
                   colour = factor(displacement))) +
  geom_line(size = 0.8) +
  scale_x_log10() +
  labs(x = "Physical duration (s)",
       y = "P('long')",
       colour = "Displacement") +
  ggtitle("Temporal-bisection GLMM – predicted curves") +
  ggsave("figure_temporal_bisection_fit.png",
         width = 6, height = 4, dpi = 300)

## ---------------------------------------------------------------------
## 3 ‒ Vierordt bias analysis
## ---------------------------------------------------------------------
if (!all(is.na(psy$reproduced))) {

  bias_tbl <- psy %>%
    mutate(bias = (reproduced - duration) / duration)

  fit_bias <- lmer(bias ~ log_dur * displacement + (1 | id),
                   data = bias_tbl)

  write_lines(capture.output(summary(fit_bias)),
              "vierordt_lmm_summary.txt")

  bias_anova <- Anova(fit_bias, type = 3)
  write_lines(capture.output(bias_anova),
              "vierordt_lmm_typeIII.txt")

  # Plot
  ggplot(bias_tbl, aes(duration, bias,
                       colour = factor(displacement))) +
    stat_summary(fun = mean, geom = "line") +
    stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.05) +
    scale_x_log10() +
    labs(x = "Physical duration (s)",
         y = "Relative bias",
         colour = "Displacement") +
    ggtitle("Vierordt bias – observed means") +
    ggsave("figure_vierordt_bias.png",
           width = 6, height = 4, dpi = 300)
}

## ---------------------------------------------------------------------
## 4 ‒ Weber variance check
## ---------------------------------------------------------------------
if (!all(is.na(psy$reproduced))) {

  weber_tbl <- psy %>%
    group_by(id, duration) %>%
    summarise(mean_rep = mean(reproduced, na.rm = TRUE),
              sd_rep   = sd(reproduced,   na.rm = TRUE),
              .groups  = "drop")

  ggplot(weber_tbl, aes(mean_rep, sd_rep)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "lm", se = FALSE, colour = "black") +
    labs(x = "Mean reproduced duration (s)",
         y = "SD reproduced duration (s)") +
    ggtitle("Weber–like variance scaling") +
    ggsave("figure_weber_variance.png",
           width = 5.5, height = 4, dpi = 300)

  lm_weber <- lm(sd_rep ~ 0 + mean_rep, data = weber_tbl)
  write_lines(capture.output(summary(lm_weber)),
              "weber_lm_noIntercept.txt")
}

## ---------------------------------------------------------------------
## 5 ‒ Agent simulation (grid-world) ------------------------------
## ---------------------------------------------------------------------
# Expected CSV columns:
#  run_id    : replicate number (factor)
#  algorithm : 'DQN', 'Clock', 'MetricChrono'     (factor)
#  episodes  : episodes to reach criterion        (numeric)

agent_path <- "../data/agent_results.csv"
if (file.exists(agent_path)) {
  agent <- read_csv(agent_path,
                    col_types = cols(
                      run_id    = col_factor(),
                      algorithm = col_factor(),
                      episodes  = col_double()
                    ))

  # Log-transform for normality
  agent$log_ep <- log(agent$episodes)

  fit_agent <- aov_car(log_ep ~ algorithm + Error(run_id/algorithm),
                       data = agent, factorize = FALSE)

  write_lines(capture.output(nice(fit_agent)),
              "agent_anova.txt")

  # Tukey HSD
  posthoc <- emmeans(fit_agent, pairwise ~ algorithm, adjust = "tukey")
  write_lines(capture.output(posthoc),
              "agent_tukey.txt")

  # Boxplot
  ggplot(agent, aes(algorithm, episodes, fill = algorithm)) +
    geom_boxplot(width = 0.6, outlier.shape = NA, alpha = 0.8) +
    scale_y_log10() +
    labs(x = "Agent", y = "Episodes (log-scale)") +
    ggtitle("Episodes to Criterion by Agent") +
    ggsave("figure_agent_episodes.png",
           width = 5, height = 4, dpi = 300)
}

## ---------------------------------------------------------------------
## 6 ‒ Session info for reproducibility -------------------------------
## ---------------------------------------------------------------------
write_lines(capture.output(sessionInfo()),
            "session_info.txt")

# End of script

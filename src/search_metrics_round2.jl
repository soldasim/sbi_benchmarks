"""
Round 2 of the metric search (2026-08-03): simple confounds + multivariate check.

Round 1 (classify_smoothness.jl global + classify_smoothness_local.jl) found no
single-metric correlation with the WarpedGP-vs-Standard margin. Round 2 checks:

  (a) Simple confounds: dx, dy (dimensionality), H_rel, n_modes (posterior shape,
      from classify_posteriors.csv) — do any of these predict WarpedGP's benefit,
      even though they're not "smoothness" metrics per se?

  (b) Multivariate OLS: does a COMBINATION of the 5 global + 5 local metrics
      explain variance even though no single metric does? (High predictor-count/
      small-n overfitting risk — reported as a rough check via adjusted R², not
      a rigorous model.)

Reads only existing plots/*.csv files; writes only new plots/*.csv|png|pdf files
under distinct prefixes. Does not modify any BOSIP experiment data or existing
classification/score CSVs.

Usage (from repo root):
    julia --project=src -e 'include("src/correlate_metrics_lib.jl"); include("src/search_metrics_round2.jl")'
"""

const CONFOUNDS = ["dx", "dy"]
const POST_SHAPE = ["n_modes", "H_rel"]

println("\n########## ROUND 2a: dimensionality confounds (dx, dy) vs WarpedGP margin ##########")
run_correlation("plots/classify_smoothness.csv", "plots/warpedgp_scores_fair.csv",
                ("warpedgp_median","standard_median"), "corr_confound_dim_vs_warpedgp"; metrics=CONFOUNDS)

println("\n########## ROUND 2a: posterior-shape confounds (n_modes, H_rel) vs WarpedGP margin ##########")
run_correlation("plots/classify_posteriors.csv", "plots/warpedgp_scores_fair.csv",
                ("warpedgp_median","standard_median"), "corr_confound_postshape_vs_warpedgp"; metrics=POST_SHAPE)

println("\n########## ROUND 2b: multivariate OLS (5 global + 5 local metrics) vs WarpedGP margin ##########")
const METRICS5 = ["cv_value", "cv_grad", "yj_dev", "skewness", "heterosced"]
multivariate_r2(
    "plots/warpedgp_scores_fair.csv", ("warpedgp_median","standard_median"),
    [
        ("plots/classify_smoothness.csv",       METRICS5, "global_"),
        ("plots/classify_smoothness_local.csv", METRICS5, "local_"),
    ],
) |> x -> (global mv_result = x)

println("\n########## ROUND 2b: multivariate OLS (global 5 metrics only) vs WarpedGP margin ##########")
multivariate_r2(
    "plots/warpedgp_scores_fair.csv", ("warpedgp_median","standard_median"),
    [("plots/classify_smoothness.csv", METRICS5, "global_")],
) |> x -> (global mv_result_global_only = x)

println("\n########## ROUND 2b: multivariate OLS (local 5 metrics only) vs WarpedGP margin ##########")
multivariate_r2(
    "plots/warpedgp_scores_fair.csv", ("warpedgp_median","standard_median"),
    [("plots/classify_smoothness_local.csv", METRICS5, "local_")],
) |> x -> (global mv_result_local_only = x)

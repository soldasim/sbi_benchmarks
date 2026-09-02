#!/bin/sh

# ASSUMES pwd == bosip_benchmarks
#
# ARGS[1] = problem_name
# ARGS[2] = run_name
# ARGS[3] = run_idx
#
# Recomputes TV-metric scores using `log_posterior_mean_safe` (src/safe_posterior_estimator.jl)
# by re-fitting the run's own model type (GaussianProcess or NonstationaryGP, dispatched by
# run_name) on each recorded data-iteration prefix. Writes to `<run_name>_<run_idx>_TVmetric_safe.jld2`
# — never overwrites the original `_TVmetric.jld2`.

~/.juliaup/bin/julialauncher --project=src src/recompute_tv_general_safe.jl $1 $2 $3

#!/bin/sh

# ASSUMES pwd == bosip_benchmarks
#
# ARGS[1] = problem: The name of the `AbstractProblem` subtype (e.g. SimpleProblem, SIRProblem).
# ARGS[2] = run_name: The run-set name (e.g. nongp).
# ARGS[3] = run_idx: The index of this particular run.
#
# Recomputes TV-metric scores using `log_posterior_mean_safe` by re-fitting the NonstationaryGP
# model on each recorded data-iteration prefix. Writes to `<run_name>_<run_idx>_TVmetric_safe.jld2`
# — never overwrites the original `_TVmetric.jld2`.

~/.juliaup/bin/julialauncher --project=src src/recompute_tv_safe.jl $1 $2 $3

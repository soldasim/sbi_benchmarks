#!/bin/sh

# ASSUMES pwd == bosip_benchmarks
#
# ARGS[1] = problem: The name of the `AbstractProblem` subtype.
# ARGS[2] = run_name: The name of the whole run-set. The folders are named after this.
# ARGS[3] = run_idx: The index of this particular run.
# ARGS[4] = iters: The target total iteration count.
# ARGS[5] = noise: The noise hyperparameter to use. Only used for "...-noise" runs.

~/.juliaup/bin/julialauncher --project=src cluster_scripts/script_continue_jld2fix.jl $1 $2 $3 $4 $5

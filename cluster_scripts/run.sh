#!/bin/sh

# ASSUMES pwd == sbi_benchmarks
#
# ARGS[1] = problem: The name of the `AbstractProblem` subtype.
# ARGS[2] = run_name: The name of the whole run-set. The folders are named after this.
# ARGS[3] = run_idx: The index of this particular run. The starting data are selected based on this.
# ARGS[4] = continue: 0 or 1, whether to continue a previous run (1) or start a new one (0).
# ARGS[5] = iters: The number of iterations to run.
# ARGS[6] = noise: The noise hyperparameter to use. Only used for "...-noise" runs.

# ensure juliaup julia is in PATH
export PATH="$HOME/.juliaup/bin:$PATH"

# start julia
julia --project=src cluster_scripts/script.jl $1 $2 $3 $4 $5 $6

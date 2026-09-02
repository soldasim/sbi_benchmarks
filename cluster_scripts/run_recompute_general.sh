#!/bin/sh

# ASSUMES pwd == bosip_benchmarks
#
# ARGS[1] = problem_name
# ARGS[2] = run_name
# ARGS[3] = run_idx

~/.juliaup/bin/julialauncher --project=src src/recompute_tv_general.jl $1 $2 $3

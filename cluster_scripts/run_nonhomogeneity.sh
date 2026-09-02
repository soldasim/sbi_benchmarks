#!/bin/sh

# ASSUMES pwd == bosip_benchmarks
#
# ARGS[1] = problem_name: reconstructible problem name (may include "_cross" suffix)
# ARGS[2] = dim_idx: 1-based output dimension index

~/.juliaup/bin/julialauncher --project=src cluster_scripts/script_nonhomogeneity.jl $1 $2

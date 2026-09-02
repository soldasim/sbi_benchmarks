#!/bin/bash
# Run precompute_grid.jl for a given problem.
# ARGS[1] = problem name (e.g. ProxySIRProblem)

# ensure juliaup julia is in PATH
export PATH="$HOME/.juliaup/bin:$PATH"

julia --project=src src/precompute_grid.jl $1

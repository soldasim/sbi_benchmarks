# Set up DuffingProblem5: generate random starts and precompute true-posterior grid.
# Run from an interactive SLURM job (not the login node).
#
# Usage (from the Julia REPL after `include("src/main.jl")`):
#   include("src/plot_marginals.jl")
#   include("src/setup_duffing5_problem.jl")
#
# Estimated runtime (4 threads, GRID_RES=50, LHC_SIZE=500):
#   ~10-30 min depending on ODE solve speed.
# For a quick test use: precompute_bip_grid(DuffingProblem5(); grid_res=10, lhc_size=20)

include("plot_marginals.jl")
include("generate_starts.jl")

prob = DuffingProblem5()
name = get_name(prob)

@info "Generating starts for $name ..."
generate_starts(prob, 20)
@info "Starts done."

@info "Precomputing true posterior grid for $name ..."
precompute_bip_grid(prob)
@info "Grid done. Saved to $(TRUE_POST_GRID_DIR)/$(name).jld2"

@info "Setup complete for $name."

# Set up DuffingProblem5 and DiffusionProblem5D for BOSIP experiments:
# generates 20 random starts and precomputes TV metric evaluation grids
# (posterior_grid.jld2 + simulator_grid.jld2) in data-bosip-norm/.
#
# Run from an interactive SLURM job after `include("src/main.jl")`:
#   include("src/setup_5d_bip_problems.jl")
#
# Estimated runtime (4 threads):
#   DuffingProblem5  — ~5-15 min  (ODE solver, 20k grid points)
#   DiffusionProblem5D — ~10-15 min  (PDE solver ~0.018s/call × 20k points)

include("generate_starts.jl")
include("precompute_grid.jl")

for problem in [DuffingProblem5(), DiffusionProblem5D()]
    name = get_name(problem)
    @info "=== Setting up $name ==="

    @info "Generating starts ..."
    generate_starts(problem, 20)
    @info "Starts done."

    @info "Precomputing TV metric grids ..."
    precompute_grid(problem)
    @info "Grids done. Saved to $(data_dir(problem))/grid/"

    @info "=== Done: $name ==="
end

@info "All 5D BIP problems set up."

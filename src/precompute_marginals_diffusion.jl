## Precompute pairwise posterior marginals for DiffusionProblem10.
## Run via submit_precompute_diffusion.sh (24 CPUs, cpufast partition).

include(joinpath(@__DIR__, "main.jl"))
include(joinpath(@__DIR__, "plot_marginals.jl"))

precompute_bip_grid(DiffusionProblem10(); force=false)

@info "Done — DiffusionProblem10 precomputed."

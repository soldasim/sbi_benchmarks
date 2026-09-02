# Generate random starts and precompute metric grids for BealeProblem_cross and
# GoldsteinPriceProblem_cross (non-proxy cross-polytope variants).
# Run from an interactive SLURM job (not the login node).

include("main.jl")
include("generate_starts.jl")
include("precompute_grid.jl")

problems = CrossPolytopeObsProblem.([
    BealeProblem(),
    GoldsteinPriceProblem(),
])

for problem in problems
    name = get_name(problem)
    @info "Setting up $name ..."
    generate_starts(problem, 20)
    precompute_grid(problem)
    @info "Done: $name"
end

@info "BealeProblem_cross and GoldsteinPriceProblem_cross set up."

# Generate random starts and precompute metric grids for the 3 hex opt-function problems.
# Run on a compute node (not the login node).

include("main.jl")
include("generate_starts.jl")
include("precompute_grid.jl")

problems = HexObsProblem.([
    RosenbrockProblem(; x_dim=2),
    StyblinskiTangProblem(; x_dim=2),
    MichalewiczProblem(; x_dim=2),
])

@assert length(problems) == 3

for problem in problems
    name = get_name(problem)
    @info "Setting up $name ..."
    generate_starts(problem, 20)
    precompute_grid(problem)
    @info "Done: $name"
end

@info "All 3 hex problems set up."

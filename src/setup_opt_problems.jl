# Generate random starts and precompute metric grids for the new opt-function problems.
# Run on a compute node (not login node).

include("main.jl")
include("generate_starts.jl")
include("precompute_grid.jl")

problems = [
    RosenbrockProblem(; x_dim=2),
    RosenbrockProblem(; x_dim=5),
    StyblinskiTangProblem(; x_dim=2),
    StyblinskiTangProblem(; x_dim=5),
    MichalewiczProblem(; x_dim=2),
    MichalewiczProblem(; x_dim=5),
]

for problem in problems
    name = get_name(problem)
    @info "Setting up $name ..."
    generate_starts(problem, 20)
    precompute_grid(problem)
    @info "Done: $name"
end

@info "All problems set up."

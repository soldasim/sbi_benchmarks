# Generate random starts and precompute metric grids for the 12 5D cross-polytope problems.
# Run from an interactive SLURM job (not the login node).

include("main.jl")
include("generate_starts.jl")
include("precompute_grid.jl")

problems = CrossPolytopeObsProblem.([
    RosenbrockProblem(; x_dim=5),
    StyblinskiTangProblem(; x_dim=5),
    MichalewiczProblem(; x_dim=5),
    AckleyProblem(; x_dim=5),
    AlpineProblem(; x_dim=5),
    ExpandedSchafferF6Problem(; x_dim=5),
    ExpandedZakharovProblem(; x_dim=5),
    GriewankProblem(; x_dim=5),
    RastriginProblem(; x_dim=5),
    SalomonProblem(; x_dim=5),
    SchwefelProblem(; x_dim=5),
    SphereProblem(; x_dim=5),
])

@assert length(problems) == 12

for problem in problems
    name = get_name(problem)
    @info "Setting up $name ..."
    generate_starts(problem, 20)
    precompute_grid(problem)
    @info "Done: $name"
end

@info "All 12 cross-polytope 5D problems set up."

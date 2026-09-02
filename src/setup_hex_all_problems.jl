# Generate random starts and precompute metric grids for the remaining 21 hex opt-function problems.
# (The first 3 — Rosenbrock, StyblinskiTang, Michalewicz — are already set up.)
# Run on a compute node (not the login node).

include("main.jl")
include("generate_starts.jl")
include("precompute_grid.jl")

problems = HexObsProblem.([
    # d-dimensional at 2D
    AckleyProblem(; x_dim=2),
    AlpineProblem(; x_dim=2),
    ExpandedSchafferF6Problem(; x_dim=2),
    ExpandedZakharovProblem(; x_dim=2),
    GriewankProblem(; x_dim=2),
    RastriginProblem(; x_dim=2),
    SalomonProblem(; x_dim=2),
    SchwefelProblem(; x_dim=2),
    SphereProblem(; x_dim=2),
    # 2D-only
    BealeProxyProblem(),
    BoothProblem(),
    CrossInTrayProblem(),
    DropWaveProblem(),
    EasomProblem(),
    GoldsteinPriceProxyProblem(),
    HimmelblauProblem(),
    HolderTableProblem(),
    LeviN13Problem(),
    MatyasProblem(),
    SchafferN2Problem(),
    ThreeHumpCamelProblem(),
])

@assert length(problems) == 21

for problem in problems
    name = get_name(problem)
    @info "Setting up $name ..."
    generate_starts(problem, 20)
    precompute_grid(problem)
    @info "Done: $name"
end

@info "All 21 remaining hex problems set up."

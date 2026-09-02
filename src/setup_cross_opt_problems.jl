# Generate random starts and precompute metric grids for all 24 2D cross-polytope problems.
# Run from an interactive SLURM job (not the login node).

include("main.jl")
include("generate_starts.jl")
include("precompute_grid.jl")

problems = CrossPolytopeObsProblem.([
    # d-dimensional at 2D (original 3)
    RosenbrockProblem(; x_dim=2),
    StyblinskiTangProblem(; x_dim=2),
    MichalewiczProblem(; x_dim=2),
    # d-dimensional at 2D (9 more)
    AckleyProblem(; x_dim=2),
    AlpineProblem(; x_dim=2),
    ExpandedSchafferF6Problem(; x_dim=2),
    ExpandedZakharovProblem(; x_dim=2),
    GriewankProblem(; x_dim=2),
    RastriginProblem(; x_dim=2),
    SalomonProblem(; x_dim=2),
    SchwefelProblem(; x_dim=2),
    SphereProblem(; x_dim=2),
    # 2D-only (12)
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

@assert length(problems) == 24

for problem in problems
    name = get_name(problem)
    @info "Setting up $name ..."
    generate_starts(problem, 20)
    precompute_grid(problem)
    @info "Done: $name"
end

@info "All 24 cross-polytope 2D problems set up."

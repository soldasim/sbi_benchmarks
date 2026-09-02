## Precompute posterior grids for all 2D problems (BIP + opt) and DuffingProblem.
## Run via submit_precompute_fast.sh (4 CPUs, cpufast partition).

include(joinpath(@__DIR__, "main.jl"))
include(joinpath(@__DIR__, "plot_marginals.jl"))

fast_problems = [
    ## 2D BIP problems
    ABProblem(),
    SimpleProblem(),
    BananaProblem(),
    BimodalProblem(),
    SIRProblem(),
    ## 3D ODE BIP problem
    DuffingProblem(),
    ## 24 optimization-function problems (all 2D)
    RosenbrockProblem(),
    StyblinskiTangProblem(),
    MichalewiczProblem(),
    AckleyProblem(),
    AlpineProblem(),
    ExpandedSchafferF6Problem(),
    ExpandedZakharovProblem(),
    GriewankProblem(),
    RastriginProblem(),
    SalomonProblem(),
    SchwefelProblem(),
    SphereProblem(),
    BealeProblem(),
    BoothProblem(),
    CrossInTrayProblem(),
    DropWaveProblem(),
    EasomProblem(),
    GoldsteinPriceProblem(),
    HimmelblauProblem(),
    HolderTableProblem(),
    LeviN13Problem(),
    MatyasProblem(),
    SchafferN2Problem(),
    ThreeHumpCamelProblem(),
]

foreach(p -> precompute_bip_grid(p; force=false), fast_problems)

@info "Done — all fast problems precomputed."

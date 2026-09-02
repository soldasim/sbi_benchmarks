# Queue MaxVar and EIV experiments for all 24 hex opt-function problems.
# 5 runs each, 200 iterations.
# Run from the repo root: julia --project=src cluster_scripts/queue_hex_all_jobs.jl

include("../src/main.jl")
include("queue_jobs.jl")

const ALL_HEX_PROBLEMS = HexObsProblem.([
    # Original 3
    RosenbrockProblem(; x_dim=2),
    StyblinskiTangProblem(; x_dim=2),
    MichalewiczProblem(; x_dim=2),
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

@assert length(ALL_HEX_PROBLEMS) == 24

const HEX_RUN_NAMES = ["maxvar", "eiv"]

for problem in ALL_HEX_PROBLEMS
    for run_name in HEX_RUN_NAMES
        queue_jobs(problem, run_name; selected_runs=collect(1:5), iters=200)
    end
end

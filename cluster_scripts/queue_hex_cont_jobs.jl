# Continue incomplete hex experiments.
# MaxVar: GoldsteinPriceProxy run 2, Himmelblau runs 1-4.
# EIV: all 24 problems x 5 runs.
# Run from the repo root: julia --project=src cluster_scripts/queue_hex_cont_jobs.jl

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

# MaxVar: only the specific incomplete runs
@info "Queuing MaxVar continuation jobs ..."
queue_jobs(HexObsProblem(GoldsteinPriceProxyProblem()), "maxvar";
    selected_runs=[2], continued=true, iters=200, time="24:00:00")
queue_jobs(HexObsProblem(HimmelblauProblem()), "maxvar";
    selected_runs=[1, 2, 3, 4], continued=true, iters=200, time="24:00:00")

# EIV: all 24 problems x 5 runs
@info "Queuing EIV continuation jobs ..."
for problem in ALL_HEX_PROBLEMS
    queue_jobs(problem, "eiv"; selected_runs=collect(1:5), continued=true, iters=200, time="24:00:00")
end

@info "Done queuing continuation jobs."

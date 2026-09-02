# Queue MaxVar and EIV experiments for the 3 hex opt-function problems.
# Runs 5 of the 20 pre-generated starts, 20 iterations each.
# Run from the repo root: julia --project=src cluster_scripts/queue_hex_jobs.jl

include("../src/main.jl")
include("queue_jobs.jl")

const HEX_PROBLEMS = HexObsProblem.([
    RosenbrockProblem(; x_dim=2),
    StyblinskiTangProblem(; x_dim=2),
    MichalewiczProblem(; x_dim=2),
])

const HEX_RUN_NAMES = ["maxvar", "eiv"]

for problem in HEX_PROBLEMS
    for run_name in HEX_RUN_NAMES
        queue_jobs(problem, run_name; selected_runs=collect(1:5), iters=20)
    end
end

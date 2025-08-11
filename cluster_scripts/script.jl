@show ARGS

# ARGS are:
# #1: problem name (matches a subtype of `AbstractProblem`)
# #2: run name (describes the used BOLFI setup)
# #3: run index
const problem_name = ARGS[1]
const run_name = ARGS[2]
const run_idx = parse(Int, ARGS[3])

# Include the main script for the run setup.
include("../src/main_scripts/main" * "_" * run_name * ".jl")

const problem = getfield(Main, Symbol(problem_name))()
const start_file = starts_dir(problem) * "/start_$(run_idx).jld2"
const data = load(start_file, "X")

main(problem; run_name, save_data=true, metric=true, data, run_idx)

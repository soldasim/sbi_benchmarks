@show ARGS

# ARGS are:
# #1: problem name (matches a subtype of `AbstractProblem`)
# #2: run name (describes the used BOSIP setup)
# #3: run index
# #4: continue? (0 or 1)
const problem_name = ARGS[1]
const run_name = ARGS[2]
const run_idx = parse(Int, ARGS[3])
const continued = (parse(Int, ARGS[4]) != 0)
const iters = parse(Int, ARGS[5])
const noise = (ARGS[6] == "nothing") ? nothing : parse(Float64, ARGS[6])

# Include the main script for the run setup.
include("../src/main_scripts/main" * "_" * run_name * ".jl")

const problem = reconstruct_problem(problem_name)
const start_file = starts_dir(problem) * "/start_$(run_idx).jld2"
const data = load(start_file, "X")

if continued
    main_continue(problem, run_name, run_idx; save_data=true, convergence=true, metric=true, iters, noise)
else
    main(problem; run_name, save_data=true, convergence=true, metric=true, data, run_idx, iters, noise)
end

# Force exit to avoid hanging in finalizers from native-code libraries (PRIMA/Fortran).
exit()

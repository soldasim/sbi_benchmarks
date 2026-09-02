@show ARGS

# Continue a run, applying JLD2 migration shims for checkpoints saved under
# older BOSIP.jl/BOSS.jl struct layouts (e.g. MAPParams{GaussianProcess}'s
# `.loglike` field renamed to `.logpost` in BOSS.jl 5f06d33). No data_dir
# override needed here (unlike script_continue_override.jl) - just the
# checkpoint-compatibility fix, for problem types whose data_dir routing is
# already correct.
#
# ARGS are:
# #1: problem name (matches a subtype of `AbstractProblem`)
# #2: run name (describes the used BOSIP setup)
# #3: run index
# #4: iters (target total iteration count)
# #5: noise
const problem_name = ARGS[1]
const run_name = ARGS[2]
const run_idx = parse(Int, ARGS[3])
const iters = parse(Int, ARGS[4])
const noise = (ARGS[5] == "nothing") ? nothing : parse(Float64, ARGS[5])

# Include the main script for the run setup.
include("../src/main_scripts/main" * "_" * run_name * ".jl")

const problem = reconstruct_problem(problem_name)

# JLD2 migration shims for checkpoints saved under older BOSIP.jl/BOSS.jl struct
# layouts (harmless no-op if the loaded checkpoint doesn't need them).
include("jld2_compat_duffing_extend.jl")

main_continue(problem, run_name, run_idx; save_data=true, convergence=true, metric=true, iters, noise)

# Force exit to avoid hanging in finalizers from native-code libraries (PRIMA/Fortran).
exit()

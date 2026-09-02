@show ARGS

# Continue a run, overriding `data_dir` for this problem type to a fixed directory.
#
# Workaround for a data_dir() routing regression: data_dir(::AbstractProblem) falls
# back to "data-convergence4/<name>" for problem types that don't have an explicit
# override, but the original 7 BIP problems' actual data (from before the TV-metric
# normalization fix moved grids to data-bosip-norm/) lives split across data-bosip/
# and data-bosip-norm/. This script pins data_dir to the correct directory for the
# single problem type being run, per job (safe: isolated SLURM process, no effect on
# data_paths.jl's routing for any other problem type/job).
#
# ARGS are:
# #1: problem name (matches a subtype of `AbstractProblem`)
# #2: run name (describes the used BOSIP setup)
# #3: run index
# #4: iters (target total iteration count)
# #5: noise
# #6: data_dir override (e.g. "data-bosip/DuffingProblem")
const problem_name = ARGS[1]
const run_name = ARGS[2]
const run_idx = parse(Int, ARGS[3])
const iters = parse(Int, ARGS[4])
const noise = (ARGS[5] == "nothing") ? nothing : parse(Float64, ARGS[5])
const override_dir = ARGS[6]

# Include the main script for the run setup.
include("../src/main_scripts/main" * "_" * run_name * ".jl")

const problem = reconstruct_problem(problem_name)

const OverrideT = typeof(problem)
@eval data_dir(::$OverrideT) = $override_dir

# JLD2 migration shims for checkpoints saved under older BOSIP.jl/BOSS.jl struct
# layouts (harmless no-op if the loaded checkpoint doesn't need them).
include("jld2_compat_duffing_extend.jl")

main_continue(problem, run_name, run_idx; save_data=true, convergence=true, metric=true, iters, noise)

# Force exit to avoid hanging in finalizers from native-code libraries (PRIMA/Fortran).
exit()

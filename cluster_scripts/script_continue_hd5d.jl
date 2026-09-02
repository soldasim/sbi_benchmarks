@show ARGS

# Continue a run for DuffingProblem5 / DiffusionProblem5D, applying the JLD2
# migration shims (MAPParams.loglike->logpost rename etc., see
# jld2_compat_duffing_extend.jl) plus the same post-load simulator-closure fix
# used for DiffusionProblem10 (script_continue_override_diffusion.jl) -
# harmless no-op if the checkpoint's `f` field already deserializes as a
# callable Function.
#
# Unlike script_continue_override*.jl, no data_dir override is needed here:
# both problems already have explicit data_dir overrides in data_paths.jl
# pointing at the correct data-bosip-norm/<name> directory.
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

# Generic JLD2 migration shims (BosipOptions.parallel_evals, MAPParams.logpost rename,
# TVMetric.true_logvals, NormalLikelihood de-parametrization) - reused as-is; its
# Duffing-specific BossProblem/act_func shims are harmless no-ops here.
include("jld2_compat_duffing_extend.jl")

# --- replicate main_continue's load, then fix the simulator closure if needed ---
const _load_dir = isdefined(Main, :warpedgp_data_dir) ? warpedgp_data_dir(problem) : data_dir(problem)
const _cont_file = isnothing(run_idx) ?
    joinpath(_load_dir, "$(run_name)_problem.jld2") :
    joinpath(_load_dir, "$(run_name)_$(run_idx)_problem.jld2")
const bosip = load(_cont_file)["problem"]
@assert bosip isa BosipProblem

if !(bosip.problem.f isa Function)
    old_bp = bosip.problem
    bosip.problem = BossProblem(simulator(problem), old_bp.domain, old_bp.y_max, old_bp.acquisition, old_bp.model, old_bp.params, old_bp.data, old_bp.consistent)
end

const estimator = log_posterior_mean
@warn "using posterior estimator: $(estimator |> nameof |> string)"

const data_count = size(bosip.problem.data.X, 2)
@assert data_count > 3
const data_max = 3 + iters

main(problem, bosip, estimator; continued=true, run_name, run_idx, save_data=true, convergence=true, metric=true, data_max, iters, noise)

# Force exit to avoid hanging in finalizers from native-code libraries (PRIMA/Fortran).
exit()

@show ARGS

# Continue a run for DiffusionProblem10 / LogDiffusionProblem, overriding `data_dir`
# (same routing trap as DuffingProblem, see script_continue_override.jl) AND fixing
# up a non-callable simulator closure after loading the checkpoint.
#
# Unlike DuffingProblem's `model_target` (zero captured fields, stateless), Diffusion's
# `model_target` closes over `p` (and, for LogDiffusionProblem, also `likelihood`) -
# JLD2 successfully constructs a `BossProblem{T}` with T = the placeholder
# reconstructed-closure type, but that placeholder isn't callable. Since `BossProblem`'s
# `f::F` field type is a free type parameter (not a fixed `Function` annotation), no
# `rconvert` hook can intercept this - so we patch it post-load instead: build a fresh
# `BossProblem` with `f = simulator(problem)` (using the already correctly-reconstructed
# live `problem` object) and swap it into the loaded `BosipProblem` (a mutable struct).
# Verified equivalent (matches stored Y to ~1e-3) for all of standard/est/eiv/nongp/
# immd/warpedgp-yja-maxvar/loglike/loglike-imiqr checkpoints, 2026-08-06.
#
# ARGS are:
# #1: problem name (matches a subtype of `AbstractProblem`)
# #2: run name (describes the used BOSIP setup)
# #3: run index
# #4: iters (target total iteration count)
# #5: noise
# #6: data_dir override (e.g. "data-bosip/DiffusionProblem10")
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

# Generic JLD2 migration shims (BosipOptions.parallel_evals, MAPParams.logpost rename,
# TVMetric.true_logvals, NormalLikelihood de-parametrization) - reused as-is, see that
# file for the itemized list; its Duffing-specific BossProblem/act_func shims are
# harmless no-ops here (they just never match Diffusion's on-disk closure shape).
include("jld2_compat_duffing_extend.jl")

# --- replicate main_continue's load, then fix the simulator closure ---
# warpedgp-yja-maxvar's own checkpoint lives under warpedgp_data_dir (hardcoded to
# data-warpedgp2/), not data_dir - only its grid/starts lookups go through data_dir.
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

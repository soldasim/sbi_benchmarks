# Recompute TV metric scores for ANY problem/run_name using the "stabilized" posterior
# estimator (`log_posterior_mean_safe`, src/safe_posterior_estimator.jl) instead of the
# original `log_posterior_mean`. Generalizes `recompute_tv_safe.jl` (which was hardcoded to
# `nongp`/`NonstationaryGP` only) to any model type, mirroring `recompute_tv_general.jl`'s
# generic `data_dir`/`make_model` dispatch.
#
# Background: `MetricCallback` can throw `DomainError` when a model's posterior variance
# comes out invalid (negative, past `_clip_var`'s tolerance) at a given TV-sample point,
# and the callback records NaN for that iteration. `log_posterior_mean_safe` wraps the
# model posterior so such failures clamp to a small positive variance instead of
# propagating, per the docstring in `src/safe_posterior_estimator.jl`.
#
# Does NOT rerun the BO loop or acquire new observations — only refits the surrogate model
# on the already-collected `_data.jld2` (X, Y) trajectory and recomputes the TV score at
# each iteration, exactly as `recompute_tv_general.jl` does, but through the safe estimator.
#
# Writes to a NEW file (`_TVmetric_safe.jld2`) — NEVER overwrites the original
# `_TVmetric.jld2` — so the old (NaN-containing) scores remain available for comparison.
#
# Reads:  <data_dir(problem)>/<run_name>_<run_idx>_data.jld2
# Writes: <data_dir(problem)>/<run_name>_<run_idx>_TVmetric_safe.jld2
#
# ARGS:
#   1: problem_name  (e.g. "BealeProblem_cross")
#   2: run_name       (e.g. "nongp", "eiv", "maxvar")
#   3: run_idx        (integer, 1-based)

using BOSS
using BOSIP
using Distributions
using KernelFunctions
using LinearAlgebra
using OptimizationPRIMA
using Bijectors
using JLD2

using Random
Random.seed!(555)

parallel() = false  # PRIMA.jl StackOverflow on Linux when parallelized

include(pwd() * "/src/include_code.jl")  # also includes safe_posterior_estimator.jl -> log_posterior_mean_safe

const problem_name = ARGS[1]
const run_name = ARGS[2]
const run_idx = parse(Int, ARGS[3])
const n_init = 3  # init_data_count hardcoded in all original main scripts

const problem = reconstruct_problem(problem_name)
const x_dim_problem = length(domain(problem).bounds[1])

const base_dir = data_dir(problem)
const data_file = base_dir * "/" * run_name * "_" * string(run_idx) * "_data.jld2"

if !isfile(data_file)
    @warn "Source file not found, skipping: $data_file"
    exit(0)
end

const out_file = base_dir * "/" * run_name * "_" * string(run_idx) * "_TVmetric_safe.jld2"

@info "Recomputing TV (safe estimator): $problem_name / $run_name / run $run_idx"

# Load all accumulated observations from the original run
X_full, Y_full = load(data_file, "data")
const n_total = size(X_full, 2)
const n_iters = n_total - n_init
@info "  $n_total obs total ($n_init init + $n_iters iters)"
size(X_full, 1) > x_dim_problem && @warn "  X has $(size(X_full,1)) rows but x_dim=$x_dim_problem — trimming to first $x_dim_problem rows (legacy BOSS artifact)"

const grid_data_loaded = load_grid(problem)
const tv_metric = TVMetric(;
    grid = grid_data_loaded.xs,
    log_ws = grid_data_loaded.log_ws,
    true_logvals = grid_data_loaded.true_logvals,
)
const ref = true_logpost(problem)

function make_model(run_name::String, problem::AbstractProblem)
    if run_name == "nongp"
        return NonstationaryGP(;
            mean = prior_mean(problem),
            lengthscale_model = BOSS.default_lengthscale_model(domain(problem).bounds, y_dim(problem)),
            amplitude_model = get_amplitude_priors(problem),
            noise_std_model = get_noise_std_priors(problem),
        )
    else  # standard/maxvar, eiv, eiig, immd all use GaussianProcess
        return GaussianProcess(;
            mean = prior_mean(problem),
            kernel = BOSS.Matern52Kernel(),
            lengthscale_priors = get_lengthscale_priors(problem),
            amplitude_priors = get_amplitude_priors(problem),
            noise_std_priors = get_noise_std_priors(problem),
        )
    end
end

const model_fitter = OptimizationMAP(;
    algorithm = NEWUOA(),
    multistart = 24,
    parallel = parallel(),
    rhoend = 1e-4,
)

# Checkpoint file — partial results saved here; deleted on success. Distinct name from
# recompute_tv_general.jl's own checkpoint so the two scripts never collide if run on the
# same (problem, run_name, idx) at the same time.
const chk_file = base_dir * "/" * run_name * "_" * string(run_idx) * "_TVmetric_safe.chk.jld2"

scores = Vector{Float64}(undef, n_iters + 1)
start_i = 0
if isfile(chk_file)
    chk_data = load(chk_file)
    start_i = chk_data["start_i"]
    scores[1:start_i] .= chk_data["scores"][1:start_i]
    @info "  Resuming from checkpoint at iter $(start_i - 1) / $n_iters"
end

for i in start_i:n_iters
    n = n_init + i
    data = ExperimentData(X_full[1:x_dim_problem, 1:n], Y_full[:, 1:n])
    model = make_model(run_name, problem)
    bosip = construct_bosip_problem(; problem, data, acquisition=LogMaxVar(), model)

    estimate_parameters!(bosip.problem, model_fitter)

    scores[i+1] = try
        post_fn = run_name in ("est", "loglike", "loglike-imiqr") ?
            log_approx_posterior(bosip) : log_posterior_mean_safe(bosip)
        calculate_metric(tv_metric, ref, post_fn)
    catch e
        @warn "  iter $i: threw $(typeof(e)), storing NaN" e
        NaN
    end
    (i % 10 == 0) && @info "  iter $i / $n_iters  TV = $(round(scores[i+1], digits=4))"

    if i % 10 == 0
        jldsave(chk_file; scores=scores, start_i=i+1)
    end
end

jldsave(out_file; score=scores)
rm(chk_file; force=true)
@info "Saved to $out_file"
exit()

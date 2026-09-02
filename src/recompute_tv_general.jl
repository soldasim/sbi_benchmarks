# Recompute TV metric scores IN PLACE for any problem, using the existing
# X/Y observation data on disk and the CURRENT (numerically-improved) BOSS.jl/
# BOSIP.jl code. Does NOT rerun the BO loop or acquire new observations —
# only refits the surrogate model to the already-collected data and
# recomputes the TV score at each iteration.
#
# Motivation: several NaN-contaminated TVmetric files predate BOSS.jl fixes
# 2315ad1 ("mitigate various numerical issues", 2026-06-23), 3a46f48 ("add
# model hyperparameter feasibility checks", 2026-06-23), and 5e57410 ("add
# iterative GP noise jitter increase", 2026-07-16). Recomputing with current
# code may resolve NaNs that were pure numerical artifacts of GP fitting,
# without needing brand-new simulator evaluations.
#
# The old TVmetric file is archived (not deleted) before being overwritten.
#
# Reads:  <data_dir(problem)>/<run_name>_<run_idx>_data.jld2
# Writes: <data_dir(problem)>/<run_name>_<run_idx>_TVmetric.jld2 (overwritten)
#         <data_dir(problem)>/archive_recompute_boss_stability_2026-08-03/<run_name>_<run_idx>_TVmetric.jld2 (old copy)
#
# ARGS:
#   1: problem_name  (e.g. "BoothProblem_cross")
#   2: run_name       (e.g. "eiv")
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

include(pwd() * "/src/include_code.jl")

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

const out_file = base_dir * "/" * run_name * "_" * string(run_idx) * "_TVmetric.jld2"
const archive_dir = base_dir * "/archive_recompute_boss_stability_2026-08-03"
mkpath(archive_dir)
const archive_file = archive_dir * "/" * run_name * "_" * string(run_idx) * "_TVmetric.jld2"

if isfile(archive_file)
    @info "Already recomputed (archive exists), skipping: $archive_file"
    exit(0)
end

@info "Recomputing TV: $problem_name / $run_name / run $run_idx"

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

# Checkpoint file — partial results saved here; deleted on success.
const chk_file = base_dir * "/" * run_name * "_" * string(run_idx) * "_TVmetric.chk.jld2"

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
            log_approx_posterior(bosip) : log_posterior_mean(bosip)
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

# Archive the old (pre-recompute) TVmetric file before overwriting.
if isfile(out_file)
    cp(out_file, archive_file; force=true)
end

jldsave(out_file; score=scores)
rm(chk_file; force=true)
@info "Saved to $out_file (old version archived to $archive_file)"
exit()

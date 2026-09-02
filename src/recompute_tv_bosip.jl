# Recompute TV metric scores for original data-bosip runs using the corrected
# importance-weighted TV formula, writing results to data-bosip-norm.
#
# The original TVmetric files in data-bosip/ were computed with a bug: calc_tv
# summed absolute differences without grid importance weights. The corrected
# formula uses proper Monte Carlo importance weighting (see BOSIP.jl tv.jl).
#
# Reads:  data-bosip/<problem>/  (X/Y observations and precomputed grid)
# Writes: data-bosip-norm/<problem>/  (corrected TVmetric score only)
#
# ARGS:
#   1: problem_name  (e.g. "ABProblem")
#   2: run_name      (e.g. "standard")
#   3: run_idx       (integer, 1-based)

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
const x_dim_problem = length(domain(problem).bounds[1])  # true parameter dimensionality

# Source: always read from data-bosip/ (the original runs)
const src_dir = "data-bosip/" * get_name(problem)
const data_file = src_dir * "/" * run_name * "_" * string(run_idx) * "_data.jld2"

if !isfile(data_file)
    @warn "Source file not found, skipping: $data_file"
    exit(0)
end

# Output: write to data-bosip-norm/
const dst_dir = "data-bosip-norm/" * get_name(problem)
mkpath(dst_dir)

const out_file = dst_dir * "/" * run_name * "_" * string(run_idx) * "_TVmetric.jld2"
if isfile(out_file)
    @info "Output already exists, skipping: $out_file"
    exit(0)
end

@info "Recomputing TV: $problem_name / $run_name / run $run_idx"

# Load all accumulated observations from the original run
X_full, Y_full = load(data_file, "data")
const n_total = size(X_full, 2)
const n_iters = n_total - n_init
@info "  $n_total obs total ($n_init init + $n_iters iters)"
size(X_full, 1) > x_dim_problem && @warn "  X has $(size(X_full,1)) rows but x_dim=$x_dim_problem — trimming to first $x_dim_problem rows (legacy BOSS artifact)"

# Load precomputed grid via data_paths.jl routing (data-bosip-norm/ for the 7
# original problems, data-bosip/ for ProxySIRProblem)
const grid_data_loaded = load_grid(problem)
const tv_metric = TVMetric(;
    grid = grid_data_loaded.xs,
    log_ws = grid_data_loaded.log_ws,
    true_logvals = grid_data_loaded.true_logvals,
)
const ref = true_logpost(problem)

# Model setup — mirrors the original main scripts
function make_model(run_name::String, problem::AbstractProblem)
    if run_name == "nongp"
        return NonstationaryGP(;
            mean = prior_mean(problem),
            lengthscale_model = BOSS.default_lengthscale_model(domain(problem).bounds, y_dim(problem)),
            amplitude_model = get_amplitude_priors(problem),
            noise_std_model = get_noise_std_priors(problem),
        )
    else  # standard, eiv, eiig all use GaussianProcess
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
const chk_file = dst_dir * "/" * run_name * "_" * string(run_idx) * "_TVmetric.chk.jld2"

# Resume from checkpoint if one exists from a previous (e.g. timed-out) run.
scores = Vector{Float64}(undef, n_iters + 1)
start_i = 0
if isfile(chk_file)
    chk_data = load(chk_file)
    start_i = chk_data["start_i"]
    scores[1:start_i] .= chk_data["scores"][1:start_i]
    @info "  Resuming from checkpoint at iter $(start_i - 1) / $n_iters"
end

# Recompute TV at each iteration by refitting GP on the observed data slice.
for i in start_i:n_iters
    n = n_init + i
    # Slice to x_dim_problem rows: ABProblem in data-bosip has a legacy extra row
    # stored by an older BOSS version; all other problems already match.
    data = ExperimentData(X_full[1:x_dim_problem, 1:n], Y_full[:, 1:n])
    model = make_model(run_name, problem)
    bosip = construct_bosip_problem(; problem, data, acquisition=LogMaxVar(), model)

    estimate_parameters!(bosip.problem, model_fitter)

    # est/loglike/loglike-imiqr use log_approx_posterior (single MAP draw of the approx
    # likelihood); all other run types (standard, eiv, eiig, nongp) use log_posterior_mean.
    # Both post_fn construction and calculate_metric are wrapped: log_approx_posterior can
    # throw PosDefException when MAP hyperparams leave the GP covariance ill-conditioned.
    scores[i+1] = try
        post_fn = run_name in ("est", "loglike", "loglike-imiqr") ?
            log_approx_posterior(bosip) : log_posterior_mean(bosip)
        calculate_metric(tv_metric, ref, post_fn)
    catch e
        @warn "  iter $i: threw $(typeof(e)), storing NaN" e
        NaN
    end
    (i % 10 == 0) && @info "  iter $i / $n_iters  TV = $(round(scores[i+1], digits=4))"

    # Save checkpoint every 10 iters so a timeout/crash doesn't lose all progress.
    if i % 10 == 0
        jldsave(chk_file; scores=scores, start_i=i+1)
    end
end

jldsave(out_file; score=scores)
rm(chk_file; force=true)
@info "Saved to $out_file"
exit()

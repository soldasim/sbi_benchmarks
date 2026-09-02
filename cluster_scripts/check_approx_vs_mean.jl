# Ad-hoc diagnostic: compare TV metric under log_approx_posterior (mean-only,
# disregards predictive uncertainty) vs log_posterior_mean (integrates GP
# predictive uncertainty), refitting NonstationaryGP at a specific data slice.
#
# Read-only diagnostic: reads existing nongp data, writes results to a NEW
# scratch file only (never touches data-bosip-norm/ or any existing output).
#
# ARGS:
#   1: problem_name (e.g. "SimpleProblem")
#   2: run_idx      (integer, e.g. 10)
#   3: n            (number of data points to slice to, e.g. 75)

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
const run_idx = parse(Int, ARGS[2])
const n = parse(Int, ARGS[3])

const sproblem = reconstruct_problem(problem_name)
const ref = true_logpost(sproblem)
const gd = load_grid(sproblem)
const tvm = TVMetric(; grid=gd.xs, log_ws=gd.log_ws, true_logvals=gd.true_logvals)

const src_dir = "data-bosip-norm/" * get_name(sproblem)
const data_file = src_dir * "/nongp_" * string(run_idx) * "_data.jld2"
X_full, Y_full = JLD2.load(data_file, "data")

@assert n <= size(X_full, 2)

model_fitter = OptimizationMAP(; algorithm=NEWUOA(), multistart=24, parallel=parallel(), rhoend=1e-4)

data = ExperimentData(X_full[:, 1:n], Y_full[:, 1:n])
model = NonstationaryGP(;
    mean = prior_mean(sproblem),
    lengthscale_model = BOSS.default_lengthscale_model(domain(sproblem).bounds, y_dim(sproblem)),
    amplitude_model = get_amplitude_priors(sproblem),
    noise_std_model = get_noise_std_priors(sproblem),
)
bosip = construct_bosip_problem(; problem=sproblem, data, acquisition=LogMaxVar(), model)
estimate_parameters!(bosip.problem, model_fitter)

tv_approx = try
    calculate_metric(tvm, ref, log_approx_posterior(bosip))
catch e
    @warn "approx_posterior threw" e
    NaN
end

tv_mean = try
    calculate_metric(tvm, ref, log_posterior_mean(bosip))
catch e
    @warn "posterior_mean threw" e
    NaN
end

@info "n=$n  TV_approx=$tv_approx  TV_mean=$tv_mean"

out_dir = "scratch_approx_vs_mean"
mkpath(out_dir)
out_file = out_dir * "/" * problem_name * "_nongp_" * string(run_idx) * "_n" * string(n) * ".jld2"
jldsave(out_file; n, tv_approx, tv_mean)
@info "Saved to $out_file"
exit()

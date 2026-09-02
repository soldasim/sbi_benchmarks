# Recompute TV-metric scores for `nongp` runs using `log_posterior_mean_safe`
# (src/safe_posterior_estimator.jl) instead of the original `log_posterior_mean`.
#
# Background: `_iters.jld2` (per-iteration BosipProblem snapshots) is NOT saved by this
# project (see `SaveCallback`'s `@warn "NOT SAVING _iters.jld2 DATA"`), so `calculate_score.jl`'s
# usual iters-file-driven recompute path is unavailable here. Instead, this script rebuilds
# each iteration's dataset by slicing the recorded `_data.jld2` (X, Y) trajectory — which IS
# saved in full, in acquisition order — and re-fits the NonstationaryGP model from scratch on
# each growing prefix, mirroring what the live BO loop did originally at that iteration.
#
# Writes to a NEW file (`..._TVmetric_safe.jld2`), never touching the original `_TVmetric.jld2`,
# so the old (NaN-containing) scores are preserved for comparison.

include(pwd() * "/src/main_scripts/main_nongp.jl")

# `calculate_score` itself (not `calculate_scores`, which drives off an `_iters.jld2` file we
# don't have) is generic and small — copied here directly from src/calculate_score.jl rather than
# `include`ing that file, since it does its own `include("main.jl")` (the plain-GP variant, which
# would conflict with this script's `main_nongp.jl` include of the same `main`/`main_continue`
# function names).
function calculate_score(cb::MetricCallback, p::BosipProblem)
    return calculate_score(cb.metric, cb.reference, cb.logpost_estimator, cb.sampler, p)
end

function calculate_score(metric::SampleMetric, ref, logpost_est, sampler::DistributionSampler, p::BosipProblem)
    if ref isa Function
        true_samples = sample_posterior_pure(sampler, ref, p.problem.domain, sample_count)
    else
        true_samples = ref
    end
    approx_samples = sample_posterior_pure(sampler, logpost_est, p.problem.domain, sample_count)
    score = calculate_metric(metric, true_samples, approx_samples)
    return score
end
function calculate_score(metric::PDFMetric, ref, logpost_est, sampler::DistributionSampler, p::BosipProblem)
    @assert ref isa Function
    true_logpdf = ref
    approx_logpdf = logpost_est(p)
    score = calculate_metric(metric, true_logpdf, approx_logpdf)
    return score
end

function recompute_nongp_tv_safe(problem::AbstractProblem, run_name::String, run_idx::Int; init_data_count::Int=3)
    dir = data_dir(problem)
    base = base_filename(problem, run_name, run_idx)

    # config template (model priors, acquisition, likelihood, domain, x_prior) from the final saved state
    bosip_final = load(joinpath(dir, "$(base)_problem.jld2"))["problem"]
    @assert bosip_final isa BosipProblem

    # full recorded (X, Y) trajectory, in acquisition order
    X, Y = load(joinpath(dir, "$(base)_data.jld2"))["data"]
    n_total = size(X, 2)

    # existing (possibly NaN-containing) scores — recompute the same number of entries
    old_scores = load(joinpath(dir, "$(base)_TVmetric.jld2"))["score"]
    n = length(old_scores)

    ### metric / sampler / estimator setup (mirrors src/calculate_score.jl) ###
    metricT = TVMetric
    metric = get_metric(metricT, problem)
    sample_count = 2 * 10^x_dim(problem)
    sampler = AMISSampler(;
        iters = 10,
        proposal_fitter = BOSIP.AnalyticalFitter(),
        gauss_mix_options = GaussMixOptions(;
            algorithm = BOBYQA(),
            multistart = 24,
            parallel = parallel(),
            cluster_ϵs = nothing,
            rel_min_weight = 1e-8,
            rhoend = 1e-4,
        ),
    )
    cb = MetricCallback(;
        reference = reference(problem),
        logpost_estimator = log_posterior_mean_safe,
        sampler,
        sample_count,
        metric,
    )

    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 24,
        parallel = parallel(),
        rhoend = 1e-4,
    )
    boss_options = BossOptions(; info=false)

    new_scores = fill(NaN, n)

    @showprogress "Recomputing $(base)..." for i in 1:n
        data_count = init_data_count + (i - 1)
        if data_count > n_total
            @warn "Iteration $i needs $data_count datapoints but only $n_total are recorded — leaving NaN."
            continue
        end

        Xi = X[:, 1:data_count]
        Yi = Y[:, 1:data_count]
        data_i = BOSS.ExperimentData(Xi, Yi)

        bosip_i = construct_bosip_problem(;
            problem,
            data = data_i,
            acquisition = bosip_final.problem.acquisition.acq,
            model = bosip_final.problem.model,
        )

        try
            BOSS.estimate_parameters!(bosip_i.problem, model_fitter; options=boss_options)
            new_scores[i] = calculate_score(cb, bosip_i)
        catch e
            @warn "Recompute failed at iteration $i (data_count=$data_count): $e"
        end

        # flush partial progress every iteration so a timed-out job still leaves usable data
        out_file = joinpath(dir, "$(base)_TVmetric_safe.jld2")
        @save out_file score=new_scores metric=cb
    end

    return new_scores
end

# --- CLI entry point ---
if abspath(PROGRAM_FILE) == @__FILE__
    problem_name = ARGS[1]
    run_name = ARGS[2]
    run_idx = parse(Int, ARGS[3])

    problem = reconstruct_problem(problem_name)
    recompute_nongp_tv_safe(problem, run_name, run_idx)

    exit()
end

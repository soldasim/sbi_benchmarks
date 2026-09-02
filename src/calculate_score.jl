include("main.jl")

# Backward-compat: old _iters.jld2 files were saved with structs that have since changed.
# These convert methods let JLD2 reconstruct those old structs when loading the files.

# NonstationaryGP uses `act_func = x -> λ_d * x + λ_lb` (linear lengthscale mapping).
# This closure is stored as a named anonymous type that no longer exists in the current
# workspace. Reconstruct it from the saved fields.
Base.convert(::Type{Function}, x::JLD2.ReconstructedStatic{Symbol("#190#193{Float64,Float64}")}) =
    let λ_d = x.λ_d, λ_lb = x.λ_lb; v -> λ_d * v + λ_lb end

# TVMetric gained true_logvals field after old runs were saved.
Base.convert(::Type{T}, x::JLD2.ReconstructedMutable{:TVMetric}) where {T<:DistributionMetric} =
    TVMetric(x.grid, x.log_ws, nothing)

# BosipOptions.parallel_evals changed from Symbol to Bool.
Base.convert(::Type{T}, x::JLD2.ReconstructedMutable{Symbol("BosipOptions{BOSIP.CombinedCallback}")}) where {T<:BosipOptions} =
    BosipOptions(; info=x.info, debug=x.debug, parallel_evals=false, callback=x.callback)

# CustomLikelihood gained δ_dim field. Use log_ψ as loaded (it's a named function, not anonymous)
# and default δ_dim=1 (correct for ProxySIR's 1D proxy output).
function Base.convert(::Type{<:Likelihood}, x::JLD2.ReconstructedMutable{:CustomLikelihood})
    return CustomLikelihood(; log_ψ=x.log_ψ, δ_dim=1, mc_samples=x.mc_samples)
end

# BossProblem type parameter may be an anonymous function that no longer exists in the
# current workspace, causing JLD2 to fall back to ReconstructedMutable. Bypass the inner
# constructor (which runs assertions) by setting fields directly.
# calculate_score never calls bosip.problem.f, so a missing f is fine.
function Base.convert(::Type{<:BossProblem}, x::JLD2.ReconstructedMutable)
    obj = ccall(:jl_new_struct_uninit, Any, (Any,), BossProblem{Missing})
    Base.setfield!(obj, :f,           missing)
    Base.setfield!(obj, :domain,      x.domain)
    Base.setfield!(obj, :y_max,       x.y_max)
    Base.setfield!(obj, :acquisition, x.acquisition)
    Base.setfield!(obj, :model,       x.model)
    Base.setfield!(obj, :params,      x.params)
    Base.setfield!(obj, :data,        x.data)
    Base.setfield!(obj, :consistent,  x.consistent)
    return obj
end

function calculate_scores(
    problem::AbstractProblem,
    estimator::Function,
    metricT::Type{<:DistributionMetric},
    run_name::String,
    run_idx::Int;
    save_score = false,
    score_suffix::String = "",
    model_override = nothing,
)
    dir = data_dir(problem)
    metric = get_metric(metricT, problem)

    # TODO
    sample_count = 2 * 10^x_dim(problem)

    # TODO
    sampler = AMISSampler(;
        iters = 10,
        proposal_fitter = BOSIP.AnalyticalFitter(), # re-fit the proposal analytically
        # proposal_fitter = OptimizationFitter(;      # re-fit the proposal by MAP optimization
        #     algorithm = NEWUOA(),
        #     multistart = 6,
        #     parallel = parallel(),
        #     static_schedule = true, # issues with PRIMA.jl
        #     rhoend = 1e-2,
        # ),
        # gauss_mix_options = nothing,                # use Laplace approximation for the 0th iteration
        gauss_mix_options = GaussMixOptions(;       # use Gaussian mixture for the 0th iteration
            algorithm = BOBYQA(),
            multistart = 24,
            parallel = parallel(),
            cluster_ϵs = nothing,
            rel_min_weight = 1e-8,
            rhoend = 1e-4,
        ),
    )

    # enclose everything in a MetricCallback (to save it compactly later)
    cb = MetricCallback(;
        reference = reference(problem),
        logpost_estimator = estimator,
        sampler,
        sample_count,
        metric,
    )

    # load the stored `BosipProblem`s at each iteration of the run
    iters_file = joinpath(dir, "$(run_name)_$(run_idx)_iters.jld2")
    data = load(iters_file)
    @assert haskey(data, "problems")
    bosip_states = data["problems"]

    # Replace possibly-stale likelihoods with a fresh instance from the current problem
    # definition. Old iters files may have missing fields or unreconstructable closures.
    try
        fresh_like = likelihood(problem)
        for state in bosip_states
            state.likelihood = fresh_like
        end
    catch
    end

    # Optionally replace the model with a fresh instance (e.g. for nongp runs where the
    # kernel contains an anonymous closure that JLD2 can't reconstruct). The fitted params
    # remain in state.problem.params and are used by BOSS independently of the model struct.
    if !isnothing(model_override)
        for state in bosip_states
            state.problem.model = model_override
        end
    end

    # calculate the metric scores
    scores = fill(NaN, length(bosip_states))
    @showprogress "Calculating scores..." for (i, p) in enumerate(bosip_states)
        try
            scores[i] = calculate_score(cb, p)
        catch e
            @warn "Score calculation failed at iteration $i: $e"
        end
    end

    # save the scores
    if save_score
        score_file = joinpath(dir, "$(run_name)_$(run_idx)_$(metric_fname(metricT))$(score_suffix).jld2")
        @save score_file score=scores metric=cb
    end

    return scores
end

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
    ### retrieve the true and approx logpdf
    @assert ref isa Function
    true_logpdf = ref
    approx_logpdf = logpost_est(p)

    ### calculate metric
    score = calculate_metric(metric, true_logpdf, approx_logpdf)
    return score
end

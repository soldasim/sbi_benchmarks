
@kwdef mutable struct MetricCallback <: BolfiCallback
    reference::Any #::Union{Function, Matrix{Float64}} true likelihood or reference samples
    metric::DistributionMetric
    sampler::DistributionSampler
    sample_count::Int
    score_history::Vector{Float64} = Float64[]
    true_samples::Union{Nothing, Matrix{Float64}} = nothing
    approx_samples::Union{Nothing, Matrix{Float64}} = nothing
    plot_callback::Union{Nothing, BolfiCallback} = nothing
end

function (cb::MetricCallback)(problem::BolfiProblem; kwargs...)
    prior = problem.x_prior
    domain = problem.problem.domain

    ### approximate likelihood
    # TODO
    # approx_like = likelihood_mean(problem)
    approx_like = approx_likelihood(problem)

    ### sample posterior
    # TODO
    if cb.reference isa Function
        true_samples, ws = sample_posterior(cb.sampler, cb.reference, prior, domain, 20 * cb.sample_count)
        true_samples = resample(true_samples, ws, cb.sample_count)
    else
        true_samples = cb.reference
    end
    approx_samples, ws = sample_posterior(cb.sampler, approx_like, prior, domain, 20 * cb.sample_count)
    approx_samples = resample(approx_samples, ws, cb.sample_count)
    
    cb.true_samples = true_samples
    cb.approx_samples = approx_samples

    ### calculate metric
    score = calculate_metric(cb.metric, true_samples, approx_samples)
    
    @show score
    push!(cb.score_history, score)

    ### plot callback
    isnothing(cb.plot_callback) || cb.plot_callback(problem, cb; kwargs...)

    nothing
end

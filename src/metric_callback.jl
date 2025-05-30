
@kwdef mutable struct MetricCallback <: BolfiCallback
    reference::Union{Function, Matrix{Float64}} # true likelihood or reference samples
    metric::DistributionMetric
    sampler::DistributionSampler
    sample_count::Int
    score_history::Vector{Float64} = Float64[]
    true_samples::Union{Nothing, Matrix{Float64}} = nothing
    approx_samples::Union{Nothing, Matrix{Float64}} = nothing
end

function (cb::MetricCallback)(problem::BolfiProblem; kwargs...)
    prior = problem.x_prior
    domain = problem.problem.domain

    ### approximate likelihood
    approx_like = likelihood_mean(problem)

    ### sample posterior
    if cb.reference isa Function
        true_samples = sample_posterior(cb.sampler, cb.reference, prior, domain, cb.sample_count)
    else
        true_samples = cb.reference
    end
    approx_samples = sample_posterior(cb.sampler, approx_like, prior, domain, cb.sample_count)
    
    cb.true_samples = true_samples
    cb.approx_samples = approx_samples

    ### calculate metric
    score = calculate_metric(cb.metric, true_samples, approx_samples)
    
    @show score
    push!(cb.score_history, score)
end

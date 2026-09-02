## Universal priors defined based on the problem domain and ampltiude & noise std estimates.

function get_lengthscale_priors(problem::AbstractProblem)
    bounds = domain(problem).bounds
    ydim = y_dim(problem)
    return get_lengthscale_priors(bounds, ydim)
end
function get_lengthscale_priors(bounds::AbstractBounds, ydim::Int)
    d = (bounds[2] .- bounds[1])

    min_λs = d ./ 20
    max_λs = d

    μs = (log.(min_λs) .+ log.(max_λs)) ./ 2
    σs = (log.(max_λs) .- log.(min_λs)) ./ 2 # bounds are within 2 stds
    dists = LogNormal.(μs, σs)

    dists = map((d, max_λ) -> truncated(d; upper=max_λ), dists, max_λs)

    return fill(product_distribution(dists), ydim)
end

function get_amplitude_priors(problem::AbstractProblem)
    est_α = est_amplitude(problem)
    return get_amplitude_priors(est_α)
end
function get_amplitude_priors(est_α::AbstractVector{<:Real})
    d = TDist(2)
    d = truncated(d; lower=0.)
    dists = transformed.(Ref(d), Bijectors.Scale.(est_α))
    return dists
end

function get_noise_std_priors(problem::AbstractProblem; noise=nothing)
    est_σ = est_noise_std(problem)

    # we know that the simulator is noiseless
    if isnothing(est_σ)
        if isnothing(noise)
            return fill(Dirac(0.), y_dim(problem))
        else
            @warn "Using increased sim. noise for improved numerical stability."
            @assert noise isa Real
            return fill(Dirac(noise), y_dim(problem))
        end
    end

    d = TDist(2)
    d = truncated(d; lower=0.)
    dists = transformed.(Ref(d), Bijectors.Scale.(est_σ))

    return dists
end

function get_grad_noise_std_priors(problem::AbstractProblem; noise=nothing)
    est_σ = est_grad_noise_std(problem)

    # we know that the simulator is noiseless
    if isnothing(est_σ)
        if isnothing(noise)
            return fill(Dirac(0.), y_dim(problem))
        else
            @warn "Using increased grad. sim. noise for improved numerical stability."
            @assert noise isa Real
            return fill(Dirac(noise), y_dim(problem))
        end
    end

    # d = TDist(2)
    # d = truncated(d; lower=0.)
    # dists = transformed.(Ref(d), Bijectors.Scale.(est_σ))
    # return dists

    return truncated.(Normal.(0., est_σ); lower=0.)
end

function get_output_warpings(problem::AbstractProblem)
    ydim = y_dim(problem)
    return [ComposedWarping(
        YeoJohnsonWarping(; λ_prior=Normal(1., 0.5)),
        SinhArcsinhWarping(; skewness_prior=Normal(0., 0.5), tailweight_prior=LogNormal(0., 0.5)),
    ) for _ in 1:ydim]
end

### MultidimProblem with fixed parameters

# function get_lengthscale_priors(problem::MultidimProblem)
#     @warn "Using fixed lengthscale priors for MultidimProblem."
#     lb, ub = domain(problem).bounds
#     diff = ub .- lb
#     return fill(product_distribution(Dirac.(diff ./ 2)), y_dim(problem))
# end
# function get_amplitude_priors(problem::MultidimProblem)
#     @warn "Using fixed amplitude priors for MultidimProblem."
#     α = est_amplitude(problem)
#     return Dirac.(α ./ 2)
# end
# function get_noise_std_priors(problem::MultidimProblem)
#     @warn "Using fixed noise std priors for MultidimProblem."
#     σ = est_noise_std(problem)
#     @assert isnothing(σ)
#     return fill(Dirac(0.), y_dim(problem))
# end

### GaussProblem with fixed parameters

# function get_lengthscale_priors(problem::GaussProblem)
#     @warn "Using fixed lengthscale priors for GaussProblem."
#     xdim = problem.x_dim
#     ydim = y_dim(problem)
#     return fill(product_distribution(fill(Dirac(1.), xdim)), ydim)
# end
# function get_amplitude_priors(problem::GaussProblem)
#     @warn "Using fixed amplitude priors for GaussProblem."
#     ydim = y_dim(problem)
#     return fill(Dirac(0.5), ydim)
# end
# function get_noise_std_priors(problem::GaussProblem)
#     @warn "Using fixed noise std priors for GaussProblem."
#     ydim = y_dim(problem)
#     return fill(Dirac(0.), ydim)
# end

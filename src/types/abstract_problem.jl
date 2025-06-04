
"""
Subtypes of `AbstractProblem` represent benchmark problems for SBI.

Each subtype of `AbstractProblem` *should* implement:

- `simulator(::AbstractProblem) -> ::Function`
- `domain(::AbstractProblem) -> ::Domain`
- `y_max(::AbstractProblem) -> ::Union{Nothing, AbstractVector{<:Real}}`: Optional, defaults to `nothing`.
- `likelihood(::AbstractProblem) -> ::Likelihood`
- `prior_mean(::AbstractProblem) -> ::AbstractVector{<:Real}`
- `x_prior(::AbstractProblem) -> ::MultivariateDistribution`
- `y_extrema(::AbstractProblem) -> ::Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}`
- `noise_std_priors(::AbstractProblem) -> ::AbstractVector{<:UnivariateDistribution}`

Each subtype of `AbstractProblem` *should* implement *at least one* of:
- `true_f(::AbstractProblem) -> ::Union{Nothing, Function}`: Defaults to `nothing`.
- `reference_samples(::AbstractProblem) -> ::Union{Nothing, Matrix{Float64}}`: Defaults to `nothing`.

The reference solution can be obtained by:
- `reference(::AbstractProblem) -> ::Union{Function, Matrix{Float64}}`

Each `AbstractProblem` additionally provides default implementations for:
- `x_dim(::AbstractProblem) -> ::Int`
- `y_dim(::AbstractProblem) -> ::Int`
"""
abstract type AbstractProblem end

"""
    simulator(::AbstractProblem) -> ::Function

Return the simulator function (i.e. the expensive simulator) of the given problem.
"""
function simulator end

"""
    domain(::AbstractProblem) -> ::Domain

Return the parameter domain of the given problem.
"""
function domain end

"""
    likelihood(::AbstractProblem) -> ::Likelihood

Return the likelihood function of the given problem.
"""
function likelihood end

"""
    prior_mean(::AbstractProblem) -> ::AbstractVector{<:Real}

Return the prior mean of the simulator outputs.
This is closely related to the `likelihood` definition.
Usually, it is reasonable to set the prior mean for the simulation outputs
such that it maximizes the likelihood. (E.g. set it to `y_obs` in case of the Gaussian likelihood.)
"""
function prior_mean end

"""
    x_prior(::AbstractProblem) -> ::MultivariateDistribution

Return the prior parameter distribution of the given problem.
"""
function x_prior end

"""
    y_extrema(::AbstractProblem) -> ::Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}

Return the (estimated) extrema of the outputs of the simulator of the given problem.

The extrema should be underconfindent to simulate the uncertainty of the user when estimating
them before running the simulator.
"""
function y_extrema end

"""
    noise_std_priors(::AbstractProblem) -> ::AbstractVector{<:UnivariateDistribution}

Return the priors for the noise standard deviations of the outputs of the simulator of the given problem.
"""
function noise_std_priors end

"""
    true_f(::AbstractProblem) -> ::Union{Nothing, Function}

Return the true noise-less simulator function or `nothing` if the true function cannot be evaluated
or is too expensive to compute.

Either `true_f` or `reference_samples` should be defined for each problem.
"""
true_f(::AbstractProblem) = nothing

"""
    reference_samples(::AbstractProblem) -> ::Union{Nothing, Matrix{Float64}}

Return reference samples from the true parameter posterior for the given problem
or `nothing` if the reference samples are not provided.

Either `reference_samples` or `true_f` should be defined for each problem.
"""
reference_samples(::AbstractProblem) = nothing


"""
    y_max(::AbstractProblem) -> ::Union{Nothing, AbstractVector{<:Real}}

Return the constraint values for the outputs of the given problem.
"""
y_max(::AbstractProblem) = nothing

"""
    reference(::AbstractProblem) -> ::Union{Function, Matrix{Float64}}

Returns either the `true_likelihood` or the `reference_samples`
depending on which of the `true_f` and `reference_samples` function have beed defined for the problem.
"""
function reference(problem::AbstractProblem)
    true_like = true_likelihood(problem)
    isnothing(true_like) || return true_like
    ref_samples = reference_samples(problem)
    isnothing(ref_samples) && error("Define `f_true` or `reference_samples` for the problem.")
    return ref_samples
end

"""
    true_likelihood(::AbstractProblem) -> ::Union{Nothing, Function}

Return the true likelihood function of the given problem
or `nothing` if the problem does not have the `true_f` function defined.

The true likelihood is used to evaluate performance metrics.
"""
function true_likelihood(problem::AbstractProblem)
    like = likelihood(problem)
    f = true_f(problem)

    isnothing(f) && return nothing

    function true_like(x)
        y = f(x)
        ll = loglike(like, y)
        return exp(ll)
    end
end

x_dim(problem::AbstractProblem) = length(domain(problem).bounds[1])
y_dim(problem::AbstractProblem) = length(y_extrema(problem)[1])

using Distributions
using Random

"""
    CombinedDistribution(base_dist::ContinuousMultivariateDistribution, scaleup::Int)

A combined distribution that samples vectors by replicating samples from a base distribution.
When sampled, returns a vector of length `scaleup * d` where `d` is the dimension of the base distribution.
Each `d`-dimensional chunk is an independent sample from `base_dist`.
"""
struct CombinedDistribution{T<:ContinuousMultivariateDistribution} <: ContinuousMultivariateDistribution
    base_dist::T
    scaleup::Int
    
    function CombinedDistribution(base_dist::T, scaleup::Int) where {T<:ContinuousMultivariateDistribution}
        @assert scaleup > 0
        new{T}(base_dist, scaleup)
    end
end

Base.length(d::CombinedDistribution) = length(d.base_dist) * d.scaleup

Distributions.mean(d::CombinedDistribution) = repeat(mean(d.base_dist), d.scaleup)

Distributions.var(d::CombinedDistribution) = repeat(var(d.base_dist), d.scaleup)

Distributions.cov(d::CombinedDistribution) = repeat(cov(d.base_dist), d.scaleup, d.scaleup)

function Distributions.logpdf(d::CombinedDistribution, x::AbstractVector{<:Real})
    @assert length(x) == length(d)
    d_base = length(d.base_dist)
    logp = 0.0
    for i in 1:d.scaleup
        x_i = @view x[((i-1)*d_base + 1):(i*d_base)]
        logp += logpdf(d.base_dist, x_i)
    end
    return logp
end

function Distributions.logpdf(d::CombinedDistribution, X::AbstractMatrix{<:Real})
    @assert size(X, 1) == length(d)
    return vec(mapslices(x -> logpdf(d, x), X; dims=1))
end

function Base.rand(rng::Random.AbstractRNG, d::CombinedDistribution)
    samples = []
    for i in 1:d.scaleup
        push!(samples, rand(rng, d.base_dist))
    end
    return vcat(samples...)
end

function Base.rand(rng::Random.AbstractRNG, d::CombinedDistribution, n::Int)
    samples = zeros(length(d), n)
    d_base = length(d.base_dist)
    for col in 1:n
        for i in 1:d.scaleup
            x_i = rand(rng, d.base_dist)
            idx_range = ((i-1)*d_base + 1):(i*d_base)
            samples[idx_range, col] .= x_i
        end
    end
    return samples
end

# Support for displaying the distribution
Base.show(io::IO, d::CombinedDistribution) = print(io, "CombinedDistribution($(d.base_dist), scaleup=$(d.scaleup))")

# Support for truncated distributions
function Distributions.truncated(d::CombinedDistribution; lower, upper)
    @assert length(lower) == length(d) && length(upper) == length(d)
    d_base = length(d.base_dist)
    
    # Extract bounds for the base distribution
    lower_base = lower[1:d_base]
    upper_base = upper[1:d_base]
    
    # Verify all chunks have the same bounds
    for i in 2:d.scaleup
        idx_range = ((i-1)*d_base + 1):(i*d_base)
        @assert lower[idx_range] ≈ lower_base && upper[idx_range] ≈ upper_base
    end
    
    # Truncate base distribution and wrap it again
    trunc_base = truncated(d.base_dist; lower=lower_base, upper=upper_base)
    return CombinedDistribution(trunc_base, d.scaleup)
end

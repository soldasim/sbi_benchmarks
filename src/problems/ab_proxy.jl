
"""
    ProxyABProblem()

The analytical toy problem of inferring the parameters `a`, `b`
given the observation `z_obs = [1.]`.

The blackbox simulator realizes the function `y = a * b`.
See also the `LogProxyABProblem` for a version of the problem,
where only the log-likelihood is returned by the simulator.

The likelihood is Gaussian.
"""
struct ProxyABProblem <: AbstractProblem end


module ProxyABProblemModule

import ..ProxyABProblem

import ..simulator
import ..domain
import ..y_max
import ..likelihood
import ..prior_mean
import ..x_prior
import ..est_amplitude
import ..est_noise_std
import ..true_f
import ..reference_samples

using BOSS
using BOSIP
using Distributions


# --- API ---

simulator(::ProxyABProblem) = model_target

domain(::ProxyABProblem) = Domain(;
    bounds = _get_bounds(),
)

# TODO proxy
# model target -> loglike
function log_ψ(δ::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    # y = δ[1]
    # y = sqrt(abs(x[1])) * sqrt(abs(x[2])) * δ[1]
    y = sign(δ[1]) * δ[1]^2
    # y = δ[1] * (abs(x[1]) * abs(x[2]))
    # y = (x[1] * x[2]) * δ[1]
    return logpdf(Normal(y, std_obs[1]), z_obs[1])
end

# TODO proxy
# likelihood(::ProxyABProblem) = NormalLikelihood(; z_obs, std_obs)
likelihood(::ProxyABProblem) = CustomLikelihood(;
    log_ψ,
    mc_samples = 1000, # TODO
)

# TODO proxy
# prior_mean(::ProxyABProblem) = z_obs
prior_mean(::ProxyABProblem) = sqrt.(z_obs)
# prior_mean(::ProxyABProblem) = sign.(z_obs)
# prior_mean(::ProxyABProblem) = [0.]
# prior_mean(::ProxyABProblem) = [1.]
# prior_mean(::ProxyABProblem) = x -> [z_obs[1] / (sqrt(abs(x[1])) * sqrt(abs(x[2])))]

x_prior(::ProxyABProblem) = _get_trunc_x_prior()

# TODO proxy
# est_amplitude(::ProxyABProblem) = [20.]
est_amplitude(::ProxyABProblem) = [sqrt(20.)]
# est_amplitude(::ProxyABProblem) = [1.]

# TODO noise
est_noise_std(::ProxyABProblem) = nothing

true_f(::ProxyABProblem) = x -> model_target(x; noise_std=zero(std_sim))


# --- UTILS ---

const z_obs = [1.]
const std_obs = [0.2]

# TODO noise
# (not using noise in order to compare with loglike modeling more fairly)
const std_sim = [0.]
# const std_sim = [0.1]

# TODO proxy
# function model_target(x)
#     y = ab_simulation(x)
#     return [y]
# end
function model_target(x; noise_std=std_sim)
    y = ab_simulation(x; noise_std)
    
    # δ = y
    
    # δ = y / (sqrt(abs(x[1])) * sqrt(abs(x[2])))
    # δ = sign(y) * sqrt(abs(x[1])) * sqrt(abs(x[2]))
    δ = sign(y) * sqrt(abs(y))
    
    # δ = y / (abs(x[1]) * abs(x[2]))
    # δ = sign(y)

    # # δ = y / (x[1] * x[2])
    # δ = 1.

    return [δ]
end

function ab_simulation(x; noise_std=std_sim)
    y = x[1] * x[2]
    y += rand(Normal(0., noise_std[1]))
    return y
end

_get_bounds() = ([-5., -5.], [5., 5.])

function _get_trunc_x_prior()
    prior = _get_x_prior()
    bounds = _get_bounds()
    return truncated(prior; lower=bounds[1], upper=bounds[2])
end
_get_x_prior() = Product(fill(Normal(0., 5/3), 2))

end # module ProxyABProblemModule

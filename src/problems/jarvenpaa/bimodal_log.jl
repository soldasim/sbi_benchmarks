
"""
    LogBimodalProblem()

The "bimodal" problem from Jarvenpaa & Gutmann's "Parallel..." paper.

This is the original version of the problem as introduced in the paper.
See also the `LogBimodalProblem` for a slightly altered version.
"""
struct LogBimodalProblem <: AbstractProblem end


module LogBimodalProblemModule

import ..LogBimodalProblem

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
using BOLFI
using Distributions
using Bijectors

# --- API ---

simulator(::LogBimodalProblem) = simulation

domain(::LogBimodalProblem) = Domain(;
    bounds = get_bounds(),
)

likelihood(::LogBimodalProblem) = get_likelihood()

# TODO loglike
prior_mean(::LogBimodalProblem) = [0.]

x_prior(::LogBimodalProblem) = get_x_prior()

# TODO loglike
est_amplitude(::LogBimodalProblem) = [1000.] # TODO ???

# TODO noise
est_noise_std(::LogBimodalProblem) = nothing

true_f(::LogBimodalProblem) = simulation


# - - - PARAMETER DOMAIN - - - - -

x_dim() = 2
get_bounds() = (fill(-6., x_dim()), fill(6., x_dim()))


# - - - EXPERIMENT - - - - -

const ρ = 0.5
const Σ = [1.; ρ;; ρ; 1.;;]
const inv_S = inv(Σ)

function f_(x)
    θ = [x[1], x[2]^2 - 2]
    return -(1/2) * θ' * inv_S * θ
end

# TODO loglike
function simulation(x)
    return [f_(x)]
end

# TODO loglike
get_likelihood() = ExpLikelihood()

# truncate the prior to the bounds
function get_x_prior()
    return Product(Uniform.(get_bounds()...))
end

end # module LogBimodalProblemModule

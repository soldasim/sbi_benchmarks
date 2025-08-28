
"""
    LogBananaProblem()

The "banana" problem from Jarvenpaa & Gutmann's "Parallel..." paper.

This is the original version of the problem as introduced in the paper.
See also the `LogBananaProblem` for a slightly altered version.
"""
struct LogBananaProblem <: AbstractProblem end


module LogBananaProblemModule

import ..LogBananaProblem

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

simulator(::LogBananaProblem) = simulation

domain(::LogBananaProblem) = Domain(;
    bounds = get_bounds(),
)

likelihood(::LogBananaProblem) = get_likelihood()

# TODO loglike
prior_mean(::LogBananaProblem) = [0.]

x_prior(::LogBananaProblem) = get_x_prior()

# TODO loglike
est_amplitude(::LogBananaProblem) = [1000.] # TODO ???

# TODO noise
est_noise_std(::LogBananaProblem) = nothing

true_f(::LogBananaProblem) = simulation


# - - - PARAMETER DOMAIN - - - - -

x_dim() = 2
get_bounds() = ([-6., -20.], [6., 2.])


# - - - EXPERIMENT - - - - -

const ρ = 0.9
const Σ = [1.; ρ;; ρ; 1.;;]
const inv_S = inv(Σ)

function f_(x)
    θ = [x[1], x[2] + x[1]^2 + 1.]
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

end # module LogBananaProblemModule

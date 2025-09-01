
"""
    LogSimpleProblem()

The "simple" problem from Jarvenpaa & Gutmann's "Parallel..." paper.

This is the original version of the problem as introduced in the paper.
See also the `LogSimpleProblem` for a slightly altered version.
"""
struct LogSimpleProblem <: AbstractProblem end


module LogSimpleProblemModule

import ..LogSimpleProblem

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
using Bijectors

# --- API ---

simulator(::LogSimpleProblem) = simulation

domain(::LogSimpleProblem) = Domain(;
    bounds = get_bounds(),
)

likelihood(::LogSimpleProblem) = get_likelihood()

# TODO loglike
prior_mean(::LogSimpleProblem) = [0.]

x_prior(::LogSimpleProblem) = get_x_prior()

# TODO loglike
est_amplitude(::LogSimpleProblem) = [1000.] # TODO ???

# TODO noise
est_noise_std(::LogSimpleProblem) = nothing

true_f(::LogSimpleProblem) = simulation


# - - - PARAMETER DOMAIN - - - - -

x_dim() = 2
get_bounds() = (fill(-16., x_dim()), fill(16., x_dim()))


# - - - EXPERIMENT - - - - -

const ρ = 0.25
const Σ = [1.; ρ;; ρ; 1.;;]
const inv_S = inv(Σ)

f_(x) = -(1/2) * x' * inv_S * x

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

end # module LogSimpleProblemModule

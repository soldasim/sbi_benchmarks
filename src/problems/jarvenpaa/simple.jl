
"""
    SimpleProblem()

The "simple" problem from Jarvenpaa & Gutmann's "Parallel..." paper.

In contrast to the problem as defined in the paper, here the input vector `x`
is returned as the simulator output (i.e. the simulator is just the identity function).
See the `LogSimpleProblem` for the original version of the problem.
"""
struct SimpleProblem <: AbstractProblem end


module SimpleProblemModule

import ..SimpleProblem

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

simulator(::SimpleProblem) = simulation

domain(::SimpleProblem) = Domain(;
    bounds = get_bounds(),
)

likelihood(::SimpleProblem) = get_likelihood()

prior_mean(::SimpleProblem) = [0., 0.]

x_prior(::SimpleProblem) = get_x_prior()

est_amplitude(::SimpleProblem) = fill(20., 2)

# TODO noise
est_noise_std(::SimpleProblem) = nothing

true_f(::SimpleProblem) = simulation


# - - - PARAMETER DOMAIN - - - - -

x_dim() = 2
get_bounds() = (fill(-16., x_dim()), fill(16., x_dim()))


# - - - EXPERIMENT - - - - -

const ρ = 0.25
const Σ = [1.; ρ;; ρ; 1.;;]
const inv_S = inv(Σ)

# f_(x) = -(1/2) * x' * inv_S * x

function simulation(x)
    return x
end

get_likelihood() = MvNormalLikelihood(;
    z_obs = [0., 0.],
    Σ_obs = Σ,
)

# truncate the prior to the bounds
function get_x_prior()
    return Product(Uniform.(get_bounds()...))
end

end # module SimpleProblemModule

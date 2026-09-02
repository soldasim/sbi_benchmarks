
"""
    Simple2Problem()

The "simple" problem from Jarvenpaa & Gutmann's "Parallel..." paper.

In contrast to the problem as defined in the paper, here the input vector `x`
is returned as the simulator output (i.e. the simulator is just the identity function).
See the `LogSimple2Problem` for the original version of the problem.
"""
struct Simple2Problem <: AbstractProblem end


module Simple2ProblemModule

import ..Simple2Problem

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

simulator(::Simple2Problem) = simulation

domain(::Simple2Problem) = Domain(;
    bounds = get_bounds(),
)

likelihood(::Simple2Problem) = get_likelihood()

prior_mean(::Simple2Problem) = [0., 0.]

x_prior(::Simple2Problem) = get_x_prior()

est_amplitude(::Simple2Problem) = fill(1., 2)

# TODO noise
est_noise_std(::Simple2Problem) = nothing

true_f(::Simple2Problem) = simulation


# - - - PARAMETER DOMAIN - - - - -

x_dim() = 2
get_bounds() = (fill(-1., x_dim()), fill(1., x_dim()))


# - - - EXPERIMENT - - - - -

const σ = 0.020
const ρ = 0.005
const Σ = [σ; ρ;; ρ; σ;;]
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

end # module Simple2ProblemModule

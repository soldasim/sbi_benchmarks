
"""
    SimpleProblem()

The "simple" problem from Jarvenpaa & Gutmann's "Parallel..." paper.
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
using BOLFI
using Distributions
using Bijectors

# --- API ---

simulator(::SimpleProblem) = simulation

domain(::SimpleProblem) = Domain(;
    bounds = get_bounds(),
)

likelihood(::SimpleProblem) = get_likelihood()

# TODO loglike
prior_mean(::SimpleProblem) = [0., 0.]
# prior_mean(::SimpleProblem) = [0.]

x_prior(::SimpleProblem) = get_x_prior()

# TODO loglike
est_amplitude(::SimpleProblem) = fill(20., 2)
# est_amplitude(::SimpleProblem) = [1000.] # TODO ???

# TODO noise
est_noise_std(::SimpleProblem) = nothing

# TODO loglike
true_f(::SimpleProblem) = x -> x
# true_f(::SimpleProblem) = x -> [f_(x)]


# - - - PARAMETER DOMAIN - - - - -

x_dim() = 2
get_bounds() = (fill(-16., x_dim()), fill(16., x_dim()))


# - - - OBSERVATION - - - - -

"""observation"""
const z_obs = [0.]
const y_dim = 1

"""simulation noise std"""
# TODO noise
# (not using noise in order to compare with loglike modeling more fairly)
const ω = fill(0., y_dim)
# const ω = fill(1., y_dim)


# - - - EXPERIMENT - - - - -

const ρ = 0.25
const Σ = [1.; ρ;; ρ; 1.;;]
const inv_S = inv(Σ)
f_(x) = -(1/2) * x' * inv_S * x

# TODO loglike
function simulation(x; noise_std=ω)
    y = x
    return y
end
# function simulation(x; noise_std=ω)
#     y1 = f_(x) + rand(Normal(0., noise_std[1]))
#     return [y1]
# end

# The objective for the GP.
obj(x) = simulation(x)

# TODO loglike
get_likelihood() = MvNormalLikelihood(;
    z_obs = [0., 0.],
    Σ_obs = Σ,
)
# get_likelihood() = ExpLikelihood()

# truncate the prior to the bounds
function get_x_prior()
    return Product(Uniform.(get_bounds()...))
end


# - - - HYPERPARAMETERS - - - - -
# THIS SECTION IS IGNORED UNLESS THE `SimpleProblemModule.get_model()` FUNCTION IS CALLED

get_kernel() = BOSS.Matern32Kernel() # TODO ???

function get_lengthscale_priors()
    ranges = (-1.) * .-(get_bounds()...)

    d = TDist(4)
    d = truncated(d; lower=0.)
    ds = transformed.(Ref(d), Bijectors.Scale.(ranges ./ 2))

    return fill(product_distribution(ds), y_dim)
end

function get_amplitude_priors()
    d = TDist(4)
    d = truncated(d; lower=0.)
    d = transformed(d, Bijectors.Scale(1000.))
    
    return fill(d, y_dim)
end

function get_noise_std_priors()
    d = TDist(4)
    d = truncated(d; lower=0.)
    d = transformed(d, Bijectors.Scale(50.))

    return fill(d, y_dim)
end

get_model() = GaussianProcess(;
    kernel = get_kernel(),
    lengthscale_priors = get_lengthscale_priors(),
    amplitude_priors = get_amplitude_priors(),
    noise_std_priors = get_noise_std_priors(),
)


end # module SimpleProblemModule

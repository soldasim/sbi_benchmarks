
"""
    ABProblem()

The analytical toy problem of inferring the parameters `a`, `b`
given the observation `z_obs = [1.]`.

The blackbox simulator realizes the function `y = a * b`.

The likelihood is Gaussian.
"""
struct ABProblem <: AbstractProblem end


module ABProblemModule

import ..ABProblem

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


# --- API ---

simulator(::ABProblem) = ab_simulation

domain(::ABProblem) = Domain(;
    bounds = _get_bounds(),
)

# TODO loglike
likelihood(::ABProblem) = NormalLikelihood(; z_obs, std_obs)
# likelihood(::ABProblem) = ExpLikelihood()

# TODO loglike
prior_mean(::ABProblem) = z_obs
# prior_mean(::ABProblem) = [0.]

x_prior(::ABProblem) = _get_trunc_x_prior()

# TODO loglike
est_amplitude(::ABProblem) = [20.]
# est_amplitude(::ABProblem) = [1000.] # TODO ???

# TODO noise
est_noise_std(::ABProblem) = nothing

true_f(::ABProblem) = x -> ab_simulation(x; noise_std=zero(std_sim))


# --- UTILS ---

const z_obs = [1.]
const std_obs = [0.2]

# TODO noise
# (not using noise in order to compare with loglike modeling more fairly)
const std_sim = [0.]
# const std_sim = [0.1]

# the true blackbox function
f_(x) = [x[1] * x[2]]

# TODO loglike
function ab_simulation(x; noise_std=std_sim)
    y = f_(x)
    y .+= rand(Normal(0., noise_std[1]))
    return y
end
# function ab_simulation(x; noise_std=std_sim)
#     y = f_(x)
#     ll = logpdf(Normal(y[1], std_obs[1]), z_obs[1])
#     ll += rand(Normal(0., noise_std[1]))
#     return [ll]
# end

_get_bounds() = ([-5., -5.], [5., 5.])

function _get_trunc_x_prior()
    prior = _get_x_prior()
    bounds = _get_bounds()
    return truncated(prior; lower=bounds[1], upper=bounds[2])
end
_get_x_prior() = Product(fill(Normal(0., 5/3), 2))

end # module ABProblemModule

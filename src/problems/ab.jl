
"""
    ABProblem()

The analytical toy problem of inferring the parameters `a`, `b`
given the observation `z_obs = [1.]`.

The blackbox simulator realizes the function `y = a * b`.
See also the `LogABProblem` for a version of the problem,
where only the log-likelihood is returned by the simulator.

The likelihood is Gaussian.
"""
@kwdef struct ABProblem <: AbstractProblem
    gradients::Bool = false
end

set_gradients(p::ABProblem, val::Bool) = ABProblem(val)


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
import ..est_grad_noise_std
import ..true_f
import ..reference_samples

using BOSS
using BOSIP
using Distributions


# --- API ---

simulator(p::ABProblem) = p.gradients ? ab_simulation_with_grads : ab_simulation

domain(::ABProblem) = Domain(;
    bounds = _get_bounds(),
)

likelihood(::ABProblem) = NormalLikelihood(; z_obs, std_obs)

prior_mean(::ABProblem) = z_obs

x_prior(::ABProblem) = _get_trunc_x_prior()

est_amplitude(::ABProblem) = [20.]

# TODO noise
est_noise_std(::ABProblem) = nothing
est_grad_noise_std(::ABProblem) = nothing

true_f(::ABProblem) = ab_simulation


# --- UTILS ---

const z_obs = [1.]
const std_obs = [0.2]


# the true blackbox function
f_(x) = [x[1] * x[2]]
J_(x) = [x[2] x[1]]

function ab_simulation(x)
    y = f_(x)
    return y
end
function ab_simulation_with_grads(x)
    y = f_(x)
    J = J_(x)
    return y, J
end

_get_bounds() = ([-5., -5.], [5., 5.])

function _get_trunc_x_prior()
    prior = _get_x_prior()
    bounds = _get_bounds()
    return truncated(prior; lower=bounds[1], upper=bounds[2])
end
_get_x_prior() = Product(fill(Normal(0., 5/3), 2))

end # module ABProblemModule

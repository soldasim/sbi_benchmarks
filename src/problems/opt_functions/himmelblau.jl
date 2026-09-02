"""
    HimmelblauProblem()

A 2D problem with the Himmelblau simulator.

Simulator: y = (x² + y − 11)² + (x + y² − 7)²

Four equal global minima at (3, 2), (−2.805, 3.131), (−3.779, −3.283),
(3.584, −1.848), all with y = 0. The four modes are at qualitatively different
locations (one in each quadrant roughly), making this a strong test for
multimodal posterior inference. The response surface is smooth.

With z_obs = 5.0 and std_obs = 1.0 the posterior has 4 modes near the regions
where the function equals 5.
"""
@kwdef struct HimmelblauProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [1.0]
end

set_gradients(p::HimmelblauProblem, val::Bool) = HimmelblauProblem(val, p.std_obs)


module HimmelblauProblemModule

import ..HimmelblauProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [5.0]
const std_obs = [1.0]
const x_true  = [3.012892, 1.351034]

# --- API ---

simulator(p::HimmelblauProblem) = p.gradients ? _sim_grads : _sim
domain(::HimmelblauProblem)     = Domain(; bounds = ([-5.0, -5.0], [5.0, 5.0]))
likelihood(p::HimmelblauProblem) = NormalLikelihood(; z_obs, std_obs=p.std_obs)
prior_mean(::HimmelblauProblem) = z_obs
x_prior(::HimmelblauProblem)    = Product([Uniform(-5.0, 5.0), Uniform(-5.0, 5.0)])
est_amplitude(::HimmelblauProblem)      = [200.0]
est_noise_std(::HimmelblauProblem)      = nothing
est_grad_noise_std(::HimmelblauProblem) = nothing
true_f(::HimmelblauProblem)    = _f
true_params(::HimmelblauProblem) = x_true

# --- Simulator ---

function _f(x)
    a = x[1]^2 + x[2] - 11
    b = x[1] + x[2]^2 - 7
    return [a^2 + b^2]
end

function _J(x)
    a = x[1]^2 + x[2] - 11
    b = x[1] + x[2]^2 - 7
    J = zeros(1, 2)
    J[1, 1] = 4*x[1]*a + 2*b
    J[1, 2] = 2*a + 4*x[2]*b
    return J
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))

end # module HimmelblauProblemModule

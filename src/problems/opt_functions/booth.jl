"""
    BoothProblem()

A 2D problem with the Booth simulator.

Simulator: y = (x + 2y - 7)² + (2x + y - 5)²

Global minimum of 0 at (1, 3). The two quadratic terms define an ellipsoidal
bowl. The level sets are ellipses with principal axes not aligned to the
coordinate axes, testing anisotropic GP kernels. Simple unimodal benchmark;
useful as a sanity check.

With z_obs = 20.0 and std_obs = 4.0 the posterior is an elongated elliptical
annular distribution offset from the true minimum.
"""
@kwdef struct BoothProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [4.0]
end

set_gradients(p::BoothProblem, val::Bool) = BoothProblem(val, p.std_obs)


module BoothProblemModule

import ..BoothProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [20.0]
const std_obs = [4.0]
const x_true  = [4.162280, -0.162275]

# --- API ---

simulator(p::BoothProblem) = p.gradients ? _sim_grads : _sim
domain(::BoothProblem)     = Domain(; bounds = ([-10.0, -10.0], [10.0, 10.0]))
likelihood(p::BoothProblem) = NormalLikelihood(; z_obs, std_obs=p.std_obs)
prior_mean(::BoothProblem) = z_obs
x_prior(::BoothProblem)    = Product([Uniform(-10.0, 10.0), Uniform(-10.0, 10.0)])
est_amplitude(::BoothProblem)      = [500.0]
est_noise_std(::BoothProblem)      = nothing
est_grad_noise_std(::BoothProblem) = nothing
true_f(::BoothProblem)    = _f
true_params(::BoothProblem) = x_true

# --- Simulator ---

function _f(x)
    return [(x[1] + 2*x[2] - 7)^2 + (2*x[1] + x[2] - 5)^2]
end

function _J(x)
    a = x[1] + 2*x[2] - 7
    b = 2*x[1] + x[2] - 5
    J = zeros(1, 2)
    J[1, 1] = 2*a + 4*b
    J[1, 2] = 4*a + 2*b
    return J
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))

end # module BoothProblemModule

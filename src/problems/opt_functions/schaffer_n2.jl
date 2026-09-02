"""
    SchafferN2Problem()

A 2D problem with the Schaffer N.2 simulator.

Simulator: y = 0.5 + (sin²(x₁² − x₂²) − 0.5) / (1 + 0.001(x₁²+x₂²))²

Global minimum of 0 at the origin. The sin²(x₁²−x₂²) term creates a
checkerboard-like oscillatory pattern along the hyperbolic level curves
x₁²−x₂² = const, while the denominator damps oscillations away from
the origin. The value is bounded in [0, 1]. Non-stationary structure
from the combination of radial damping and diagonal oscillation.

With z_obs = 0.5 and std_obs = 0.05 the posterior is multimodal with
many small modes on the alternating checkerboard ridges.
"""
@kwdef struct SchafferN2Problem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [0.05]
end

set_gradients(p::SchafferN2Problem, val::Bool) = SchafferN2Problem(val, p.std_obs)


module SchafferN2ProblemModule

import ..SchafferN2Problem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [0.5]
const std_obs = [0.05]
const x_true  = [3.443052, -3.327042]

# --- API ---

simulator(p::SchafferN2Problem) = p.gradients ? _sim_grads : _sim
domain(::SchafferN2Problem)     = Domain(; bounds = ([-10.0, -10.0], [10.0, 10.0]))
likelihood(p::SchafferN2Problem) = NormalLikelihood(; z_obs, std_obs=p.std_obs)
prior_mean(::SchafferN2Problem) = z_obs
x_prior(::SchafferN2Problem)    = Product([Uniform(-10.0, 10.0), Uniform(-10.0, 10.0)])
est_amplitude(::SchafferN2Problem)      = [1.0]
est_noise_std(::SchafferN2Problem)      = nothing
est_grad_noise_std(::SchafferN2Problem) = nothing
true_f(::SchafferN2Problem)    = _f
true_params(::SchafferN2Problem) = x_true

# --- Simulator ---

function _f(x)
    u = x[1]^2 - x[2]^2
    p = 1.0 + 0.001*(x[1]^2 + x[2]^2)
    return [0.5 + (sin(u)^2 - 0.5) / p^2]
end

function _J(x)
    u  = x[1]^2 - x[2]^2
    r2 = x[1]^2 + x[2]^2
    p  = 1.0 + 0.001*r2
    q  = p^2
    su = sin(u); cu = cos(u)
    # d[(sin²u-0.5)/q]/dx₁: quotient rule, du/dx₁=2x₁, du/dx₂=-2x₂, dq/dxᵢ=2p·0.002·xᵢ
    num_val = sin(u)^2 - 0.5
    J = zeros(1, 2)
    J[1, 1] = (2*su*cu*2*x[1] * q  - num_val * 2*p * 0.002*x[1]) / q^2
    J[1, 2] = (2*su*cu*(-2*x[2]) * q - num_val * 2*p * 0.002*x[2]) / q^2
    return J
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))

end # module SchafferN2ProblemModule

"""
    DropWaveProblem()

A 2D problem with the Drop-Wave simulator.

Simulator: y = −(1 + cos(12 √(x₁²+x₂²))) / (0.5(x₁²+x₂²) + 2)

Global minimum of −1 at the origin. The function creates concentric wave-like
rings that decrease in amplitude with radius; the denominator ensures the
function rises toward 0 far from the origin. Highly multimodal with rings of
local minima, each ring slightly higher than the previous one. Good benchmark
for testing acquisition functions that must balance exploration (outer rings)
vs. exploitation (central minimum).

With z_obs = −0.7 and std_obs = 0.05 the posterior has multiple ring-shaped
modes corresponding to the concentric local minima near level −0.7.
"""
@kwdef struct DropWaveProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [0.05]
end

set_gradients(p::DropWaveProblem, val::Bool) = DropWaveProblem(val, p.std_obs)


module DropWaveProblemModule

import ..DropWaveProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [-0.7]
const std_obs = [0.05]
const x_true  = [-0.775270, 0.777145]

# --- API ---

simulator(p::DropWaveProblem) = p.gradients ? _sim_grads : _sim
domain(::DropWaveProblem)     = Domain(; bounds = ([-5.12, -5.12], [5.12, 5.12]))
likelihood(p::DropWaveProblem) = NormalLikelihood(; z_obs, std_obs=p.std_obs)
prior_mean(::DropWaveProblem) = z_obs
x_prior(::DropWaveProblem)    = Product([Uniform(-5.12, 5.12), Uniform(-5.12, 5.12)])
est_amplitude(::DropWaveProblem)      = [1.0]
est_noise_std(::DropWaveProblem)      = nothing
est_grad_noise_std(::DropWaveProblem) = nothing
true_f(::DropWaveProblem)    = _f
true_params(::DropWaveProblem) = x_true

# --- Simulator ---

function _f(x)
    r2 = x[1]^2 + x[2]^2
    r  = sqrt(r2)
    q  = 0.5 * r2 + 2.0
    return [-(1.0 + cos(12.0 * r)) / q]
end

function _J(x)
    r2 = x[1]^2 + x[2]^2
    r  = sqrt(r2)
    q  = 0.5 * r2 + 2.0
    J  = zeros(1, 2)
    if r > 1e-12
        # f = -num/q, num = 1+cos(12r), q = 0.5r²+2
        # df/dr = -(num'q - num·q')/q² = -(-12sin(12r)·q - num·r)/q²
        #       = (12sin(12r)·q + num·r) / q²
        df_dr = (12.0 * sin(12.0 * r) * q + (1.0 + cos(12.0 * r)) * r) / q^2
        J[1, 1] = df_dr * x[1] / r
        J[1, 2] = df_dr * x[2] / r
    end
    return J
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))

end # module DropWaveProblemModule

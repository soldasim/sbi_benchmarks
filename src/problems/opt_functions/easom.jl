"""
    EasomProblem()

A 2D problem with the Easom simulator.

Simulator: y = −cos(x₁) cos(x₂) exp(−(x₁−π)² − (x₂−π)²)

Global minimum of −1 at (π, π). The function is nearly flat (≈ 0) everywhere
except in a very narrow region around (π, π) where it drops sharply to −1.
The Gaussian envelope exp(−(x−π)² − (y−π)²) localizes all variation to a
small region, creating a needle-like peak surrounded by a vast flat plain.
Domain restricted to [0, 6]² to capture the interesting structure while
remaining tractable for GP-based inference.

With z_obs = −0.5 and std_obs = 0.1 the posterior concentrates in the narrow
Gaussian well around (π, π).
"""
@kwdef struct EasomProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [0.1]
end

set_gradients(p::EasomProblem, val::Bool) = EasomProblem(val, p.std_obs)


module EasomProblemModule

import ..EasomProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [-0.5]
const std_obs = [0.1]
const x_true  = [2.975515, 3.792155]

# --- API ---

simulator(p::EasomProblem) = p.gradients ? _sim_grads : _sim
domain(::EasomProblem)     = Domain(; bounds = ([0.0, 0.0], [6.0, 6.0]))
likelihood(p::EasomProblem) = NormalLikelihood(; z_obs, std_obs=p.std_obs)
prior_mean(::EasomProblem) = z_obs
x_prior(::EasomProblem)    = Product([Uniform(0.0, 6.0), Uniform(0.0, 6.0)])
est_amplitude(::EasomProblem)      = [1.0]
est_noise_std(::EasomProblem)      = nothing
est_grad_noise_std(::EasomProblem) = nothing
true_f(::EasomProblem)    = _f
true_params(::EasomProblem) = x_true

# --- Simulator ---

function _f(x)
    e = exp(-(x[1] - π)^2 - (x[2] - π)^2)
    return [-cos(x[1]) * cos(x[2]) * e]
end

function _J(x)
    e  = exp(-(x[1] - π)^2 - (x[2] - π)^2)
    c1 = cos(x[1]); s1 = sin(x[1])
    c2 = cos(x[2]); s2 = sin(x[2])
    J  = zeros(1, 2)
    # df/dx₁ = -(−sin(x₁)cos(x₂)e + cos(x₁)cos(x₂)e·(−2(x₁−π)))
    #        = e·cos(x₂)·(sin(x₁) + 2(x₁−π)cos(x₁))  ... wait let me redo
    # f = -c1*c2*e
    # df/dx₁ = -(dc1/dx₁ * c2 * e + c1 * c2 * de/dx₁)
    #        = -((-s1)*c2*e + c1*c2*e*(-2*(x[1]-π)))
    #        = e*c2*(s1 + 2*(x[1]-π)*c1)
    # df/dx₁ = e·cos(x₂)·(sin(x₁) + 2(x₁-π)cos(x₁)), similarly for x₂
    J[1, 1] = e * c2 * (s1 + 2.0*(x[1]-π)*c1)
    J[1, 2] = e * c1 * (s2 + 2.0*(x[2]-π)*c2)
    return J
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))

end # module EasomProblemModule

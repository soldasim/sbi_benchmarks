"""
    HolderTableProblem()

A 2D problem with the Holder Table simulator.

Simulator: y = −|sin(x₁) cos(x₂) exp(|1 − √(x₁²+x₂²)/π|)|

Four equal global minima at approximately (±8.0550, ±9.6646) with y ≈ −19.2085.
The exp(|1 − r/π|) term amplifies values where r ≈ π, creating a "table" of
elevated values near that radius. The absolute value and four-fold symmetry
give rise to four isolated global-minima clusters. Good benchmark for testing
methods on posteriors with well-separated modes.

With z_obs = −15.0 and std_obs = 1.0 the posterior has 4 distinct modes near
each global minimum.
"""
@kwdef struct HolderTableProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [1.0]
end

set_gradients(p::HolderTableProblem, val::Bool) = HolderTableProblem(val, p.std_obs)


module HolderTableProblemModule

import ..HolderTableProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [-15.0]
const std_obs = [1.0]
const x_true  = [-7.586684, 9.148047]

# --- API ---

simulator(p::HolderTableProblem) = p.gradients ? _sim_grads : _sim
domain(::HolderTableProblem)     = Domain(; bounds = ([-10.0, -10.0], [10.0, 10.0]))
likelihood(p::HolderTableProblem) = NormalLikelihood(; z_obs, std_obs=p.std_obs)
prior_mean(::HolderTableProblem) = z_obs
x_prior(::HolderTableProblem)    = Product([Uniform(-10.0, 10.0), Uniform(-10.0, 10.0)])
est_amplitude(::HolderTableProblem)      = [20.0]
est_noise_std(::HolderTableProblem)      = nothing
est_grad_noise_std(::HolderTableProblem) = nothing
true_f(::HolderTableProblem)    = _f
true_params(::HolderTableProblem) = x_true

# --- Simulator ---

function _f(x)
    r = sqrt(x[1]^2 + x[2]^2)
    e = exp(abs(1.0 - r / π))
    return [-abs(sin(x[1]) * cos(x[2]) * e)]
end

function _J(x)
    r  = sqrt(x[1]^2 + x[2]^2)
    e  = exp(abs(1.0 - r / π))
    s1 = sin(x[1]); c1 = cos(x[1])
    c2 = cos(x[2]); s2 = sin(x[2])
    raw = s1 * c2 * e
    sg  = sign(raw)          # sign of raw (dA/d(raw))
    # df/dx₁: d/dx₁[-|raw|] = -sg * d(raw)/dx₁
    # d(raw)/dx₁ = c1*c2*e + s1*c2*e * d|1-r/π|/dx₁ * sign(1-r/π)
    # d|1-r/π|/dx₁ = -x₁/(π*r) * sign(1-r/π)
    J = zeros(1, 2)
    if r > 1e-12
        sign_term = sign(1.0 - r / π)
        de_dx1 = e * sign_term * (-x[1] / (π * r))
        de_dx2 = e * sign_term * (-x[2] / (π * r))
        draw_dx1 = c1*c2*e + s1*c2*de_dx1
        draw_dx2 = -s1*s2*e + s1*c2*de_dx2
        J[1, 1] = -sg * draw_dx1
        J[1, 2] = -sg * draw_dx2
    end
    return J
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))

end # module HolderTableProblemModule

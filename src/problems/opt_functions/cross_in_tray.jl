"""
    CrossInTrayProblem()

A 2D problem with the Cross-in-Tray simulator.

Simulator: y = -0.0001 (|sin(x₁) sin(x₂) exp(|100 − √(x₁²+x₂²)/π|)| + 1)^0.1

Four equal global minima at approximately (±1.3491, ±1.3491) with y ≈ −2.0626.
The function is nearly 0 everywhere except near these four troughs. The exp term
produces very large intermediate values that the outer 0.1 power and 0.0001
scale compress into a bounded range. The tray-like shape with 4 symmetric modes
makes this a strong test for multimodal posterior inference.

With z_obs = -1.9 and std_obs = 0.05 the posterior is 4-modal with one mode
near each global minimum.
"""
@kwdef struct CrossInTrayProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [0.05]
end

set_gradients(p::CrossInTrayProblem, val::Bool) = CrossInTrayProblem(val, p.std_obs)


module CrossInTrayProblemModule

import ..CrossInTrayProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [-1.9]
const std_obs = [0.05]
const x_true  = [0.425753, -0.841386]

# --- API ---

simulator(p::CrossInTrayProblem) = p.gradients ? _sim_grads : _sim
domain(::CrossInTrayProblem)     = Domain(; bounds = ([-10.0, -10.0], [10.0, 10.0]))
likelihood(p::CrossInTrayProblem) = NormalLikelihood(; z_obs, std_obs=p.std_obs)
prior_mean(::CrossInTrayProblem) = z_obs
x_prior(::CrossInTrayProblem)    = Product([Uniform(-10.0, 10.0), Uniform(-10.0, 10.0)])
est_amplitude(::CrossInTrayProblem)      = [2.0]
est_noise_std(::CrossInTrayProblem)      = nothing
est_grad_noise_std(::CrossInTrayProblem) = nothing
true_f(::CrossInTrayProblem)    = _f
true_params(::CrossInTrayProblem) = x_true

# --- Simulator ---

function _f(x)
    r   = sqrt(x[1]^2 + x[2]^2)
    # domain [-10,10]: r ≤ √200 < 15, so 100 - r/π > 0 always
    e   = exp(100.0 - r / π)
    A   = abs(sin(x[1]) * sin(x[2]) * e)
    return [-0.0001 * (A + 1.0)^0.1]
end

function _J(x)
    r = sqrt(x[1]^2 + x[2]^2)
    e = exp(100.0 - r / π)
    raw = sin(x[1]) * sin(x[2]) * e
    A   = abs(raw)
    f_val = -0.0001 * (A + 1.0)^0.1
    # df/dA = -0.0001 * 0.1 * (A+1)^(-0.9) = f_val * 0.1 / (A+1)
    df_dA = f_val * 0.1 / (A + 1.0)
    # dA/d(raw) = sign(raw)
    s = sign(raw)
    J = zeros(1, 2)
    if r > 1e-12
        # d(raw)/dx₁ = cos(x₁)*sin(x₂)*e - sin(x₁)*sin(x₂)*e*x₁/(π*r)
        # d(raw)/dx₂ = sin(x₁)*cos(x₂)*e - sin(x₁)*sin(x₂)*e*x₂/(π*r)
        J[1, 1] = df_dA * s * e * (cos(x[1])*sin(x[2]) - sin(x[1])*sin(x[2])*x[1]/(π*r))
        J[1, 2] = df_dA * s * e * (sin(x[1])*cos(x[2]) - sin(x[1])*sin(x[2])*x[2]/(π*r))
    else
        J[1, 1] = 0.0
        J[1, 2] = 0.0
    end
    return J
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))

end # module CrossInTrayProblemModule

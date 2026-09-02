"""
    GoldsteinPriceProblem()

A 2D problem with the Goldstein-Price simulator.

Simulator: y = [1 + (x+y+1)²(19−14x+3x²−14y+6xy+3y²)] ×
               [30 + (2x−3y)²(18−32x+12x²+48y−36xy+27y²)]

Global minimum of 3 at (0, −1). The function has many local minima and a
highly non-stationary response surface with large-scale structure and sharp
local features. The standard version has a 4-order-of-magnitude variation
in scale. Good benchmark for proxy design (log transform often recommended).

With z_obs = 30.0 and std_obs = 5.0 the posterior is multimodal near the
level set where the product of the two factors equals 30.
"""
@kwdef struct GoldsteinPriceProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [5.0]
end

set_gradients(p::GoldsteinPriceProblem, val::Bool) = GoldsteinPriceProblem(val, p.std_obs)


module GoldsteinPriceProblemModule

import ..GoldsteinPriceProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [30.0]
const std_obs = [5.0]
const x_true  = [-0.267916, -0.719521]   # same reference point as GoldsteinPriceProxyProblem

# --- API ---

simulator(p::GoldsteinPriceProblem) = p.gradients ? _sim_grads : _sim
domain(::GoldsteinPriceProblem)     = Domain(; bounds = ([-2.0, -2.0], [2.0, 2.0]))
likelihood(p::GoldsteinPriceProblem) = NormalLikelihood(; z_obs, std_obs=p.std_obs)
prior_mean(::GoldsteinPriceProblem) = z_obs
x_prior(::GoldsteinPriceProblem)    = Product([Uniform(-2.0, 2.0), Uniform(-2.0, 2.0)])
est_amplitude(::GoldsteinPriceProblem)      = [300.0]
est_noise_std(::GoldsteinPriceProblem)      = nothing
est_grad_noise_std(::GoldsteinPriceProblem) = nothing
true_f(::GoldsteinPriceProblem)     = _f
true_params(::GoldsteinPriceProblem) = x_true

# --- Simulator ---

function _f(x)
    x1, x2 = x[1], x[2]
    s = x1 + x2 + 1
    Q = 19 - 14*x1 + 3*x1^2 - 14*x2 + 6*x1*x2 + 3*x2^2
    A = 1 + s^2 * Q
    t = 2*x1 - 3*x2
    R = 18 - 32*x1 + 12*x1^2 + 48*x2 - 36*x1*x2 + 27*x2^2
    B = 30 + t^2 * R
    return [A * B]
end

function _J(x)
    x1, x2 = x[1], x[2]
    s = x1 + x2 + 1
    Q  = 19 - 14*x1 + 3*x1^2 - 14*x2 + 6*x1*x2 + 3*x2^2
    dQ1 = -14 + 6*x1 + 6*x2
    dQ2 = -14 + 6*x1 + 6*x2
    A  = 1 + s^2 * Q
    dA1 = 2*s*Q + s^2*dQ1
    dA2 = 2*s*Q + s^2*dQ2
    t = 2*x1 - 3*x2
    R  = 18 - 32*x1 + 12*x1^2 + 48*x2 - 36*x1*x2 + 27*x2^2
    dR1 = -32 + 24*x1 - 36*x2
    dR2 =  48 - 36*x1 + 54*x2
    B  = 30 + t^2 * R
    dB1 = 4*t*R + t^2*dR1
    dB2 = -6*t*R + t^2*dR2
    J = zeros(1, 2)
    J[1, 1] = dA1*B + A*dB1
    J[1, 2] = dA2*B + A*dB2
    return J
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))

end # module GoldsteinPriceProblemModule

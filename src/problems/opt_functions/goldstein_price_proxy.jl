"""
    GoldsteinPriceProxyProblem()

A 2D problem with the Goldstein-Price simulator using a log(f − 2) proxy variable.

Proxy: δ = log(f(x) − 2)

The raw Goldstein-Price function has global minimum 3 at (0, −1) and spans
[3, ~166 000] on [−2, 2]², causing severe GP ill-conditioning. Since f ≥ 3 > 2
everywhere, the shift f − 2 ≥ 1 is always positive and log(f − 2) ∈ [0, ~12]
gives a well-conditioned GP response surface.

The likelihood uses CustomLikelihood: it inverts the proxy transformation
(f = exp(δ) + 2) and evaluates the original NormalLikelihood, giving the
same true posterior as GoldsteinPriceProblem.

Compare with GoldsteinPriceProblem (no proxy) to assess the effect of proxy design.
"""
@kwdef struct GoldsteinPriceProxyProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [5.0]
end

set_gradients(p::GoldsteinPriceProxyProblem, val::Bool) = GoldsteinPriceProxyProblem(val, p.std_obs)


module GoldsteinPriceProxyProblemModule

import ..GoldsteinPriceProxyProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..proxy_offset; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const _z_obs_raw   = 30.0
const _std_obs_raw = 5.0
const _C           = -2.0   # proxy = log(f + C) = log(f - 2)
const x_true       = [-0.267916, -0.719521]

# --- API ---

simulator(p::GoldsteinPriceProxyProblem) = p.gradients ? _sim_grads : _sim
domain(::GoldsteinPriceProxyProblem)     = Domain(; bounds = ([-2.0, -2.0], [2.0, 2.0]))
prior_mean(::GoldsteinPriceProxyProblem) = [log(_z_obs_raw + _C)]  # proxy-space expected value = log(28)
x_prior(::GoldsteinPriceProxyProblem)    = Product([Uniform(-2.0, 2.0), Uniform(-2.0, 2.0)])
est_amplitude(::GoldsteinPriceProxyProblem)      = [5.0]
est_noise_std(::GoldsteinPriceProxyProblem)      = nothing
est_grad_noise_std(::GoldsteinPriceProxyProblem) = nothing
true_f(::GoldsteinPriceProxyProblem)    = _true_f
true_params(::GoldsteinPriceProxyProblem) = x_true
proxy_offset(::GoldsteinPriceProxyProblem) = _C

function _log_ψ(std_obs_raw::Float64)
    return (δ, x) -> begin
        f_val = exp(δ[1]) - _C
        logpdf(Normal(_z_obs_raw, std_obs_raw), f_val)
    end
end

function likelihood(p::GoldsteinPriceProxyProblem)
    return CustomLikelihood(;
        log_ψ      = _log_ψ(p.std_obs[1]),
        δ_dim      = 1,
        mc_samples = 1000,
    )
end

# --- Raw Goldstein-Price function ---

function _f_raw(x)
    x1, x2 = x[1], x[2]
    s = x1 + x2 + 1
    Q = 19 - 14*x1 + 3*x1^2 - 14*x2 + 6*x1*x2 + 3*x2^2
    A = 1 + s^2 * Q
    t = 2*x1 - 3*x2
    R = 18 - 32*x1 + 12*x1^2 + 48*x2 - 36*x1*x2 + 27*x2^2
    B = 30 + t^2 * R
    return A * B
end

function _J_raw(x)
    x1, x2 = x[1], x[2]
    s = x1 + x2 + 1
    Q  = 19 - 14*x1 + 3*x1^2 - 14*x2 + 6*x1*x2 + 3*x2^2
    dQ = -14 + 6*x1 + 6*x2
    A  = 1 + s^2 * Q
    dA1 = 2*s*Q + s^2*dQ;  dA2 = 2*s*Q + s^2*dQ
    t = 2*x1 - 3*x2
    R  = 18 - 32*x1 + 12*x1^2 + 48*x2 - 36*x1*x2 + 27*x2^2
    dR1 = -32 + 24*x1 - 36*x2;  dR2 = 48 - 36*x1 + 54*x2
    B  = 30 + t^2 * R
    dB1 = 4*t*R + t^2*dR1;  dB2 = -6*t*R + t^2*dR2
    J = zeros(1, 2)
    J[1,1] = dA1*B + A*dB1
    J[1,2] = dA2*B + A*dB2
    return J
end

# --- Proxy simulator ---

function _f(x)
    return [log(_f_raw(x) + _C)]
end

function _J(x)
    f_val = _f_raw(x)
    return _J_raw(x) ./ (f_val + _C)
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))
_true_f    = (x) -> _f(x)

end # module GoldsteinPriceProxyProblemModule

"""
    BealeProxyProblem()

A 2D problem with the Beale simulator using a log(1 + f) proxy variable.

Proxy: δ = log(1 + f(x))  where  f(x) = (1.5−x+xy)² + (2.25−x+xy²)² + (2.625−x+xy³)²

The raw Beale function spans [0, ~100 000] on [-4.5, 4.5]², causing severe GP
ill-conditioning. The log(1 + f) proxy compresses this to [0, ~11.5], making
the response surface well-suited for a stationary GP.

The likelihood uses CustomLikelihood: it inverts the proxy transformation
(f = exp(δ) − 1) and evaluates the original NormalLikelihood, giving the
same true posterior as BealeProblem.

Compare with BealeProblem (no proxy) to assess the effect of proxy design.
"""
@kwdef struct BealeProxyProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [1.0]
end

set_gradients(p::BealeProxyProblem, val::Bool) = BealeProxyProblem(val, p.std_obs)


module BealeProxyProblemModule

import ..BealeProxyProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..proxy_offset; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const _z_obs_raw   = 5.0
const _std_obs_raw = 1.0
const _C           = 1.0   # proxy = log(f + C)
const x_true       = [1.057710, -0.879811]

# --- API ---

simulator(p::BealeProxyProblem) = p.gradients ? _sim_grads : _sim
domain(::BealeProxyProblem)     = Domain(; bounds = ([-4.5, -4.5], [4.5, 4.5]))
prior_mean(::BealeProxyProblem) = [log(_z_obs_raw + _C)]  # proxy-space expected value = log(6)
x_prior(::BealeProxyProblem)    = Product([Uniform(-4.5, 4.5), Uniform(-4.5, 4.5)])
est_amplitude(::BealeProxyProblem)      = [5.0]
est_noise_std(::BealeProxyProblem)      = nothing
est_grad_noise_std(::BealeProxyProblem) = nothing
true_f(::BealeProxyProblem)    = _true_f
true_params(::BealeProxyProblem) = x_true
proxy_offset(::BealeProxyProblem) = _C

function _log_ψ(std_obs_raw::Float64)
    return (δ, x) -> begin
        f_val = exp(δ[1]) - _C
        logpdf(Normal(_z_obs_raw, std_obs_raw), f_val)
    end
end

function likelihood(p::BealeProxyProblem)
    return CustomLikelihood(;
        log_ψ      = _log_ψ(p.std_obs[1]),
        δ_dim      = 1,
        mc_samples = 1000,
    )
end

# --- Raw Beale function ---

function _f_raw(x)
    a = 1.5   - x[1] + x[1]*x[2]
    b = 2.25  - x[1] + x[1]*x[2]^2
    c = 2.625 - x[1] + x[1]*x[2]^3
    return a^2 + b^2 + c^2
end

function _J_raw(x)
    a = 1.5   - x[1] + x[1]*x[2];   da1 = -1+x[2];   da2 = x[1]
    b = 2.25  - x[1] + x[1]*x[2]^2; db1 = -1+x[2]^2; db2 = 2*x[1]*x[2]
    c = 2.625 - x[1] + x[1]*x[2]^3; dc1 = -1+x[2]^3; dc2 = 3*x[1]*x[2]^2
    J = zeros(1, 2)
    J[1,1] = 2*a*da1 + 2*b*db1 + 2*c*dc1
    J[1,2] = 2*a*da2 + 2*b*db2 + 2*c*dc2
    return J
end

# --- Proxy simulator ---

function _f(x)
    return [log(_C + _f_raw(x))]
end

function _J(x)
    f_val = _f_raw(x)
    return _J_raw(x) ./ (_C + f_val)
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))
_true_f    = (x) -> _f(x)

end # module BealeProxyProblemModule

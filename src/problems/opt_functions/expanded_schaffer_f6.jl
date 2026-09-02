"""
    ExpandedSchafferF6Problem(; x_dim=2)

A d-dimensional problem with the Expanded Schaffer F6 simulator.

Simulator: y = Σᵢ₌₁ᵈ g(xᵢ, xᵢ₊₁)  (indices wrap: xₐ₊₁ = x₁)

where g(a,b) = 0.5 + (sin²(√(a²+b²)) - 0.5) / (1 + 0.001(a²+b²))²

Each term g is bounded in [0, 1], so y ∈ [0, d]. The global minimum is 0
at the origin. The function has a highly oscillatory ring structure; the
denominator damps oscillations away from the origin, making the surface
non-stationary.

With z_obs = 0.8 and std_obs = 0.1 the posterior concentrates on multiple
ring-like regions where the sum of consecutive-pair contributions hits 0.8.
"""
@kwdef struct ExpandedSchafferF6Problem <: AbstractOptFunctionProblem
    gradients::Bool = false
    x_dim::Int = 2
    std_obs::Vector{Float64} = [0.1]
end

set_gradients(p::ExpandedSchafferF6Problem, val::Bool) = ExpandedSchafferF6Problem(val, p.x_dim, p.std_obs)

get_name(p::ExpandedSchafferF6Problem) = (p |> typeof |> string) * string(p.x_dim)


module ExpandedSchafferF6ProblemModule

import ..ExpandedSchafferF6Problem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const x_true  = [-8.178575, -5.905071]
const x_true_5 = [-8.178575, -5.905071, 3.0, -6.0, 7.0]

# --- API ---

simulator(p::ExpandedSchafferF6Problem) = p.gradients ? _sim_grads(p.x_dim) : _sim(p.x_dim)
domain(p::ExpandedSchafferF6Problem)    = Domain(; bounds = (fill(-10.0, p.x_dim), fill(10.0, p.x_dim)))
_z_obs(p::ExpandedSchafferF6Problem) = true_f(p)(true_params(p))
likelihood(p::ExpandedSchafferF6Problem) = NormalLikelihood(; z_obs=_z_obs(p), std_obs=p.std_obs)
prior_mean(p::ExpandedSchafferF6Problem) = _z_obs(p)
x_prior(p::ExpandedSchafferF6Problem)   = Product(fill(Uniform(-10.0, 10.0), p.x_dim))
est_amplitude(::ExpandedSchafferF6Problem)      = [2.0]
est_noise_std(::ExpandedSchafferF6Problem)      = nothing
est_grad_noise_std(::ExpandedSchafferF6Problem) = nothing
true_f(p::ExpandedSchafferF6Problem)    = _true_f(p.x_dim)
function true_params(p::ExpandedSchafferF6Problem)
    p.x_dim == 2 && return x_true
    p.x_dim == 5 && return x_true_5
    error("true_params not defined for $(p.x_dim)D $(typeof(p))")
end

# --- Simulator ---

# g(a,b) and its partial derivatives w.r.t. a and b (both equal via chain rule through r)
function _g_and_dg(a, b)
    r2 = a^2 + b^2
    r  = sqrt(r2)
    p  = 1.0 + 0.001 * r2
    q  = p^2
    sr = sin(r)
    cr = cos(r)
    g  = 0.5 + (sr^2 - 0.5) / q
    if r < 1e-12
        dg_da = 0.0
        dg_db = 0.0
    else
        # dg/dr = (sin(2r)*q - (sin²r - 0.5)*0.004*p*r) / q²
        dg_dr = (2*sr*cr * q - (sr^2 - 0.5) * 0.004 * p * r) / q^2
        dg_da = a / r * dg_dr
        dg_db = b / r * dg_dr
    end
    return g, dg_da, dg_db
end

function _f(x)
    d = length(x)
    y = 0.0
    for i in 1:d
        j = mod1(i + 1, d)
        g, _, _ = _g_and_dg(x[i], x[j])
        y += g
    end
    return [y]
end

function _J(x)
    d = length(x)
    J = zeros(1, d)
    for i in 1:d
        j = mod1(i + 1, d)
        _, dg_da, dg_db = _g_and_dg(x[i], x[j])
        J[1, i] += dg_da           # contribution from g(xᵢ, xᵢ₊₁) w.r.t. first arg
        J[1, j] += dg_db           # contribution from g(xᵢ, xᵢ₊₁) w.r.t. second arg
    end
    return J
end

_sim(d)       = (x) -> _f(x)
_sim_grads(d) = (x) -> (_f(x), _J(x))
_true_f(d)    = (x) -> _f(x)

end # module ExpandedSchafferF6ProblemModule

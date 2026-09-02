"""
    MichalewiczProblem(; x_dim=2, m=10)

A d-dimensional problem with the Michalewicz simulator.

Simulator: y = −Σᵢ₌₁ᵈ sin(xᵢ) sin²ᵐ(i⋅xᵢ²/π)  (scalar output)

The steepness parameter m (default 10) controls how sharp the ridges are.
Higher m produces narrower, more needle-like peaks. Peaks occur where
i·xᵢ²/π = π/2 + nπ, i.e. xᵢ = π√((2n+1)/(2i)), giving multiple local
minima per dimension. Each dimension contributes independently but with
different frequencies (proportional to i), so the response surface is highly
heterogeneous — fine structure at high-index dimensions, broader structure
at low-index dimensions.

This extreme gradient heterogeneity violates GP stationarity assumptions
severely. It serves as a strong test case for proxy smoothing and nonstationary
surrogate models.

For d=2, the two deepest modes are at approximately (2.22, 1.57) with
f ≈ −1.79 and (2.22, 2.72) with f ≈ −1.20. With z_obs = −1.5 and
std_obs = 0.3, the posterior is bimodal with roughly equal weights on both.
"""
@kwdef struct MichalewiczProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    x_dim::Int = 2
    m::Int = 10
    std_obs::Vector{Float64} = [0.3]
end

set_gradients(p::MichalewiczProblem, val::Bool) = MichalewiczProblem(val, p.x_dim, p.m, p.std_obs)

get_name(p::MichalewiczProblem) = (p |> typeof |> string) * string(p.x_dim)


module MichalewiczProblemModule

import ..MichalewiczProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const x_true  = [2.097956, 1.632165]
const x_true_5 = [2.097956, 1.632165, 1.28, 1.11, 0.99]

# --- API ---

simulator(p::MichalewiczProblem) = p.gradients ? _sim_grads(p.m) : _sim(p.m)
domain(p::MichalewiczProblem)    = Domain(; bounds = (fill(0.0, p.x_dim), fill(π, p.x_dim)))
_z_obs(p::MichalewiczProblem) = true_f(p)(true_params(p))
likelihood(p::MichalewiczProblem) = NormalLikelihood(; z_obs=_z_obs(p), std_obs=p.std_obs)
prior_mean(p::MichalewiczProblem) = _z_obs(p)
x_prior(p::MichalewiczProblem)   = Product(fill(Uniform(0.0, π), p.x_dim))
est_amplitude(::MichalewiczProblem)      = [1.0]
est_noise_std(::MichalewiczProblem)      = nothing
est_grad_noise_std(::MichalewiczProblem) = nothing
true_f(p::MichalewiczProblem)    = _true_f(p.m)
function true_params(p::MichalewiczProblem)
    p.x_dim == 2 && return x_true
    p.x_dim == 5 && return x_true_5
    error("true_params not defined for $(p.x_dim)D $(typeof(p))")
end

# --- Simulator ---

function _f(x, m)
    d = length(x)
    y = 0.0
    for i in 1:d
        y -= sin(x[i]) * sin(i * x[i]^2 / π)^(2*m)
    end
    return [y]
end

function _J(x, m)
    d = length(x)
    J = zeros(1, d)
    for i in 1:d
        s  = sin(x[i])
        c  = cos(x[i])
        si = sin(i * x[i]^2 / π)
        ci = cos(i * x[i]^2 / π)
        # d/dxᵢ [ -sin(xᵢ) sin²ᵐ(i xᵢ²/π) ]
        J[1, i] = -(c * si^(2*m) + s * 2*m * si^(2*m - 1) * ci * (2*i*x[i] / π))
    end
    return J
end

_sim(m)       = (x) -> _f(x, m)
_sim_grads(m) = (x) -> (_f(x, m), _J(x, m))
_true_f(m)    = (x) -> _f(x, m)

end # module MichalewiczProblemModule

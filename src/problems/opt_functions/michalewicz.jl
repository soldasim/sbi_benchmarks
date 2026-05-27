"""
    MichalewiczProblem(; x_dim=2, m=10)

A d-dimensional problem with the Michalewicz simulator.

Simulator: y = −Σᵢ₌₁ᵈ sin(xᵢ) sin²ᵐ(i⋅xᵢ/π)  (scalar output)

The steepness parameter m (default 10) controls how sharp the ridges are.
Higher m produces narrower, more needle-like peaks. The function has d! local
minima in d dimensions. Each dimension's contribution varies in frequency
(proportional to i), so the response surface is highly heterogeneous — fine
structure at high-index dimensions, broader structure at low-index dimensions.

This extreme gradient heterogeneity (gradients near zero almost everywhere,
huge near the peaks) violates GP stationarity assumptions severely. It serves
as a strong test case for proxy smoothing (e.g. a log transform to compress
the dynamic range) and for nonstationary surrogate models.

With z_obs = −1.0 and small Gaussian noise, the posterior concentrates near
the deepest peaks of the function.
"""
@kwdef struct MichalewiczProblem <: AbstractProblem
    gradients::Bool = false
    x_dim::Int = 2
    m::Int = 10
end

set_gradients(p::MichalewiczProblem, val::Bool) = MichalewiczProblem(val, p.x_dim, p.m)

get_name(p::MichalewiczProblem) = (p |> typeof |> string) * string(p.x_dim)


module MichalewiczProblemModule

import ..MichalewiczProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [-1.0]
const std_obs = [0.2]

# --- API ---

simulator(p::MichalewiczProblem) = p.gradients ? _sim_grads(p.m) : _sim(p.m)
domain(p::MichalewiczProblem)    = Domain(; bounds = (fill(0.0, p.x_dim), fill(π, p.x_dim)))
likelihood(::MichalewiczProblem) = NormalLikelihood(; z_obs, std_obs)
prior_mean(::MichalewiczProblem) = z_obs
x_prior(p::MichalewiczProblem)   = Product(fill(Uniform(0.0, π), p.x_dim))
est_amplitude(::MichalewiczProblem)      = [1.0]
est_noise_std(::MichalewiczProblem)      = nothing
est_grad_noise_std(::MichalewiczProblem) = nothing
true_f(p::MichalewiczProblem)    = _true_f(p.m)

# --- Simulator ---

function _f(x, m)
    d = length(x)
    y = 0.0
    for i in 1:d
        y -= sin(x[i]) * sin(i * x[i] / π)^(2*m)
    end
    return [y]
end

function _J(x, m)
    d = length(x)
    J = zeros(1, d)
    for i in 1:d
        s  = sin(x[i])
        c  = cos(x[i])
        si = sin(i * x[i] / π)
        ci = cos(i * x[i] / π)
        # d/dxᵢ [ -sin(xᵢ) sin²ᵐ(i xᵢ/π) ]
        J[1, i] = -(c * si^(2*m) + s * 2*m * si^(2*m - 1) * ci * (i / π))
    end
    return J
end

_sim(m)       = (x) -> _f(x, m)
_sim_grads(m) = (x) -> (_f(x, m), _J(x, m))
_true_f(m)    = (x) -> _f(x, m)

end # module MichalewiczProblemModule

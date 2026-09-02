"""
    AckleyProblem(; x_dim=2)

A d-dimensional problem with the Ackley simulator.

Simulator: y = -20 exp(-0.2 √(1/d Σxᵢ²)) - exp(1/d Σcos(2πxᵢ)) + 20 + e

The function has a single global minimum at the origin (y=0) surrounded by
a nearly flat outer region with concentric ridges. The exponential envelope
and cosine modulation make the response surface non-stationary: smooth near
the origin, oscillatory elsewhere. Good benchmark for testing proxy design
under mixed smooth/oscillatory structure.

With z_obs = 3.5 and std_obs = 0.5 the posterior forms a ring-shaped
distribution with mild multimodality from the cosine ridges.
"""
@kwdef struct AckleyProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    x_dim::Int = 2
    std_obs::Vector{Float64} = [0.5]
end

set_gradients(p::AckleyProblem, val::Bool) = AckleyProblem(val, p.x_dim, p.std_obs)

get_name(p::AckleyProblem) = (p |> typeof |> string) * string(p.x_dim)


module AckleyProblemModule

import ..AckleyProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const x_true  = [1.032593, 0.187300]
const x_true_5 = [1.032593, 0.187300, 0.8, -1.2, 1.5]

# --- API ---

simulator(p::AckleyProblem) = p.gradients ? _sim_grads(p.x_dim) : _sim(p.x_dim)
domain(p::AckleyProblem)    = Domain(; bounds = (fill(-5.0, p.x_dim), fill(5.0, p.x_dim)))
_z_obs(p::AckleyProblem) = true_f(p)(true_params(p))
likelihood(p::AckleyProblem) = NormalLikelihood(; z_obs=_z_obs(p), std_obs=p.std_obs)
prior_mean(p::AckleyProblem) = _z_obs(p)
x_prior(p::AckleyProblem)   = Product(fill(Uniform(-5.0, 5.0), p.x_dim))
est_amplitude(::AckleyProblem)      = [14.0]
est_noise_std(::AckleyProblem)      = nothing
est_grad_noise_std(::AckleyProblem) = nothing
true_f(p::AckleyProblem)    = _true_f(p.x_dim)
function true_params(p::AckleyProblem)
    p.x_dim == 2 && return x_true
    p.x_dim == 5 && return x_true_5
    error("true_params not defined for $(p.x_dim)D $(typeof(p))")
end

# --- Simulator ---

const _a = 20.0
const _b = 0.2
const _c = 2π

function _f(x)
    d = length(x)
    r = sqrt(sum(xi^2 for xi in x) / d)
    s = sum(cos(_c * xi) for xi in x) / d
    return [-_a * exp(-_b * r) - exp(s) + _a + exp(1.0)]
end

function _J(x)
    d = length(x)
    sum_sq  = sum(xi^2 for xi in x)
    sum_cos = sum(cos(_c * xi) for xi in x)
    r = sqrt(sum_sq / d)
    term1_exp = exp(-_b * r)
    term2_exp = exp(sum_cos / d)
    J = zeros(1, d)
    for j in 1:d
        d1 = r > 1e-12 ? _a * _b * x[j] / (d * r) * term1_exp : 0.0
        d2 = _c / d * sin(_c * x[j]) * term2_exp
        J[1, j] = d1 + d2
    end
    return J
end

_sim(d)      = (x) -> _f(x)
_sim_grads(d) = (x) -> (_f(x), _J(x))
_true_f(d)   = (x) -> _f(x)

end # module AckleyProblemModule

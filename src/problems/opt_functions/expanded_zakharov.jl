"""
    ExpandedZakharovProblem(; x_dim=2)

A d-dimensional problem with the Zakharov simulator.

Simulator: y = Σxᵢ² + (Σ 0.5i xᵢ)² + (Σ 0.5i xᵢ)⁴

The function is unimodal with a global minimum of 0 at the origin. The
asymmetric weighting of dimensions (coefficient 0.5i grows with index i)
makes higher-indexed dimensions more influential on the quartic term,
creating an anisotropic bowl that challenges isotropic GP surrogates.

With z_obs = 20.0 and std_obs = 4.0 the posterior is an anisotropic
ring-shaped distribution.
"""
@kwdef struct ExpandedZakharovProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    x_dim::Int = 2
    std_obs::Vector{Float64} = [4.0]
end

set_gradients(p::ExpandedZakharovProblem, val::Bool) = ExpandedZakharovProblem(val, p.x_dim, p.std_obs)

get_name(p::ExpandedZakharovProblem) = (p |> typeof |> string) * string(p.x_dim)


module ExpandedZakharovProblemModule

import ..ExpandedZakharovProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const x_true  = [-0.245732, 2.000000]
const x_true_5 = [-0.245732, 2.0, 0.5, -1.0, 1.5]

# --- API ---

simulator(p::ExpandedZakharovProblem) = p.gradients ? _sim_grads(p.x_dim) : _sim(p.x_dim)
domain(p::ExpandedZakharovProblem)    = Domain(; bounds = (fill(-2.0, p.x_dim), fill(2.0, p.x_dim)))
_z_obs(p::ExpandedZakharovProblem) = true_f(p)(true_params(p))
likelihood(p::ExpandedZakharovProblem) = NormalLikelihood(; z_obs=_z_obs(p), std_obs=p.std_obs)
prior_mean(p::ExpandedZakharovProblem) = _z_obs(p)
x_prior(p::ExpandedZakharovProblem)   = Product(fill(Uniform(-2.0, 2.0), p.x_dim))
est_amplitude(::ExpandedZakharovProblem)      = [100.0]
est_noise_std(::ExpandedZakharovProblem)      = nothing
est_grad_noise_std(::ExpandedZakharovProblem) = nothing
true_f(p::ExpandedZakharovProblem)    = _true_f(p.x_dim)
function true_params(p::ExpandedZakharovProblem)
    p.x_dim == 2 && return x_true
    p.x_dim == 5 && return x_true_5
    error("true_params not defined for $(p.x_dim)D $(typeof(p))")
end

# --- Simulator ---

function _f(x)
    d = length(x)
    w = sum(0.5 * i * x[i] for i in 1:d)
    return [sum(xi^2 for xi in x) + w^2 + w^4]
end

function _J(x)
    d = length(x)
    w = sum(0.5 * i * x[i] for i in 1:d)
    J = zeros(1, d)
    for j in 1:d
        J[1, j] = 2*x[j] + (2*w + 4*w^3) * 0.5 * j
    end
    return J
end

_sim(d)       = (x) -> _f(x)
_sim_grads(d) = (x) -> (_f(x), _J(x))
_true_f(d)    = (x) -> _f(x)

end # module ExpandedZakharovProblemModule

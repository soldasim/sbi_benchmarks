"""
    AlpineProblem(; x_dim=2)

A d-dimensional problem with the Alpine N.1 simulator.

Simulator: y = Σᵢ |xᵢ sin(xᵢ) + 0.1 xᵢ|

The function has multiple local minima separated by ridges. The absolute-value
structure creates a non-smooth response surface with a global minimum of 0 at
the origin. Good benchmark for proxy design under non-smooth landscapes.

With z_obs = 5.0 and std_obs = 1.0 the posterior is multimodal, concentrating
near the level sets of the function.
"""
@kwdef struct AlpineProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    x_dim::Int = 2
    std_obs::Vector{Float64} = [1.0]
end

set_gradients(p::AlpineProblem, val::Bool) = AlpineProblem(val, p.x_dim, p.std_obs)

get_name(p::AlpineProblem) = (p |> typeof |> string) * string(p.x_dim)


module AlpineProblemModule

import ..AlpineProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const x_true  = [-5.300579, 0.200829]
const x_true_5 = [-5.300579, 0.200829, -3.5, 5.0, -7.0]

# --- API ---

simulator(p::AlpineProblem) = p.gradients ? _sim_grads(p.x_dim) : _sim(p.x_dim)
domain(p::AlpineProblem)    = Domain(; bounds = (fill(-10.0, p.x_dim), fill(10.0, p.x_dim)))
_z_obs(p::AlpineProblem) = true_f(p)(true_params(p))
likelihood(p::AlpineProblem) = NormalLikelihood(; z_obs=_z_obs(p), std_obs=p.std_obs)
prior_mean(p::AlpineProblem) = _z_obs(p)
x_prior(p::AlpineProblem)   = Product(fill(Uniform(-10.0, 10.0), p.x_dim))
est_amplitude(::AlpineProblem)      = [20.0]
est_noise_std(::AlpineProblem)      = nothing
est_grad_noise_std(::AlpineProblem) = nothing
true_f(p::AlpineProblem)    = _true_f(p.x_dim)
function true_params(p::AlpineProblem)
    p.x_dim == 2 && return x_true
    p.x_dim == 5 && return x_true_5
    error("true_params not defined for $(p.x_dim)D $(typeof(p))")
end

# --- Simulator ---

function _f(x)
    return [sum(abs(xi * sin(xi) + 0.1 * xi) for xi in x)]
end

function _J(x)
    d = length(x)
    J = zeros(1, d)
    for j in 1:d
        inner = x[j] * sin(x[j]) + 0.1 * x[j]
        d_inner = sin(x[j]) + x[j] * cos(x[j]) + 0.1
        J[1, j] = sign(inner) * d_inner
    end
    return J
end

_sim(d)       = (x) -> _f(x)
_sim_grads(d) = (x) -> (_f(x), _J(x))
_true_f(d)    = (x) -> _f(x)

end # module AlpineProblemModule

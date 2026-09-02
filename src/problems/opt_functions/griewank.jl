"""
    GriewankProblem(; x_dim=2)

A d-dimensional problem with the Griewank simulator.

Simulator: y = 1 + Σ(xᵢ²/4000) - Π cos(xᵢ/√i)

The quadratic term creates a broad bowl while the product-of-cosines creates
a regular grid of local minima. The global minimum is 0 at the origin. As
dimension increases, the local minima become denser. The response surface is
non-stationary (different oscillation frequencies per dimension due to the
1/√i scaling).

With z_obs = 0.5 and std_obs = 0.1 the posterior is multimodal, spread across
the many near-equal local minima of the function.
"""
@kwdef struct GriewankProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    x_dim::Int = 2
    std_obs::Vector{Float64} = [0.1]
end

set_gradients(p::GriewankProblem, val::Bool) = GriewankProblem(val, p.x_dim, p.std_obs)

get_name(p::GriewankProblem) = (p |> typeof |> string) * string(p.x_dim)


module GriewankProblemModule

import ..GriewankProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const x_true  = [8.991318, -5.785064]
const x_true_5 = [8.991318, -5.785064, 3.0, -7.0, 5.0]

# --- API ---

simulator(p::GriewankProblem) = p.gradients ? _sim_grads(p.x_dim) : _sim(p.x_dim)
domain(p::GriewankProblem)    = Domain(; bounds = (fill(-10.0, p.x_dim), fill(10.0, p.x_dim)))
_z_obs(p::GriewankProblem) = true_f(p)(true_params(p))
likelihood(p::GriewankProblem) = NormalLikelihood(; z_obs=_z_obs(p), std_obs=p.std_obs)
prior_mean(p::GriewankProblem) = _z_obs(p)
x_prior(p::GriewankProblem)   = Product(fill(Uniform(-10.0, 10.0), p.x_dim))
est_amplitude(::GriewankProblem)      = [2.0]
est_noise_std(::GriewankProblem)      = nothing
est_grad_noise_std(::GriewankProblem) = nothing
true_f(p::GriewankProblem)    = _true_f(p.x_dim)
function true_params(p::GriewankProblem)
    p.x_dim == 2 && return x_true
    p.x_dim == 5 && return x_true_5
    error("true_params not defined for $(p.x_dim)D $(typeof(p))")
end

# --- Simulator ---

function _f(x)
    d = length(x)
    s = sum(xi^2 / 4000.0 for xi in x)
    p = prod(cos(x[i] / sqrt(Float64(i))) for i in 1:d)
    return [1.0 + s - p]
end

function _J(x)
    d = length(x)
    p = prod(cos(x[i] / sqrt(Float64(i))) for i in 1:d)
    J = zeros(1, d)
    for j in 1:d
        cos_j = cos(x[j] / sqrt(Float64(j)))
        if abs(cos_j) > 1e-12
            prod_rest = p / cos_j
        else
            prod_rest = prod(cos(x[i] / sqrt(Float64(i))) for i in 1:d if i != j; init = 1.0)
        end
        J[1, j] = x[j] / 2000.0 + sin(x[j] / sqrt(Float64(j))) / sqrt(Float64(j)) * prod_rest
    end
    return J
end

_sim(d)       = (x) -> _f(x)
_sim_grads(d) = (x) -> (_f(x), _J(x))
_true_f(d)    = (x) -> _f(x)

end # module GriewankProblemModule

"""
    RastriginProblem(; x_dim=2)

A d-dimensional problem with the Rastrigin simulator.

Simulator: y = 10d + Σᵢ (xᵢ² - 10 cos(2π xᵢ))

One of the most widely used multimodal benchmarks. The global minimum is 0
at the origin; local minima form a regular grid at integer points, each
separated by the cosine term. The quadratic baseline grows away from the
origin while the cosine modulation creates O(10^d) local minima. Highly
challenging for surrogate models due to the combination of global trend
and local oscillations.

With z_obs = 20.0 and std_obs = 3.0 the posterior is strongly multimodal,
with modes near each local minimum of the function at the relevant level set.
"""
@kwdef struct RastriginProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    x_dim::Int = 2
    std_obs::Vector{Float64} = [3.0]
end

set_gradients(p::RastriginProblem, val::Bool) = RastriginProblem(val, p.x_dim, p.std_obs)

get_name(p::RastriginProblem) = (p |> typeof |> string) * string(p.x_dim)


module RastriginProblemModule

import ..RastriginProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const x_true  = [1.857216, 2.838278]
const x_true_5 = [1.857216, 2.838278, -1.5, 3.0, -2.5]

# --- API ---

simulator(p::RastriginProblem) = p.gradients ? _sim_grads(p.x_dim) : _sim(p.x_dim)
domain(p::RastriginProblem)    = Domain(; bounds = (fill(-5.12, p.x_dim), fill(5.12, p.x_dim)))
_z_obs(p::RastriginProblem) = true_f(p)(true_params(p))
likelihood(p::RastriginProblem) = NormalLikelihood(; z_obs=_z_obs(p), std_obs=p.std_obs)
prior_mean(p::RastriginProblem) = _z_obs(p)
x_prior(p::RastriginProblem)   = Product(fill(Uniform(-5.12, 5.12), p.x_dim))
est_amplitude(::RastriginProblem)      = [80.0]
est_noise_std(::RastriginProblem)      = nothing
est_grad_noise_std(::RastriginProblem) = nothing
true_f(p::RastriginProblem)    = _true_f(p.x_dim)
function true_params(p::RastriginProblem)
    p.x_dim == 2 && return x_true
    p.x_dim == 5 && return x_true_5
    error("true_params not defined for $(p.x_dim)D $(typeof(p))")
end

# --- Simulator ---

function _f(x)
    d = length(x)
    return [10.0 * d + sum(xi^2 - 10.0 * cos(2π * xi) for xi in x)]
end

function _J(x)
    d = length(x)
    J = zeros(1, d)
    for j in 1:d
        J[1, j] = 2*x[j] + 20π * sin(2π * x[j])
    end
    return J
end

_sim(d)       = (x) -> _f(x)
_sim_grads(d) = (x) -> (_f(x), _J(x))
_true_f(d)    = (x) -> _f(x)

end # module RastriginProblemModule

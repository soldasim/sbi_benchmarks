"""
    SphereProblem(; x_dim=2)

A d-dimensional problem with the Sphere simulator.

Simulator: y = Σxᵢ²

The simplest possible unimodal benchmark. The response surface is a perfect
paraboloid with global minimum 0 at the origin. Level sets are exact spheres,
producing a ring/sphere-shaped posterior. Used as a baseline: any method that
fails on Sphere has fundamental problems. GP surrogates fit the smooth quadratic
surface easily, making it an easy benchmark for proxy design.

With z_obs = 4.0 and std_obs = 0.5 the posterior is an annular distribution
concentrated on the circle {x: Σxᵢ² ≈ 4}.
"""
@kwdef struct SphereProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    x_dim::Int = 2
    std_obs::Vector{Float64} = [0.5]
end

set_gradients(p::SphereProblem, val::Bool) = SphereProblem(val, p.x_dim, p.std_obs)

get_name(p::SphereProblem) = (p |> typeof |> string) * string(p.x_dim)


module SphereProblemModule

import ..SphereProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const x_true  = [1.478150, -1.347246]
const x_true_5 = [1.0, -1.0, 1.2, -0.8, 1.0]

# --- API ---

simulator(p::SphereProblem) = p.gradients ? _sim_grads(p.x_dim) : _sim(p.x_dim)
domain(p::SphereProblem)    = Domain(; bounds = (fill(-5.12, p.x_dim), fill(5.12, p.x_dim)))
_z_obs(p::SphereProblem) = true_f(p)(true_params(p))
likelihood(p::SphereProblem) = NormalLikelihood(; z_obs=_z_obs(p), std_obs=p.std_obs)
prior_mean(p::SphereProblem) = _z_obs(p)
x_prior(p::SphereProblem)   = Product(fill(Uniform(-5.12, 5.12), p.x_dim))
est_amplitude(::SphereProblem)      = [52.0]
est_noise_std(::SphereProblem)      = nothing
est_grad_noise_std(::SphereProblem) = nothing
true_f(p::SphereProblem)    = _true_f(p.x_dim)
function true_params(p::SphereProblem)
    p.x_dim == 2 && return x_true
    p.x_dim == 5 && return x_true_5
    error("true_params not defined for $(p.x_dim)D $(typeof(p))")
end

# --- Simulator ---

function _f(x)
    return [sum(xi^2 for xi in x)]
end

function _J(x)
    d = length(x)
    J = zeros(1, d)
    for j in 1:d
        J[1, j] = 2*x[j]
    end
    return J
end

_sim(d)       = (x) -> _f(x)
_sim_grads(d) = (x) -> (_f(x), _J(x))
_true_f(d)    = (x) -> _f(x)

end # module SphereProblemModule

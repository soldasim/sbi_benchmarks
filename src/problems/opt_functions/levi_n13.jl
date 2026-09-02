"""
    LeviN13Problem()

A 2D problem with the Lévi N.13 simulator.

Simulator: y = sin²(3πx₁) + (x₁−1)²(1 + sin²(3πx₂)) + (x₂−1)²(1 + sin²(2πx₂))

Global minimum of 0 at (1, 1). The function has many local minima arranged
along lines in the domain due to the sin² terms. The two different frequencies
(3π and 2π) create an irregular lattice of local minima.

With z_obs = 2.0 and std_obs = 0.3 the posterior is multimodal with modes
distributed along the level sets near the many local minima.
"""
@kwdef struct LeviN13Problem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [0.3]
end

set_gradients(p::LeviN13Problem, val::Bool) = LeviN13Problem(val, p.std_obs)


module LeviN13ProblemModule

import ..LeviN13Problem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [2.0]
const std_obs = [0.3]
const x_true  = [0.000000, 0.000000]

# --- API ---

simulator(p::LeviN13Problem) = p.gradients ? _sim_grads : _sim
domain(::LeviN13Problem)     = Domain(; bounds = ([-10.0, -10.0], [10.0, 10.0]))
likelihood(p::LeviN13Problem) = NormalLikelihood(; z_obs, std_obs=p.std_obs)
prior_mean(::LeviN13Problem) = z_obs
x_prior(::LeviN13Problem)    = Product([Uniform(-10.0, 10.0), Uniform(-10.0, 10.0)])
est_amplitude(::LeviN13Problem)      = [50.0]
est_noise_std(::LeviN13Problem)      = nothing
est_grad_noise_std(::LeviN13Problem) = nothing
true_f(::LeviN13Problem)    = _f
true_params(::LeviN13Problem) = x_true

# --- Simulator ---

function _f(x)
    s3x1 = sin(3π * x[1])
    s3x2 = sin(3π * x[2])
    s2x2 = sin(2π * x[2])
    return [s3x1^2 + (x[1]-1)^2 * (1 + s3x2^2) + (x[2]-1)^2 * (1 + s2x2^2)]
end

function _J(x)
    s3x1 = sin(3π * x[1]); c3x1 = cos(3π * x[1])
    s3x2 = sin(3π * x[2]); c3x2 = cos(3π * x[2])
    s2x2 = sin(2π * x[2]); c2x2 = cos(2π * x[2])
    J = zeros(1, 2)
    J[1, 1] = 6π * s3x1 * c3x1 + 2*(x[1]-1) * (1 + s3x2^2)
    J[1, 2] = (x[1]-1)^2 * 6π * s3x2 * c3x2 +
              2*(x[2]-1) * (1 + s2x2^2) +
              (x[2]-1)^2 * 4π * s2x2 * c2x2
    return J
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))

end # module LeviN13ProblemModule

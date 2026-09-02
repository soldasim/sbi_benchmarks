"""
    ThreeHumpCamelProblem()

A 2D problem with the Three-Hump Camel simulator.

Simulator: y = 2x² − 1.05x⁴ + x⁶/6 + xy + y²

Global minimum of 0 at the origin. Three local minima total: the global
minimum at (0,0) and two others. The function has a distinct anisotropy
between x and y — the x-axis exhibits a triple-well structure from the
polynomial terms while y is a simple parabola. The combined shape creates
a saddle-like landscape with three "humps". Domain restricted to [−2,2]² to
capture the three-hump structure while keeping scale manageable.

With z_obs = 2.0 and std_obs = 0.3 the posterior has multiple modes near the
three humps.
"""
@kwdef struct ThreeHumpCamelProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [0.3]
end

set_gradients(p::ThreeHumpCamelProblem, val::Bool) = ThreeHumpCamelProblem(val, p.std_obs)


module ThreeHumpCamelProblemModule

import ..ThreeHumpCamelProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [2.0]
const std_obs = [0.3]
const x_true  = [1.236365, 0.469524]

# --- API ---

simulator(p::ThreeHumpCamelProblem) = p.gradients ? _sim_grads : _sim
domain(::ThreeHumpCamelProblem)     = Domain(; bounds = ([-2.0, -2.0], [2.0, 2.0]))
likelihood(p::ThreeHumpCamelProblem) = NormalLikelihood(; z_obs, std_obs=p.std_obs)
prior_mean(::ThreeHumpCamelProblem) = z_obs
x_prior(::ThreeHumpCamelProblem)    = Product([Uniform(-2.0, 2.0), Uniform(-2.0, 2.0)])
est_amplitude(::ThreeHumpCamelProblem)      = [20.0]
est_noise_std(::ThreeHumpCamelProblem)      = nothing
est_grad_noise_std(::ThreeHumpCamelProblem) = nothing
true_f(::ThreeHumpCamelProblem)    = _f
true_params(::ThreeHumpCamelProblem) = x_true

# --- Simulator ---

function _f(x)
    return [2*x[1]^2 - 1.05*x[1]^4 + x[1]^6/6 + x[1]*x[2] + x[2]^2]
end

function _J(x)
    J = zeros(1, 2)
    J[1, 1] = 4*x[1] - 4.2*x[1]^3 + x[1]^5 + x[2]
    J[1, 2] = x[1] + 2*x[2]
    return J
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))

end # module ThreeHumpCamelProblemModule

"""
    MatyasProblem()

A 2D problem with the Matyas simulator.

Simulator: y = 0.26(x² + y²) − 0.48xy

Global minimum of 0 at the origin. The function is a simple bowl with the
minimum along the line x = y, creating a narrow elongated level-set ellipse
at the minimum but broader ellipses away from it. The off-diagonal coefficient
−0.48xy means the principal axes are rotated 45° relative to the coordinate
axes. Tests anisotropic GP kernels.

With z_obs = 10.0 and std_obs = 2.0 the posterior is an elongated
elliptical annular distribution.
"""
@kwdef struct MatyasProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [2.0]
end

set_gradients(p::MatyasProblem, val::Bool) = MatyasProblem(val, p.std_obs)


module MatyasProblemModule

import ..MatyasProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [10.0]
const std_obs = [2.0]
const x_true  = [5.843671, -0.385979]

# --- API ---

simulator(p::MatyasProblem) = p.gradients ? _sim_grads : _sim
domain(::MatyasProblem)     = Domain(; bounds = ([-10.0, -10.0], [10.0, 10.0]))
likelihood(p::MatyasProblem) = NormalLikelihood(; z_obs, std_obs=p.std_obs)
prior_mean(::MatyasProblem) = z_obs
x_prior(::MatyasProblem)    = Product([Uniform(-10.0, 10.0), Uniform(-10.0, 10.0)])
est_amplitude(::MatyasProblem)      = [50.0]
est_noise_std(::MatyasProblem)      = nothing
est_grad_noise_std(::MatyasProblem) = nothing
true_f(::MatyasProblem)    = _f
true_params(::MatyasProblem) = x_true

# --- Simulator ---

function _f(x)
    return [0.26*(x[1]^2 + x[2]^2) - 0.48*x[1]*x[2]]
end

function _J(x)
    J = zeros(1, 2)
    J[1, 1] = 0.52*x[1] - 0.48*x[2]
    J[1, 2] = 0.52*x[2] - 0.48*x[1]
    return J
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))

end # module MatyasProblemModule

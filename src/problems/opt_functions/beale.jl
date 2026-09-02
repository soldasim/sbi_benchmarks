"""
    BealeProblem()

A 2D problem with the Beale simulator.

Simulator: y = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²

Global minimum of 0 at (3, 0.5). The three terms create a narrow curving
valley running roughly along x ≈ 3, with the valley floor dropping sharply
to the minimum. The response surface is smooth but highly anisotropic, with
very different curvatures along and across the valley.

With z_obs = 5.0 and std_obs = 1.0 the posterior concentrates in the valley
region where the function is near 5.
"""
@kwdef struct BealeProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    std_obs::Vector{Float64} = [1.0]
end

set_gradients(p::BealeProblem, val::Bool) = BealeProblem(val, p.std_obs)


module BealeProblemModule

import ..BealeProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const z_obs   = [5.0]
const std_obs = [1.0]
const x_true  = [1.057710, -0.879811]   # same reference point as BealeProxyProblem

# --- API ---

simulator(p::BealeProblem) = p.gradients ? _sim_grads : _sim
domain(::BealeProblem)    = Domain(; bounds = ([-4.5, -4.5], [4.5, 4.5]))
likelihood(p::BealeProblem) = NormalLikelihood(; z_obs, std_obs=p.std_obs)
prior_mean(::BealeProblem) = z_obs
x_prior(::BealeProblem)    = Product([Uniform(-4.5, 4.5), Uniform(-4.5, 4.5)])
est_amplitude(::BealeProblem)      = [50.0]
est_noise_std(::BealeProblem)      = nothing
est_grad_noise_std(::BealeProblem) = nothing
true_f(::BealeProblem)     = _f
true_params(::BealeProblem) = x_true

# --- Simulator ---

function _f(x)
    a = 1.5  - x[1] + x[1]*x[2]
    b = 2.25 - x[1] + x[1]*x[2]^2
    c = 2.625 - x[1] + x[1]*x[2]^3
    return [a^2 + b^2 + c^2]
end

function _J(x)
    a = 1.5  - x[1] + x[1]*x[2]
    b = 2.25 - x[1] + x[1]*x[2]^2
    c = 2.625 - x[1] + x[1]*x[2]^3
    da_dx1 = -1 + x[2];  da_dx2 = x[1]
    db_dx1 = -1 + x[2]^2; db_dx2 = 2*x[1]*x[2]
    dc_dx1 = -1 + x[2]^3; dc_dx2 = 3*x[1]*x[2]^2
    J = zeros(1, 2)
    J[1, 1] = 2*a*da_dx1 + 2*b*db_dx1 + 2*c*dc_dx1
    J[1, 2] = 2*a*da_dx2 + 2*b*db_dx2 + 2*c*dc_dx2
    return J
end

_sim       = (x) -> _f(x)
_sim_grads = (x) -> (_f(x), _J(x))

end # module BealeProblemModule

"""
    RosenbrockProblem(; x_dim=2)

A d-dimensional problem with the Rosenbrock simulator.

Simulator: y = Σᵢ₌₁ᵈ⁻¹ [100(xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)²]  (scalar output)

The global minimum is 0 at x = (1,...,1). The response surface is a narrow
banana-shaped valley — flat along the valley, steeply curved across it — which
violates the stationarity assumptions of a standard GP. This makes the choice of
proxy variable (log transform, clipping, etc.) highly impactful.

With z_obs = 1 and Gaussian likelihood, the posterior concentrates along the
banana-shaped level set {f(x) ≈ 1}, yielding a curved, non-Gaussian posterior.
"""
@kwdef struct RosenbrockProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    x_dim::Int = 2
    std_obs::Vector{Float64} = [0.5]
end

set_gradients(p::RosenbrockProblem, val::Bool) = RosenbrockProblem(val, p.x_dim, p.std_obs)

get_name(p::RosenbrockProblem) = (p |> typeof |> string) * string(p.x_dim)


module RosenbrockProblemModule

import ..RosenbrockProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const x_true   = [0.000000, 0.000000]
const x_true_5 = [0.0, 0.0, 0.0, 0.0, 0.0]

# --- API ---

simulator(p::RosenbrockProblem) = p.gradients ? _sim_grads(p.x_dim) : _sim(p.x_dim)
domain(p::RosenbrockProblem)    = Domain(; bounds = (fill(-2.0, p.x_dim), fill(2.0, p.x_dim)))
_z_obs(p::RosenbrockProblem) = true_f(p)(true_params(p))
likelihood(p::RosenbrockProblem) = NormalLikelihood(; z_obs=_z_obs(p), std_obs=p.std_obs)
prior_mean(p::RosenbrockProblem) = _z_obs(p)
x_prior(p::RosenbrockProblem)   = Product(fill(Uniform(-2.0, 2.0), p.x_dim))
est_amplitude(::RosenbrockProblem)      = [100.0]
est_noise_std(::RosenbrockProblem)      = nothing
est_grad_noise_std(::RosenbrockProblem) = nothing
true_f(p::RosenbrockProblem)    = _true_f(p.x_dim)
function true_params(p::RosenbrockProblem)
    p.x_dim == 2 && return x_true
    p.x_dim == 5 && return x_true_5
    error("true_params not defined for $(p.x_dim)D $(typeof(p))")
end

# --- Simulator ---

function _f(x)
    d = length(x)
    y = 0.0
    for i in 1:(d-1)
        y += 100.0 * (x[i+1] - x[i]^2)^2 + (1.0 - x[i])^2
    end
    return [y]
end

function _J(x)
    d = length(x)
    J = zeros(1, d)
    for i in 1:d
        if i < d
            J[1, i] += -400.0 * x[i] * (x[i+1] - x[i]^2) + 2.0 * (x[i] - 1.0)
        end
        if i > 1
            J[1, i] += 200.0 * (x[i] - x[i-1]^2)
        end
    end
    return J
end

_sim(d) = (x) -> _f(x)
_sim_grads(d) = (x) -> (_f(x), _J(x))
_true_f(d) = (x) -> _f(x)

end # module RosenbrockProblemModule

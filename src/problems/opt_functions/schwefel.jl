"""
    SchwefelProblem(; x_dim=2)

A d-dimensional problem with the Schwefel simulator.

Simulator: y = 418.9829d - Σᵢ xᵢ sin(√|xᵢ|)

The global minimum is ≈ 0 at x = (420.9687, ..., 420.9687). The deceptive
property of Schwefel is that the second-best local minimum is far from the
global minimum (in the opposite corner of the domain), making it highly
misleading for local search. The oscillatory sin(√|x|) term creates a
complex landscape with many local minima.

With z_obs = 500.0 and std_obs = 80.0 the posterior is multimodal with
modes spread across multiple local-minima regions.
"""
@kwdef struct SchwefelProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    x_dim::Int = 2
    std_obs::Vector{Float64} = [80.0]
end

set_gradients(p::SchwefelProblem, val::Bool) = SchwefelProblem(val, p.x_dim, p.std_obs)

get_name(p::SchwefelProblem) = (p |> typeof |> string) * string(p.x_dim)


module SchwefelProblemModule

import ..SchwefelProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const x_true  = [-289.074269, 70.869565]
const x_true_5 = [-289.074269, 70.869565, 200.0, -150.0, 350.0]

# --- API ---

simulator(p::SchwefelProblem) = p.gradients ? _sim_grads(p.x_dim) : _sim(p.x_dim)
domain(p::SchwefelProblem)    = Domain(; bounds = (fill(-500.0, p.x_dim), fill(500.0, p.x_dim)))
_z_obs(p::SchwefelProblem) = true_f(p)(true_params(p))
likelihood(p::SchwefelProblem) = NormalLikelihood(; z_obs=_z_obs(p), std_obs=p.std_obs)
prior_mean(p::SchwefelProblem) = _z_obs(p)
x_prior(p::SchwefelProblem)   = Product(fill(Uniform(-500.0, 500.0), p.x_dim))
est_amplitude(::SchwefelProblem)      = [840.0]
est_noise_std(::SchwefelProblem)      = nothing
est_grad_noise_std(::SchwefelProblem) = nothing
true_f(p::SchwefelProblem)    = _true_f(p.x_dim)
function true_params(p::SchwefelProblem)
    p.x_dim == 2 && return x_true
    p.x_dim == 5 && return x_true_5
    error("true_params not defined for $(p.x_dim)D $(typeof(p))")
end

# --- Simulator ---

function _f(x)
    d = length(x)
    return [418.9829 * d - sum(xi * sin(sqrt(abs(xi))) for xi in x)]
end

function _J(x)
    d = length(x)
    J = zeros(1, d)
    for j in 1:d
        xj = x[j]
        if abs(xj) < 1e-12
            J[1, j] = 0.0
        else
            sqrtabs = sqrt(abs(xj))
            # d/dxⱼ[xⱼ sin(√|xⱼ|)] = sin(√|xⱼ|) + sign(xⱼ)·√|xⱼ|/2·cos(√|xⱼ|)
            J[1, j] = -(sin(sqrtabs) + sign(xj) * sqrtabs / 2.0 * cos(sqrtabs))
        end
    end
    return J
end

_sim(d)       = (x) -> _f(x)
_sim_grads(d) = (x) -> (_f(x), _J(x))
_true_f(d)    = (x) -> _f(x)

end # module SchwefelProblemModule

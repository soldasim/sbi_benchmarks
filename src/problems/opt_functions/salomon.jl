"""
    SalomonProblem(; x_dim=2)

A d-dimensional problem with the Salomon simulator.

Simulator: y = 1 - cos(2π √(Σxᵢ²)) + 0.1 √(Σxᵢ²)

Radially symmetric with a global minimum of 0 at the origin. Concentric
rings of local minima expand outward; the 0.1r term gradually raises the
baseline with distance. The radial symmetry means the level sets of the
function are exact circles/spheres, producing ring/sphere-shaped posterior
distributions that test the ability of the surrogate to model symmetric
multimodal posteriors.

With z_obs = 1.0 and std_obs = 0.1 the posterior concentrates on concentric
rings where the function value is near 1.0.
"""
@kwdef struct SalomonProblem <: AbstractOptFunctionProblem
    gradients::Bool = false
    x_dim::Int = 2
    std_obs::Vector{Float64} = [0.1]
end

set_gradients(p::SalomonProblem, val::Bool) = SalomonProblem(val, p.x_dim, p.std_obs)

get_name(p::SalomonProblem) = (p |> typeof |> string) * string(p.x_dim)


module SalomonProblemModule

import ..SalomonProblem
import ..simulator; import ..domain; import ..likelihood; import ..prior_mean
import ..x_prior; import ..est_amplitude; import ..est_noise_std
import ..est_grad_noise_std; import ..true_f; import ..true_params; import ..reference_samples; import ..y_max

using BOSS; using BOSIP; using Distributions

const x_true  = [7.961685, 4.034466]
const x_true_5 = [6.0, 4.0, 3.0, -4.0, 2.0]

# --- API ---

simulator(p::SalomonProblem) = p.gradients ? _sim_grads(p.x_dim) : _sim(p.x_dim)
domain(p::SalomonProblem)    = Domain(; bounds = (fill(-10.0, p.x_dim), fill(10.0, p.x_dim)))
_z_obs(p::SalomonProblem) = true_f(p)(true_params(p))
likelihood(p::SalomonProblem) = NormalLikelihood(; z_obs=_z_obs(p), std_obs=p.std_obs)
prior_mean(p::SalomonProblem) = _z_obs(p)
x_prior(p::SalomonProblem)   = Product(fill(Uniform(-10.0, 10.0), p.x_dim))
est_amplitude(::SalomonProblem)      = [3.0]
est_noise_std(::SalomonProblem)      = nothing
est_grad_noise_std(::SalomonProblem) = nothing
true_f(p::SalomonProblem)    = _true_f(p.x_dim)
function true_params(p::SalomonProblem)
    p.x_dim == 2 && return x_true
    p.x_dim == 5 && return x_true_5
    error("true_params not defined for $(p.x_dim)D $(typeof(p))")
end

# --- Simulator ---

function _f(x)
    r = sqrt(sum(xi^2 for xi in x))
    return [1.0 - cos(2π * r) + 0.1 * r]
end

function _J(x)
    d = length(x)
    r = sqrt(sum(xi^2 for xi in x))
    J = zeros(1, d)
    if r > 1e-12
        df_dr = 2π * sin(2π * r) + 0.1
        for j in 1:d
            J[1, j] = df_dr * x[j] / r
        end
    end
    return J
end

_sim(d)       = (x) -> _f(x)
_sim_grads(d) = (x) -> (_f(x), _J(x))
_true_f(d)    = (x) -> _f(x)

end # module SalomonProblemModule

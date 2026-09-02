"""
    DuffingProblem5()

5-parameter Duffing oscillator problem for simulation-based inference.

All 5 physical parameters of the forced Duffing equation are inferred:
    ẍ + δẋ + αx + βx³ = γcos(ωt)

- δ: damping coefficient        ∈ [0.05, 1.0]
- α: linear stiffness           ∈ [-2.0, 2.0]
- β: nonlinear stiffness        ∈ [0.05, 2.0]
- γ: driving amplitude          ∈ [0.1, 1.5]
- ω: driving frequency (rad/s)  ∈ [π, 4π]

Uses 5 equally-spaced displacement observations at t = 2, 4, 6, 8, 10.
Reference parameters match DuffingProblem (δ,α,β) with the previously
fixed values γ = 0.65, ω = 2π used as the ground truth.
"""
@kwdef struct DuffingProblem5 <: AbstractProblem
    gradients::Bool = false
end

set_gradients(p::DuffingProblem5, val::Bool) = DuffingProblem5(val)


module DuffingModule5

import ..DuffingProblem5

import ..simulator
import ..domain
import ..likelihood
import ..prior_mean
import ..x_prior
import ..est_amplitude
import ..est_noise_std
import ..est_grad_noise_std
import ..true_f
import ..true_params
import ..reference_samples

using BOSS
using BOSIP
using Distributions
using DifferentialEquations
using JLD2
using ForwardDiff


# --- API ---

simulator(p::DuffingProblem5) = p.gradients ? _get_model_target_with_grads() : _get_model_target()

domain(::DuffingProblem5) = Domain(;
    bounds = _get_bounds(),
)

likelihood(::DuffingProblem5) = NormalLikelihood(; z_obs, std_obs)

prior_mean(::DuffingProblem5) = _get_prior_mean()

x_prior(::DuffingProblem5) = _get_trunc_x_prior()

est_amplitude(::DuffingProblem5) = _get_est_amplitude()

est_noise_std(::DuffingProblem5) = nothing
est_grad_noise_std(::DuffingProblem5) = fill(1., n_obs)

true_f(::DuffingProblem5) = _get_model_target()
true_params(::DuffingProblem5) = x_ref
reference_samples(::DuffingProblem5) = nothing


# --- UTILS ---

# Reference parameters: δ, α, β, γ, ω
# δ/α/β match DuffingProblem; γ/ω are the previously fixed values.
const x_ref = [0.15, -1.0, 0.5, 0.65, 2π]

# Observation parameters
const t_span = (0.0, 10.0)
const t_transient = 0.0
const save_freq = 0.05
const n_obs = 5   # observations at t = 2, 4, 6, 8, 10

# Noise parameters
const std_obs = fill(0.1, n_obs)

"""
Duffing oscillator ODE system with all 5 parameters inferred.
p = [δ, α, β, γ, ω]
"""
function _duffing5_ode!(du, u, p, t)
    δ, α, β, γ, ω = p
    x, x_dot = u
    du[1] = x_dot
    du[2] = -δ*x_dot - α*x - β*x^3 + γ*cos(ω*t)
end

function _get_model_target()
    function model_target(x_)
        sol = _duffing5_simulation(x_)
        positions, _, _ = _extract_measurements(sol)
        return positions
    end
end

function _get_model_target_with_grads()
    f = _get_model_target()
    function model_target_with_grads(x_)
        y = f(x_)
        J = ForwardDiff.jacobian(f, x_)
        return y, J
    end
end

function _duffing5_simulation(x_)
    u0 = [0.0, 0.0]
    prob = ODEProblem(_duffing5_ode!, u0, t_span, x_)
    return solve(prob, Tsit5(), saveat=save_freq)
end

function _extract_measurements(sol)
    t_indices = findall(t -> t > t_transient, sol.t)
    positions  = [sol.u[i][1] for i in t_indices]
    velocities = [sol.u[i][2] for i in t_indices]
    step = length(t_indices) ÷ n_obs
    @assert step > 0
    obs_indices = step:step:(step*n_obs)
    positions  = positions[obs_indices[1:n_obs]]
    velocities = velocities[obs_indices[1:n_obs]]
    times      = sol.t[obs_indices[1:n_obs]]
    return positions, velocities, times
end

function _generate_reference_data()
    sol = _duffing5_simulation(x_ref)
    positions, _, _ = _extract_measurements(sol)
    return positions
end

const z_obs = _generate_reference_data()

_get_bounds() = (
    [0.05, -2.0, 0.05, 0.10,  π],
    [1.00,  2.0, 2.00, 1.50, 4π],
)

_get_prior_mean() = z_obs

_get_est_amplitude() = fill(2.0, n_obs)

function _get_trunc_x_prior()
    prior  = _get_x_prior()
    bounds = _get_bounds()
    return truncated(prior; lower=bounds[1], upper=bounds[2])
end

function _get_x_prior()
    return Product([
        Normal(0.10, 0.50),  # δ: damping coefficient
        Normal(0.00, 1.00),  # α: linear stiffness
        Normal(0.00, 1.00),  # β: nonlinear stiffness
        Normal(0.65, 0.50),  # γ: driving amplitude
        Normal(2π,   π),     # ω: driving frequency
    ])
end

end # module DuffingModule5

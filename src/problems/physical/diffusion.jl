"""
    DiffusionProblem()

The 1D diffusion equation problem for simulation-based inference.

The diffusion equation is a partial differential equation:
    ∂u/∂t = D ∂²u/∂x² + S(x,t)

where:
- u(x,t): concentration field
- D: diffusion coefficient (to be inferred)
- S(x,t): source term

This implementation solves the 1D diffusion equation with:
- Initial condition: Gaussian pulse at the center
- Boundary conditions: zero flux at boundaries
- Source term: S(x,t) = A*exp(-((x-x_s)²)/(2σ_s²)) for t_s < t < t_s + Δt, 0 otherwise

The parameters to infer are [x_s, t_s, A] where:
- x_s: source location
- t_s: source activation time
- A: source amplitude

The source starts radiating at unknown time t_s at unknown location x_s with unknown amplitude A and continues for a duration Δt,
where t_s ∈ [-10.0, -0.5], x_s ∈ [-5.0, 5.0], and A ∈ [0.0, 3.0].

The observations are concentration measurements at uniformly spaced locations at t=10.
"""
struct DiffusionProblem <: AbstractProblem end

module DiffusionModule

import ..DiffusionProblem

import ..simulator
import ..domain
import ..y_max
import ..likelihood
import ..prior_mean
import ..x_prior
import ..est_amplitude
import ..est_noise_std
import ..true_f
import ..reference_samples

using BOSS
using BOSIP
using Distributions
using DifferentialEquations


# --- API ---

simulator(::DiffusionProblem) = diffusion_simulation

domain(::DiffusionProblem) = Domain(;
    bounds = _get_bounds(),
)

likelihood(::DiffusionProblem) = NormalLikelihood(; z_obs, std_obs)

prior_mean(::DiffusionProblem) = _get_prior_mean()

x_prior(::DiffusionProblem) = _get_trunc_x_prior()

est_amplitude(::DiffusionProblem) = _get_est_amplitude()

# TODO noise
est_noise_std(::DiffusionProblem) = nothing

true_f(::DiffusionProblem) = x -> diffusion_simulation(x)


# --- UTILS ---

# Spatial discretization
const x_grid = collect(range(-5., 5.; length=101))
const nx = length(x_grid)               # number of spatial points
const L = (x_grid[end] - x_grid[begin]) # domain length
const dx = L / (nx - 1)                 # spatial step size

# Time parameters
const t_span = (-10, 0.0)  # simulation time span
const save_freq = 0.1      # time step for saving solution
const t_grid = collect(t_span[1]:save_freq:t_span[2])

# Fixed parameters
const D = 2.0           # diffusion coefficient (fixed)

# Source parameters
const Δt_source = 0.5   # source duration
const σ_s = 0.5         # source width

# Observation parameters
const obs_locations = collect(-4.5:1:4.5)   # spatial observation points
const n_obs = length(obs_locations)

# Noise parameters
const std_obs = fill(0.02, n_obs)  # observation noise

# [x_s, t_s, A]: position, activation time, and amplitude of the source
const x_ref = [-3.0, -4.0, 1.0]

"""
    diffusion_pde!(du, u, p, t)

1D diffusion PDE discretized in space using finite differences.
u: concentration field at spatial grid points
p = [x_s, t_s, A]: parameters (source location, activation time, and amplitude)
"""
function diffusion_pde!(du, u, p, t)
    x_s, t_s, A = p
    
    # Interior points (finite difference for second derivative)
    for i in 2:(nx-1)
        d2u_dx2 = (u[i+1] - 2*u[i] + u[i-1]) / dx^2
        # Source is active from t_s to t_s + Δt_source
        source = (t_s < t < t_s + Δt_source) ? A * exp(-((x_grid[i] - x_s)^2) / (2*σ_s^2)) : 0.0
        du[i] = D * d2u_dx2 + source
    end
    
    # Boundary conditions (zero flux: du/dx = 0)
    du[1] = du[2]     # left boundary
    du[nx] = du[nx-1] # right boundary
end

"""
    diffusion_simulation(x)

Simulate the 1D diffusion equation with parameters x = [x_s, t_s, A].
Returns observations at specified spatial locations and time points.
"""
function diffusion_simulation(x)
    x_s, t_s, A = x
    
    # Initial condition: Gaussian pulse at center
    u0 = 0.1 * exp.(-((x_grid .- L/2).^2) / (2*1.0^2))
    
    # Set up and solve PDE
    prob = ODEProblem(diffusion_pde!, u0, t_span, [x_s, t_s, A])
    sol = solve(prob, Tsit5(), saveat=save_freq)
    
    # Extract observations at specified locations at final time
    obs_indices = [findfirst(x -> abs(x - loc) < dx/2, x_grid) for loc in obs_locations]
    final_time_idx = length(sol.t)  # last time point
    
    y = Float64[]
    for space_idx in obs_indices
        push!(y, sol.u[final_time_idx][space_idx])
    end
    
    return y
end

"""
Generate reference observation data with known parameters.
"""
function _generate_reference_data()
    return diffusion_simulation(x_ref)
end

# Generate synthetic observation data
const z_obs = _generate_reference_data()

"""
Parameter bounds: [x_s_min, t_s_min, A_min], [x_s_max, t_s_max, A_max]
"""
_get_bounds() = ([-5.0, -10.0, 0.0], [5.0, -0.5, 3.0])

"""
Prior mean based on typical parameter values.
"""
_get_prior_mean() = z_obs

"""
Estimated amplitude for each observation dimension.
"""
_get_est_amplitude() = fill(1.0, n_obs)

"""
Truncated prior distribution for parameters.
"""
function _get_trunc_x_prior()
    prior = _get_x_prior()
    bounds = _get_bounds()
    return truncated(prior; lower=bounds[1], upper=bounds[2])
end

"""
Prior distribution:
- x_s ~ Uniform (source location can vary across domain)
- t_s ~ Uniform (source activation time in the past)
- A ~ LogNormal (source amplitude is positive)
"""
function _get_x_prior()
    # the priors are truncated to the domain bounds automatically
    return Product([
        Uniform(-5.0, 5.0),   # x_s: source location
        Normal(-1.0, 5.0),    # t_s: source activation time (closer to 0 more probable)
        LogNormal(0., 1.),    # A: source amplitude
    ])
end

end # module DiffusionModule

"""
    DiffusionProblem()

The 1D diffusion equation problem for simulation-based inference.

The diffusion equation is a partial differential equation:
    ∂u/∂t = D (∂²u/∂x² + ∂²u/∂y²) + S(x,y,t)

where:
- u(x,y,t): concentration field
- D: diffusion coefficient (fixed)
- S(x,y,t): source term

This implementation solves the 2D diffusion equation with:
- Initial condition: zero concentration everywhere
- Boundary conditions: zero flux at boundaries
- Source term: S(x,y,t) = A*exp(-((x-x_s)² + (y-y_s)²)/(2σ_s²)) for t_s < t < t_s + Δt, 0 otherwise

The parameters to infer are [x_s, y_s, t_s, A] where:
- x_s: source x-location
- y_s: source y-location
- t_s: source activation time
- A: source amplitude

The source starts radiating at unknown time t_s at unknown location (x_s, y_s) with unknown amplitude A and continues for a duration Δt,
where t_s ∈ [-10.0, -0.5], x_s ∈ [-5.0, 5.0], y_s ∈ [-5.0, 5.0], and A ∈ [0.0, 3.0].

The observations are concentration measurements at 3 locations arranged in a triangle at t=0.
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

simulator(::DiffusionProblem) = model_target

domain(::DiffusionProblem) = Domain(;
    bounds = _get_bounds(),
)

likelihood(::DiffusionProblem) = NormalLikelihood(; z_obs, std_obs)

prior_mean(::DiffusionProblem) = _get_prior_mean()

x_prior(::DiffusionProblem) = _get_trunc_x_prior()

est_amplitude(::DiffusionProblem) = _get_est_amplitude()

# TODO noise
est_noise_std(::DiffusionProblem) = nothing

true_f(::DiffusionProblem) = model_target


# --- UTILS ---

# Spatial discretization
const x_grid = collect(range(-5., 5.; length=21))  # reduced for 2D
const y_grid = collect(range(-5., 5.; length=21))  # reduced for 2D
const nx = length(x_grid)               # number of x grid points
const ny = length(y_grid)               # number of y grid points
const Lx = (x_grid[end] - x_grid[begin]) # x domain length
const Ly = (y_grid[end] - y_grid[begin]) # y domain length
const dx = Lx / (nx - 1)                 # x spatial step size
const dy = Ly / (ny - 1)                 # y spatial step size

# Time parameters
const t_span = (-10, 0.0)  # simulation time span
const save_freq = 0.1      # time step for saving solution

# Fixed parameters
const D = 1.0           # diffusion coefficient (fixed)

# Source parameters
const Δt_source = 0.5   # source duration
const σ_s = 0.5         # source width

# Observation parameters (3 points in triangular arrangement)
const obs_points = [(-3.0, -2.0), (3.0, -2.0), (0.0, 3.0)]  # triangular observation points
const n_obs = length(obs_points)  # total observations (3)

# Assert that observation points are exactly at grid points
for (x_loc, y_loc) in obs_points
    @assert x_loc in x_grid "Observation x-coordinate $x_loc not in grid"
    @assert y_loc in y_grid "Observation y-coordinate $y_loc not in grid"
end

# Noise parameters
const std_obs = fill(1e-2, n_obs)  # observation noise

# [x_s, y_s, t_s, A]: position, activation time, and amplitude of the source
const x_ref = [-3.0, 2.0, -4.0, 1.0]



"""
    diffusion_pde!(du, u, p, t)

2D diffusion PDE discretized in space using finite differences.
u: concentration field at spatial grid points (flattened 2D array)
p = [x_s, y_s, t_s, A]: parameters (source location, activation time, and amplitude)
"""
function diffusion_pde!(du, u, p, t)
    x_s, y_s, t_s, A = p
    
    # Initialize all derivatives to zero
    fill!(du, 0.0)
    
    # Check if source is active
    source_active = (t_s <= t < t_s + Δt_source)
    
    # Process all grid points
    for j in 1:ny, i in 1:nx
        idx = (j-1)*nx + i
        
        # Compute diffusion terms with zero-flux boundary conditions
        d2u_dx2 = 0.0
        d2u_dy2 = 0.0
        
        # X-direction second derivative
        if i == 1  # Left boundary
            d2u_dx2 = 0.0  # Zero flux
        elseif i == nx  # Right boundary  
            d2u_dx2 = 0.0  # Zero flux
        else  # Interior
            d2u_dx2 = (u[idx+1] - 2*u[idx] + u[idx-1]) / dx^2
        end
        
        # Y-direction second derivative
        if j == 1  # Bottom boundary
            d2u_dy2 = 0.0  # Zero flux
        elseif j == ny  # Top boundary
            d2u_dy2 = 0.0  # Zero flux
        else  # Interior
            d2u_dy2 = (u[idx+nx] - 2*u[idx] + u[idx-nx]) / dy^2
        end
        
        # Source term
        source = 0.0
        if source_active
            r_squared = (x_grid[i] - x_s)^2 + (y_grid[j] - y_s)^2
            source = A * exp(-r_squared / (2*σ_s^2))
        end
        
        # Update derivative
        du[idx] = D * (d2u_dx2 + d2u_dy2) + source
    end
end

"""
    model_target(x)

The diffusion problem simulator together with the mapping to the model target variable.
"""
function model_target(x)
    sol = diffusion_simulation(x)
    observations = extract_measurements(sol)
    return observations
end

"""
    diffusion_simulation(x)

Simulate the 2D diffusion equation with parameters x = [x_s, y_s, t_s, A].
Returns the full ODE solution.
"""
function diffusion_simulation(x)
    x_s, y_s, t_s, A = x
    
    # Initial condition: zero concentration at all grid points
    u0 = zeros(Float64, nx * ny)
    
    # Set up and solve PDE
    # only start simulation when source activates
    t_span_ = max(t_span[1], t_s), t_span[2]
    prob = ODEProblem(diffusion_pde!, u0, t_span_, [x_s, y_s, t_s, A])
    sol = solve(prob, Tsit5(), saveat=save_freq)
    
    return sol
end

"""
    extract_measurements(sol)

Extract concentration measurements from the simulation solution at specified observation points.
"""
function extract_measurements(sol)
    # Extract observations at specified 2D locations at final time
    final_time_idx = length(sol.t)  # last time point
    final_state = sol.u[final_time_idx]
    
    y = Float64[]
    for (x_loc, y_loc) in obs_points
        # Get exact grid indices (since points are asserted to be on grid)
        i = findfirst(==(x_loc), x_grid)
        j = findfirst(==(y_loc), y_grid)
        idx = (j-1)*nx + i
        push!(y, final_state[idx])
    end
    
    return y
end

"""
Generate reference observation data with known parameters.
"""
function _generate_reference_data()
    sol = diffusion_simulation(x_ref)
    observations = extract_measurements(sol)
    return observations
end

# Generate synthetic observation data
const z_obs = _generate_reference_data()

"""
Parameter bounds: [x_s_min, y_s_min, t_s_min, A_min], [x_s_max, y_s_max, t_s_max, A_max]
"""
_get_bounds() = ([-5.0, -5.0, -10.0, 0.0], [5.0, 5.0, -0.5, 3.0])

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
- x_s ~ Uniform (source x-location can vary across domain)
- y_s ~ Uniform (source y-location can vary across domain)
- t_s ~ Normal (source activation time in the past)
- A ~ LogNormal (source amplitude is positive)
"""
function _get_x_prior()
    # the priors are truncated to the domain bounds automatically
    return Product([
        Uniform(-5.0, 5.0),   # x_s: source x-location
        Uniform(-5.0, 5.0),   # y_s: source y-location
        Normal(-1.0, 5.0),    # t_s: source activation time (closer to 0 more probable)
        LogNormal(0., 1.),    # A: source amplitude
    ])
end

end # module DiffusionModule

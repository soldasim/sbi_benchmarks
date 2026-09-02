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

The observations are integrated concentration measurements at 3 locations arranged in a triangle over configurable time intervals during the full simulation time (-10, 0). Reference data uses 10-second intervals (entire simulation), while model outputs use 2-second intervals by default.
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

# likelihood(::DiffusionProblem) = NormalLikelihood(; z_obs, std_obs)
likelihood(::DiffusionProblem) = _get_likelihood()

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

# Measurement intervals
const observation_interval = 10.0  # interval length for reference data (entire simulation)
const model_interval = 10.0         # interval length for model outputs

# Simulation fidelity
const sim_step = 0.05

function _get_likelihood()
    ratio = Int(observation_interval / model_interval)
    sum_lengths = fill(ratio, n_obs_locs * n_obs_times)
    return SumNormalLikelihood(; sum_lengths, z_obs, std_obs)
end

# Fixed parameters
const D = 1.0           # diffusion coefficient (fixed)

# Source parameters
const Δt_source = 0.5   # source duration
const σ_s = 0.5         # source width

# Observation parameters (3 points in triangular arrangement)
const obs_points = [(-3.0, -2.0), (3.0, -2.0), (0.0, 3.0)]  # triangular observation points
const n_obs_locs = length(obs_points)  # total observations for reference data (3 points × 1 interval = 3)
const n_obs_times = Int((t_span[2] - t_span[1]) / observation_interval)  # number of observation intervals
const n_modeled_times = Int((t_span[2] - t_span[1]) / model_interval)  # number of 2-second intervals (5)

# Assert that observation points are exactly at grid points
for (x_loc, y_loc) in obs_points
    @assert x_loc in x_grid "Observation x-coordinate $x_loc not in grid"
    @assert y_loc in y_grid "Observation y-coordinate $y_loc not in grid"
end

# Noise parameters
const std_obs = fill(1e-2, n_obs_locs)  # observation noise

# [x_s, y_s, t_s, A]: position, activation time, and amplitude of the source
const x_ref = [1.5, 1.5, -4.0, 1.0]


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
    model_target(x;)

The diffusion problem simulator together with the mapping to the model target variable.
Passes keyword arguments through to extract_measurements.
"""
function model_target(x;)
    sol = diffusion_simulation(x)
    observations = extract_measurements(sol; interval_length=model_interval)
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
    # start the simulation when source activates
    t_span_ = (max(t_span[1], t_s), t_span[2])
    prob = ODEProblem(diffusion_pde!, u0, t_span_, [x_s, y_s, t_s, A])
    sol = solve(prob, Tsit5(), saveat=sim_step)
    
    return sol
end

"""
    extract_measurements(sol; interval_length)

Extract concentration measurements from the simulation solution at specified observation points.
Integrates concentration over specified intervals using trapezoidal integration.
"""
function extract_measurements(sol; interval_length)
    n_intervals_local = Int((t_span[2] - t_span[1]) / interval_length)
    n_total = n_intervals_local * length(obs_points)
    y = zeros(Float64, n_total)
    
    # Pre-compute observation point indices
    obs_indices = [(findfirst(==(x), x_grid), findfirst(==(y), y_grid)) for (x, y) in obs_points]
    
    for interval_idx in 1:n_intervals_local
        # Define time bounds
        t_start = t_span[1] + (interval_idx - 1) * interval_length
        t_end = t_start + interval_length

        # concentrations before simulation start are zero and can be skipped from integration
        (t_end <= sol.t[begin]) && continue
        t_start = max(t_start, sol.t[begin])
        
        # Find time indices, handle edge cases
        start_idx = findfirst(t -> t >= t_start, sol.t)
        end_idx = findfirst(t -> t >= t_end, sol.t)
        @assert !isnothing(t_start)
        end_idx = something(end_idx, length(sol.t)+1)
        end_idx -= 1
        
        # Integrate for each observation point
        for (obs_idx, (i, j)) in enumerate(obs_indices)
            grid_idx = (j-1)*nx + i
            integrated = 0.0
            
            # ADD initial trapezoid segment from t_start to first integrated interval
            if start_idx == 1
                @assert t_start == sol.t[start_idx]
            else
                c1, c2 = sol.u[start_idx-1][grid_idx], sol.u[start_idx][grid_idx]
                integrated += 0.5 * (c1 + c2) * (sol.t[start_idx] - t_start)
            end

            for t_idx in start_idx:end_idx
                dt = sol.t[t_idx+1] - sol.t[t_idx]
                c1, c2 = sol.u[t_idx][grid_idx], sol.u[t_idx+1][grid_idx]
                integrated += 0.5 * (c1 + c2) * dt
            end

            # SUBSTRACT final trapezoid segment from t_end to end of last integrated interval
            if end_idx == length(sol.t)
                @assert t_end == sol.t[end_idx]
            else
                c1, c2 = sol.u[end_idx][grid_idx], sol.u[end_idx+1][grid_idx]
                integrated -= 0.5 * (c1 + c2) * (sol.t[end_idx+1] - t_end)
            end

            y[(interval_idx-1)*length(obs_points) + obs_idx] = integrated
        end
    end
    
    return y
end

"""
Generate reference observation data with known parameters.
Uses interval_length=10.0 to measure over the entire simulation period.
"""
function _generate_reference_data()
    sol = diffusion_simulation(x_ref)
    observations = extract_measurements(sol; interval_length=observation_interval)
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
function _get_prior_mean()
    ratio = Int(observation_interval / model_interval)
    μ = similar(z_obs, length(z_obs) * ratio)
    for i in eachindex(z_obs)
        idx = range((i-1)*ratio + 1, i*ratio)
        μ[idx] .= z_obs[i] / ratio
    end
    return μ
end

"""
Estimated amplitude for each observation dimension.
"""
_get_est_amplitude() = fill(1. * model_interval, n_obs_locs * n_modeled_times)

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
        Normal(-2.0, 5.0),    # t_s: source activation time (closer to 0 more probable)
        LogNormal(0., 0.5),    # A: source amplitude
    ])
end

end # module DiffusionModule

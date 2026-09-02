"""
    DiffusionProblem5D()

5-parameter advection-diffusion problem for simulation-based inference.

Two physical parameters are added to the original 3D DiffusionProblem:
- x_s:   source x-location          ∈ [-5.0,  5.0]
- y_s:   source y-location          ∈ [-5.0,  5.0]
- t_s:   source activation time     ∈ [-10.0, -0.5]
- D:     diffusion coefficient      ∈ [0.01,  0.20]
- v_max: wind amplitude             ∈ [0.10,  2.00]

Observations: 3 sensor locations × 5 integration windows of 2 time units
= 15 observations total (vs. 3 in DiffusionProblem10).

Reference parameters match DiffusionProblem (x_s, y_s, t_s) with the
previously fixed values D = 0.05, v_max = 0.8 used as ground truth.
"""
@kwdef struct DiffusionProblem5D <: AbstractProblem
    gradients::Bool = false
end

set_gradients(p::DiffusionProblem5D, val::Bool) = DiffusionProblem5D(val)


module DiffusionModule5D

import ..DiffusionProblem5D

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
using ForwardDiff


# --- API ---

simulator(p::DiffusionProblem5D) = _get_model_target()

domain(::DiffusionProblem5D) = Domain(; bounds = _get_bounds())

likelihood(::DiffusionProblem5D) = LogNormalLikelihood(; log_z_obs = log.(z_obs), CV)

prior_mean(::DiffusionProblem5D) = log.(z_obs)

x_prior(::DiffusionProblem5D) = _get_trunc_x_prior()

est_amplitude(::DiffusionProblem5D) = _get_est_amplitude()

est_noise_std(::DiffusionProblem5D) = nothing
est_grad_noise_std(::DiffusionProblem5D) = fill(1., n_obs)

true_f(::DiffusionProblem5D) = _get_model_target()
true_params(::DiffusionProblem5D) = x_ref
reference_samples(::DiffusionProblem5D) = nothing


# --- UTILS ---

# Spatial grid (same as DiffusionProblem)
const x_grid = collect(range(-5., 5.; length=51))
const y_grid = collect(range(-5., 5.; length=51))
const nx = length(x_grid)
const ny = length(y_grid)
const Lx = x_grid[end] - x_grid[begin]
const Ly = y_grid[end] - y_grid[begin]
const dx = Lx / (nx - 1)
const dy = Ly / (ny - 1)

# Time parameters
const t_span = (-10.0, 0.0)
const sim_step = 0.1

# Observation setup: 3 sensors × 5 windows of 2 time units = 15 observations
const obs_points = [(-2.0, -2.0), (2.0, -2.0), (0.0, 1.4)]
const n_obs_locs = length(obs_points)
const model_interval = 2.0
const n_intervals = Int((t_span[2] - t_span[1]) / model_interval)  # = 5
const n_obs = n_obs_locs * n_intervals  # = 15

for (x_loc, y_loc) in obs_points
    @assert x_loc in x_grid
    @assert y_loc in y_grid
end

# Noise
const CV = fill(0.1, n_obs)

# Fixed physical constants
const A_adv = 10.0
const Δt_source = 0.5
const σ_source = 0.5
const A_source = 100.0
const INIT_CONCENTRATION = 1e-2
const MIN_MEASURED_VALUE_PER_TIME_UNIT = 1e-2 / 10.
const wind_scale = 2.0

# Unit wind field (v_max = 1); scaled by the inferred v_max at simulation time
const unit_v_x_field = [sin(wind_scale * y_grid[j] / 5.0) * cos(wind_scale * x_grid[i] / 5.0)
                        for j in 1:ny, i in 1:nx]
const unit_v_y_field = [-cos(wind_scale * y_grid[j] / 5.0) * sin(wind_scale * x_grid[i] / 5.0)
                        for j in 1:ny, i in 1:nx]

# Reference parameters: x_s, y_s, t_s match DiffusionProblem; D, v_max are the previously fixed values
const x_ref = [1.5, 1.5, -4.0, 0.05, 0.8]

"""
5-parameter advection-diffusion PDE.
p = [x_s, y_s, t_s, D, v_max]
"""
function diffusion5d_pde!(du, u, p, t)
    x_s, y_s, t_s, D_, v_max_ = p

    fill!(du, 0.0)
    source_active = (t_s <= t < t_s + Δt_source)

    for j in 1:ny, i in 1:nx
        idx = (j-1)*nx + i

        # Diffusion (zero-flux BCs)
        d2u_dx2 = if i == 1
            (u[idx+1] - 2*u[idx] + u[idx]) / dx^2
        elseif i == nx
            (u[idx] - 2*u[idx] + u[idx-1]) / dx^2
        else
            (u[idx+1] - 2*u[idx] + u[idx-1]) / dx^2
        end

        d2u_dy2 = if j == 1
            (u[idx+nx] - 2*u[idx] + u[idx]) / dy^2
        elseif j == ny
            (u[idx] - 2*u[idx] + u[idx-nx]) / dy^2
        else
            (u[idx+nx] - 2*u[idx] + u[idx-nx]) / dy^2
        end

        v_x = v_max_ * unit_v_x_field[j, i]
        v_y = v_max_ * unit_v_y_field[j, i]

        # Advection (upwind, zero-flux BCs)
        du_dx = if i == 1
            v_x >= 0 ? zero(eltype(u)) : (u[idx+1] - u[idx]) / dx
        elseif i == nx
            v_x >= 0 ? (u[idx] - u[idx-1]) / dx : zero(eltype(u))
        else
            v_x >= 0 ? (u[idx] - u[idx-1]) / dx : (u[idx+1] - u[idx]) / dx
        end

        du_dy = if j == 1
            v_y >= 0 ? zero(eltype(u)) : (u[idx+nx] - u[idx]) / dy
        elseif j == ny
            v_y >= 0 ? (u[idx] - u[idx-nx]) / dy : zero(eltype(u))
        else
            v_y >= 0 ? (u[idx] - u[idx-nx]) / dy : (u[idx+nx] - u[idx]) / dy
        end

        source = if source_active
            r2 = (x_grid[i] - x_s)^2 + (y_grid[j] - y_s)^2
            A_source * exp(-r2 / (2*σ_source^2))
        else
            zero(eltype(u))
        end

        du[idx] = D_ * (d2u_dx2 + d2u_dy2) - A_adv * (v_x * du_dx + v_y * du_dy) + source
    end
end

function _diffusion5d_simulation(x_)
    x_s, y_s, t_s = x_[1], x_[2], x_[3]
    u0 = INIT_CONCENTRATION * ones(eltype(x_), nx * ny)
    t_span_ = (max(t_span[1], t_s), t_span[2])
    prob = ODEProblem(diffusion5d_pde!, u0, t_span_, x_)
    return solve(prob, Tsit5(), saveat=sim_step)
end

function _extract_measurements(sol)
    obs_indices = [(findfirst(==(x), x_grid), findfirst(==(y), y_grid)) for (x, y) in obs_points]
    y = zeros(eltype(sol.u[1]), n_obs)

    for interval_idx in 1:n_intervals
        t_start = t_span[1] + (interval_idx - 1) * model_interval
        t_end   = t_start + model_interval

        (t_end <= sol.t[begin]) && continue
        t_start = max(t_start, sol.t[begin])

        start_idx = findfirst(t -> t >= t_start, sol.t)
        end_idx   = findfirst(t -> t >= t_end,   sol.t)
        @assert !isnothing(start_idx)
        end_idx = something(end_idx, length(sol.t)+1) - 1

        for (obs_idx, (i, j)) in enumerate(obs_indices)
            grid_idx = (j-1)*nx + i
            integrated = 0.0

            if start_idx > 1
                c1, c2 = sol.u[start_idx-1][grid_idx], sol.u[start_idx][grid_idx]
                integrated += 0.5 * (c1 + c2) * (sol.t[start_idx] - t_start)
            end
            for t_idx in start_idx:end_idx
                dt = sol.t[t_idx+1] - sol.t[t_idx]
                c1, c2 = sol.u[t_idx][grid_idx], sol.u[t_idx+1][grid_idx]
                integrated += 0.5 * (c1 + c2) * dt
            end
            if end_idx < length(sol.t)
                c1, c2 = sol.u[end_idx][grid_idx], sol.u[end_idx+1][grid_idx]
                integrated -= 0.5 * (c1 + c2) * (sol.t[end_idx+1] - t_end)
            end

            y[(obs_idx-1)*n_intervals + interval_idx] = integrated
        end
    end

    min_val = MIN_MEASURED_VALUE_PER_TIME_UNIT * model_interval
    return max.(min_val, y)
end

function _get_model_target()
    function model_target(x_)
        sol = _diffusion5d_simulation(x_)
        return log.(_extract_measurements(sol))
    end
end

function _generate_reference_data()
    sol = _diffusion5d_simulation(x_ref)
    return _extract_measurements(sol)
end

const z_obs = _generate_reference_data()

_get_bounds() = (
    [-5.0, -5.0, -10.0, 0.01, 0.10],
    [ 5.0,  5.0,  -0.5, 0.20, 2.00],
)

_get_est_amplitude() = fill(log(A_source * model_interval), n_obs)

function _get_trunc_x_prior()
    prior  = _get_x_prior()
    bounds = _get_bounds()
    return truncated(prior; lower=bounds[1], upper=bounds[2])
end

function _get_x_prior()
    return Product([
        Uniform(-5.0,  5.0),    # x_s
        Uniform(-5.0,  5.0),    # y_s
        Normal(-2.0,   5.0),    # t_s
        Normal(0.05,  0.05),    # D: diffusion coefficient
        Normal(0.80,  0.50),    # v_max: wind amplitude
    ])
end

end # module DiffusionModule5D

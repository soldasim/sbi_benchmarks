using CairoMakie
using Printf

include("main.jl")

function animate_diffusion_sim(x=DiffusionModule.x_ref; framerate=10, filename="diffusion_animation.gif", use_log_scale=false)
    sol = DiffusionModule.diffusion_simulation(x)
    
    # Get grid parameters from DiffusionModule
    x_grid = DiffusionModule.x_grid
    y_grid = DiffusionModule.y_grid
    nx = DiffusionModule.nx
    ny = DiffusionModule.ny
    obs_points = DiffusionModule.obs_points
    
    x_s, y_s, t_s = x
    
    # Calculate global max for consistent colorscale
    global_max = maximum(maximum.(sol.u))
    global_max = max(global_max, 1e-8)  # avoid zero
    global_min = use_log_scale ? 1e-10 : 0.0  # minimum value for scale
    
    # Create figure and axis
    fig = Figure(size=(600, 600))
    ax = Axis(fig[1, 1], 
             xlabel="x", ylabel="y",
             aspect=1.0,
             title="t = "*@sprintf("%.2f", sol.t[1]))
    
    # Initialize with first time step
    u_2d = reshape(sol.u[1], nx, ny)
    # Check for negative values and clamp them
    if any(u_2d .< 0)
        @warn "Found negative concentrations in frame 1: $(count(u_2d .< 0)) points, min = $(minimum(u_2d))"
        u_2d = max.(u_2d, 0.0)
    end
    
    # Apply scaling based on user choice
    if use_log_scale
        u_2d_scaled = log10.(u_2d .+ global_min)
        colorrange_val = (log10(global_min), log10(global_max + global_min))
        colorbar_label = "log₁₀(Concentration)"
    else
        u_2d_scaled = u_2d
        colorrange_val = (global_min, global_max)
        colorbar_label = "Concentration"
    end
    
    hm = heatmap!(ax, x_grid, y_grid, u_2d_scaled, 
            colormap = :plasma,
            colorrange = colorrange_val)
    
    obs_x = [pt[1] for pt in obs_points]
    obs_y = [pt[2] for pt in obs_points]
    scatter!(ax, obs_x, obs_y, color=:white, markersize=12, 
            strokecolor=:black, strokewidth=2)
    
    # Add source location marker (static)
    scatter!(ax, [x_s], [y_s], color=:red, marker=:star5, markersize=18,
            strokecolor=:white, strokewidth=2)
    
    # Add colorbar with appropriate labels
    cb = Colorbar(fig[1, 2], hm, label=colorbar_label)
    # Add custom ticks for better readability
    if use_log_scale
        log_ticks = [log10(global_min), (log10(global_min) + log10(global_max + global_min))/2, log10(global_max + global_min)]
        cb.ticks = log_ticks
    end
    
    # Add overall title with parameters
    fig[0, :] = Label(fig, "Advection-Diffusion: Source at ($x_s, $y_s), t_s=$t_s", 
                     fontsize=16)
    
    # Create animation
    record(fig, filename, 1:length(sol.t); framerate=framerate) do frame
        # Update heatmap data with chosen scaling
        u_2d = reshape(sol.u[frame], nx, ny)
        # Check for negative values and clamp them
        if any(u_2d .< 0)
            @warn "Found negative concentrations in frame $frame: $(count(u_2d .< 0)) points, min = $(minimum(u_2d))"
            u_2d = max.(u_2d, 0.0)
        end
        
        # Apply scaling based on user choice
        if use_log_scale
            u_2d_scaled = log10.(u_2d .+ global_min)
        else
            u_2d_scaled = u_2d
        end
        hm[3] = u_2d_scaled  # Update the heatmap data
        
        # Update title with current time
        ax.title = "t = "*@sprintf("%.2f", sol.t[frame])
    end
    
    println("Animation saved to: $filename")
    return filename
end

# Keep the original function as well for static plots
function plot_diffusion_sim(x; n_plots=5, use_log_scale=false)
    sol = DiffusionModule.diffusion_simulation(x)
    
    # Get grid parameters from DiffusionModule
    x_grid = DiffusionModule.x_grid
    y_grid = DiffusionModule.y_grid
    nx = DiffusionModule.nx
    ny = DiffusionModule.ny
    obs_points = DiffusionModule.obs_points
    
    fig = Figure(size=(240 * n_plots, 800))
    
    # Plot at several time points
    if n_plots == 1
        time_indices = [length(sol.t)]  # Only final time
    elseif n_plots == 2
        time_indices = [1, length(sol.t)]  # First and last
    else
        # Distribute time points evenly across the simulation
        step = (length(sol.t) - 1) ÷ (n_plots - 1)
        time_indices = [1 + i*step for i in 0:(n_plots-1)]
        time_indices[end] = length(sol.t)  # Ensure we get the final time
    end
    
    x_s, y_s, t_s = x
    
    for (idx, t_idx) in enumerate(time_indices)
        ax = Axis(fig[1, idx], 
                 title="t = $(round(sol.t[t_idx], digits=2))",
                 xlabel="x", ylabel="y",
                 aspect=1.0)
        
        # Reshape flattened solution to 2D grid
        u_2d = reshape(sol.u[t_idx], nx, ny)
        # Check for negative values
        if any(u_2d .< 0)
            @warn "Found negative concentrations: $(count(u_2d .< 0)) points, min = $(minimum(u_2d))"
            # Clamp to zero or add debugging
            u_2d = max.(u_2d, 0.0)
        end
        
        # Create heatmap with chosen scaling
        global_max_static = maximum(maximum.(sol.u))
        global_max_static = max(global_max_static, 1e-8) # avoid zero
        global_min_static = use_log_scale ? 1e-10 : 0.0  # minimum value for scale
        
        # Apply scaling based on user choice
        if use_log_scale
            u_2d_scaled = log10.(u_2d .+ global_min_static)
            colorrange_val = (log10(global_min_static), log10(global_max_static + global_min_static))
            colorbar_label = "log₁₀(Concentration)"
        else
            u_2d_scaled = u_2d
            colorrange_val = (global_min_static, global_max_static)
            colorbar_label = "Concentration"
        end
        
        hm = heatmap!(ax, x_grid, y_grid, u_2d_scaled, 
                colormap = :plasma,
                colorrange = colorrange_val)
        
        # Add observation points
        obs_x = [pt[1] for pt in obs_points]
        obs_y = [pt[2] for pt in obs_points]
        scatter!(ax, obs_x, obs_y, color=:white, markersize=10, 
                strokecolor=:black, strokewidth=2)
        
        # Add source location marker
        scatter!(ax, [x_s], [y_s], color=:red, marker=:star5, markersize=15,
                strokecolor=:white, strokewidth=1)
        
        # Add colorbar for the last subplot
        if idx == length(time_indices)
            cb = Colorbar(fig[1, idx+1], hm, label=colorbar_label)
            # Add custom ticks for better readability (only for log scale)
            if use_log_scale
                log_ticks = [log10(global_min_static), (log10(global_min_static) + log10(global_max_static + global_min_static))/2, log10(global_max_static + global_min_static)]
                cb.ticks = log_ticks
            end
        end
    end
    
    # Add overall title with parameters
    fig[0, :] = Label(fig, "Advection-Diffusion: Source at ($x_s, $y_s), t_s=$t_s", 
                     fontsize=16)
    
    # Display the figure
    display(fig)
    return fig
end

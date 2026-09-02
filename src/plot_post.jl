using CairoMakie

include("main.jl")

# Set larger font sizes for better readability
function set_theme_fonts!(; base_fontsize=20)
    set_theme!(
        fontsize = base_fontsize,
        Axis = (
            titlesize = base_fontsize + 2,
            xlabelsize = base_fontsize,
            ylabelsize = base_fontsize,
            xticklabelsize = base_fontsize - 2,
            yticklabelsize = base_fontsize - 2,
        ),
        Legend = (
            titlesize = base_fontsize,
            labelsize = base_fontsize - 1,
        ),
        Label = (
            textsize = base_fontsize,
        )
    )
end

function plot_all_posteriors(problems=nothing; base_fontsize=20, save_plot=false, resolution=50)
    if isnothing(problems)
        problems = [ABProblem(), SimpleProblem(), BananaProblem(), BimodalProblem()]
    end
    plot_type = :auto

    # Set theme with larger fonts
    set_theme_fonts!(base_fontsize=base_fontsize)
    
    # Calculate dimensions for single row layout
    n_cols = length(problems)
    base_size = 300  # Base size per subplot
    fig_width = n_cols * base_size
    fig_height = base_size
    
    # Create main figure
    fig = Figure(size=(fig_width, fig_height))
    
    # Reduce gaps between subplots
    colgap!(fig.layout, 10)  # Small gap between columns
    
    # Plot each problem in a single row
    for (col_idx, problem) in enumerate(problems)
        try
            # Create axis directly in the grid and use plot_posterior with axis
            labels = get_param_labels(problem)
            title = "$(typeof(problem) |> nameof |> string)"[1:end-7] # remove "Problem"
            
            # Create new axis in the grid
            new_ax = Axis(fig[1, col_idx],
                xlabel = labels[1],
                ylabel = labels[2],
                title = title,
                aspect = AxisAspect(1),  # Keep plots square
                titlesize = base_fontsize,
                xlabelsize = base_fontsize - 2,
                ylabelsize = base_fontsize - 2,
                xticklabelsize = base_fontsize - 4,
                yticklabelsize = base_fontsize - 4
            )
            
            # Use plot_posterior with the axis (only works for 2D problems)
            if x_dim(problem) == 2
                plot_posterior(problem; axis=new_ax, plot_type=:twodim, base_fontsize=base_fontsize, resolution=resolution)
            else
                # For n-dimensional problems, show a message
                text!(new_ax, 0.5, 0.5, 
                      text="$(x_dim(problem))D Problem\n$(title)", 
                      align=(:center, :center),
                      fontsize=base_fontsize-2,
                      space=:relative)
            end
            
        catch e
            # If plotting fails, show error message
            Label(fig[1, col_idx], 
                  "Error: $(typeof(problem) |> nameof |> string)\n$e", 
                  fontsize=base_fontsize-4,
                  halign=:center, valign=:center)
            @warn "Failed to plot $(typeof(problem)): $e"
        end
    end
    
    # Save if requested
    if save_plot
        save("all_posteriors.png", fig)
        println("Saved plot as all_posteriors.png")
    end
    
    return fig
end



# plot_type = :auto, :twodim, :ndim
function plot_posterior(problem::AbstractProblem; axis=nothing, plot_type=:auto, base_fontsize=20, resolution=50)
    # Set theme with larger fonts (only if creating new figure)
    if axis === nothing
        set_theme_fonts!(base_fontsize=base_fontsize)
    end
    
    post = reference(problem)
    @assert post isa Function

    x_dim_ = x_dim(problem)

    (plot_type == :auto) && (plot_type = x_dim_ == 2 ? :twodim : :ndim)
    
    if plot_type == :twodim
        @assert x_dim_ == 2
        # Get bounds and labels from the problem
        bounds = domain(problem).bounds
        labels = get_param_labels(problem)
        title = "$(typeof(problem) |> nameof |> string)"[1:end-7] # remove "Problem"
        
        # Use plot_2d_posterior with optional axis parameter
        return plot_2d_posterior(post; axis=axis, bounds=bounds, labels=labels, title=title, resolution=resolution)
    elseif plot_type == :ndim
        # Get bounds and labels from the problem
        bounds = domain(problem).bounds
        labels = get_param_labels(problem)
        title = "$(typeof(problem) |> nameof |> string)"[1:end-7] # remove "Problem"
        return plot_nd_posterior(post; bounds=bounds, labels=labels, title=title, resolution=resolution)
    else
        error("Unsupported plot_type: $plot_type")
    end
end

# Helper function to get parameter labels for different problems
function get_param_labels(problem::AbstractProblem)
    # Define parameter labels for known problem types
    problem_type = typeof(problem) |> nameof |> string
    
    if occursin("Duffing", problem_type)
        return ["δ", "α", "β"]
    elseif occursin("Diffusion", problem_type)
        return ["xₛ", "yₛ", "tₛ"]
    elseif occursin("SIR", problem_type)
        return ["β", "γ"]
    elseif occursin("AB", problem_type)
        return ["a", "b"]
    else
        # Fallback to generic labels
        n = x_dim(problem)
        return ["x$i" for i in 1:n]
    end
end

function plot_2d_posterior(post::Function; 
                          axis=nothing,
                          bounds=([-5.0, -5.0], [5.0, 5.0]), 
                          labels=["Parameter 1", "Parameter 2"],
                          resolution=100,
                          title="2D Posterior",
                          colormap=:viridis,
                          levels=20,
                          base_fontsize=20)
    
    # Set theme with larger fonts only when creating new figure
    if axis === nothing
        set_theme_fonts!(base_fontsize=base_fontsize)
    end
    
    lb, ub = bounds
    @assert length(lb) == 2 && length(ub) == 2 "Bounds must be 2D for 2D posterior plot"
    @assert length(labels) == 2 "Must provide exactly 2 parameter labels"
    
    # Create point vectors for triangulated contour
    x_points = Float64[]
    y_points = Float64[]
    z_log = Float64[]
    
    # Create regular grid and evaluate posterior
    x_range = range(lb[1], ub[1], length=resolution)
    y_range = range(lb[2], ub[2], length=resolution)
    
    # Evaluate log-posterior and store as points
    for xi in x_range
        for yi in y_range
            log_val = post([xi, yi])
            push!(x_points, xi)
            push!(y_points, yi)
            push!(z_log, log_val)
        end
    end
    
    # Convert to probability density (subtract maximum for numerical stability)
    valid_mask = z_log .> -Inf
    if any(valid_mask)
        M = maximum(z_log[valid_mask])
        z_prob = exp.(z_log .- M)
        z_prob[.!valid_mask] .= 0.0
    else
        z_prob = zeros(length(z_log))
        @warn "No valid posterior values found - all log probabilities are -Inf"
    end
    
    if axis === nothing
        # Create new figure and axis
        fig = Figure(size=(600, 600))  # Make figure square
        ax = Axis(fig[1, 1],
            xlabel = labels[1],
            ylabel = labels[2],
            title = title,
            aspect = AxisAspect(1)  # Keep plots square
        )
        limits!(ax, lb[1], ub[1], lb[2], ub[2])  # Set exact domain bounds
        
        # Plot triangulated contour
        if any(z_prob .> 0)
            tricontourf!(ax, x_points, y_points, z_prob, colormap=colormap, levels=levels)
        else
            @warn "No valid posterior values to plot - all probabilities are zero"
        end
        
        return fig
    else
        # Use provided axis
        ax = axis
        ax.xlabel = labels[1]
        ax.ylabel = labels[2]
        ax.aspect = AxisAspect(1)  # Keep plots square, consistent with new figure case
        limits!(ax, lb[1], ub[1], lb[2], ub[2])  # Set exact domain bounds
        
        # Plot triangulated contour
        if any(z_prob .> 0)
            tricontourf!(ax, x_points, y_points, z_prob, colormap=colormap, levels=levels)
        else
            @warn "No valid posterior values to plot - all probabilities are zero"
        end
        
        return ax
    end
end

function plot_nd_posterior(post::Function;
                          bounds=nothing,
                          labels=nothing,
                          resolution=50,
                          title="N-D Posterior",
                          colormap=:viridis,
                          levels=15,
                          base_fontsize=16,
                          marginal_method=:mean)
    
    # Set theme with larger fonts
    set_theme_fonts!(base_fontsize=base_fontsize)
    
    lb, ub = bounds
    n = length(lb)
    @assert length(ub) == n "Lower and upper bounds must have same dimension"
    @assert length(labels) == n "Must provide exactly $n parameter labels"
    
    # Create an n×n grid of plots (marginals on diagonal, 2D slices off-diagonal)
    fig_size = min(1200, max(600, n * 200))  # Scale figure size with dimensions
    fig = Figure(size=(fig_size, fig_size))
    
    # Calculate marginal posteriors for all pairs of dimensions (full matrix)
    # Create all off-diagonal plots
    for i in 1:n
        for j in 1:n
            if i == j
                continue  # Skip diagonal, will handle separately
            end
            
            dim1, dim2 = j, i  # Transpose: x-axis follows column (j), y-axis follows row (i)
        
        # Create 2D marginal by integrating/averaging over all other dimensions
        other_dims = setdiff(1:n, [dim1, dim2])
        
        # Create grid for the two dimensions of interest
        x_range = range(lb[dim1], ub[dim1], length=resolution)
        y_range = range(lb[dim2], ub[dim2], length=resolution)
        
        # Create ranges for other dimensions (fewer points for marginalization)
        marg_resolution = max(5, resolution÷4)  # Adaptive resolution for marginalization
        other_ranges = [range(lb[d], ub[d], length=marg_resolution) for d in other_dims]
        
        x_points = Float64[]
        y_points = Float64[]
        z_log = Float64[]
        
        # Evaluate posterior on grid and marginalize over other dimensions
        for xi in x_range
            for yi in y_range
                # Marginalize over all other dimensions
                marginal_vals = Float64[]
                
                # Generate all combinations of other dimension values
                if isempty(other_dims)
                    # No marginalization needed (n=2 case)
                    params = zeros(n)
                    params[dim1] = xi
                    params[dim2] = yi
                    log_val = post(params)
                    if log_val > -Inf
                        push!(marginal_vals, exp(log_val))
                    end
                else
                    # Use Cartesian product for all other dimensions
                    for other_vals in Iterators.product(other_ranges...)
                        # Create parameter vector with correct ordering
                        params = zeros(n)
                        params[dim1] = xi
                        params[dim2] = yi  
                        
                        # Assign values for other dimensions
                        for (idx, d) in enumerate(other_dims)
                            params[d] = other_vals[idx]
                        end
                        
                        log_val = post(params)
                        if log_val > -Inf
                            push!(marginal_vals, exp(log_val))
                        end
                    end
                end
                
                # Compute marginal value
                if !isempty(marginal_vals)
                    if marginal_method == :mean
                        marginal_prob = mean(marginal_vals)
                    elseif marginal_method == :max
                        marginal_prob = maximum(marginal_vals)
                    else  # :sum or integration
                        marginal_prob = sum(marginal_vals) * (ub[dim3] - lb[dim3]) / length(z_range)
                    end
                    
                    push!(x_points, xi)
                    push!(y_points, yi)
                    push!(z_log, log(marginal_prob + 1e-12))  # Small epsilon to avoid log(0)
                else
                    push!(x_points, xi)
                    push!(y_points, yi)
                    push!(z_log, -Inf)
                end
            end
        end
        
        # Convert to probability density (subtract maximum for numerical stability)
        valid_mask = z_log .> -Inf
        if any(valid_mask)
            M = maximum(z_log[valid_mask])
            z_prob = exp.(z_log .- M)
            z_prob[.!valid_mask] .= 0.0
        else
            z_prob = zeros(length(z_log))
        end
        
            # Create subplot at position (i,j)
            ax = Axis(fig[i, j],
                xlabel = labels[dim1],
                ylabel = labels[dim2],
                aspect = AxisAspect(1),  # Force square aspect ratio
                limits = (lb[dim1], ub[dim1], lb[dim2], ub[dim2])
            )
        
            # Plot marginal contour
            if any(z_prob .> 0)
                tricontourf!(ax, x_points, y_points, z_prob, colormap=colormap, levels=levels)
            else
                @warn "No valid posterior values for dimensions $dim1, $dim2"
            end
        end
    end
    
    # Add 1D marginals on the diagonal
    for dim in 1:n
        # Calculate 1D marginal
        x_range = range(lb[dim], ub[dim], length=resolution)
        marginal_1d = Float64[]
        
        # Other dimensions for marginalization
        other_dims = setdiff(1:n, [dim])
        
        for xi in x_range
            # Marginalize over other dimensions using a sparse grid
            sparse_resolution = max(3, resolution÷4)
            other_ranges = [range(lb[d], ub[d], length=sparse_resolution) for d in other_dims]
            
            vals = Float64[]
            if isempty(other_dims)
                # No marginalization needed (n=1 case)
                params = [xi]
                log_val = post(params)
                if log_val > -Inf
                    push!(vals, exp(log_val))
                end
            else
                # Marginalize over all other dimensions
                for other_vals in Iterators.product(other_ranges...)
                    params = zeros(n)
                    params[dim] = xi
                    
                    # Assign values for other dimensions
                    for (idx, d) in enumerate(other_dims)
                        params[d] = other_vals[idx]
                    end
                    
                    log_val = post(params)
                    if log_val > -Inf
                        push!(vals, exp(log_val))
                    end
                end
            end
            
            if !isempty(vals)
                push!(marginal_1d, mean(vals))
            else
                push!(marginal_1d, 0.0)
            end
        end
        
        # Plot 1D marginal
        ax_1d = Axis(fig[dim, dim],
            xlabel = labels[dim],
            aspect = AxisAspect(1),  # Make 1D plots same size as 2D plots
            limits = (lb[dim], ub[dim], nothing, nothing),
            yticksvisible = false,  # Hide y-axis ticks
            yticklabelsvisible = false  # Hide y-axis tick labels
        )
        
        if any(marginal_1d .> 0)
            lines!(ax_1d, collect(x_range), marginal_1d, color=:blue, linewidth=2)
        end
    end
    
    # Add overall title
    Label(fig[0, :], title, fontsize=base_fontsize + 2, font=:bold)
    
    return fig
end

# Backward compatibility alias
plot_3d_posterior(post::Function; kwargs...) = plot_nd_posterior(post; kwargs...)

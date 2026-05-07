using CairoMakie
using Statistics
using Random
include("main.jl")

function compute_grads(problem::AbstractProblem; num_samples::Int=1000)
    """
    Compute gradient statistics for a problem with arbitrary y_dim.
    
    Returns a dictionary with statistics for each output dimension:
    - mean: mean gradient for each input dimension
    - std: standard deviation of gradient for each input dimension
    - min: minimum gradient for each input dimension
    - max: maximum gradient for each input dimension
    - all_grads: all gradient values [input_dim, output_dim, sample]
    """
    sim = simulator(problem)
    xdim = x_dim(problem)
    ydim = y_dim(problem)
    
    # Get domain bounds from problem
    bounds = domain(problem).bounds
    lower = bounds[1]
    upper = bounds[2]
    
    # Sample random points from the domain
    Random.seed!(42)
    samples = rand(xdim, num_samples)
    for i in 1:xdim
        samples[i, :] .= lower[i] .+ samples[i, :] .* (upper[i] - lower[i])
    end
    
    # Compute gradients at all sample points
    grads = Array{Float64, 3}(undef, xdim, ydim, num_samples)
    
    for sample_idx in 1:num_samples
        result = sim(samples[:, sample_idx])
        for output_dim in 1:ydim
            for input_dim in 1:xdim
                idx = (output_dim - 1) * xdim + input_dim
                grads[input_dim, output_dim, sample_idx] = result[2][idx]
            end
        end
    end
    
    # Compute statistics
    stats = Dict()
    
    for output_dim in 1:ydim
        stats[output_dim] = Dict(
            :mean => [mean(grads[input_dim, output_dim, :]) for input_dim in 1:xdim],
            :std => [std(grads[input_dim, output_dim, :]) for input_dim in 1:xdim],
            :min => [minimum(grads[input_dim, output_dim, :]) for input_dim in 1:xdim],
            :max => [maximum(grads[input_dim, output_dim, :]) for input_dim in 1:xdim],
            :all_grads => grads[:, output_dim, :]
        )
    end
    
    return stats
end

function meshgrid(x::AbstractVector, y::AbstractVector)
    X = repeat(x', length(y), 1)
    Y = repeat(y, 1, length(x))
    return X, Y
end

function plot_grads(problem::AbstractProblem)
    @assert x_dim(problem) == 2 "Plotting only implemented for 2D problems."
    
    sim = simulator(problem)
    xdim = x_dim(problem)
    ydim = y_dim(problem)
    
    # Get domain bounds from problem
    bounds = domain(problem).bounds
    lower = bounds[1]
    upper = bounds[2]

    x = range(lower[1], upper[1], length=100)
    y = range(lower[2], upper[2], length=100)
    X, Y = meshgrid(x, y)
    
    # Compute all function values and gradients once
    Z_all = Array{Float64, 3}(undef, size(X, 1), size(X, 2), ydim)
    Z_grads_all = [Matrix{Float64}(undef, size(X)) for _ in 1:xdim, _ in 1:ydim]
    
    for i in 1:size(X, 1), j in 1:size(X, 2)
        result = sim([X[i, j], Y[i, j]])
        for output_dim in 1:ydim
            Z_all[i, j, output_dim] = result[1][output_dim]
            for input_dim in 1:xdim
                idx = (output_dim - 1) * xdim + input_dim
                Z_grads_all[input_dim, output_dim][i, j] = result[2][idx]
            end
        end
    end
    
    # Plot each output dimension
    mkpath("plots")
    for output_dim in 1:ydim
        Z = Z_all[:, :, output_dim]
        Z_grads = [Z_grads_all[input_dim, output_dim] for input_dim in 1:xdim]
        
        # Create figure with subplots (1 for function + xdim for gradients, each with colorbar)
        fig = Figure(size=(550 * (1 + xdim), 400))
        
        ax1 = Axis(fig[1, 1]; title="Function Value (Output $output_dim)", xlabel="x", ylabel="y")
        Z_range = extrema(Z)
        Z_range = Z_range[1] == Z_range[2] ? (Z_range[1] - 1, Z_range[2] + 1) : Z_range
        hm1 = heatmap!(ax1, x, y, Z; colorrange=Z_range, colormap=:viridis)
        Colorbar(fig[1, 2], hm1; label="Value")
        
        for input_dim in 1:xdim
            ax = Axis(fig[1, 2*input_dim + 1]; title="∂f[$output_dim]/∂x[$input_dim]", xlabel="x", ylabel="y")
            Z_grad = Z_grads[input_dim]
            grad_range = extrema(Z_grad)
            grad_range = grad_range[1] == grad_range[2] ? (grad_range[1] - 1, grad_range[2] + 1) : grad_range
            hm = heatmap!(ax, x, y, Z_grad; colorrange=grad_range, colormap=:viridis)
            Colorbar(fig[1, 2*input_dim + 2], hm; label="Gradient")
        end
        
        save("plots/gradient_plot_output_$(output_dim).png", fig)
    end
end

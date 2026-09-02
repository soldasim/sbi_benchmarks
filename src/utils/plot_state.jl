module PlotModule

using BOSIP, BOSS
using CairoMakie

import ..AbstractProblem
import ..MultidimProblem
import ..GaussProblem
import ..MeanGauss
import ..ABProblem
import ..SimpleProblem
import ..BananaProblem
import ..BimodalProblem
import ..SIRProblem
import ..DuffingProblem
import ..DuffingProblem5
import ..DiffusionProblem
import ..DiffusionProblem5D
import ..ProxySIRProblem
import ..LogABProblem
import ..LogSimpleProblem
import ..LogBananaProblem
import ..LogBimodalProblem
import ..LogSIRProblem
import ..LogDuffingProblem
import ..LogDiffusionProblem
import ..reference
import ..AbstractOptFunctionProblem
import ..SharpProblem
import ..HexObsProblem
import ..CrossPolytopeObsProblem

include("../data_paths.jl")

@kwdef mutable struct PlotCB <: BosipCallback
    problem::AbstractProblem
    estimator::Function
    sampler::DistributionSampler
    sample_count::Int
    plot_each::Int = 10
    resolution::Int = 500
    save_plots::Bool = false
    iters::Int = 0
end

function (cb::PlotCB)(bosip::BosipProblem; term_cond, first, kwargs...)
    first || (cb.iters += 1)
    (cb.iters % cb.plot_each == 0) || return

    plot_state(bosip, cb.estimator, cb.problem, cb.sampler, cb.sample_count, cb.iters; cb.resolution, cb.save_plots)
end

function plot_state(bosip::BosipProblem, estimator::Function, p::AbstractProblem, sampler::DistributionSampler, sample_count::Int, iter::Int; resolution=500, save_plots=false)
    domain = bosip.problem.domain
    lb, ub = domain.bounds
    X = bosip.problem.data.X

    # TODO rem
    # # Get the observed value
    # @assert bosip.likelihood isa NormalLikelihood
    # @assert length(bosip.likelihood.z_obs) == 1
    # @assert length(bosip.likelihood.std_obs) == 1
    # z_obs = bosip.likelihood.z_obs[1]
    # std_obs = bosip.likelihood.std_obs[1]
    
    # Get model posterior for mean/std of predictions
    model_posterior = BOSS.model_posterior(bosip.problem)
    
    # Get log-posterior mean and variance
    log_post_mean = BOSIP.log_posterior_mean(bosip)
    log_post_var = BOSIP.log_posterior_variance(bosip)
    
    # Compute grid for evaluation
    x = range(lb[1], ub[1], length=resolution)
    y = range(lb[2], ub[2], length=resolution)
    
    # # Compute model mean and std
    # Z_model_mean = Matrix{Float64}(undef, resolution, resolution)
    # Z_model_std = Matrix{Float64}(undef, resolution, resolution)
    
    # # Compute posterior mean and std
    # Z_post_mean = Matrix{Float64}(undef, resolution, resolution)
    # Z_post_std = Matrix{Float64}(undef, resolution, resolution)

    # for (i, xi) in enumerate(x), (j, yi) in enumerate(y)
    #     xij = [xi, yi]
        
    #     # Model mean and std
    #     m, v = mean_and_var(model_posterior, xij)
    #     Z_model_mean[j, i] = m[1]
    #     Z_model_std[j, i] = sqrt(v[1])
        
    #     # Posterior mean and std
    #     Z_post_mean[j, i] = exp(log_post_mean(xij))
    #     Z_post_std[j, i] = sqrt(exp(log_post_var(xij)))
    # end
    
    # Batch computation
    xs = hcat([[t...] for t in Iterators.product(x, y)]...)
    M, V = mean_and_var(model_posterior, xs)
    log_post_means = log_post_mean(xs)
    log_post_vars = log_post_var(xs)

    Z_model_mean = reshape(M, (resolution, resolution))'
    Z_model_std = reshape(sqrt.(V), (resolution, resolution))'
    Z_post_mean = reshape(exp.(log_post_means), (resolution, resolution))'
    Z_post_std = reshape(sqrt.(exp.(log_post_vars)), (resolution, resolution))'
    
    # Create figure with 2x2 grid
    fig = Figure(; size = (1000, 1000))
    
    # Row 1: Model predictions
    ax1 = Axis(fig[1, 1]; xlabel="x₁", ylabel="x₂", title="Model Mean", aspect=AxisAspect(1))
    hm1 = heatmap!(ax1, y, x, Z_model_mean; colormap=:thermal)
    # contour!(ax1, y, x, Z_model_mean; levels=[z_obs], color=:lime, linewidth=2) # TODO rem
    scatter!(ax1, X[2,:], X[1,:]; color=:white, markersize=5, strokewidth=1, strokecolor=:black)
    scatter!(ax1, X[2,end], X[1,end]; color=:red, markersize=6, strokewidth=1, strokecolor=:darkred)
    Colorbar(fig[1, 2], hm1)
    
    ax2 = Axis(fig[1, 3]; xlabel="x₁", ylabel="x₂", title="Model Std", aspect=AxisAspect(1))
    hm2 = heatmap!(ax2, y, x, Z_model_std; colormap=:thermal)
    # contour!(ax2, y, x, Z_model_std; levels=[std_obs], color=:cyan, linewidth=2) # TODO rem
    scatter!(ax2, X[2,:], X[1,:]; color=:white, markersize=5, strokewidth=1, strokecolor=:black)
    scatter!(ax2, X[2,end], X[1,end]; color=:red, markersize=6, strokewidth=1, strokecolor=:darkred)
    Colorbar(fig[1, 4], hm2)
    
    # Row 2: Posterior mean and std
    ax3 = Axis(fig[2, 1]; xlabel="x₁", ylabel="x₂", title="Posterior Mean", aspect=AxisAspect(1))
    hm3 = heatmap!(ax3, y, x, Z_post_mean; colormap=:thermal)
    scatter!(ax3, X[2,:], X[1,:]; color=:white, markersize=5, strokewidth=1, strokecolor=:black)
    scatter!(ax3, X[2,end], X[1,end]; color=:red, markersize=6, strokewidth=1, strokecolor=:darkred)
    Colorbar(fig[2, 2], hm3)
    
    ax4 = Axis(fig[2, 3]; xlabel="x₁", ylabel="x₂", title="Posterior Std", aspect=AxisAspect(1))
    hm4 = heatmap!(ax4, y, x, Z_post_std; colormap=:thermal)
    # TODO rem contours
    # contour!(ax4, y, x, Z_model_mean; levels=[z_obs], color=:lime, linewidth=2)
    # contour!(ax4, y, x, Z_model_std; levels=[std_obs], color=:cyan, linewidth=2)
    scatter!(ax4, X[2,:], X[1,:]; color=:white, markersize=5, strokewidth=1, strokecolor=:black)
    scatter!(ax4, X[2,end], X[1,end]; color=:red, markersize=6, strokewidth=1, strokecolor=:darkred)
    # Create dummy lines for legend
    lines!(ax4, [NaN], [NaN]; color=:lime, linewidth=2, label="model mean == obs. value")
    lines!(ax4, [NaN], [NaN]; color=:cyan, linewidth=2, label="model std == obs. std")
    axislegend(ax4; position = :rb)
    Colorbar(fig[2, 4], hm4)
    
    if save_plots
        dir = plot_dir() * "/state_plots"
        mkpath(dir)
        save(dir * "/" * string(typeof(p)) * "_$iter.png", fig)
    else
        display(fig)
    end
end

end # module PlotModule

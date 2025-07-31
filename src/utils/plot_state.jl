module PlotModule

using BOLFI
using CairoMakie

import ..AbstractProblem
import ..reference
import ..log_posterior_estimate

include("../data_paths.jl")

@kwdef struct PlotCB <: BolfiCallback
    problem::AbstractProblem
    sampler::DistributionSampler
    sample_count::Int
    plot_each::Int = 1000
    save_plots::Bool = false
end

function (cb::PlotCB)(bolfi::BolfiProblem; term_cond, first, kwargs...)
    first && return
    (term_cond.iter % cb.plot_each == 0) || return

    plot_state(bolfi, cb.problem, cb.sampler, cb.sample_count, term_cond.iter; cb.save_plots)
end

function plot_state(bolfi::BolfiProblem, p::AbstractProblem, sampler::DistributionSampler, sample_count::Int, iter::Int; save_plots=false)
    est_logpost = log_posterior_estimate()(bolfi)

    domain = bolfi.problem.domain
    lb, ub = domain.bounds
    X = bolfi.problem.data.X

    x = range(lb[1], ub[1], length=100)
    y = range(lb[2], ub[2], length=100)
    Z = [est_logpost([xi, yi]) for xi in x, yi in y]

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title=string(log_posterior_estimate()))
    contourf!(ax, x, y, Z)

    ### sample posterior
    ref = reference(p)
    if ref isa Function
        true_samples = BOLFI.pure_sample_posterior(sampler, ref, domain, sample_count)
    else
        true_samples = ref
    end

    approx_samples = BOLFI.pure_sample_posterior(sampler, est_logpost, domain, sample_count) 
    ###

    scatter!(ax, true_samples[1, :], true_samples[2, :], color=:blue, marker=:x, markersize=4, label="Reference Samples")
    scatter!(ax, approx_samples[1, :], approx_samples[2, :], color=:red, marker=:x, markersize=4, label="Approx. Samples")

    scatter!(ax, X[1,:], X[2,:], color=:white, markersize=6)
    scatter!(ax, X[1,:], X[2,:], color=:black, markersize=4, label="Data")
    
    axislegend(ax)

    if save_plots
        dir = plot_dir() * "/state_plots"
        mkpath(dir)
        save(dir * "/" * string(typeof(p)) * "_$iter.png", fig)
    else
        display(fig)
    end
end

end # module PlotModule

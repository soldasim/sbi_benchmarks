module PlotModule

using BOLFI
using CairoMakie

import ..MetricCallback
import ..AbstractProblem
import ..reference

@kwdef struct PlotCB <: BolfiCallback
    problem::AbstractProblem
    plot_each::Int = 1
end

function (cb::PlotCB)(bolfi::BolfiProblem, metric::MetricCallback; term_cond, first, kwargs...)
    first && return
    (term_cond.iter % cb.plot_each == 0) || return

    plot_state(bolfi, cb.problem, metric)
end

function plot_state(bolfi::BolfiProblem, p::AbstractProblem, metric::MetricCallback)
    exp_post = posterior_mean(bolfi)
    bounds = bolfi.problem.domain.bounds
    X = bolfi.problem.data.X

    x = range(bounds[1][1], bounds[2][1], length=100)
    y = range(bounds[1][2], bounds[2][2], length=100)
    Z = [exp_post([xi, yi]) for xi in x, yi in y]

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title="Posterior Mean")
    contourf!(ax, x, y, Z)

    xs_ref = metric.true_samples
    scatter!(ax, xs_ref[1, :], xs_ref[2, :], color=:blue, marker=:x, markersize=4, label="Reference Samples")

    xs_approx = metric.approx_samples
    scatter!(ax, xs_approx[1, :], xs_approx[2, :], color=:red, marker=:x, markersize=4, label="Approx. Samples")

    scatter!(ax, X[1,:], X[2,:], color=:white, markersize=6)
    scatter!(ax, X[1,:], X[2,:], color=:black, markersize=4, label="Data")
    
    axislegend(ax)
    display(fig)
end

end # module PlotModule

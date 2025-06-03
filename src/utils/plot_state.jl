module PlotModule

using BOLFI
using CairoMakie

@kwdef struct PlotCB <: BolfiCallback
    plot_each::Int = 1
end

function (cb::PlotCB)(bolfi::BolfiProblem; term_cond, first, kwargs...)
    first && return
    (term_cond.iter % cb.plot_each == 0) || return

    plot_state(bolfi)
end

function plot_state(bolfi::BolfiProblem)
    exp_post = posterior_mean(bolfi)
    bounds = bolfi.problem.domain.bounds
    X = bolfi.problem.data.X

    x = range(bounds[1][1], bounds[2][1], length=100)
    y = range(bounds[1][2], bounds[2][2], length=100)
    Z = [exp_post([xi, yi]) for xi in x, yi in y]

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title="Posterior Mean")
    contourf!(ax, x, y, Z)

    scatter!(ax, X[1,:], X[2,:], color=:red, markersize=8, label="Data")
    axislegend(ax)
    
    display(fig)
end

end # module PlotModule

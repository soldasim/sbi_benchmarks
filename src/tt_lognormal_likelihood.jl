using StatsPlots

function plot_lognormal(y; CV=0.1)
    σ_log = sqrt(log(1 + CV^2))
    μ_log = log(y) - σ_log^2 / 2
    dist = LogNormal(μ_log, σ_log)

    StatsPlots.plot(dist)
end

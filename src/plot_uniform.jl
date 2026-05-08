# Generate convergence plots for the uniform (random) acquisition benchmark.
# Tests Teckentrup Thm 3.5 (L2/convergence metric) and Thm 5.1/5.2 (TV metric).

include("plots.jl")

mkpath(plot_dir())

problems_meangauss = [MeanGauss(; x_dim=d) for d in 1:6]
problems_ab = [MultidimProblem(ABProblem(), d) for d in 1:3]
all_problems = vcat(problems_meangauss, problems_ab)

groups = ["uniform"]

# TV metric (posterior convergence — Thm 5.1/5.2)
for problem in all_problems
    name = get_name(problem)
    @info "Plotting TV metric for $name ..."
    fig = plot_results([problem]; save_plot=false, metric=:tv, plotted_groups=groups)
    save(plot_dir() * "/uniform_tv_$(name).png", fig)
    save(plot_dir() * "/uniform_tv_$(name).pdf", fig)
end

# L2 convergence metric (simulator convergence — Thm 3.5)
for problem in all_problems
    name = get_name(problem)
    @info "Plotting convergence metric for $name ..."
    fig = plot_results([problem]; save_plot=false, metric=:convergence, plotted_groups=groups)
    save(plot_dir() * "/uniform_conv_$(name).png", fig)
    save(plot_dir() * "/uniform_conv_$(name).pdf", fig)
end

# Combined grid: all problems, TV metric
@info "Plotting combined TV grid ..."
fig_tv = plot_results(; save_plot=false, metric=:tv, plotted_groups=groups)
save(plot_dir() * "/uniform_tv_all.png", fig_tv)
save(plot_dir() * "/uniform_tv_all.pdf", fig_tv)

# Combined grid: all problems, convergence metric
@info "Plotting combined convergence grid ..."
fig_conv = plot_results(; save_plot=false, metric=:convergence, plotted_groups=groups)
save(plot_dir() * "/uniform_conv_all.png", fig_conv)
save(plot_dir() * "/uniform_conv_all.pdf", fig_conv)

@info "All plots saved to $(plot_dir())/"

# Generate convergence plots for the maxvar (LogMaxVar) acquisition benchmark.
# Tests Teckentrup Thm 3.5 (L2/convergence metric) and Thm 5.1/5.2 (TV metric).
# Compare against plot_uniform.jl (random acquisition).

include("plots.jl")

mkpath(plot_dir())

problems_meangauss = [MeanGauss(; x_dim=d) for d in 1:6]
problems_ab = [MultidimProblem(ABProblem(), d) for d in 1:3]
problems_square = [MultidimProblem(SquareProblem(), d) for d in 1:6]
problems_sine   = [MultidimProblem(SineProblem(),   d) for d in 1:6]
problems_cubic  = [MultidimProblem(CubicProblem(),  d) for d in 1:6]
all_problems = vcat(problems_meangauss, problems_ab, problems_square, problems_sine, problems_cubic)

groups = ["maxvar"]

# Combined grid: all problems, TV metric
@info "Plotting combined TV grid ..."
fig_tv = plot_results(; save_plot=false, metric=:tv, plotted_groups=groups, plot_individual_runs=true)
save(plot_dir() * "/maxvar_tv_all.png", fig_tv)
save(plot_dir() * "/maxvar_tv_all.pdf", fig_tv)

# Combined grid: all problems, convergence metric
@info "Plotting combined convergence grid ..."
fig_conv = plot_results(; save_plot=false, metric=:convergence, plotted_groups=groups, plot_individual_runs=true)
save(plot_dir() * "/maxvar_conv_all.png", fig_conv)
save(plot_dir() * "/maxvar_conv_all.pdf", fig_conv)

@info "All plots saved to $(plot_dir())/"

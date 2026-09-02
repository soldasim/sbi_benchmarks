# Plot TV-metric convergence results for all 24 sharp opt-function benchmark problems (4×6 grid).

include("plots.jl")

mkpath(plot_dir())

all_problems = SharpProblem.([
    RosenbrockProblem(; x_dim=2),
    StyblinskiTangProblem(; x_dim=2),
    MichalewiczProblem(; x_dim=2),
    AckleyProblem(; x_dim=2),
    AlpineProblem(; x_dim=2),
    ExpandedSchafferF6Problem(; x_dim=2),
    ExpandedZakharovProblem(; x_dim=2),
    GriewankProblem(; x_dim=2),
    RastriginProblem(; x_dim=2),
    SalomonProblem(; x_dim=2),
    SchwefelProblem(; x_dim=2),
    SphereProblem(; x_dim=2),
    BealeProxyProblem(),
    BoothProblem(),
    CrossInTrayProblem(),
    DropWaveProblem(),
    EasomProblem(),
    GoldsteinPriceProxyProblem(),
    HimmelblauProblem(),
    HolderTableProblem(),
    LeviN13Problem(),
    MatyasProblem(),
    SchafferN2Problem(),
    ThreeHumpCamelProblem(),
])

@assert length(all_problems) == 24

groups = ["maxvar", "eiv"]

@info "Plotting combined 4×6 TV-metric grid (sharp) ..."
set_theme_fonts!(; base_fontsize=16)

nrows, ncols = 4, 6
ax_width, ax_height = axis_size()
fig = Figure(; size = (ax_width * ncols, ax_height * nrows))

for (i, prob) in enumerate(all_problems)
    row = div(i - 1, ncols) + 1
    col = mod(i - 1, ncols) + 1
    plot_result_axis!(fig[row, col], [prob];
        legend = (i == 1),
        metric = :tv,
        plotted_groups = groups,
        plot_individual_runs = true,
    )
end

save(plot_dir() * "/all_opt_results_tv_grid_sharp.png", fig)
save(plot_dir() * "/all_opt_results_tv_grid_sharp.pdf", fig)
@info "Done. Saved to $(plot_dir())/all_opt_results_tv_grid_sharp.png"

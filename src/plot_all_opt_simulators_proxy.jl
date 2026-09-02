# Plot the simulator response surface f(x) (or proxy δ(x) where applicable) for all 24
# opt-function benchmark problems.  Same 4×6 grid ordering as plot_all_opt_simulators.jl,
# but Beale and GoldsteinPrice are replaced by their proxy variants so the colour shows
# δ = log(f + C) instead of the raw simulator output f.

include("main.jl")

using CairoMakie
using Random

Random.seed!(42)

all_problems = [
    # Original 3
    RosenbrockProblem(; x_dim=2),
    StyblinskiTangProblem(; x_dim=2),
    MichalewiczProblem(; x_dim=2),
    # New d-dimensional at 2D
    AckleyProblem(; x_dim=2),
    AlpineProblem(; x_dim=2),
    ExpandedSchafferF6Problem(; x_dim=2),
    ExpandedZakharovProblem(; x_dim=2),
    GriewankProblem(; x_dim=2),
    RastriginProblem(; x_dim=2),
    SalomonProblem(; x_dim=2),
    SchwefelProblem(; x_dim=2),
    SphereProblem(; x_dim=2),
    # New 2D-only (proxy variants for Beale and GoldsteinPrice)
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
]

@assert length(all_problems) == 24

# Strip "Problem" suffix and replace "Proxy" with a short indicator
function _panel_title(problem::AbstractProblem)
    s = get_name(problem)
    s = replace(s, "Proxy" => " (proxy)")
    s = replace(s, "Problem" => "")
    return s
end

function add_simulator_axis!(fig_pos, problem::AbstractProblem; grid_size=80)
    bounds = domain(problem).bounds
    lb, ub = bounds
    f = true_f(problem)

    xs1 = range(lb[1], ub[1]; length=grid_size) |> collect
    xs2 = range(lb[2], ub[2]; length=grid_size) |> collect
    zs  = [f([x1, x2])[1] for x1 in xs1, x2 in xs2]

    z_obs_val = prior_mean(problem)[1]

    ax = Axis(fig_pos;
        title      = _panel_title(problem),
        xlabel     = "x₁", ylabel = "x₂",
        titlesize  = 13,
        xlabelsize = 11, ylabelsize = 11,
        xticklabelsize = 9, yticklabelsize = 9,
    )
    heatmap!(ax, xs1, xs2, zs; colormap=:viridis)
    contour!(ax, xs1, xs2, zs; levels=[z_obs_val], color=:red, linewidth=1.5)
    return ax
end

mkpath(plot_dir())

@info "Plotting 4×6 grid of all 24 simulator response surfaces (proxy variants for Beale & GoldsteinPrice) ..."

nrows, ncols = 4, 6
cell_size = 220
fig = Figure(; size = (cell_size * ncols, cell_size * nrows))

for (i, prob) in enumerate(all_problems)
    row = div(i - 1, ncols) + 1
    col = mod(i - 1, ncols) + 1
    @info "  [$row,$col] $(_panel_title(prob)) ..."
    add_simulator_axis!(fig[row, col], prob; grid_size=80)
end

save(plot_dir() * "/all_opt_simulators_proxy_grid.png", fig)
save(plot_dir() * "/all_opt_simulators_proxy_grid.pdf", fig)
@info "Done. Saved to $(plot_dir())/all_opt_simulators_proxy_grid.png"

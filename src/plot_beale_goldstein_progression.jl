###
### 4-column × 6-row scatter progression plot for Beale and Goldstein problems.
###
### Rows:
###   1. Beale — MaxVar (no proxy)
###   2. Beale — EIV (no proxy)
###   3. Beale — MaxVar with proxy (BealeProxyProblem)
###   4. Goldstein-Price — MaxVar (no proxy)
###   5. Goldstein-Price — EIV (no proxy)
###   6. Goldstein-Price — MaxVar with proxy (GoldsteinPriceProxyProblem)
###
### Columns: snapshots at cumulative evals 25+3, 50+3, 100+3, 200+3
###
### Run from: ~/repos/bosip_benchmarks/
###   ~/.juliaup/bin/julia --project=src src/plot_beale_goldstein_progression.jl
###

using JLD2
using CairoMakie

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

const DATA_ROOT = "data-opt-functions"
const PLOT_DIR  = "plots"
const INIT_DATA = 3   # initial evaluations before BO starts

data_path(problem_dir, method, run_idx, suffix) =
    joinpath(DATA_ROOT, problem_dir, "$(method)_$(run_idx)_$(suffix).jld2")

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

"""
Load X matrix (parameter locations) for a single run.
Returns (n_params × n_evals) Float64 matrix, or nothing if not found.
"""
function load_X(problem_dir::String, method::String, run_idx::Int)
    fpath = data_path(problem_dir, method, run_idx, "data")
    isfile(fpath) || return nothing
    d = load(fpath)
    return d["data"][1]   # (X, Y)[1] = X, shape (n_params, n_evals)
end

# ─────────────────────────────────────────────
# Color / style palette
# ─────────────────────────────────────────────

# Match the palette from plot_sir_duffing_comparison.jl
const PALETTE = Dict(
    "maxvar" => Makie.wong_colors()[2],   # orange
    "eiv"    => Makie.wong_colors()[1],   # blue
    "proxy"  => Makie.wong_colors()[3],   # green  (proxy variant of maxvar)
)

# ─────────────────────────────────────────────
# Scatter panel
# ─────────────────────────────────────────────

"""
Add a single scatter panel to fig_pos.
- prob_dir:    subdirectory in DATA_ROOT (e.g. "BealeProblem")
- method:      file prefix (e.g. "maxvar", "eiv")
- palette_key: key into PALETTE for the colour
- n_eval:      total evaluations to show (INIT_DATA + snap)
- param_names: axis-label tuple, e.g. ("x₁", "x₂")
- run_idx:     which run file to load (default 1)
"""
function add_scatter_panel!(pos, prob_dir::String, method::String, palette_key::String,
                             n_eval::Int, param_names::Tuple; run_idx::Int=1)
    ax = Axis(pos;
        xlabel          = param_names[1],
        ylabel          = param_names[2],
        xticklabelsize  = 8,
        yticklabelsize  = 8,
        xlabelsize      = 9,
        ylabelsize      = 9,
    )

    X = load_X(prob_dir, method, run_idx)
    if isnothing(X)
        text!(ax, 0.5, 0.5; text="no data", align=(:center, :center),
              space=:relative, fontsize=10, color=:red)
        return ax
    end

    n_avail = size(X, 2)
    n_use   = min(n_eval, n_avail)

    xs1 = X[1, 1:n_use]
    xs2 = X[2, 1:n_use]

    col = get(PALETTE, palette_key, :gray)

    # Initial data in light gray
    n_init = min(INIT_DATA, n_use)
    scatter!(ax, xs1[1:n_init], xs2[1:n_init];
             color=:lightgray, markersize=5, strokewidth=0.3, strokecolor=:gray50)

    # BO-acquired points with colour fading from light to saturated
    if n_use > n_init
        n_bo = n_use - n_init
        for i in 1:n_bo
            frac = i / max(n_bo, 1)
            pt_col = (col, 0.3 + 0.7*frac)
            scatter!(ax, [xs1[n_init+i]], [xs2[n_init+i]];
                     color=pt_col, markersize=4)
        end
    end

    return ax
end

# ─────────────────────────────────────────────
# Build figure
# ─────────────────────────────────────────────

function plot_data_progression()
    snaps       = [25, 50, 100, 200]
    snap_labels = ["iter 25", "iter 50", "iter 100", "iter 200"]

    # (prob_dir, method, palette_key, row_label, param_names, run_idx)
    # No "true params" star: both problems have distributed/multimodal posteriors.
    groups = [
        ("BealeProblem",               "maxvar", "maxvar", "Beale MaxVar",        ("x₁", "x₂"), 1),
        ("BealeProblem",               "eiv",    "eiv",    "Beale EIV",           ("x₁", "x₂"), 1),
        ("BealeProxyProblem",          "maxvar", "proxy",  "Beale MaxVar+proxy",  ("x₁", "x₂"), 1),
        ("GoldsteinPriceProblem",      "maxvar", "maxvar", "Goldstein MaxVar",    ("x₁", "x₂"), 1),
        ("GoldsteinPriceProblem",      "eiv",    "eiv",    "Goldstein EIV",       ("x₁", "x₂"), 1),
        ("GoldsteinPriceProxyProblem", "maxvar", "proxy",  "Goldstein MaxVar+proxy", ("x₁", "x₂"), 1),
    ]

    nrows    = length(groups)
    ncols    = length(snaps) + 1   # +1 for row labels
    cell_w   = 140
    cell_h   = 140

    fig = Figure(size=(cell_w * ncols + 40, cell_h * nrows + 60))

    # Column headers
    for (ci, lbl) in enumerate(snap_labels)
        Label(fig[1, ci+1]; text=lbl, font=:bold, fontsize=11, tellwidth=false)
    end

    for (ri, (prob_dir, method, pkey, row_lbl, pnames, run_idx)) in enumerate(groups)
        Label(fig[ri+1, 1]; text=row_lbl, fontsize=9, halign=:right, tellheight=false)

        for (ci, snap) in enumerate(snaps)
            n_eval = INIT_DATA + snap
            add_scatter_panel!(fig[ri+1, ci+1], prob_dir, method, pkey,
                               n_eval, pnames; run_idx)
        end
    end

    rowgap!(fig.layout, 4)
    colgap!(fig.layout, 4)

    return fig
end

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

mkpath(PLOT_DIR)

@info "Generating Beale/Goldstein data progression figure..."
fig = plot_data_progression()
save(joinpath(PLOT_DIR, "beale_goldstein_progression.pdf"), fig)
save(joinpath(PLOT_DIR, "beale_goldstein_progression.png"), fig)
@info "Saved $(PLOT_DIR)/beale_goldstein_progression.pdf"
@info "Saved $(PLOT_DIR)/beale_goldstein_progression.png"
@info "Done."

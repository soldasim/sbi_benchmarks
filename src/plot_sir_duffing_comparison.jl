###
### Comparison plots: MaxVar (standard/LogMaxVar) vs EIV acquisitions
### on SIR and Duffing problems.
###
### Figure 1 (TV convergence): 2 rows (SIR, Duffing) × 2 columns (no proxy, with proxy)
### Figure 2 (Data progression): scatter plots at iter 25, 50, 100, 200
###
### Run from: ~/repos/bosip_benchmarks/
###   julia --project=src src/plot_sir_duffing_comparison.jl
###

using JLD2
using CairoMakie
using Statistics

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

const DATA_DIR = "data-bosip"
const PLOT_DIR = "plots"

data_path(problem, method, run_idx, suffix) =
    joinpath(DATA_DIR, problem, "$(method)_$(run_idx)_$(suffix).jld2")

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

"""
Load TV score vector for a single run.
Returns `nothing` if file not found.
"""
function load_tv_scores(problem::String, method::String, run_idx::Int)
    fpath = data_path(problem, method, run_idx, "TVmetric")
    isfile(fpath) || return nothing
    return load(fpath, "score")
end

"""
Load all TV score vectors for a method across all 20 runs.
Returns only those runs for which data exist.
"""
function load_all_tv_scores(problem::String, method::String; max_runs::Int=20)
    scores = Vector{Float64}[]
    for i in 1:max_runs
        s = load_tv_scores(problem, method, i)
        isnothing(s) || push!(scores, s)
    end
    return scores
end

"""
Load X matrix (parameter locations) for a single run.
Returns (n_params × n_evals) Float64 matrix (Julia column-major convention),
or `nothing` if not found.
Data is saved as: save(..., "data" => (X, Y)) where X is (n_params × n_evals).
"""
function load_X(problem::String, method::String, run_idx::Int)
    fpath = data_path(problem, method, run_idx, "data")
    isfile(fpath) || return nothing
    d = load(fpath)
    return d["data"][1]   # (X, Y)[1] = X, shape (n_params, n_evals)
end

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

const INIT_DATA = 3   # 3 initial evaluations before BO starts

"""
Aggregate a list of score vectors (possibly different lengths) into
(xs, medians, q10, q90) using only iterations where ≥ half the runs have data.
"""
function aggregate_scores(scores::Vector{<:AbstractVector{Float64}})
    isempty(scores) && return nothing
    maxlen = maximum(length.(scores))
    # build a matrix padded with NaN
    mat = fill(NaN, maxlen, length(scores))
    for (j, s) in enumerate(scores)
        mat[1:length(s), j] = s
    end
    xs = (INIT_DATA):(INIT_DATA + maxlen - 1)
    medians = Float64[]
    q10_vec = Float64[]
    q90_vec = Float64[]
    xs_out = Int[]
    for i in 1:maxlen
        row = filter(!isnan, mat[i, :])
        if length(row) >= max(1, length(scores) ÷ 2)
            push!(xs_out, xs[i])
            push!(medians, median(row))
            push!(q10_vec, quantile(row, 0.10))
            push!(q90_vec, quantile(row, 0.90))
        end
    end
    return xs_out, medians, q10_vec, q90_vec
end

# ─────────────────────────────────────────────
# Color / style palette
# ─────────────────────────────────────────────

# Consistent colors across all panels
const PALETTE = Dict(
    "standard"  => Makie.wong_colors()[2],   # orange
    "eiv"       => Makie.wong_colors()[1],   # blue
    "grads"     => Makie.wong_colors()[3],   # green
)

method_label(m) = Dict(
    "standard"  => "MaxVar (LogMaxVar)",
    "eiv"       => "EIV",
    "grads"     => "MaxVar + grads",
)[m]

# ─────────────────────────────────────────────
# Figure 1: TV convergence
# ─────────────────────────────────────────────

function plot_tv_convergence()
    # Layout: 2 rows (SIR, Duffing) × 2 columns (no proxy, with proxy)
    # Each cell: multiple method curves with median ± IQR band

    # Define what to plot per cell
    # (problem_dir, method_list, title)
    cells = [
        ("SIRProblem",      ["standard", "eiv"],   "SIR (no proxy)"),
        ("ProxySIRProblem", ["standard"],           "SIR (with proxy)"),
        ("DuffingProblem",  ["standard", "eiv"],   "Duffing (no proxy)"),
        ("DuffingProblem",  ["grads"],              "Duffing (with proxy / grads)"),
    ]

    fig = Figure(size=(900, 600))
    Label(fig[0, 1], "Without proxies"; font=:bold, tellwidth=false)
    Label(fig[0, 2], "With proxies"; font=:bold, tellwidth=false)
    Label(fig[1, 0], "SIR"; font=:bold, rotation=π/2, tellheight=false)
    Label(fig[2, 0], "Duffing"; font=:bold, rotation=π/2, tellheight=false)

    # row, col pairs for each cell
    positions = [(1,1), (1,2), (2,1), (2,2)]

    for (cell_idx, (prob, methods, title)) in enumerate(cells)
        r, c = positions[cell_idx]
        ax = Axis(fig[r, c];
            xlabel = "Evaluations",
            ylabel = "TV distance",
            title  = title,
            xscale = log10,
            yscale = log10,
        )

        for method in methods
            scores = load_all_tv_scores(prob, method)
            if isempty(scores)
                @warn "No data for $prob / $method"
                continue
            end
            agg = aggregate_scores(scores)
            isnothing(agg) && continue
            xs, meds, q10, q90 = agg

            col = get(PALETTE, method, :gray)
            lbl = method_label(method)
            n_runs = length(scores)
            lbl_full = "$lbl (n=$n_runs)"

            band!(ax, xs, q10, q90; color=(col, 0.20))
            lines!(ax, xs, meds; color=col, linewidth=2, label=lbl_full)
        end

        axislegend(ax; position=:lb, labelsize=10)
    end

    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 10)

    return fig
end

# ─────────────────────────────────────────────
# Figure 2: Data progression scatter plots
# ─────────────────────────────────────────────

"""
Plot scatter of parameter locations at snapshot `n_eval` for run 1
of the given method/problem.
`param_idx`: which two parameters to plot (1-indexed tuple, e.g. (1,2) for 2D SIR)
`true_params`: (x, y) for the true parameter star marker
`param_names`: axis labels
"""
function add_scatter_panel!(pos, prob, method, n_eval, param_idx, true_params, param_names;
                             run_idx=1)
    ax = Axis(pos;
        xlabel = param_names[1],
        ylabel = param_names[2],
        xticklabelsize = 8,
        yticklabelsize = 8,
        xlabelsize = 9,
        ylabelsize = 9,
    )

    X = load_X(prob, method, run_idx)
    if isnothing(X)
        text!(ax, 0.5, 0.5; text="no data", align=(:center,:center), space=:relative, fontsize=10, color=:red)
        return ax
    end

    # X is (n_params, n_evals) in Julia column-major convention
    n_avail = size(X, 2)
    n_use   = min(n_eval, n_avail)

    pi1, pi2 = param_idx
    xs1 = X[pi1, 1:n_use]
    xs2 = X[pi2, 1:n_use]

    col = get(PALETTE, method, :gray)

    # colour-code by acquisition order: init data in gray, BO points in method colour
    n_init = min(INIT_DATA, n_use)
    scatter!(ax, xs1[1:n_init], xs2[1:n_init];
             color=:lightgray, markersize=5, strokewidth=0.3, strokecolor=:black)
    if n_use > n_init
        # gradient from light to saturated to show acquisition order
        n_bo = n_use - n_init
        for i in 1:n_bo
            frac = i / max(n_bo, 1)
            pt_col = (col, 0.3 + 0.7*frac)
            scatter!(ax, [xs1[n_init+i]], [xs2[n_init+i]];
                     color=pt_col, markersize=4)
        end
    end

    # true param star
    scatter!(ax, [true_params[1]], [true_params[2]];
             marker=:star5, markersize=14, color=:red, strokewidth=1, strokecolor=:black)

    return ax
end

function plot_data_progression()
    # Snapshots (as number of BO evaluations, i.e. total = INIT_DATA + snap)
    snaps = [25, 50, 100, 200]  # total evals = INIT_DATA + snap
    snap_labels = ["iter 25", "iter 50", "iter 100", "iter 200"]

    # We need:
    # Row structure:
    #   SIR no-proxy: standard + eiv    → 2 method rows
    #   SIR proxy:    standard           → 1 method row
    #   Duffing no-proxy: standard + eiv → 2 method rows
    #   Duffing proxy:    grads           → 1 method row

    # (problem_dir, method, display_label, param_idx, param_names, true_params, run_idx)
    groups = [
        ("SIRProblem",      "standard", "SIR MaxVar",           (1,2), ("β","γ"),  (0.6148, 0.1917), 1),
        ("SIRProblem",      "eiv",      "SIR EIV",              (1,2), ("β","γ"),  (0.6148, 0.1917), 1),
        ("ProxySIRProblem", "standard", "SIR MaxVar+proxy",     (1,2), ("β","γ"),  (0.6148, 0.1917), 1),
        ("DuffingProblem",  "standard", "Duffing MaxVar",       (1,2), ("δ","α"),  (0.15, -1.0),     1),
        ("DuffingProblem",  "eiv",      "Duffing EIV",          (1,2), ("δ","α"),  (0.15, -1.0),     1),
        # Duffing grads: only runs 3,5,8,10,14,15,20 exist; use run 3
        ("DuffingProblem",  "grads",    "Duffing MaxVar+grads", (1,2), ("δ","α"),  (0.15, -1.0),     3),
    ]

    nrows = length(groups)
    ncols = length(snaps) + 1   # +1 for row labels
    cell_w, cell_h = 140, 140

    fig = Figure(size=(cell_w * (ncols) + 40, cell_h * nrows + 60))

    # Column headers
    for (ci, lbl) in enumerate(snap_labels)
        Label(fig[1, ci+1]; text=lbl, font=:bold, fontsize=11, tellwidth=false)
    end

    for (ri, (prob, method, row_lbl, pidx, pnames, true_p, run_idx)) in enumerate(groups)
        Label(fig[ri+1, 1]; text=row_lbl, fontsize=9, halign=:right, tellheight=false)

        for (ci, snap) in enumerate(snaps)
            n_eval = INIT_DATA + snap
            add_scatter_panel!(fig[ri+1, ci+1], prob, method, n_eval, pidx, true_p, pnames; run_idx)
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

@info "Generating TV convergence figure..."
fig1 = plot_tv_convergence()
save(joinpath(PLOT_DIR, "sir_duffing_tv.pdf"), fig1)
save(joinpath(PLOT_DIR, "sir_duffing_tv.png"), fig1)
@info "Saved $(PLOT_DIR)/sir_duffing_tv.pdf"

@info "Generating data progression figure..."
fig2 = plot_data_progression()
save(joinpath(PLOT_DIR, "sir_duffing_progression.pdf"), fig2)
save(joinpath(PLOT_DIR, "sir_duffing_progression.png"), fig2)
@info "Saved $(PLOT_DIR)/sir_duffing_progression.pdf"

@info "Done."

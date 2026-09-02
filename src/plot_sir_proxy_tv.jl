###
### TV metric comparison: SIR without proxy vs SIR with proxy
### Methods: GP+MaxVar (standard), GP+EIV (eiv), NonstationaryGP+MaxVar (nongp)
###
### Produces two figures (one per problem variant), saved as PDF + PNG.
###
### Run from: ~/repos/bosip_benchmarks/
###   include("src/plot_sir_proxy_tv.jl")   (from live Julia session)
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

function load_tv_scores(problem::String, method::String, run_idx::Int)
    # Prefer the recomputed normalized variant; fall back to original
    norm_path = data_path(problem, method, run_idx, "TVmetric_norm")
    fpath = isfile(norm_path) ? norm_path : data_path(problem, method, run_idx, "TVmetric")
    isfile(fpath) || return nothing
    return load(fpath, "score")
end

function load_all_tv_scores(problem::String, method::String; max_runs::Int=20)
    scores = Vector{Float64}[]
    for i in 1:max_runs
        s = load_tv_scores(problem, method, i)
        isnothing(s) || push!(scores, s)
    end
    return scores
end

# ─────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────

const INIT_DATA = 3

function aggregate_scores(scores::Vector{<:AbstractVector{Float64}})
    isempty(scores) && return nothing
    maxlen = maximum(length.(scores))
    mat = fill(NaN, maxlen, length(scores))
    for (j, s) in enumerate(scores)
        mat[1:length(s), j] = s
    end
    xs = INIT_DATA:(INIT_DATA + maxlen - 1)
    medians = Float64[]
    q10_vec = Float64[]
    q90_vec = Float64[]
    xs_out  = Int[]
    for i in 1:maxlen
        row = filter(!isnan, mat[i, :])
        if length(row) >= max(1, length(scores) ÷ 2)
            push!(xs_out,  xs[i])
            push!(medians, median(row))
            push!(q10_vec, quantile(row, 0.10))
            push!(q90_vec, quantile(row, 0.90))
        end
    end
    return xs_out, medians, q10_vec, q90_vec
end

# ─────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────

const METHODS = ["standard", "eiv", "nongp"]

const PALETTE = Dict(
    "standard" => Makie.wong_colors()[2],   # orange — GP+MaxVar
    "eiv"      => Makie.wong_colors()[1],   # blue   — GP+EIV
    "nongp"    => Makie.wong_colors()[4],   # purple — NonstationaryGP+MaxVar
)

const LABELS = Dict(
    "standard" => "GP + MaxVar",
    "eiv"      => "GP + EIV",
    "nongp"    => "NonstationaryGP + MaxVar",
)

# ─────────────────────────────────────────────
# Single-panel plot
# ─────────────────────────────────────────────

function plot_tv_panel!(ax, problem::String; max_runs::Int=20)
    for method in METHODS
        scores = load_all_tv_scores(problem, method; max_runs)
        if isempty(scores)
            @warn "No data for $problem / $method"
            continue
        end
        agg = aggregate_scores(scores)
        isnothing(agg) && continue
        xs, meds, q10, q90 = agg

        col  = PALETTE[method]
        lbl  = "$(LABELS[method]) (n=$(length(scores)))"

        band!(ax, xs, q10, q90; color=(col, 0.20))
        lines!(ax, xs, meds; color=col, linewidth=2, label=lbl)
    end
    axislegend(ax; position=:lb, labelsize=10, framevisible=true)
end

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

mkpath(PLOT_DIR)

# Figure 1 — SIR without proxy
@info "Plotting SIR (no proxy)..."
fig1 = Figure(size=(600, 420))
ax1 = Axis(fig1[1, 1];
    title  = "SIR — no proxy",
    xlabel = "Evaluations",
    ylabel = "TV distance",
    xscale = log10,
    yscale = log10,
)
plot_tv_panel!(ax1, "SIRProblem"; max_runs=20)
save(joinpath(PLOT_DIR, "sir_tv_noproxy.pdf"), fig1)
save(joinpath(PLOT_DIR, "sir_tv_noproxy.png"), fig1)
@info "Saved plots/sir_tv_noproxy.{pdf,png}"

# Figure 2 — SIR with proxy
@info "Plotting SIR (with proxy)..."
fig2 = Figure(size=(600, 420))
ax2 = Axis(fig2[1, 1];
    title  = "SIR — with proxy",
    xlabel = "Evaluations",
    ylabel = "TV distance",
    xscale = log10,
    yscale = log10,
)
plot_tv_panel!(ax2, "ProxySIRProblem"; max_runs=20)
save(joinpath(PLOT_DIR, "sir_tv_proxy.pdf"), fig2)
save(joinpath(PLOT_DIR, "sir_tv_proxy.png"), fig2)
@info "Saved plots/sir_tv_proxy.{pdf,png}"

@info "Done."

"""
Plot and precompute true posterior marginals for all 31 benchmark problems
(7 BIP + 24 optimization-function).

## Two-phase workflow

**Phase 1 — precompute** (expensive; run once on the cluster):

    include("src/plot_marginals.jl")
    precompute_all_grids()                          # all 31 problems
    precompute_all_bip_grids()                      # 7 BIP only
    precompute_bip_grid(DuffingProblem())            # single problem
    precompute_bip_grid(DiffusionProblem10(); grid_res=3, lhc_size=3)  # low-res test

Storage per problem under `data/true_posterior_grids/`:

  2-D problems — full grid:
    `xs_per_dim`   :: Vector{Vector{Float64}}   — grid axes (grid_res points each)
    `logvals`      :: Matrix{Float64}           — log-posterior on the grid_res² grid

  3-D problems — pairwise marginals via LHC:
    `xs_per_dim`     :: Vector{Vector{Float64}} — grid axes for each parameter
    `pair_dims`      :: Vector{Vector{Int}}     — e.g. [[1,2],[1,3],[2,3]]
    `pair_marginals` :: Vector{Matrix{Float64}} — log-marginal for each pair (grid_res²)

**Phase 2 — plot** (cheap; re-run freely):

    include("src/plot_marginals.jl")
    fig = plot_bip_true_posteriors()

## Grid resolution

Default: `GRID_RES = 50` (50² per panel).  For 3-D problems, `LHC_SIZE = 500` LHC
samples marginalise over the third parameter at each display-grid point.
Override per call: `precompute_bip_grid(p; grid_res=5, lhc_size=5)` for testing.
"""

include("main.jl")

using JLD2
using CairoMakie
using Random: randperm

## ── Parameters ────────────────────────────────────────────────────────────────

const GRID_RES = 50      # grid resolution per axis (2-D full grid; 3-D display grid per pair)
const LHC_SIZE = 500     # LHC samples for marginalising over extra dimensions in 3-D

const GRID_RES_OVERRIDES = Dict{String,Int}()

const TRUE_POST_GRID_DIR = "data/true_posterior_grids"

## ── Problem lists ─────────────────────────────────────────────────────────────
## (prefixed _PM_ to avoid const conflicts when classify_posteriors.jl is co-loaded)

const _PM_BIP_PROBLEMS = [
    ABProblem(),
    SimpleProblem(),
    BananaProblem(),
    BimodalProblem(),
    SIRProblem(),
    DuffingProblem(),
    DuffingProblem5(),
    DiffusionProblem10(),
    DiffusionProblem5D(),
]

const _PM_OPT_PROBLEMS = [
    RosenbrockProblem(),
    StyblinskiTangProblem(),
    MichalewiczProblem(),
    AckleyProblem(),
    AlpineProblem(),
    ExpandedSchafferF6Problem(),
    ExpandedZakharovProblem(),
    GriewankProblem(),
    RastriginProblem(),
    SalomonProblem(),
    SchwefelProblem(),
    SphereProblem(),
    BealeProblem(),
    BoothProblem(),
    CrossInTrayProblem(),
    DropWaveProblem(),
    EasomProblem(),
    GoldsteinPriceProblem(),
    HimmelblauProblem(),
    HolderTableProblem(),
    LeviN13Problem(),
    MatyasProblem(),
    SchafferN2Problem(),
    ThreeHumpCamelProblem(),
]

## ── Grid paths ────────────────────────────────────────────────────────────────

_grid_path(problem::AbstractProblem) =
    joinpath(TRUE_POST_GRID_DIR, get_name(problem) * ".jld2")

## ── LHC sampler ───────────────────────────────────────────────────────────────

function _lhc(lb, ub, n)
    d = length(lb)
    X = Matrix{Float64}(undef, d, n)
    for k in 1:d
        perm    = randperm(n)
        X[k, :] .= lb[k] .+ (ub[k] - lb[k]) .* ((perm .- rand(n)) ./ n)
    end
    return X
end

## ── LHC 1-D marginal ─────────────────────────────────────────────────────────

function _lhc_1d_marginal(logpost, lb, ub, xs_k, dim_k, lhc_size)
    nk  = length(xs_k)
    lhc = _lhc(lb, ub, lhc_size)

    logval = Vector{Float64}(undef, nk)
    Threads.@threads for ik in 1:nk
        col_local = copy(lhc)
        col_local[dim_k, :] .= xs_k[ik]
        lp     = [logpost(col_local[:, j]) for j in 1:lhc_size]
        lp_max = maximum(lp)
        logval[ik] = log(mean(exp.(lp .- lp_max))) + lp_max
    end
    return logval
end

## ── LHC pairwise marginal ─────────────────────────────────────────────────────

function _lhc_marginal(logpost, lb, ub, xs_a, xs_b, dim_a, dim_b, lhc_size)
    na  = length(xs_a)
    nb  = length(xs_b)
    lhc = _lhc(lb, ub, lhc_size)

    logval = Matrix{Float64}(undef, na, nb)
    Threads.@threads for ia in 1:na
        col_local = copy(lhc)
        col_local[dim_a, :] .= xs_a[ia]
        for ib in 1:nb
            col_local[dim_b, :] .= xs_b[ib]
            lp     = [logpost(col_local[:, k]) for k in 1:lhc_size]
            lp_max = maximum(lp)
            logval[ia, ib] = log(mean(exp.(lp .- lp_max))) + lp_max
        end
    end
    return logval
end

## ── Phase 1: precompute ───────────────────────────────────────────────────────

"""
    precompute_bip_grid(problem; force=false, grid_res=GRID_RES, lhc_size=LHC_SIZE)

Evaluate true posterior marginals and save to JLD2.
Skips if the file already exists unless `force=true`.

Keyword arguments `grid_res` and `lhc_size` override the module-level defaults and
are useful for quick testing (e.g. `grid_res=5, lhc_size=5`).
"""
function precompute_bip_grid(problem::AbstractProblem;
                              force::Bool=false,
                              grid_res::Int=GRID_RES,
                              lhc_size::Int=LHC_SIZE)
    name    = get_name(problem)
    outpath = _grid_path(problem)
    d       = x_dim(problem)

    if isfile(outpath) && !force
        @info "$name: already exists at $outpath — skipping (pass force=true to recompute)"
        return
    end

    res     = get(GRID_RES_OVERRIDES, name, grid_res)
    lb, ub  = domain(problem).bounds
    logpost = true_logpost(problem)

    xs_per_dim = [collect(range(lb[k], ub[k]; length=res)) for k in 1:d]

    mkpath(TRUE_POST_GRID_DIR)

    if d == 2
        @info "$name: computing 2D grid ($(res)×$(res)) ..."
        logvals = Matrix{Float64}(undef, res, res)
        Threads.@threads for i1 in 1:res
            for i2 in 1:res
                logvals[i1, i2] = logpost([xs_per_dim[1][i1], xs_per_dim[2][i2]])
            end
        end
        save(outpath, Dict("dim" => 2, "xs_per_dim" => xs_per_dim, "logvals" => logvals))

    elseif d >= 3
        pairs   = [(da, db) for da in 1:d for db in (da+1):d]
        npairs  = length(pairs)
        @info "$name: computing $d 1-D marginals ($(res) grid, $(lhc_size) LHC samples each) ..."
        marg1d = Vector{Vector{Float64}}(undef, d)
        for k in 1:d
            @info "  dim $k"
            marg1d[k] = _lhc_1d_marginal(logpost, lb, ub, xs_per_dim[k], k, lhc_size)
        end
        @info "$name: computing $npairs pairwise marginals ($(res)² grid, $(lhc_size) LHC samples each) ..."
        marginals = Vector{Matrix{Float64}}(undef, npairs)
        for (pi, (da, db)) in enumerate(pairs)
            @info "  pair ($da,$db)"
            marginals[pi] = _lhc_marginal(logpost, lb, ub, xs_per_dim[da], xs_per_dim[db], da, db, lhc_size)
        end
        pair_dims = [[da, db] for (da, db) in pairs]
        save(outpath, Dict("dim" => d, "xs_per_dim" => xs_per_dim,
                           "pair_dims" => pair_dims, "pair_marginals" => marginals,
                           "marg1d" => marg1d))

    end

    @info "$name: saved to $outpath"
end

"""
    precompute_all_bip_grids(; force=false, kwargs...)

Precompute posterior grids for the 7 BIP problems.
"""
precompute_all_bip_grids(; force::Bool=false, kwargs...) =
    foreach(p -> precompute_bip_grid(p; force, kwargs...), _PM_BIP_PROBLEMS)

"""
    precompute_all_grids(; force=false, kwargs...)

Precompute posterior grids for all 31 benchmark problems (7 BIP + 24 opt-function).
"""
precompute_all_grids(; force::Bool=false, kwargs...) =
    foreach(p -> precompute_bip_grid(p; force, kwargs...), [_PM_BIP_PROBLEMS; _PM_OPT_PROBLEMS])

## ── Grid loading ──────────────────────────────────────────────────────────────

function _load_grid(problem::AbstractProblem)
    path = _grid_path(problem)
    isfile(path) || error("Grid not found for $(get_name(problem)): $path\n" *
                          "Run precompute_bip_grid($(typeof(problem))()) first.")
    return load(path)
end

## ── Phase 2: plotting ─────────────────────────────────────────────────────────

const _BIP_PARAM_LABELS = Dict{String, Vector{String}}(
    "ABProblem"          => ["a", "b"],
    "SimpleProblem"      => ["x₁", "x₂"],
    "BananaProblem"      => ["x₁", "x₂"],
    "BimodalProblem"     => ["x₁", "x₂"],
    "SIRProblem"         => ["β", "γ"],
    "DuffingProblem"     => ["δ", "α", "β"],
    "DuffingProblem5"    => ["δ", "α", "β", "γ", "ω"],
    "DiffusionProblem10" => ["xₛ", "yₛ", "tₛ"],
    "DiffusionProblem5D" => ["xₛ", "yₛ", "tₛ", "D", "v_max"],
)

_param_labels(problem::AbstractProblem) =
    get(_BIP_PARAM_LABELS, get_name(problem), ["x$i" for i in 1:x_dim(problem)])

function _add_2d_panel!(figpos, problem::AbstractProblem)
    data = _load_grid(problem)
    xs   = data["xs_per_dim"]
    lv   = data["logvals"]
    lv   = lv .- maximum(lv)
    labs = _param_labels(problem)

    ax = Axis(figpos;
        title          = get_name(problem),
        xlabel         = labs[1],
        ylabel         = labs[2],
        titlesize      = 14,
        xlabelsize     = 12,
        ylabelsize     = 12,
        xticklabelsize = 10,
        yticklabelsize = 10,
    )
    heatmap!(ax, xs[1], xs[2], exp.(lv); colormap=:matter)
    return ax
end

function _add_hd_panels!(fig, row_offset, col_offset, problem::AbstractProblem)
    data  = _load_grid(problem)
    xs    = data["xs_per_dim"]
    pdims = data["pair_dims"]
    pmarg = data["pair_marginals"]
    labs  = _param_labels(problem)
    name  = get_name(problem)

    for (pi, pd) in enumerate(pdims)
        da, db = pd[1], pd[2]
        lv     = pmarg[pi] .- maximum(pmarg[pi])
        row    = row_offset + da
        col    = col_offset + db
        ax = Axis(fig[row, col];
            title          = "$name  ($(labs[da])–$(labs[db]))",
            xlabel         = labs[da],
            ylabel         = labs[db],
            titlesize      = 12,
            xlabelsize     = 11,
            ylabelsize     = 11,
            xticklabelsize = 9,
            yticklabelsize = 9,
        )
        heatmap!(ax, xs[da], xs[db], exp.(lv); colormap=:matter)
    end
end

"""
    plot_bip_true_posteriors(; save_path=nothing)

Plot true posterior marginals for all 7 BIP problems from precomputed grids.

Layout (upper-triangle convention for 3-D problems):
  Row 1 (cols 1–5): AB, Simple, Banana, Bimodal, SIR — one 2-D panel each.
  Rows 2–3:         Duffing (cols 2–3) and Diffusion (cols 5–6) in upper-triangle
                    arrangement: pair (i,j) with i<j at figure row (1+i), col (offset+j).
"""
function plot_bip_true_posteriors(; save_path::Union{String,Nothing}=nothing)
    fig = Figure(; size=(1500, 700))

    for (col, prob) in enumerate([ABProblem(), SimpleProblem(), BananaProblem(),
                                   BimodalProblem(), SIRProblem()])
        _add_2d_panel!(fig[1, col], prob)
    end

    _add_hd_panels!(fig, 1, 0, DuffingProblem())
    _add_hd_panels!(fig, 1, 3, DiffusionProblem10())

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Saved to $save_path"
    end

    return fig
end

"""
Classify all 31 benchmark problems (7 BIP + 24 opt-function) by two posterior properties:
  1. Number of modes  (unimodal vs multimodal)
  2. Domain-normalised discrete entropy  H_rel = H_disc / log(N_cells)  ∈ [0,1]

Mode counting uses connected components of the superlevel set at `PEAK_REL_THRESHOLD`
× the global maximum density.  This is robust to:
  - Curved ridges (e.g. BananaProblem): one connected banana → 1 mode
  - Symmetry-straddling peaks (e.g. SimpleProblem): one connected blob → 1 mode
  - Genuine multimodality: separated high-density islands → correct count

Entropy uses discrete normalisation H_disc / log(N_cells), which avoids divide-by-zero
when the domain volume happens to be 1 (e.g. SIRProblem).

2-D problems  →  regular grid, exact logpost evaluation (threaded).
3-D problems  →  MC samples from the precomputed posterior grid:
    entropy    :  importance-sampling estimator
    mode count :  Gaussian KDE from weighted MC samples evaluated on a coarse
                  regular grid, then connected components.  No simulator re-calls.

Run from an interactive Julia session after `include("src/main.jl")`:
    include("src/classify_posteriors.jl")

Results are printed to stdout and saved to plots/classify_posteriors.csv.

Tunable parameters
------------------
DEFAULT_GRID_RES   : per-dimension resolution for 2-D regular grids
GRID_RES_OVERRIDES : per-problem resolution override (lower for ODE simulators)
PEAK_REL_THRESHOLD : superlevel-set threshold as a fraction of the global max density
KDE_GRID_RES       : per-dimension resolution of the KDE evaluation grid (3-D problems)
KDE_BW_FRAC        : KDE bandwidth as a fraction of the shortest domain edge (3-D)
"""

# ─── Parameters ──────────────────────────────────────────────────────────────

const DEFAULT_GRID_RES   = 300
const PEAK_REL_THRESHOLD = 0.15   # 15 % of global max density

const GRID_RES_OVERRIDES = Dict(
    "SIRProblem" => 150,   # ODE solver, but 2-D so higher res than before
)

const KDE_GRID_RES = 25    # 25^3 = 15 625 KDE query points for 3-D problems
const KDE_BW_FRAC  = 0.05  # bandwidth = 5 % of shortest domain edge
                            # 15 % was too wide: merged all Duffing modes into one blob

# ─── Problem list ────────────────────────────────────────────────────────────

const _BIP_PROBLEMS = [
    ABProblem(),
    SimpleProblem(),
    BananaProblem(),
    BimodalProblem(),
    SIRProblem(),
    DuffingProblem(),
    DiffusionProblem10(),
]

const _OPT_PROBLEMS = CrossPolytopeObsProblem.([
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
])

const _ALL_PROBLEMS = [_BIP_PROBLEMS; _OPT_PROBLEMS]

# ─── Shared helpers ──────────────────────────────────────────────────────────

_logsumexp(v) = (m = maximum(v); m + log(sum(exp, v .- m)))

# ─── 2-D regular-grid helpers ────────────────────────────────────────────────

function _make_2d_grid(problem, res)
    lb, ub = domain(problem).bounds
    x1s = range(lb[1], ub[1]; length=res)
    x2s = range(lb[2], ub[2]; length=res)
    xs  = Matrix{Float64}(undef, 2, res^2)
    k   = 1
    for x1 in x1s, x2 in x2s
        xs[1, k] = x1;  xs[2, k] = x2;  k += 1
    end
    return xs, step(x1s), step(x2s)
end

# Discrete entropy normalised by log(N_cells) — always in [0, 1]
function _discrete_entropy(logvals_flat)
    logZ  = _logsumexp(logvals_flat)
    logp  = logvals_flat .- logZ          # discrete log-probs
    p     = exp.(logp)
    H     = -sum(p_i * lp_i for (p_i, lp_i) in zip(p, logp) if isfinite(lp_i))
    H_rel = H / log(length(logvals_flat))
    return H, H_rel
end

# Connected components of superlevel set via BFS (4-connectivity)
function _count_components_2d(logvals_grid, rel_threshold)
    logmax  = maximum(logvals_grid)
    log_min = logmax + log(rel_threshold)
    mask    = logvals_grid .>= log_min
    res1, res2 = size(mask)
    visited = falses(res1, res2)
    n = 0
    queue = Tuple{Int,Int}[]
    for i1 in 1:res1, i2 in 1:res2
        mask[i1,i2] && !visited[i1,i2] || continue
        n += 1
        push!(queue, (i1, i2));  visited[i1,i2] = true
        while !isempty(queue)
            ci, cj = popfirst!(queue)
            for (di, dj) in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1))
                ni, nj = ci+di, cj+dj
                1 <= ni <= res1 && 1 <= nj <= res2 || continue
                mask[ni,nj] && !visited[ni,nj] || continue
                visited[ni,nj] = true;  push!(queue, (ni, nj))
            end
        end
    end
    return n
end

function _analyse_2d(problem, name)
    res       = get(GRID_RES_OVERRIDES, name, DEFAULT_GRID_RES)
    @info "Analysing $name  (2-D grid: $(res)×$(res))"
    xs, dx1, dx2 = _make_2d_grid(problem, res)
    logvals   = reshape(true_logpost(problem)(xs), res, res)
    n_modes   = _count_components_2d(logvals, PEAK_REL_THRESHOLD)
    H, H_rel  = _discrete_entropy(vec(logvals))
    return (name=name, n_modes=n_modes, H=H, H_rel=H_rel)
end

# ─── 3-D MC-grid helpers ─────────────────────────────────────────────────────

# Importance-sampling discrete entropy from the MC posterior grid
function _mc_entropy(log_ws, true_logvals)
    log_iw = log_ws .+ true_logvals      # log unnorm. IS weights ∝ p/q
    log_w̃  = log_iw .- _logsumexp(log_iw)
    w̃      = exp.(log_w̃)
    H      = -sum(w_i * lw_i for (w_i, lw_i) in zip(w̃, log_w̃) if isfinite(lw_i))
    H_rel  = H / log(length(w̃))
    return H, H_rel
end

# Build a d-dimensional regular evaluation grid; returns (d, prod(sizes)) matrix
function _make_nd_grid(lb, ub, res)
    d = length(lb)
    ranges = [range(lb[k], ub[k]; length=res) for k in 1:d]
    strides = [res^(k-1) for k in 1:d]
    M = res^d
    xs = Matrix{Float64}(undef, d, M)
    for j in 1:M
        for k in 1:d
            xs[k, j] = ranges[k][mod((j-1) ÷ strides[k], res) + 1]
        end
    end
    return xs
end

# Gaussian KDE evaluated at query_pts using weighted samples.
# Returns log(p_KDE) at each query point (up to an additive constant).
# Bandwidth h is isotropic.
function _kde_logvals(query_pts, sample_pts, log_w̃, h)
    d, M = size(query_pts)
    N    = size(sample_pts, 2)
    h2   = 2.0 * h^2
    log_terms = Vector{Float64}(undef, N)
    logvals   = Vector{Float64}(undef, M)
    for j in 1:M
        q = @view query_pts[:, j]
        for i in 1:N
            d2 = 0.0
            for k in 1:d; d2 += (q[k] - sample_pts[k, i])^2; end
            log_terms[i] = log_w̃[i] - d2 / h2
        end
        logvals[j] = _logsumexp(log_terms)
    end
    return logvals
end

# Connected components of superlevel set on a d-dimensional regular grid (4-connectivity)
function _count_components_nd(logvals_flat, sizes, rel_threshold)
    logmax  = maximum(logvals_flat)
    log_min = logmax + log(rel_threshold)
    mask    = logvals_flat .>= log_min
    N       = length(mask)
    visited = falses(N)
    d       = length(sizes)
    strides = [prod(sizes[1:k-1]) for k in 1:d]   # column-major strides

    function neighbours(idx)
        nbrs = Int[]
        multi = [(((idx-1) ÷ strides[k]) % sizes[k]) + 1 for k in 1:d]
        for k in 1:d
            for delta in (-1, 1)
                new_k = multi[k] + delta
                1 <= new_k <= sizes[k] || continue
                n_idx = idx + delta * strides[k]
                push!(nbrs, n_idx)
            end
        end
        return nbrs
    end

    n = 0
    queue = Int[]
    for i in 1:N
        mask[i] && !visited[i] || continue
        n += 1
        push!(queue, i);  visited[i] = true
        while !isempty(queue)
            ci = popfirst!(queue)
            for ni in neighbours(ci)
                1 <= ni <= N && mask[ni] && !visited[ni] || continue
                visited[ni] = true;  push!(queue, ni)
            end
        end
    end
    return n
end

function _analyse_highd(problem, name)
    d  = x_dim(problem)
    lb, ub = domain(problem).bounds
    @info "Analysing $name  ($(d)-D, MC grid + KDE)"

    g = load_grid(problem)

    # Entropy via IS on the MC grid
    H, H_rel = _mc_entropy(g.log_ws, g.true_logvals)

    # Normalised IS weights for KDE
    log_iw = g.log_ws .+ g.true_logvals
    log_w̃  = log_iw .- _logsumexp(log_iw)

    # KDE on a coarse regular grid
    h           = KDE_BW_FRAC * minimum(ub .- lb)
    query_pts   = _make_nd_grid(lb, ub, KDE_GRID_RES)
    kde_vals    = _kde_logvals(query_pts, g.xs, log_w̃, h)
    sizes       = fill(KDE_GRID_RES, d)
    n_modes     = _count_components_nd(kde_vals, sizes, PEAK_REL_THRESHOLD)

    return (name=name, n_modes=n_modes, H=H, H_rel=H_rel)
end

# ─── Dispatch & run ──────────────────────────────────────────────────────────

# Strip _cross suffix and normalise proxy names so CSV keys match compute_acq_scores.jl
const _PROXY_DISPLAY = Dict(
    "BealeProxyProblem"          => "BealeProblem",
    "GoldsteinPriceProxyProblem" => "GoldsteinPriceProblem",
)
function _display_name(p)
    nm = replace(get_name(p), "_cross" => "")
    return get(_PROXY_DISPLAY, nm, nm)
end

_analyse_problem(p) = (nm = _display_name(p); x_dim(p) == 2 ? _analyse_2d(p, nm) : _analyse_highd(p, nm))

const classify_results = map(_analyse_problem, _ALL_PROBLEMS)

println()
println(rpad("Problem", 38), "│ Modes │   H_rel")
println(repeat('─', 38), "─┼───────┼──────────")
for r in classify_results
    println(rpad(r.name, 38), "│   $(lpad(r.n_modes, 3)) │  $(round(r.H_rel, digits=3))")
end

mkpath("plots")
open("plots/classify_posteriors.csv", "w") do io
    println(io, "problem,n_modes,H,H_rel")
    for r in classify_results
        println(io, "$(r.name),$(r.n_modes),$(r.H),$(r.H_rel)")
    end
end
println("\nSaved → plots/classify_posteriors.csv")

"""
Extend classify_posteriors.csv with H_rel and n_modes for 6 high-dimensional problems:

  - Group C (5-D BIP): DuffingProblem5, DiffusionProblem5D
      Uses precomputed LHC pairwise marginal grids from data/true_posterior_grids/<name>.jld2.
      H_rel = max pairwise-marginal H_rel; n_modes = max connected components over pairs.

  - Group D (5-D cross-opt): Rosenbrock5, StyblinskiTang5, Michalewicz5, Sphere5
      Uses precomputed IS posterior grids (posterior_grid.jld2) via load_grid.
      H_rel = IS entropy (full 5-D joint). n_modes = max connected components over
      pairwise 2-D KDE marginals projected from IS samples (avoids infeasible 25^5 KDE).

## Usage

Must be run after `include("src/classify_posteriors.jl")` so that helper functions
(_discrete_entropy, _count_components_2d, _mc_entropy, _kde_logvals, _logsumexp,
PEAK_REL_THRESHOLD, KDE_GRID_RES, KDE_BW_FRAC) are in scope.
Use run_classify_posteriors_hd.jl as the entry point.

Appends 6 rows to plots/classify_posteriors.csv without overwriting existing rows.
"""

## ─── Group C: analysis from precomputed LHC pairwise marginals ───────────────

function _analyse_bip5d(problem)
    name = get_name(problem)
    path = joinpath("data", "true_posterior_grids", name * ".jld2")
    isfile(path) || error("Grid not found: $path — run precompute_bip_grid($(typeof(problem))()) first")
    @info "Analysing $name  (5-D BIP, pairwise marginals)"

    data      = load(path)
    pair_marg = data["pair_marginals"]  # Vector{Matrix{Float64}}: C(5,2)=10 log-marginals

    # H_rel: max over pairwise 2-D marginal H_rel values
    pair_entropy = [_discrete_entropy(m) for m in pair_marg]
    H_idx  = argmax(e[2] for e in pair_entropy)
    H      = pair_entropy[H_idx][1]
    H_rel  = pair_entropy[H_idx][2]

    # n_modes: max connected components over all pairwise 2-D marginals
    n_modes = maximum(_count_components_2d(m, PEAK_REL_THRESHOLD) for m in pair_marg)

    return (name=name, n_modes=n_modes, H=H, H_rel=H_rel)
end

## ─── Group D: analysis from IS posterior grids ───────────────────────────────

# Compute a 2-D KDE marginal for a specific pair of dimensions from IS samples.
# Returns a (KDE_GRID_RES × KDE_GRID_RES) matrix of log-density values.
function _pairwise_kde_2d(xs, log_w̃, lb, ub, da, db)
    xa = xs[da, :]
    xb = xs[db, :]
    h  = KDE_BW_FRAC * min(ub[da] - lb[da], ub[db] - lb[db])
    h2 = 2.0 * h^2
    N  = length(log_w̃)

    ra  = range(lb[da], ub[da]; length=KDE_GRID_RES)
    rb  = range(lb[db], ub[db]; length=KDE_GRID_RES)
    out = Matrix{Float64}(undef, KDE_GRID_RES, KDE_GRID_RES)
    log_terms = Vector{Float64}(undef, N)

    for ia in 1:KDE_GRID_RES
        qa = ra[ia]
        for ib in 1:KDE_GRID_RES
            qb = rb[ib]
            @inbounds for i in 1:N
                log_terms[i] = log_w̃[i] - ((qa - xa[i])^2 + (qb - xb[i])^2) / h2
            end
            out[ia, ib] = _logsumexp(log_terms)
        end
    end
    return out
end

function _analyse_opt5d(problem)
    name   = _display_name(problem)   # strips _cross suffix for CSV key
    lb, ub = domain(problem).bounds
    d      = x_dim(problem)
    @info "Analysing $name  (5-D cross-opt, IS posterior grid)"

    g = load_grid(problem)   # (xs, log_ws, true_logvals), N = 20_000 samples

    # IS entropy over the full 5-D joint
    H, H_rel = _mc_entropy(g.log_ws, g.true_logvals)

    # Normalised IS weights for pairwise KDE
    log_iw = g.log_ws .+ g.true_logvals
    log_w̃  = log_iw .- _logsumexp(log_iw)

    # n_modes: max connected components over all C(d,2) pairwise 2-D KDE marginals
    pairs   = [(da, db) for da in 1:d for db in (da+1):d]
    n_modes = maximum(
        _count_components_2d(_pairwise_kde_2d(g.xs, log_w̃, lb, ub, da, db), PEAK_REL_THRESHOLD)
        for (da, db) in pairs
    )

    return (name=name, n_modes=n_modes, H=H, H_rel=H_rel)
end

## ─── Problem lists ────────────────────────────────────────────────────────────

const _HD_BIP_PROBLEMS = [DuffingProblem5(), DiffusionProblem5D()]

const _HD_OPT_PROBLEMS = CrossPolytopeObsProblem.([
    RosenbrockProblem(; x_dim=5),
    StyblinskiTangProblem(; x_dim=5),
    MichalewiczProblem(; x_dim=5),
    SphereProblem(; x_dim=5),
])

## ─── Run ──────────────────────────────────────────────────────────────────────

println("\nClassifying 5-D BIP problems from pairwise marginal grids ...")
hd_bip_results = [_analyse_bip5d(p) for p in _HD_BIP_PROBLEMS]

println("\nClassifying 5-D cross-opt problems from IS posterior grids ...")
hd_opt_results = [_analyse_opt5d(p) for p in _HD_OPT_PROBLEMS]

hd_all_results = [hd_bip_results; hd_opt_results]

## ─── Print table ──────────────────────────────────────────────────────────────

println()
println(rpad("Problem", 38), "│ Modes │   H_rel")
println(repeat('─', 38), "─┼───────┼──────────")
for r in hd_all_results
    println(rpad(r.name, 38), "│   $(lpad(r.n_modes, 3)) │  $(round(r.H_rel, digits=3))")
end

## ─── Append to classify_posteriors.csv ───────────────────────────────────────

open("plots/classify_posteriors.csv", "a") do io
    for r in hd_all_results
        println(io, "$(r.name),$(r.n_modes),$(r.H),$(r.H_rel)")
    end
end
println("\nAppended $(length(hd_all_results)) rows → plots/classify_posteriors.csv")

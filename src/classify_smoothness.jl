"""
Classify all 37 benchmark problems (31 original + 6 HD) by proxy *value-surface*
smoothness properties, complementing the posterior-shape metrics in
`classify_posteriors.jl` (H_rel, n_modes).

Two original metrics (2026-07-14):

  1. `cv_value` — coefficient of variation of the proxy value δ(x) = true_f(problem)(x)
     over the domain. Targets the "heterogeneous/huge output scale" failure mode
     that WarpedGP automatically fixes (e.g. BealeProblem, GoldsteinPriceProblem).

  2. `cv_grad`  — coefficient of variation of the proxy gradient norm ‖∇δ_j(x)‖ over
     the domain. Targets the "sharp ridge / spatially varying lengthscale" failure
     mode that the nonstationary GP addresses (e.g. SIRProblem).

Three more metrics (2026-08-03), added because cv_value showed no apparent
correlation with actual WarpedGP-vs-Standard performance — each targets a more
specific distributional pathology than plain dispersion:

  3. `yj_dev`     — |λ̂ − 1| where λ̂ is the MLE-optimal Yeo-Johnson transform
     parameter fit to δ(x) (profile log-likelihood, grid search). This is the
     most direct proxy possible: it estimates almost exactly what WarpedGP's
     own Yeo-Johnson warping layer would learn, without running BO/GP at all.

  4. `skewness`   — third standardized moment of δ(x). Decouples asymmetry from
     raw spread (cv_value can be high for a symmetric heavy-tailed distribution,
     which Yeo-Johnson does little for).

  5. `heterosced` — local heteroscedasticity: for each domain sample, take its
     k nearest neighbours (in x-space) and compute the local mean and local std
     of δ over that neighbourhood; the metric is corr(local_std, |local_mean|)
     across all samples. A GP with one global amplitude/noise can already
     absorb "uniformly large values" after fitting hyperparameters — what it
     can't absorb is variance that scales *with* the local signal (Poisson-like/
     multiplicative noise), which is exactly what a power transform fixes. This
     is a local/spatial statistic, unlike the other four which are purely
     marginal (domain-pooled).

All five are computed from the same Latin Hypercube sample of the domain
(shared `LHC_SIZE` points), evaluating `true_f` (noiseless proxy) and its
Jacobian via `ForwardDiff.jacobian`. For multi-dimensional outputs (y_dim > 1),
each metric is computed per output dimension and aggregated with `maximum`
(or `maximum ∘ abs` for signed metrics) — a single badly-scaled/skewed/
ridge-y/heteroscedastic output dimension is enough to justify the advanced
surrogate.

Run from an interactive Julia session after `include("src/main.jl")`:
    include("src/classify_smoothness.jl")

Results are printed to stdout and saved to plots/classify_smoothness.csv.
"""

using ForwardDiff
using Statistics: mean, std, var, cor, median
using LinearAlgebra: eigvals, Symmetric

# ─── Parameters ──────────────────────────────────────────────────────────────

const LHC_SIZE = 1000   # domain samples shared by both metrics
const HESS_BUDGET = 150 # total Hessian evaluations per problem (n_pts × dy, roughly)

# ─── Problem list (31 original + 6 HD = 37, matches acq_scores_final.csv) ────

const _SM_BIP_PROBLEMS = [
    ABProblem(),
    SimpleProblem(),
    BananaProblem(),
    BimodalProblem(),
    SIRProblem(),
    DuffingProblem(),
    DiffusionProblem10(),
]

const _SM_OPT_PROBLEMS = CrossPolytopeObsProblem.([
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

const _SM_HD_BIP_PROBLEMS = [DuffingProblem5(), DiffusionProblem5D()]

const _SM_HD_OPT_PROBLEMS = CrossPolytopeObsProblem.([
    RosenbrockProblem(; x_dim=5),
    StyblinskiTangProblem(; x_dim=5),
    MichalewiczProblem(; x_dim=5),
    SphereProblem(; x_dim=5),
])

const _SM_ALL_PROBLEMS = [_SM_BIP_PROBLEMS; _SM_OPT_PROBLEMS; _SM_HD_BIP_PROBLEMS; _SM_HD_OPT_PROBLEMS]

# ─── Naming (same normalisation as classify_posteriors.jl) ──────────────────

const _SM_PROXY_DISPLAY = Dict(
    "BealeProxyProblem"          => "BealeProblem",
    "GoldsteinPriceProxyProblem" => "GoldsteinPriceProblem",
)
function _sm_display_name(p)
    nm = replace(get_name(p), "_cross" => "")
    return get(_SM_PROXY_DISPLAY, nm, nm)
end

# ─── LHC sampler (self-contained; mirrors plot_marginals.jl's `_lhc`) ────────

function _sm_lhc(lb, ub, n)
    d = length(lb)
    X = Matrix{Float64}(undef, d, n)
    for k in 1:d
        perm    = randperm(n)
        X[k, :] .= lb[k] .+ (ub[k] - lb[k]) .* ((perm .- rand(n)) ./ n)
    end
    return X
end

# ─── Coefficient of variation (scale-invariant spread) ───────────────────────
#
# Uses RMS (not |mean|) in the denominator: std/|mean| blows up whenever the
# raw values straddle zero (mean ≈ 0 despite a well-behaved surface — e.g.
# SimpleProblem's y=x is symmetric about 0). RMS = sqrt(mean(v.^2)) is always
# ≥ std (since RMS² = mean² + var), so this variant is bounded in [0, 1] and
# only degenerates when all values are exactly zero.

_cv(v) = std(v) / max(sqrt(mean(v .^ 2)), 1e-12)

# ─── Skewness (third standardized moment) ────────────────────────────────────

function _skewness(v)
    m = mean(v)
    s = std(v; corrected=false)
    s <= 1e-12 && return 0.0
    return mean((v .- m) .^ 3) / s^3
end

# ─── Yeo-Johnson MLE λ, deviation from identity (λ=1) ────────────────────────
# Standard Yeo-Johnson transform; λ=1 is the identity for both signs. Fit via
# profile log-likelihood grid search (assumes transformed values ~ Gaussian).

function _yj_transform(y, λ)
    if y >= 0
        λ == 0.0 ? log1p(y) : (((1 + y)^λ - 1) / λ)
    else
        λ == 2.0 ? -log1p(-y) : -(((1 - y)^(2 - λ) - 1) / (2 - λ))
    end
end

function _yj_profile_ll(v, λ)
    t  = _yj_transform.(v, λ)
    σ2 = var(t; corrected=false)
    σ2 <= 1e-300 && return -Inf
    n   = length(v)
    jac = sum(sign(y) * log1p(abs(y)) for y in v)
    return -0.5 * n * log(σ2) + (λ - 1) * jac
end

function _yj_lambda_dev(v; grid = -5.0:0.02:5.0)
    lls = [_yj_profile_ll(v, λ) for λ in grid]
    λ̂   = grid[argmax(lls)]
    return abs(λ̂ - 1.0)
end

# ─── Local heteroscedasticity via k-NN neighbourhoods ────────────────────────
# corr(local_std, |local_mean|) across domain samples — high when local spread
# scales with local signal magnitude (a failure mode a power transform fixes,
# distinct from a merely large-but-constant-scale output).

function _sm_knn_indices(X; k=20)
    d, N = size(X)
    kk = min(k + 1, N)   # includes the point itself (distance 0)
    idxs = Matrix{Int}(undef, kk, N)
    Threads.@threads for i in 1:N
        dists = Vector{Float64}(undef, N)
        @inbounds for j in 1:N
            s = 0.0
            for a in 1:d
                diff = X[a, j] - X[a, i]
                s += diff * diff
            end
            dists[j] = s
        end
        idxs[:, i] = partialsortperm(dists, 1:kk)
    end
    return idxs
end

function _heterosced(idxs, y)
    N = size(idxs, 2)
    means = Vector{Float64}(undef, N)
    stds  = Vector{Float64}(undef, N)
    for i in 1:N
        vals     = @view y[idxs[:, i]]
        means[i] = mean(vals)
        stds[i]  = std(vals)
    end
    c = cor(stds, abs.(means))
    return isnan(c) ? 0.0 : c
end

# ─── Excess kurtosis (tail-heaviness distinct from skewness) ─────────────────

function _kurtosis(v)
    m = mean(v)
    s = std(v; corrected=false)
    s <= 1e-12 && return 0.0
    return mean((v .- m) .^ 4) / s^4 - 3.0
end

# ─── Hessian anisotropy: condition number of curvature eigenvalues ───────────
# Nested ForwardDiff Hessians are expensive (esp. through ODE/PDE simulators),
# so this is estimated on a subsample sized to keep total Hessian evals per
# problem roughly constant (HESS_BUDGET) regardless of dy: n_pts = HESS_BUDGET/dy.
# Median condition number over the subsample, then max over output dims.
# Points/dims where the Hessian AD fails (rare, e.g. through adaptive ODE
# solvers) are silently skipped.

function _hessian_anisotropy(f, X, dy)
    n_pts = clamp(HESS_BUDGET ÷ max(dy, 1), 5, size(X, 2))
    conds_per_dim = Float64[]
    for j in 1:dy
        fj = x -> f(x)[j]
        conds = Float64[]
        for i in 1:n_pts
            xi = @view X[:, i]
            H = try
                ForwardDiff.hessian(fj, xi)
            catch
                continue
            end
            ev_abs = abs.(eigvals(Symmetric(H)))
            lo = max(minimum(ev_abs), 1e-12)
            push!(conds, maximum(ev_abs) / lo)
        end
        isempty(conds) || push!(conds_per_dim, median(conds))
    end
    isempty(conds_per_dim) && return 0.0
    return maximum(conds_per_dim)
end

# ─── Empirical Lipschitz constant estimate ───────────────────────────────────
# max pairwise |δ(x_i) - δ(x_j)| / ‖x_i - x_j‖ over the LHC sample — worst-case
# sharpness, distinct from cv_grad's average-case gradient-magnitude spread.
# Free: reuses the already-computed X, Y (no new simulator calls).

function _lipschitz(X, y)
    N = length(y)
    dxdim = size(X, 1)
    best = 0.0
    @inbounds for i in 1:N
        for j in (i+1):N
            s = 0.0
            for a in 1:dxdim
                d = X[a, i] - X[a, j]
                s += d * d
            end
            dist = sqrt(s)
            dist < 1e-9 && continue
            slope = abs(y[i] - y[j]) / dist
            slope > best && (best = slope)
        end
    end
    return best
end

# ─── Path roughness: total variation along random 1-D lines through domain ───
# Scale-normalised (divided by RMS value along the line) so it's comparable
# across problems with different output scales. Costs n_lines*n_pts extra
# simulator evaluations.

function _path_roughness(f, lb, ub, dy; n_lines=10, n_pts=50)
    d = length(lb)
    tv_per_dim = zeros(dy)
    for _ in 1:n_lines
        p0 = lb .+ rand(d) .* (ub .- lb)
        p1 = lb .+ rand(d) .* (ub .- lb)
        vals = Matrix{Float64}(undef, dy, n_pts)
        for (k, t) in enumerate(range(0.0, 1.0; length=n_pts))
            vals[:, k] = f(p0 .+ t .* (p1 .- p0))
        end
        for j in 1:dy
            scale = max(sqrt(mean(@view(vals[j, :]) .^ 2)), 1e-12)
            tv_per_dim[j] += sum(abs, diff(@view vals[j, :])) / scale
        end
    end
    tv_per_dim ./= n_lines
    return maximum(tv_per_dim)
end

# ─── Output redundancy: average |cross-output correlation| ──────────────────
# For multi-dim outputs (dy>1): are the output dimensions redundant (highly
# correlated) or largely independent? Free: reuses already-computed Y.
# dy=1 problems have no cross-output structure — defined as 0.0.

function _output_redundancy(Y)
    dy = size(Y, 1)
    dy < 2 && return 0.0
    C = cor(Y')
    s, cnt = 0.0, 0
    for i in 1:dy, j in 1:dy
        i == j && continue
        s += abs(C[i, j])
        cnt += 1
    end
    return s / cnt
end

# ─── Core analysis ────────────────────────────────────────────────────────────

function _analyse_smoothness(problem, name)
    println("Analysing $name  (LHC_SIZE=$LHC_SIZE)")
    lb, ub = domain(problem).bounds
    dx = length(lb)
    f  = true_f(problem)
    isnothing(f) && error("$name has no true_f defined — cannot evaluate proxy surface.")

    X = _sm_lhc(lb, ub, LHC_SIZE)

    # Evaluate proxy value + Jacobian at every sample.
    y1 = f(@view X[:, 1])
    dy = length(y1)
    Y  = Matrix{Float64}(undef, dy, LHC_SIZE)
    Y[:, 1] = y1
    G  = Array{Float64}(undef, dy, LHC_SIZE)   # per-sample, per-output gradient norm
    G[:, 1] = [norm(@view ForwardDiff.jacobian(f, X[:, 1])[j, :]) for j in 1:dy]

    Threads.@threads for i in 2:LHC_SIZE
        xi = X[:, i]
        Y[:, i] = f(xi)
        J = ForwardDiff.jacobian(f, xi)
        for j in 1:dy
            G[j, i] = norm(@view J[j, :])
        end
    end

    cv_value = maximum(_cv(@view Y[j, :]) for j in 1:dy)
    cv_grad  = maximum(_cv(@view G[j, :]) for j in 1:dy)

    yj_dev     = maximum(_yj_lambda_dev(@view Y[j, :]) for j in 1:dy)
    skewness   = maximum(abs(_skewness(@view Y[j, :])) for j in 1:dy)
    knn_idxs   = _sm_knn_indices(X; k=20)
    heterosced = maximum(abs(_heterosced(knn_idxs, @view Y[j, :])) for j in 1:dy)

    kurtosis         = maximum(_kurtosis(@view Y[j, :]) for j in 1:dy)
    hessian_cond     = _hessian_anisotropy(f, X, dy)
    lipschitz        = maximum(_lipschitz(X, @view Y[j, :]) for j in 1:dy)
    path_roughness   = _path_roughness(f, lb, ub, dy)
    output_redundancy = _output_redundancy(Y)

    return (name=name, dx=dx, dy=dy, cv_value=cv_value, cv_grad=cv_grad,
            yj_dev=yj_dev, skewness=skewness, heterosced=heterosced,
            kurtosis=kurtosis, hessian_cond=hessian_cond, lipschitz=lipschitz,
            path_roughness=path_roughness, output_redundancy=output_redundancy)
end

_sm_analyse_problem(p) = _analyse_smoothness(p, _sm_display_name(p))

# ─── Run (call `run_classify_smoothness()` explicitly, e.g. from
#      run_classify_smoothness.jl — including this file only defines things,
#      so a single problem can be smoke-tested first via _sm_analyse_problem) ──

function run_classify_smoothness()
    results = map(_sm_analyse_problem, _SM_ALL_PROBLEMS)

    println()
    for r in results
        println("$(r.name): cv_value=$(round(r.cv_value,digits=3)) cv_grad=$(round(r.cv_grad,digits=3)) " *
                "yj_dev=$(round(r.yj_dev,digits=3)) skewness=$(round(r.skewness,digits=3)) " *
                "heterosced=$(round(r.heterosced,digits=3)) kurtosis=$(round(r.kurtosis,digits=3)) " *
                "hessian_cond=$(round(r.hessian_cond,digits=3)) lipschitz=$(round(r.lipschitz,digits=3)) " *
                "path_roughness=$(round(r.path_roughness,digits=3)) output_redundancy=$(round(r.output_redundancy,digits=3))")
    end

    mkpath("plots")
    open("plots/classify_smoothness.csv", "w") do io
        println(io, "problem,dx,dy,cv_value,cv_grad,yj_dev,skewness,heterosced,kurtosis,hessian_cond,lipschitz,path_roughness,output_redundancy")
        for r in results
            println(io, "$(r.name),$(r.dx),$(r.dy),$(r.cv_value),$(r.cv_grad),$(r.yj_dev),$(r.skewness),$(r.heterosced)," *
                        "$(r.kurtosis),$(r.hessian_cond),$(r.lipschitz),$(r.path_roughness),$(r.output_redundancy)")
        end
    end
    println("\nSaved → plots/classify_smoothness.csv")
    return results
end

"""
Non-homogeneity criterion `H` — Suggestion 1 from `~/Documents/phd/bosip/paper1_suggestions.md`
("A non-homogeneity criterion: predicting when the warped GP pays off").

Distinguishes two properties of the (proxy) response surface δ(θ) = (φ∘f)(θ) that are easy to
conflate:
  - **Anisotropy**  — different lengthscales per input dimension, but constant across the domain.
    A stationary ARD GP is the correct model class here; warping/nonstationary kernels have
    nothing to fix.
  - **Non-homogeneity** — the local lengthscale/curvature *varies with θ* (e.g. SIR's sharp
    ridge embedded in an otherwise flat surface). A single global lengthscale is misspecified
    everywhere. This is exactly the misspecification warped GP / nonstationary GP repair.

`H` estimates the second property directly, distinct from `classify_smoothness.jl`'s `cv_grad`
(a cheap closed-form proxy for the same idea, already correlation-tested with a null result —
see [[project-bosip-reviews]]): `H` fits actual local stationary GPs by MAP (not a closed-form
gradient-norm statistic) and calibrates significance against a *parametric bootstrap null*
drawn from ONE global stationary GP fit to the same data, so a raw-magnitude false positive
(a surface that's just "hard", not non-homogeneous) is explicitly controlled for.

## Algorithm (offline; no new simulations beyond one space-filling design D_N)

  1. Draw a Latin Hypercube design D_N = {(θ_i, δ_i)}, i=1..N over the domain, one proxy output
     dimension at a time (δ = true_f(problem)(θ)[j]).
  2. Choose `m` local centers among the N design points (index subset).
  3. For each center c_j: take its `k` nearest neighbours in θ-space, fit a stationary ARD GP
     locally by MAP (shared lengthscale/amplitude/noise priors across all centers — reusing this
     project's standard `prior_mean`/`get_*_priors` helpers), record the log-lengthscales
     λ_j = log ℓ_j ∈ R^d.
  4. H = max_dim std_j(λ_j[dim])  — aggregate spread of the per-center log-lengthscales, worst
     dimension.
  5. Calibrate: fit ONE global stationary ARD GP to the full D_N (same priors), then for
     b = 1..B draw a synthetic dataset from that global GP's *prior* at the same θ locations
     (`BOSS.finite_gp` + `rand`), recompute H_b via steps 2-4 (same centers/neighbourhoods, only
     y resampled), and report p = #{H_b >= H_obs} / B — the fraction of homogeneous-null
     replicates at least as spread out as the real fit. Rejecting the null (small p) is evidence
     of genuine non-homogeneity, not just small-k estimation noise (which the raw H alone can't
     distinguish, per Karvonen & Oates 2023's ML-in-noiseless-GP-regression ill-posedness result
     — hence MAP with priors, not plain ML, at every fit in this file).

For multi-output problems (dy > 1), H/p are computed independently per output dimension and
aggregated via `maximum(H)` (worst dimension), matching classify_smoothness.jl's convention —
the reported p is the one belonging to that argmax dimension.

Must be run after:
    include("src/main.jl")
    include("src/classify_smoothness.jl")
so that the problem lists (_SM_ALL_PROBLEMS), `_sm_display_name`, `_sm_lhc` and
`_sm_knn_indices` are in scope.

Usage (in-process, smoke tests / small sweeps):
    include("src/classify_nonhomogeneity.jl")
    run_classify_nonhomogeneity()                       # all 37 problems, full spec settings
    run_classify_nonhomogeneity(_SM_BIP_PROBLEMS[1:1])   # smoke test on one cheap problem
    run_classify_nonhomogeneity(...; n_design=30, m_centers=4, n_bootstrap=5)  # cheap dry run

Usage (per-(problem, output dim) cluster split — see cluster_scripts/script_nonhomogeneity.jl,
cluster_scripts/submit_nonhomogeneity.sh, src/merge_classify_nonhomogeneity.jl):
    run_classify_nonhomogeneity_dim(problem, get_name(problem), j)   # saves to plots/nonhomogeneity/

**Cost warning**: at the current defaults (N=200, m=20, B=200, multistart=24/24) this is
~m + 1 + B*m ≈ 4021 local GP-MAP fits PER OUTPUT DIMENSION per problem, each fit now also
4x (local) / 1.5x (global) more expensive than the original smoke-tested defaults due to the
higher multistart — expensive across all 37 problems. Smoke-test with reduced
`n_design`/`m_centers`/`n_bootstrap`/`multistart_*` on 1-2 problems before running the full
sweep; this is implementation only, not yet validated at scale (see the deliverable's
pre-registered predictions P1-P3 in paper1_suggestions.md for what the full run should check).
"""

using Statistics: mean, std

# ─── Parameters (spec's "N ≈ 50-200", "m ≈ 10-20", "k ≈ 3d-5d", "B ≈ 200") ───

const H_N_DESIGN         = 200   # |D_N|, space-filling design size
const H_M_CENTERS        = 20    # number of local centers m
const H_K_MULT           = 4     # k = H_K_MULT * d nearest neighbours per center
const H_N_BOOTSTRAP      = 200   # B, parametric bootstrap replications
const H_MULTISTART_LOCAL = 24    # MAP multistart for the m small local fits
const H_MULTISTART_GLOBAL = 24   # MAP multistart for the one global (bootstrap-null) fit

# ─── Local/global stationary-GP MAP fit, single output dimension ────────────
# `model_j` must already be the y_dim=1 slice of the project's standard model
# (BOSS.slice(full_model, j)) so mean/priors are correctly indexed for dim j.

function _h_fit_params(problem, model_j, X_local, y_row, model_fitter, boss_options)
    data = BOSS.ExperimentData(X_local, y_row)
    prob = BOSS.BossProblem(;
        f = _ -> zeros(1),
        domain = domain(problem),
        acquisition = missing,
        model = model_j,
        data,
    )
    BOSS.estimate_parameters!(prob, model_fitter; options=boss_options)
    return prob.params.params   # GaussianProcessParams: λ, α, σ
end

function _h_try_fit_lengthscales(problem, model_j, X_local, y_row, model_fitter, boss_options)
    try
        return vec(_h_fit_params(problem, model_j, X_local, y_row, model_fitter, boss_options).λ)
    catch e
        @warn "Local GP MAP fit failed, skipping this center: $e"
        return nothing
    end
end

function _h_try_fit_params(problem, model_j, X_local, y_row, model_fitter, boss_options)
    try
        return _h_fit_params(problem, model_j, X_local, y_row, model_fitter, boss_options)
    catch e
        @warn "Global GP MAP fit failed: $e"
        return nothing
    end
end

# ─── H statistic from a set of per-center log-lengthscale vectors ───────────
# H = max over input dims of the across-center std of log-lengthscales.

function _h_statistic(log_lambdas::AbstractVector{<:AbstractVector{<:Real}})
    length(log_lambdas) < 2 && return NaN
    Λ = reduce(hcat, log_lambdas)   # d × (#successful centers)
    return maximum(std(@view Λ[dim, :]) for dim in 1:size(Λ, 1))
end

# ─── One full H (+ synthetic-replicate) computation given fixed centers/kNN ─
# `y_row` is 1×N (a single problem output dimension evaluated at the shared design X).

function _h_from_dataset(problem, model_j, X, y_row, center_idxs, knn_idxs, model_fitter, boss_options)
    log_lambdas = Vector{Float64}[]
    for c in center_idxs
        nb = @view knn_idxs[:, c]
        λ = _h_try_fit_lengthscales(problem, model_j, X[:, nb], y_row[:, nb], model_fitter, boss_options)
        isnothing(λ) || push!(log_lambdas, log.(λ))
    end
    return _h_statistic(log_lambdas)
end

# ─── Full pipeline for one problem, one output dimension ─────────────────────

function _h_analyse_dim(problem, model_j, X, y_row; m_centers, k_neighbors, n_bootstrap,
                          multistart_local, multistart_global)
    N = size(X, 2)
    fitter_local = OptimizationMAP(; algorithm=NEWUOA(), multistart=multistart_local,
                                     parallel=parallel(), rhoend=1e-4)
    fitter_global = OptimizationMAP(; algorithm=NEWUOA(), multistart=multistart_global,
                                      parallel=parallel(), rhoend=1e-4)
    boss_options = BossOptions(; info=false)

    center_idxs = sort(randperm(N)[1:min(m_centers, N)])
    knn_idxs = _sm_knn_indices(X; k=k_neighbors)   # (k+1) × N, includes self

    H_obs = _h_from_dataset(problem, model_j, X, y_row, center_idxs, knn_idxs, fitter_local, boss_options)
    if isnan(H_obs)
        @warn "Too few successful local fits to compute H_obs — skipping this dimension."
        return (H=NaN, p=NaN)
    end

    # One global stationary GP fit to the full D_N, same priors — the homogeneous null model.
    params_g = _h_try_fit_params(problem, model_j, X, y_row, fitter_global, boss_options)
    if isnothing(params_g)
        @warn "Global GP MAP fit failed — cannot calibrate bootstrap null, returning uncalibrated H."
        return (H=H_obs, p=NaN)
    end
    λ_g, α_g, σ_g = vec(params_g.λ), only(params_g.α), only(params_g.σ)

    mean_fn = BOSS.mean_getindex(model_j.mean, 1)
    fgp = BOSS.finite_gp(X, mean_fn, model_j.kernel, λ_g, α_g, σ_g)

    H_boot = Vector{Float64}(undef, n_bootstrap)
    for b in 1:n_bootstrap
        y_synth = reshape(rand(fgp), 1, N)
        H_boot[b] = _h_from_dataset(problem, model_j, X, y_synth, center_idxs, knn_idxs, fitter_local, boss_options)
    end
    valid = filter(!isnan, H_boot)
    if isempty(valid)
        @warn "All bootstrap replicates failed — returning uncalibrated H."
        return (H=H_obs, p=NaN)
    end
    p = count(>=(H_obs), valid) / length(valid)
    return (H=H_obs, p=p)
end

# ─── Per-(problem, output dimension) entry point ─────────────────────────────
# This is the unit of work a single SLURM job computes (see
# cluster_scripts/script_nonhomogeneity.jl) — valid because GaussianProcess is
# `sliceable`: the joint MAP objective across output dims is a SUM of independent
# per-dimension terms (no shared parameters, no cross-output covariance), so
# fitting dimension j in isolation gives exactly the same λ_j as fitting all
# dimensions jointly. `problem_name` is used both for the RNG seed (so this is
# reproducible and independent across dims/problems run as separate processes)
# and for the saved output filename — pass the RECONSTRUCTIBLE name (e.g.
# `get_name(problem)`, which may include a "_cross" suffix), not the display name.

function run_classify_nonhomogeneity_dim(problem, problem_name::AbstractString, j::Int;
                                           n_design = H_N_DESIGN,
                                           m_centers = H_M_CENTERS,
                                           k_mult = H_K_MULT,
                                           n_bootstrap = H_N_BOOTSTRAP,
                                           multistart_local = H_MULTISTART_LOCAL,
                                           multistart_global = H_MULTISTART_GLOBAL,
                                           save = true,
                                           out_dir = "plots/nonhomogeneity")
    Random.seed!(hash((problem_name, j)))

    lb, ub = domain(problem).bounds
    dx = length(lb)
    f = true_f(problem)
    isnothing(f) && error("$problem_name has no true_f defined — cannot evaluate proxy surface.")

    X = _sm_lhc(lb, ub, n_design)
    y1 = f(@view X[:, 1])
    dy = length(y1)
    @assert 1 <= j <= dy "dim index $j out of range for $problem_name (dy=$dy)"
    Y = Matrix{Float64}(undef, dy, n_design)
    Y[:, 1] = y1
    for i in 2:n_design
        Y[:, i] = f(@view X[:, i])
    end

    full_model = GaussianProcess(;
        mean = prior_mean(problem),
        kernel = BOSS.Matern52Kernel(),
        lengthscale_priors = get_lengthscale_priors(problem),
        amplitude_priors = get_amplitude_priors(problem),
        noise_std_priors = get_noise_std_priors(problem),
    )
    model_j = BOSS.slice(full_model, j)
    k_neighbors = clamp(k_mult * dx, dx + 2, n_design - 1)

    println("Analysing $problem_name dim $j/$dy  (N=$n_design, m=$m_centers, B=$n_bootstrap, k=$k_neighbors)")
    res = _h_analyse_dim(problem, model_j, X, @view(Y[j:j, :]);
                          m_centers, k_neighbors, n_bootstrap, multistart_local, multistart_global)
    println("$problem_name dim $j: H=$(round(res.H,digits=3)) p=$(round(res.p,digits=3))")

    result = (problem=problem_name, dim=j, dx=dx, dy=dy, H=res.H, p=res.p,
              n_design=n_design, m_centers=m_centers, k_neighbors=k_neighbors, n_bootstrap=n_bootstrap)

    if save
        mkpath(out_dir)
        out_path = joinpath(out_dir, "$(problem_name)_dim$(j).jld2")
        @save out_path result
        println("Saved → $out_path")
    end
    return result
end

# ─── Full pipeline for one problem, all output dimensions, aggregated ────────
# In-process convenience wrapper (smoke tests / small sweeps) — loops
# `run_classify_nonhomogeneity_dim` per dimension WITHOUT saving. Gives
# identical results to running each dimension as a separate SLURM job (same
# per-(name,j) RNG seeding), which is what makes the cluster split exact
# rather than an approximation of this function.

function _h_analyse_problem(problem, name; n_design, m_centers, k_mult, n_bootstrap,
                              multistart_local, multistart_global)
    dy = y_dim(problem)
    results = [run_classify_nonhomogeneity_dim(problem, name, j; n_design, m_centers, k_mult,
                                                 n_bootstrap, multistart_local, multistart_global, save=false)
               for j in 1:dy]

    H_per_dim = [r.H for r in results]
    p_per_dim = [r.p for r in results]
    j_worst = argmax(replace(H_per_dim, NaN => -Inf))

    return (name=name, dx=results[1].dx, dy=dy, H=H_per_dim[j_worst], p=p_per_dim[j_worst],
            n_design=n_design, m_centers=m_centers, k_neighbors=results[1].k_neighbors, n_bootstrap=n_bootstrap)
end

_h_analyse(p; kwargs...) = _h_analyse_problem(p, _sm_display_name(p); kwargs...)

# ─── Run ──────────────────────────────────────────────────────────────────────

function run_classify_nonhomogeneity(problems = _SM_ALL_PROBLEMS;
                                       n_design = H_N_DESIGN,
                                       m_centers = H_M_CENTERS,
                                       k_mult = H_K_MULT,
                                       n_bootstrap = H_N_BOOTSTRAP,
                                       multistart_local = H_MULTISTART_LOCAL,
                                       multistart_global = H_MULTISTART_GLOBAL)
    results = [_h_analyse(p; n_design, m_centers, k_mult, n_bootstrap, multistart_local, multistart_global)
               for p in problems]

    println()
    for r in results
        println("$(r.name): H=$(round(r.H,digits=3)) p=$(round(r.p,digits=3)) " *
                "(dx=$(r.dx), dy=$(r.dy), k=$(r.k_neighbors))")
    end

    mkpath("plots")
    open("plots/classify_nonhomogeneity.csv", "w") do io
        println(io, "problem,dx,dy,H,p,n_design,m_centers,k_neighbors,n_bootstrap")
        for r in results
            println(io, "$(r.name),$(r.dx),$(r.dy),$(r.H),$(r.p),$(r.n_design),$(r.m_centers),$(r.k_neighbors),$(r.n_bootstrap)")
        end
    end
    println("\nSaved → plots/classify_nonhomogeneity.csv")
    return results
end

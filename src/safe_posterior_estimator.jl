# A `logpost_estimator` for `MetricCallback` (TV metric) and `PlotCB` (diagnostic plotting) ONLY.
#
# Background: `NonstationaryGP`'s posterior variance can occasionally come out substantially
# negative (not just floating-point noise) far from training data, tripping `_clip_var`'s
# `DomainError` (`BOSS.jl/src/models/gaussian_process.jl`). Currently, `MetricCallback` catches
# this and records NaN for that iteration's TV score — this file avoids that by clamping the
# offending variance to a small positive epsilon instead, with a `@warn` so it stays visible.
#
# Scope guarantee: this file is NEVER used by the actual BO loop (model fitting, acquisition
# optimization) — only by `main.jl`'s `estimator` variable, which itself is only consumed by
# `MetricCallback`/`PlotCB` (see `main.jl`'s use of `estimator`). Swap it in per-run_name (e.g.
# `main_nongp.jl`) by setting `estimator = log_posterior_mean_safe` instead of BOSIP.jl's
# `log_posterior_mean`. Nothing here modifies BOSS.jl/BOSIP.jl.
#
# General for any `Likelihood`: instead of duplicating any likelihood's formula, this wraps the
# `ModelPosterior`/`ModelPosteriorSlice` object itself so `mean`/`var`/`mean_and_var` are the only
# methods that need overriding — every likelihood implementation calls these polymorphically on
# whatever posterior object it's handed, so wrapping the posterior fixes all of them at once.
#
# NOTE ON QUALIFICATION: all `mean`/`var`/`mean_and_var` method definitions below explicitly
# qualify as `BOSS.mean`/`BOSS.var`/`BOSS.mean_and_var` rather than bare names. This is required,
# not stylistic — in a long-lived shared session (this file is `include`d into whatever `Main`
# already has loaded), an unqualified `function mean(...)` can silently extend a *different*
# generic function (e.g. `Statistics.mean`) than the one BOSS.jl/BOSIP.jl's internals actually
# dispatch through, in which case this wrapper would silently never trigger. Verified live that
# the unqualified form fails this way; the qualified form is the fix.

const SAFE_VAR_EPS = 1e-10

## ── Slice-level wrapper (one output dimension) ──────────────────────────────────────

struct SafeModelPosteriorSlice{M<:BOSS.SurrogateModel, P<:BOSS.ModelPosteriorSlice{M}} <: BOSS.ModelPosteriorSlice{M}
    inner::P
end

BOSS.mean(s::SafeModelPosteriorSlice, x::AbstractVecOrMat{<:Real}) = BOSS.mean(s.inner, x)

function BOSS.var(s::SafeModelPosteriorSlice, x::AbstractVector{<:Real})
    try
        return BOSS.var(s.inner, x)
    catch e
        e isa DomainError || rethrow(e)
        @warn "SafeModelPosteriorSlice: clamping variance to $SAFE_VAR_EPS after $(e)."
        return SAFE_VAR_EPS
    end
end
function BOSS.var(s::SafeModelPosteriorSlice, X::AbstractMatrix{<:Real})
    try
        return BOSS.var(s.inner, X)
    catch e
        e isa DomainError || rethrow(e)
        @warn "SafeModelPosteriorSlice: clamping variance to $SAFE_VAR_EPS after $(e)."
        return fill(SAFE_VAR_EPS, size(X, 2))
    end
end

function BOSS.mean_and_var(s::SafeModelPosteriorSlice, x::AbstractVector{<:Real})
    try
        return BOSS.mean_and_var(s.inner, x)
    catch e
        e isa DomainError || rethrow(e)
        @warn "SafeModelPosteriorSlice: clamping variance to $SAFE_VAR_EPS after $(e)."
        return BOSS.mean(s.inner, x), SAFE_VAR_EPS
    end
end
function BOSS.mean_and_var(s::SafeModelPosteriorSlice, X::AbstractMatrix{<:Real})
    try
        return BOSS.mean_and_var(s.inner, X)
    catch e
        e isa DomainError || rethrow(e)
        @warn "SafeModelPosteriorSlice: clamping variance to $SAFE_VAR_EPS after $(e)."
        return BOSS.mean(s.inner, X), fill(SAFE_VAR_EPS, size(X, 2))
    end
end

## ── Joint-level wrapper (models that implement `model_posterior` directly, not via slices) ──

struct SafeModelPosterior{M<:BOSS.SurrogateModel, P<:BOSS.ModelPosterior{M}} <: BOSS.ModelPosterior{M}
    inner::P
end

BOSS.mean(s::SafeModelPosterior, x::AbstractVecOrMat{<:Real}) = BOSS.mean(s.inner, x)

function BOSS.var(s::SafeModelPosterior, x::AbstractVector{<:Real})
    try
        return BOSS.var(s.inner, x)
    catch e
        e isa DomainError || rethrow(e)
        @warn "SafeModelPosterior: clamping variance to $SAFE_VAR_EPS after $(e)."
        return fill(SAFE_VAR_EPS, length(BOSS.mean(s.inner, x)))
    end
end
function BOSS.var(s::SafeModelPosterior, X::AbstractMatrix{<:Real})
    try
        return BOSS.var(s.inner, X)
    catch e
        e isa DomainError || rethrow(e)
        @warn "SafeModelPosterior: clamping variance to $SAFE_VAR_EPS after $(e)."
        return fill(SAFE_VAR_EPS, size(BOSS.mean(s.inner, X)))
    end
end

function BOSS.mean_and_var(s::SafeModelPosterior, x::AbstractVector{<:Real})
    try
        return BOSS.mean_and_var(s.inner, x)
    catch e
        e isa DomainError || rethrow(e)
        @warn "SafeModelPosterior: clamping variance to $SAFE_VAR_EPS after $(e)."
        μ = BOSS.mean(s.inner, x)
        return μ, fill(SAFE_VAR_EPS, length(μ))
    end
end
function BOSS.mean_and_var(s::SafeModelPosterior, X::AbstractMatrix{<:Real})
    try
        return BOSS.mean_and_var(s.inner, X)
    catch e
        e isa DomainError || rethrow(e)
        @warn "SafeModelPosterior: clamping variance to $SAFE_VAR_EPS after $(e)."
        μ = BOSS.mean(s.inner, X)
        return μ, fill(SAFE_VAR_EPS, size(μ))
    end
end

# NOTE: `cov`/`mean_and_cov` are NOT given the same safety net (out of scope for now — the only
# likelihood actually used with `nongp` in this project, `NormalLikelihood`, never calls them;
# only `mean`/`var`/`mean_and_var` are exercised). `std`/`mean_and_std` need no separate handling
# either — their generic defaults (`BOSS.jl/src/posterior.jl`) just call `var`/`mean_and_var` and
# take `sqrt`, so they inherit this clamping automatically via normal dispatch.

## ── Dispatch: rewrap slices in place for sliceable models, else wrap the whole posterior ──

# `DefaultModelPosterior` is the generic container BOSS.jl uses for any model that only
# implements `model_posterior_slice` (e.g. `NonstationaryGP`) — rewrapping its `slices` in place
# lets its own existing per-dimension aggregation loop (`var.(post.slices, Ref(x))`, unchanged)
# isolate a bad dimension to just that dimension, instead of clamping the whole output vector.
make_safe(post::BOSS.DefaultModelPosterior) =
    BOSS.DefaultModelPosterior([SafeModelPosteriorSlice(s) for s in post.slices])

# Fallback for models that implement `model_posterior` directly (e.g. `GaussianProcess`,
# `WarpedGaussianProcess`) — no per-dimension decomposition available generically, so a failure
# anywhere in the joint `var`/`mean_and_var` call clamps the whole output.
make_safe(post::BOSS.ModelPosterior) = SafeModelPosterior(post)

## ── Custom `logpost_estimator`, mirroring BOSIP.jl's `log_posterior_mean`/`log_likelihood_mean` ──
## dispatch exactly, but wrapping the model posterior in `make_safe` right after it's obtained.

function log_posterior_mean_safe(bosip::BOSIP.BosipProblem)
    x_prior = bosip.x_prior
    log_like_mean = log_likelihood_mean_safe(bosip)
    log_post_mean(x) = BOSIP._log_prior(x_prior, x) .+ log_like_mean(x)
    return log_post_mean
end

log_likelihood_mean_safe(bosip::BOSIP.BosipProblem) =
    log_likelihood_mean_safe(typeof(bosip.problem.params), bosip)

function log_likelihood_mean_safe(::Type{<:BOSS.UniFittedParams}, bosip::BOSIP.BosipProblem)
    model_post = make_safe(BOSS.model_posterior(bosip.problem))
    return BOSIP.log_likelihood_mean(bosip.likelihood, model_post)
end

function log_likelihood_mean_safe(::Type{<:BOSS.MultiFittedParams}, bosip::BOSIP.BosipProblem)
    model_posts = make_safe.(BOSS.model_posterior(bosip.problem))
    sample_count = length(model_posts)
    log_like_means = BOSIP.log_likelihood_mean.(Ref(bosip.likelihood), model_posts)

    function log_like_mean(x)
        return log.(mapreduce(f -> exp.(f(x)), .+, log_like_means) ./ sample_count)
    end
    return log_like_mean
end

# JLD2 migration shims for continuing old checkpoints (eiv/nongp/immd runs on
# DuffingProblem, saved before two BOSIP.jl/BOSS.jl struct changes) under the
# current package code. Purely additive (new `rconvert` methods for the exact
# old on-disk field layout) - does not alter the live TVMetric/MAPParams types
# or any other run/loading path.
#
# 1. BOSIP.jl commit b2e6302 ("precalculate TVMetric reference logpdf values",
#    2025-10-06) added a `true_logvals::Union{Nothing,Vector{Float64}}` field to
#    TVMetric (previously just `grid`, `log_ws`). Old checkpoints lack it -
#    reconstruct with `true_logvals = nothing` (equivalent behavior: values get
#    recomputed each call instead of cached, just a bit slower).
#
# 2. BOSS.jl commit 5f06d33 ("rename loglike to correct Bayesian nomenclature")
#    renamed `MAPParams.loglike` -> `MAPParams.logpost`. Old checkpoints have
#    the old field name - just pass it through positionally.

import JLD2

# NB: JLD2 calls `rconvert(T, x)` with T = the *field's declared type* (often an
# abstract supertype), not the on-disk struct's own type name - so these methods
# must be keyed on the abstract field types (DistributionMetric, Union{Nothing,
# FittedParams}), confirmed from the actual failing call sites' stacktraces.

JLD2.rconvert(::Type{DistributionMetric}, x::JLD2.ReconstructedMutable{:TVMetric, (:grid, :log_ws), Tuple{Any, Any}}) =
    TVMetric(x.grid, x.log_ws, nothing)

const _OldMAPParamsGP = JLD2.ReconstructedMutable{Symbol("MAPParams{GaussianProcess}"), (:params, :loglike), Tuple{Any, Any}}
const _OldMAPParamsNonstatGP = JLD2.ReconstructedMutable{Symbol("MAPParams{NonstationaryGP}"), (:params, :loglike), Tuple{Any, Any}}

_migrate_mapparams(x::Union{_OldMAPParamsGP, _OldMAPParamsNonstatGP}) = MAPParams(x.params, x.loglike)
_migrate_mapparams(x) = x  # already-converted / not this legacy shape - pass through

JLD2.rconvert(::Type{Union{Nothing, FittedParams}}, x::_OldMAPParamsGP) =
    _migrate_mapparams(x)
JLD2.rconvert(::Type{Union{Nothing, FittedParams}}, x::_OldMAPParamsNonstatGP) =
    _migrate_mapparams(x)

# 3. BOSIP.jl commit 734e0cc-adjacent history: BosipOptions.parallel_evals changed
#    from Symbol (:serial/:parallel/:distributed) to Bool. Old default was :parallel.
JLD2.rconvert(::Type{BosipOptions}, x::JLD2.ReconstructedMutable{Symbol("BosipOptions{BOSIP.CombinedCallback}"), (:info, :debug, :parallel_evals, :callback), Tuple{Bool, Bool, Symbol, BOSIP.CombinedCallback}}) =
    BosipOptions(; info=x.info, debug=x.debug, parallel_evals=(x.parallel_evals != :serial), callback=x.callback)

# 4. BossProblem.f holds the simulator closure (e.g. DuffingModule's `model_target`).
# JLD2 cannot deserialize closures across Julia sessions in general - but this
# specific closure is stateless (captures nothing external, see
# src/problems/physical/duffing.jl's `_get_model_target`), so it's semantically
# identical to freshly re-obtaining it via `simulator(problem)` on the live,
# already-reconstructed `problem` (a global set earlier in this script, before this
# file is included) - nothing is actually lost.
JLD2.rconvert(::Type{BossProblem}, x::JLD2.ReconstructedMutable{Symbol("BossProblem{#model_target}"), (:f, :domain, :y_max, :acquisition, :model, :params, :data, :consistent), Tuple{JLD2.ReconstructedSingleton{Symbol("#model_target")}, Any, Any, Any, Any, Any, Any, Bool}}) =
    BossProblem(simulator(problem), x.domain, x.y_max, x.acquisition, x.model, _migrate_mapparams(x.params), x.data, x.consistent)

# 5. BOSIP.jl de-parametrized NormalLikelihood (dropped the Vector{Float64} type
# param) - same two fields, just no longer generic.
JLD2.rconvert(::Type{Likelihood}, x::JLD2.ReconstructedMutable{Symbol("NormalLikelihood{Vector{Float64}}"), (:z_obs, :std_obs), Tuple{Any, Any}}) =
    NormalLikelihood(x.z_obs, x.std_obs)

# 6. ParametrizedGP.act_func (nonstationary_gp, i.e. "nongp" runs) is a closure
# `x -> λ_d * x + λ_lb` (maps the activation output from [0,1] to [λ_lb, λ_ub],
# see BOSS.jl src/models/nonstationary_gp/nonstationary_gp.jl:463 - same formula
# in the current code, unchanged). The compiler-assigned anonymous closure name
# (e.g. "#190#193") is process-specific and can differ per saved checkpoint, so
# this matches on the captured field names/types only (`where {S}`), not the name.
JLD2.rconvert(::Type{Function}, x::JLD2.ReconstructedStatic{S, (:λ_d, :λ_lb), Tuple{Float64, Float64}}) where {S} =
    (v -> x.λ_d * v + x.λ_lb)

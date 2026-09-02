"""
UNPAIRED, iteration-normalized, log-iteration-weighted AUC-difference scoring —
supersedes the "final" (each run's own last iteration) and "fair" (single
shared-budget point) conventions for every comparative/"winner" plot (5a, 5b, 5c).

Renamed 2026-09-01 from `compute_paired_auc_scores.jl` — that name was WRONG:
despite the file/function names, the live implementation had already stopped
doing per-run pairing on 2026-08-05 (see "Why" below) and was computing a
DIFFERENCE OF MEDIAN CURVES instead, an unpaired statistic. This file keeps
that unpaired computation (unchanged, just correctly named); the genuinely
paired variant (median of per-run AUC differences, restored from the dead code
that used to sit commented out in the old file) now lives in its own file,
`compute_paired_run_auc_scores.jl`, with its own output CSVs (`*_paired.csv`).

## Metric (per problem, per pair of configs candidate vs baseline) — UNPAIRED

DIFFERENCE OF MEDIAN CURVES (not median of per-run differences — see "Why" below):
  1. T = shared window = min raw score-array length over ALL loaded runs of BOTH
     configs (the widest window for which every run of both configs has data).
  2. median_candidate(t) = median, across ALL of the candidate's runs that have a
     finite/non-negative reading at iteration t, of that reading — for each
     t = 1..T. (Independently for the baseline: median_baseline(t).) These are
     two single "median curves", not one number per run. Each config's runs are
     treated as an UNPAIRED pool here — run index i of the candidate is not
     matched against run index i of the baseline.
  3. AUC_candidate = trapezoidal integral of median_candidate(t) over
     LOG(iteration index) — i.e. weighted by 1/t — from iteration 1 to T.
     (Likewise AUC_baseline from median_baseline(t).)
  4. mean_candidate = AUC_candidate / log(T)  — a LOG-ITERATION-weighted
     time-average of the median curve, bounded in [0,1] since TV itself is
     bounded in [0,1].
  5. score = mean_candidate - mean_baseline  (bounded in [-1,1])

Negative → candidate wins (lower TV). Positive → baseline wins.

## Why "difference of median curves", not "median of differences" (2026-08-05)

The original version of this metric computed, per matched run index i (paired by
shared warm-start seed), a per-run AUC-diff, then took median(diff_i) over runs.
That is the statistically more defensible choice for genuinely paired data (it
cancels out whatever makes a given seed easy/hard for BOTH configs) — see
`compute_paired_run_auc_scores.jl` for that restored variant. This version was
tried as an explicit alternative: it discards the run-level pairing and instead
asks "what's the gap between the two configs' typical (median) trajectories,"
treating each config's 20 runs as an unpaired sample at every iteration.
`median(A) - median(B) ≠ median(A - B)` in general, so this is a genuinely
different quantity, not just a reordering of the same computation. This is the
version currently used as the default for plots 5a/5b/5c.

## Why log(iteration), not log(TV)

TV is kept on its RAW/linear scale deliberately — log(TV) is unbounded (→ -∞ as
TV→0), which breaks the "value bounded in [0,1]" property the whole [-1,1] score
range depends on. But a PLAIN iteration-linear average weighs iteration 101-200
exactly as heavily as iterations 1-100, even though most of the interesting BO
behavior — and most of what a practitioner cares about ("how fast did it get
good?") — happens early. Weighting by 1/t (integrating over log(t) instead of
t) gives that emphasis without touching TV's linear scale.

## Outputs
  plots/acq_scores_auc.csv       — eiv_vs_maxvar and immd_vs_maxvar, all 37 problems
  plots/warpedgp_scores_auc.csv  — warpedgp_vs_standard, all 37 problems
  plots/nongp_scores_auc.csv     — nongp_vs_standard, all 37 problems

## Usage
    include("src/compute_acq_scores_final.jl")   # for _GROUP_A/_GROUP_B_PROBS/_GROUP_C/_GROUP_D_PROBS
    include("src/compute_median_curve_auc_scores.jl")
"""

using JLD2
using Statistics: median

## ── Raw-score loading ────────────────────────────────────────────────────────

function _pa_load(prob_dir, run_name, idx)
    fpath = joinpath(prob_dir, "$(run_name)_$(idx)_TVmetric.jld2")
    isfile(fpath) || return nothing
    try
        return load(fpath, "score")
    catch
        return nothing
    end
end

function _pa_collect(dir, key; n::Int)
    d = Dict{Int,Vector{Float64}}()
    for i in 1:n
        v = _pa_load(dir, key, i)
        v === nothing || (d[i] = v)
    end
    return d
end

## ── Trapezoidal AUC over a raw score vector, restricted to indices 1:T ───────
## Integrated over LOG(iteration), not raw iteration — equivalent to weighting
## each point by 1/t, so earlier iterations contribute more (see module
## docstring for the "why"). TV itself stays linear/raw (not log).
## Skips non-finite/negative entries, connecting directly across any gaps.
## Returns `nothing` if fewer than 2 valid points exist (can't integrate).

function _pa_auc(score::AbstractVector{<:Real}, T::Int)
    Tc = min(T, length(score))
    idxs = [i for i in 1:Tc if isfinite(score[i]) && score[i] >= 0.0]
    length(idxs) < 2 && return nothing
    auc = 0.0
    for k in 1:(length(idxs) - 1)
        i, j = idxs[k], idxs[k+1]
        auc += 0.5 * (score[i] + score[j]) * (log(j) - log(i))
    end
    return auc
end

## ── Per-iteration median curve across a group's runs ──────────────────────────
## curve[t] = median of all runs' finite/non-negative readings at iteration t
## (NaN where no run has a valid reading at that iteration — _pa_auc already
## skips NaN entries, so this composes directly with it).

function _median_curve(vecs::Dict{Int,Vector{Float64}}, T::Int)
    curve = fill(NaN, T)
    for t in 1:T
        vals = Float64[]
        for v in values(vecs)
            (t <= length(v) && isfinite(v[t]) && v[t] >= 0.0) && push!(vals, v[t])
        end
        isempty(vals) || (curve[t] = median(vals))
    end
    return curve
end

## ── AUC-difference score for one problem/comparison ───────────────────────────
## UNPAIRED: difference of median curves (see module docstring for the "why" vs.
## the genuinely paired median-of-per-run-differences version, which now lives
## in compute_paired_run_auc_scores.jl's `paired_run_auc_diff`).

function median_curve_auc_diff(dir_cand, key_cand, dir_base, key_base; n_cand::Int=20, n_base::Int=20)
    vecs_c = _pa_collect(dir_cand, key_cand; n=n_cand)
    vecs_b = _pa_collect(dir_base, key_base; n=n_base)
    matched = sort(collect(intersect(keys(vecs_c), keys(vecs_b))))   # kept for quality reporting only

    empty_result = (median_diff=NaN, n_pairs=0, T=0,
                     cand_n=length(vecs_c), base_n=length(vecs_b),
                     cand_nan=0, base_nan=0,
                     cand_miniter=0, cand_maxiter=0, base_miniter=0, base_maxiter=0)
    (isempty(vecs_c) || isempty(vecs_b)) && return empty_result

    all_vecs = vcat(collect(values(vecs_c)), collect(values(vecs_b)))
    T = minimum(length.(all_vecs))
    denom = T > 1 ? log(T) : 1.0

    curve_c = _median_curve(vecs_c, T)
    curve_b = _median_curve(vecs_b, T)
    auc_c = _pa_auc(curve_c, T)
    auc_b = _pa_auc(curve_b, T)
    diff  = (auc_c === nothing || auc_b === nothing) ? NaN : (auc_c / denom - auc_b / denom)

    c_lens = length.(values(vecs_c)); b_lens = length.(values(vecs_b))
    c_nan  = sum(count(isnan, v) for v in values(vecs_c); init=0)
    b_nan  = sum(count(isnan, v) for v in values(vecs_b); init=0)

    return (median_diff = diff,
            n_pairs = length(matched), T = T,
            cand_n = length(vecs_c), base_n = length(vecs_b),
            cand_nan = c_nan, base_nan = b_nan,
            cand_miniter = minimum(c_lens), cand_maxiter = maximum(c_lens),
            base_miniter = minimum(b_lens), base_maxiter = maximum(b_lens))
end

## ── compute_acq_scores_auc: EIV vs MaxVar, IMMD vs MaxVar (all 4 groups) ──────

function compute_acq_scores_auc()
    rows = NamedTuple[]

    function add_row!(prob, typ, mv_dir, mv_key, mv_n, ei_dir, ei_key, ei_n, im_dir, im_key, im_n)
        ei = median_curve_auc_diff(ei_dir, ei_key, mv_dir, mv_key; n_cand=ei_n, n_base=mv_n)
        im = median_curve_auc_diff(im_dir, im_key, mv_dir, mv_key; n_cand=im_n, n_base=mv_n)
        push!(rows, (problem=prob, type=typ, eiv=ei, immd=im))
        @info "AUC $prob  eiv_diff=$(round(ei.median_diff, digits=4)) (T=$(ei.T), n=$(ei.n_pairs))  immd_diff=$(round(im.median_diff, digits=4)) (T=$(im.T), n=$(im.n_pairs))"
    end

    for prob in _GROUP_A
        d = joinpath("data-bosip-norm", prob)
        isdir(d) || (@warn "Missing: $d"; continue)
        add_row!(prob, "BIP", d, "standard", 20, d, "eiv", 20, d, "immd", 20)
    end
    for prob in _GROUP_B_PROBS
        d = joinpath("data-opt-functions", prob * "_cross")
        isdir(d) || (@warn "Missing: $d"; continue)
        add_row!(prob, "Opt", d, "maxvar", 20, d, "eiv", 20, d, "immd", 20)
    end
    for prob in _GROUP_C
        d = joinpath("data-bosip-norm", prob)
        isdir(d) || (@warn "Missing: $d"; continue)
        add_row!(prob, "BIP", d, "standard", 20, d, "eiv", 5, d, "immd", 5)
    end
    for prob in _GROUP_D_PROBS
        d = joinpath("data-opt-functions", prob * "_cross")
        isdir(d) || (@warn "Missing: $d"; continue)
        add_row!(prob, "Opt", d, "maxvar", 20, d, "eiv", 5, d, "immd", 5)
    end

    mkpath("plots")
    open("plots/acq_scores_auc.csv", "w") do io
        println(io, "problem,type," *
                    "eiv_diff,eiv_T,eiv_npairs,maxvar_n,maxvar_miniter,maxvar_maxiter,maxvar_nan," *
                    "eiv_n,eiv_miniter,eiv_maxiter,eiv_nan," *
                    "immd_diff,immd_T,immd_npairs,immd_n,immd_miniter,immd_maxiter,immd_nan")
        for r in rows
            println(io, "$(r.problem),$(r.type)," *
                        "$(r.eiv.median_diff),$(r.eiv.T),$(r.eiv.n_pairs),$(r.eiv.base_n),$(r.eiv.base_miniter),$(r.eiv.base_maxiter),$(r.eiv.base_nan)," *
                        "$(r.eiv.cand_n),$(r.eiv.cand_miniter),$(r.eiv.cand_maxiter),$(r.eiv.cand_nan)," *
                        "$(r.immd.median_diff),$(r.immd.T),$(r.immd.n_pairs),$(r.immd.cand_n),$(r.immd.cand_miniter),$(r.immd.cand_maxiter),$(r.immd.cand_nan)")
        end
    end
    @info "Saved $(length(rows)) rows → plots/acq_scores_auc.csv"
    return rows
end

## ── compute_warpedgp_scores_auc: WarpedGP vs Standard (all 4 groups) ─────────

function compute_warpedgp_scores_auc()
    rows = NamedTuple[]

    function add_row!(prob, typ, std_dir, std_key, std_n, wgp_dir)
        isdir(std_dir) || (@warn "Missing: $std_dir"; return)
        isdir(wgp_dir) || (@warn "Missing: $wgp_dir"; return)
        r = median_curve_auc_diff(wgp_dir, "warpedgp-yja-maxvar", std_dir, std_key; n_cand=20, n_base=std_n)
        push!(rows, (problem=prob, type=typ, wgp=r))
        @info "AUC-WGP $prob  diff=$(round(r.median_diff, digits=4)) (T=$(r.T), n=$(r.n_pairs))"
    end

    for prob in _GROUP_A
        add_row!(prob, "BIP", joinpath("data-bosip-norm", prob), "standard", 20, joinpath("data-warpedgp2", prob))
    end
    for prob in _GROUP_B_PROBS
        add_row!(prob, "Opt", joinpath("data-opt-functions", prob * "_cross"), "maxvar", 20, joinpath("data-warpedgp2", prob * "_cross"))
    end
    for prob in _GROUP_C
        add_row!(prob, "BIP", joinpath("data-bosip-norm", prob), "standard", 20, joinpath("data-warpedgp2", prob))
    end
    for prob in _GROUP_D_PROBS
        add_row!(prob, "Opt", joinpath("data-opt-functions", prob * "_cross"), "maxvar", 20, joinpath("data-warpedgp2", prob * "_cross"))
    end

    mkpath("plots")
    open("plots/warpedgp_scores_auc.csv", "w") do io
        println(io, "problem,type,warpedgp_diff,warpedgp_T,warpedgp_npairs," *
                    "standard_n,standard_miniter,standard_maxiter,standard_nan," *
                    "warpedgp_n,warpedgp_miniter,warpedgp_maxiter,warpedgp_nan")
        for r in rows
            w = r.wgp
            println(io, "$(r.problem),$(r.type),$(w.median_diff),$(w.T),$(w.n_pairs)," *
                        "$(w.base_n),$(w.base_miniter),$(w.base_maxiter),$(w.base_nan)," *
                        "$(w.cand_n),$(w.cand_miniter),$(w.cand_maxiter),$(w.cand_nan)")
        end
    end
    @info "Saved $(length(rows)) rows → plots/warpedgp_scores_auc.csv"
    return rows
end

## ── compute_nongp_scores_auc: NonstatGP vs Standard (all 4 groups) ───────────
## DiffusionProblem5D: exclude the 6 crashed nongp runs (idx 6,8,10,11,13,16),
## same as compute_fair_scores.jl — otherwise T collapses to ~4 for the problem.

const _DIFFUSION5D_NONGP_CRASHED_AUC = [6, 8, 10, 11, 13, 16]

function _pa_collect_excl(dir, key; n::Int, exclude::Vector{Int}=Int[])
    d = Dict{Int,Vector{Float64}}()
    for i in 1:n
        i in exclude && continue
        v = _pa_load(dir, key, i)
        v === nothing || (d[i] = v)
    end
    return d
end

## UNPAIRED, mirrors median_curve_auc_diff above (see that function/the module
## docstring). Genuinely paired version: `paired_run_auc_diff_excl` in
## compute_paired_run_auc_scores.jl.

function median_curve_auc_diff_excl(dir_cand, key_cand, dir_base, key_base;
                               n_cand::Int=20, n_base::Int=20, exclude_cand::Vector{Int}=Int[])
    vecs_c = _pa_collect_excl(dir_cand, key_cand; n=n_cand, exclude=exclude_cand)
    vecs_b = _pa_collect(dir_base, key_base; n=n_base)
    matched = sort(collect(intersect(keys(vecs_c), keys(vecs_b))))   # kept for quality reporting only
    empty_result = (median_diff=NaN, n_pairs=0, T=0, cand_n=length(vecs_c), base_n=length(vecs_b),
                     cand_nan=0, base_nan=0, cand_miniter=0, cand_maxiter=0, base_miniter=0, base_maxiter=0)
    (isempty(vecs_c) || isempty(vecs_b)) && return empty_result
    all_vecs = vcat(collect(values(vecs_c)), collect(values(vecs_b)))
    T = minimum(length.(all_vecs))
    denom = T > 1 ? log(T) : 1.0

    curve_c = _median_curve(vecs_c, T)
    curve_b = _median_curve(vecs_b, T)
    auc_c = _pa_auc(curve_c, T)
    auc_b = _pa_auc(curve_b, T)
    diff  = (auc_c === nothing || auc_b === nothing) ? NaN : (auc_c / denom - auc_b / denom)

    c_lens = length.(values(vecs_c)); b_lens = length.(values(vecs_b))
    c_nan  = sum(count(isnan, v) for v in values(vecs_c); init=0)
    b_nan  = sum(count(isnan, v) for v in values(vecs_b); init=0)
    return (median_diff = diff, n_pairs = length(matched), T = T,
            cand_n = length(vecs_c), base_n = length(vecs_b), cand_nan = c_nan, base_nan = b_nan,
            cand_miniter = minimum(c_lens), cand_maxiter = maximum(c_lens),
            base_miniter = minimum(b_lens), base_maxiter = maximum(b_lens))
end

function compute_nongp_scores_auc()
    rows = NamedTuple[]

    function add_row!(prob, typ, std_dir, std_key, std_n, nongp_dir; exclude_nongp::Vector{Int}=Int[])
        isdir(std_dir) || (@warn "Missing: $std_dir"; return)
        isdir(nongp_dir) || (@warn "Missing: $nongp_dir"; return)
        r = median_curve_auc_diff_excl(nongp_dir, "nongp", std_dir, std_key; n_cand=20, n_base=std_n, exclude_cand=exclude_nongp)
        push!(rows, (problem=prob, type=typ, ng=r))
        @info "AUC-NonGP $prob  diff=$(round(r.median_diff, digits=4)) (T=$(r.T), n=$(r.n_pairs))"
    end

    for prob in _GROUP_A
        add_row!(prob, "BIP", joinpath("data-bosip-norm", prob), "standard", 20, joinpath("data-bosip-norm", prob))
    end
    for prob in _GROUP_B_PROBS
        add_row!(prob, "Opt", joinpath("data-opt-functions", prob * "_cross"), "maxvar", 20, joinpath("data-opt-functions", prob * "_cross"))
    end
    for prob in _GROUP_C
        excl = prob == "DiffusionProblem5D" ? _DIFFUSION5D_NONGP_CRASHED_AUC : Int[]
        add_row!(prob, "BIP", joinpath("data-bosip-norm", prob), "standard", 20, joinpath("data-bosip-norm", prob); exclude_nongp=excl)
    end
    for prob in _GROUP_D_PROBS
        add_row!(prob, "Opt", joinpath("data-opt-functions", prob * "_cross"), "maxvar", 20, joinpath("data-opt-functions", prob * "_cross"))
    end

    mkpath("plots")
    open("plots/nongp_scores_auc.csv", "w") do io
        println(io, "problem,type,nongp_diff,nongp_T,nongp_npairs," *
                    "standard_n,standard_miniter,standard_maxiter,standard_nan," *
                    "nongp_n,nongp_miniter,nongp_maxiter,nongp_nan")
        for r in rows
            n = r.ng
            println(io, "$(r.problem),$(r.type),$(n.median_diff),$(n.T),$(n.n_pairs)," *
                        "$(n.base_n),$(n.base_miniter),$(n.base_maxiter),$(n.base_nan)," *
                        "$(n.cand_n),$(n.cand_miniter),$(n.cand_maxiter),$(n.cand_nan)")
        end
    end
    @info "Saved $(length(rows)) rows → plots/nongp_scores_auc.csv"
    return rows
end

## ── Run immediately on include ───────────────────────────────────────────────

acq_auc_rows      = compute_acq_scores_auc()
warpedgp_auc_rows = compute_warpedgp_scores_auc()
nongp_auc_rows    = compute_nongp_scores_auc()

println()
println(rpad("Problem", 32), "│ ", rpad("Type", 4), "│  T  │ n │ ", rpad("EIV-MaxVar", 11), "│  T  │ n │ ", rpad("IMMD-MaxVar", 11))
for r in acq_auc_rows
    e, m = r.eiv, r.immd
    println(rpad(r.problem, 32), "│ ", rpad(r.type, 4), "│ ", lpad(e.T,3), " │",
            lpad(e.n_pairs,2), " │ ", lpad(round(e.median_diff; digits=4), 9), " │ ",
            lpad(m.T,3), " │", lpad(m.n_pairs,2), " │ ", lpad(round(m.median_diff; digits=4), 9))
end

println()
println(rpad("Problem", 32), "│ ", rpad("Type", 4), "│  T  │ n │ WarpedGP-Standard")
for r in warpedgp_auc_rows
    w = r.wgp
    println(rpad(r.problem, 32), "│ ", rpad(r.type, 4), "│ ", lpad(w.T,3), " │", lpad(w.n_pairs,2), " │ ", round(w.median_diff; digits=4))
end

println()
println(rpad("Problem", 32), "│ ", rpad("Type", 4), "│  T  │ n │ NonstatGP-Standard")
for r in nongp_auc_rows
    n = r.ng
    println(rpad(r.problem, 32), "│ ", rpad(r.type, 4), "│ ", lpad(n.T,3), " │", lpad(n.n_pairs,2), " │ ", round(n.median_diff; digits=4))
end

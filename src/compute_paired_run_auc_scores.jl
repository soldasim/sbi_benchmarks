"""
PAIRED, iteration-normalized, log-iteration-weighted AUC-difference scoring —
the genuinely paired sibling of `compute_median_curve_auc_scores.jl`.

Restored 2026-09-01 from the dead code that used to sit commented out (unused)
inside the old, misleadingly-named `compute_paired_auc_scores.jl` — that file's
LIVE implementation had actually been unpaired since 2026-08-05 despite its
name; it's now correctly renamed to `compute_median_curve_auc_scores.jl`. This
file restores the ORIGINAL paired methodology as its own first-class script,
so both variants can be computed and plotted side by side without one silently
shadowing the other.

## Metric (per problem, per pair of configs candidate vs baseline) — PAIRED

MEDIAN OF PER-RUN DIFFERENCES (paired by shared warm-start seed/run index):
  1. T = shared window = min raw score-array length over ALL loaded runs of BOTH
     configs (the widest window for which every run of both configs has data).
  2. For each run index i present in BOTH configs' run sets: integrate the RAW
     (not log-TV) score over iteration 1..T via `_pa_auc` (log-iteration-
     weighted trapezoidal AUC — see "Why log(iteration)" below), divide by
     log(T) to get a per-run, log-iteration-weighted mean-TV value bounded in
     [0,1], and take candidate_i − baseline_i.
  3. score = median of that per-run difference over all matched run indices i —
     bounded in [-1,1] by construction (TV itself is bounded in [0,1]).

Negative → candidate wins (lower TV). Positive → baseline wins.

This is the statistically more defensible choice for genuinely paired data (it
cancels out whatever makes a given seed easy/hard for BOTH configs), as opposed
to `compute_median_curve_auc_scores.jl`'s "difference of median curves", which
discards the run-index pairing and instead asks "what's the gap between the two
configs' typical (median) trajectories." `median(A) - median(B) ≠ median(A - B)`
in general, so these are genuinely different quantities, not two computations
of the same thing — see that file's docstring for the fuller discussion of why
the unpaired version became the current default for plots 5a/5b/5c despite this
one being closer to standard paired-comparison practice.

## Why log(iteration), not log(TV)

Same reasoning as compute_median_curve_auc_scores.jl: TV stays on its RAW/linear
scale (log(TV) is unbounded, breaks the [-1,1] score-range guarantee), but a
plain iteration-linear average would weigh iteration 101-200 exactly as heavily
as iterations 1-100. Weighting by 1/t (integrating over log(t) instead of t)
emphasizes early iterations — "how fast did it get good?" — without touching
TV's linear scale.

## Outputs
  plots/acq_scores_auc_paired.csv       — eiv_vs_maxvar and immd_vs_maxvar, all 37 problems
  plots/warpedgp_scores_auc_paired.csv  — warpedgp_vs_standard, all 37 problems
  plots/nongp_scores_auc_paired.csv     — nongp_vs_standard, all 37 problems

## Usage
    include("src/compute_acq_scores_final.jl")   # for _GROUP_A/_GROUP_B_PROBS/_GROUP_C/_GROUP_D_PROBS
    include("src/compute_paired_run_auc_scores.jl")
"""

using JLD2
using Statistics: median

## ── Raw-score loading (identical to compute_median_curve_auc_scores.jl; kept
## self-contained here rather than shared, so this file works as a standalone
## copy — see that file for the sibling unpaired implementation) ─────────────

function _pra_load(prob_dir, run_name, idx)
    fpath = joinpath(prob_dir, "$(run_name)_$(idx)_TVmetric.jld2")
    isfile(fpath) || return nothing
    try
        return load(fpath, "score")
    catch
        return nothing
    end
end

function _pra_collect(dir, key; n::Int)
    d = Dict{Int,Vector{Float64}}()
    for i in 1:n
        v = _pra_load(dir, key, i)
        v === nothing || (d[i] = v)
    end
    return d
end

## ── Trapezoidal AUC over a raw score vector, restricted to indices 1:T ───────
## Integrated over LOG(iteration), not raw iteration (see module docstring).
## Skips non-finite/negative entries, connecting directly across any gaps.
## Returns `nothing` if fewer than 2 valid points exist (can't integrate).

function _pra_auc(score::AbstractVector{<:Real}, T::Int)
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

## ── AUC-difference score for one problem/comparison ───────────────────────────
## PAIRED: median of per-run AUC differences, matched by run index (shared
## warm-start seed). See module docstring for the "why" vs. the unpaired
## difference-of-median-curves version in compute_median_curve_auc_scores.jl.

function paired_run_auc_diff(dir_cand, key_cand, dir_base, key_base; n_cand::Int=20, n_base::Int=20)
    vecs_c = _pra_collect(dir_cand, key_cand; n=n_cand)
    vecs_b = _pra_collect(dir_base, key_base; n=n_base)
    matched = sort(collect(intersect(keys(vecs_c), keys(vecs_b))))

    empty_result = (median_diff=NaN, n_pairs=0, T=0,
                     cand_n=length(vecs_c), base_n=length(vecs_b),
                     cand_nan=0, base_nan=0,
                     cand_miniter=0, cand_maxiter=0, base_miniter=0, base_maxiter=0)
    isempty(matched) && return empty_result

    all_vecs = vcat(collect(values(vecs_c)), collect(values(vecs_b)))
    T = minimum(length.(all_vecs))
    denom = T > 1 ? log(T) : 1.0

    diffs = Float64[]
    for i in matched
        auc_c = _pra_auc(vecs_c[i], T)
        auc_b = _pra_auc(vecs_b[i], T)
        (auc_c === nothing || auc_b === nothing) && continue
        push!(diffs, (auc_c / denom) - (auc_b / denom))
    end

    c_lens = length.(values(vecs_c)); b_lens = length.(values(vecs_b))
    c_nan  = sum(count(isnan, v) for v in values(vecs_c); init=0)
    b_nan  = sum(count(isnan, v) for v in values(vecs_b); init=0)

    return (median_diff = isempty(diffs) ? NaN : median(diffs),
            n_pairs = length(diffs), T = T,
            cand_n = length(vecs_c), base_n = length(vecs_b),
            cand_nan = c_nan, base_nan = b_nan,
            cand_miniter = minimum(c_lens), cand_maxiter = maximum(c_lens),
            base_miniter = minimum(b_lens), base_maxiter = maximum(b_lens))
end

## ── compute_acq_scores_paired: EIV vs MaxVar, IMMD vs MaxVar (all 4 groups) ──

function compute_acq_scores_paired()
    rows = NamedTuple[]

    function add_row!(prob, typ, mv_dir, mv_key, mv_n, ei_dir, ei_key, ei_n, im_dir, im_key, im_n)
        ei = paired_run_auc_diff(ei_dir, ei_key, mv_dir, mv_key; n_cand=ei_n, n_base=mv_n)
        im = paired_run_auc_diff(im_dir, im_key, mv_dir, mv_key; n_cand=im_n, n_base=mv_n)
        push!(rows, (problem=prob, type=typ, eiv=ei, immd=im))
        @info "AUC-paired $prob  eiv_diff=$(round(ei.median_diff, digits=4)) (T=$(ei.T), n=$(ei.n_pairs))  immd_diff=$(round(im.median_diff, digits=4)) (T=$(im.T), n=$(im.n_pairs))"
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
    open("plots/acq_scores_auc_paired.csv", "w") do io
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
    @info "Saved $(length(rows)) rows → plots/acq_scores_auc_paired.csv"
    return rows
end

## ── compute_warpedgp_scores_paired: WarpedGP vs Standard (all 4 groups) ──────

function compute_warpedgp_scores_paired()
    rows = NamedTuple[]

    function add_row!(prob, typ, std_dir, std_key, std_n, wgp_dir)
        isdir(std_dir) || (@warn "Missing: $std_dir"; return)
        isdir(wgp_dir) || (@warn "Missing: $wgp_dir"; return)
        r = paired_run_auc_diff(wgp_dir, "warpedgp-yja-maxvar", std_dir, std_key; n_cand=20, n_base=std_n)
        push!(rows, (problem=prob, type=typ, wgp=r))
        @info "AUC-paired-WGP $prob  diff=$(round(r.median_diff, digits=4)) (T=$(r.T), n=$(r.n_pairs))"
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
    open("plots/warpedgp_scores_auc_paired.csv", "w") do io
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
    @info "Saved $(length(rows)) rows → plots/warpedgp_scores_auc_paired.csv"
    return rows
end

## ── compute_nongp_scores_paired: NonstatGP vs Standard (all 4 groups) ────────
## DiffusionProblem5D: exclude the 6 crashed nongp runs (idx 6,8,10,11,13,16),
## same as compute_fair_scores.jl / compute_median_curve_auc_scores.jl —
## otherwise T collapses to ~4 for the problem.

const _DIFFUSION5D_NONGP_CRASHED_AUC_PAIRED = [6, 8, 10, 11, 13, 16]

function _pra_collect_excl(dir, key; n::Int, exclude::Vector{Int}=Int[])
    d = Dict{Int,Vector{Float64}}()
    for i in 1:n
        i in exclude && continue
        v = _pra_load(dir, key, i)
        v === nothing || (d[i] = v)
    end
    return d
end

function paired_run_auc_diff_excl(dir_cand, key_cand, dir_base, key_base;
                                   n_cand::Int=20, n_base::Int=20, exclude_cand::Vector{Int}=Int[])
    vecs_c = _pra_collect_excl(dir_cand, key_cand; n=n_cand, exclude=exclude_cand)
    vecs_b = _pra_collect(dir_base, key_base; n=n_base)
    matched = sort(collect(intersect(keys(vecs_c), keys(vecs_b))))
    empty_result = (median_diff=NaN, n_pairs=0, T=0, cand_n=length(vecs_c), base_n=length(vecs_b),
                     cand_nan=0, base_nan=0, cand_miniter=0, cand_maxiter=0, base_miniter=0, base_maxiter=0)
    isempty(matched) && return empty_result
    all_vecs = vcat(collect(values(vecs_c)), collect(values(vecs_b)))
    T = minimum(length.(all_vecs))
    denom = T > 1 ? log(T) : 1.0
    diffs = Float64[]
    for i in matched
        auc_c = _pra_auc(vecs_c[i], T)
        auc_b = _pra_auc(vecs_b[i], T)
        (auc_c === nothing || auc_b === nothing) && continue
        push!(diffs, (auc_c / denom) - (auc_b / denom))
    end
    c_lens = length.(values(vecs_c)); b_lens = length.(values(vecs_b))
    c_nan  = sum(count(isnan, v) for v in values(vecs_c); init=0)
    b_nan  = sum(count(isnan, v) for v in values(vecs_b); init=0)
    return (median_diff = isempty(diffs) ? NaN : median(diffs), n_pairs = length(diffs), T = T,
            cand_n = length(vecs_c), base_n = length(vecs_b), cand_nan = c_nan, base_nan = b_nan,
            cand_miniter = minimum(c_lens), cand_maxiter = maximum(c_lens),
            base_miniter = minimum(b_lens), base_maxiter = maximum(b_lens))
end

function compute_nongp_scores_paired()
    rows = NamedTuple[]

    function add_row!(prob, typ, std_dir, std_key, std_n, nongp_dir; exclude_nongp::Vector{Int}=Int[])
        isdir(std_dir) || (@warn "Missing: $std_dir"; return)
        isdir(nongp_dir) || (@warn "Missing: $nongp_dir"; return)
        r = paired_run_auc_diff_excl(nongp_dir, "nongp", std_dir, std_key; n_cand=20, n_base=std_n, exclude_cand=exclude_nongp)
        push!(rows, (problem=prob, type=typ, ng=r))
        @info "AUC-paired-NonGP $prob  diff=$(round(r.median_diff, digits=4)) (T=$(r.T), n=$(r.n_pairs))"
    end

    for prob in _GROUP_A
        add_row!(prob, "BIP", joinpath("data-bosip-norm", prob), "standard", 20, joinpath("data-bosip-norm", prob))
    end
    for prob in _GROUP_B_PROBS
        add_row!(prob, "Opt", joinpath("data-opt-functions", prob * "_cross"), "maxvar", 20, joinpath("data-opt-functions", prob * "_cross"))
    end
    for prob in _GROUP_C
        excl = prob == "DiffusionProblem5D" ? _DIFFUSION5D_NONGP_CRASHED_AUC_PAIRED : Int[]
        add_row!(prob, "BIP", joinpath("data-bosip-norm", prob), "standard", 20, joinpath("data-bosip-norm", prob); exclude_nongp=excl)
    end
    for prob in _GROUP_D_PROBS
        add_row!(prob, "Opt", joinpath("data-opt-functions", prob * "_cross"), "maxvar", 20, joinpath("data-opt-functions", prob * "_cross"))
    end

    mkpath("plots")
    open("plots/nongp_scores_auc_paired.csv", "w") do io
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
    @info "Saved $(length(rows)) rows → plots/nongp_scores_auc_paired.csv"
    return rows
end

## ── Run immediately on include ───────────────────────────────────────────────

acq_auc_paired_rows      = compute_acq_scores_paired()
warpedgp_auc_paired_rows = compute_warpedgp_scores_paired()
nongp_auc_paired_rows    = compute_nongp_scores_paired()

println()
println(rpad("Problem", 32), "│ ", rpad("Type", 4), "│  T  │ n │ ", rpad("EIV-MaxVar", 11), "│  T  │ n │ ", rpad("IMMD-MaxVar", 11))
for r in acq_auc_paired_rows
    e, m = r.eiv, r.immd
    println(rpad(r.problem, 32), "│ ", rpad(r.type, 4), "│ ", lpad(e.T,3), " │",
            lpad(e.n_pairs,2), " │ ", lpad(round(e.median_diff; digits=4), 9), " │ ",
            lpad(m.T,3), " │", lpad(m.n_pairs,2), " │ ", lpad(round(m.median_diff; digits=4), 9))
end

println()
println(rpad("Problem", 32), "│ ", rpad("Type", 4), "│  T  │ n │ WarpedGP-Standard")
for r in warpedgp_auc_paired_rows
    w = r.wgp
    println(rpad(r.problem, 32), "│ ", rpad(r.type, 4), "│ ", lpad(w.T,3), " │", lpad(w.n_pairs,2), " │ ", round(w.median_diff; digits=4))
end

println()
println(rpad("Problem", 32), "│ ", rpad("Type", 4), "│  T  │ n │ NonstatGP-Standard")
for r in nongp_auc_paired_rows
    n = r.ng
    println(rpad(r.problem, 32), "│ ", rpad(r.type, 4), "│ ", lpad(n.T,3), " │", lpad(n.n_pairs,2), " │ ", round(n.median_diff; digits=4))
end

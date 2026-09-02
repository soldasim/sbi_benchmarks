"""
Compute per-run final log-TV scores and aggregate per problem–acquisition pair.

Covers the four paper problem groups (A, B, C, D):
  A — 7 original BIP   (data-bosip-norm/, n=20 each acq)
  B — 24 cross 2D opt  (data-opt-functions/<prob>_cross/, n=20 each acq)
  C — 2 HD BIP 5D      (data-bosip-norm/, maxvar n=20, eiv/immd n=5)
  D — 4 HD cross 5D    (data-opt-functions/<prob>_cross/, maxvar n=20, eiv/immd n=5)

The score for a single run is the log of the last finite, positive TV value:

    score = log(TV(T))

where T is the last iteration with a valid (finite, >0) TV reading.
This directly measures final solution quality without trajectory averaging.

## Outputs

1. `plots/acq_scores_final.csv`
   Columns: problem, type, maxvar_n, maxvar_median, eiv_n, eiv_median,
            immd_n, immd_median, winner

2. `plots/warpedgp_scores_final.csv`
   Columns: problem, type, standard_n, standard_median, warpedgp_n,
            warpedgp_median, winner

## Usage

    include("src/compute_acq_scores_final.jl")
"""

using JLD2
using Statistics: median

const _ACSF_DRAW_THRESHOLD = log(1.20)   # 20 % margin to count as a win

## ── Per-run helpers ──────────────────────────────────────────────────────────

function _last_valid_log_tv(prob_dir, run_name, idx)
    tv_path = joinpath(prob_dir, "$(run_name)_$(idx)_TVmetric.jld2")
    isfile(tv_path) || return nothing
    try
        score = load(tv_path, "score")
        i_last = findlast(v -> isfinite(v) && v > 0.0, score)
        i_last === nothing && return nothing
        return log(score[i_last])
    catch
        return nothing
    end
end

function _collect_scores(prob_dir, run_name; n::Int=20)
    scores = Float64[]
    for idx in 1:n
        v = _last_valid_log_tv(prob_dir, run_name, idx)
        v !== nothing && push!(scores, v)
    end
    return scores
end

function _med_and_n(scores)
    isempty(scores) ? (NaN, 0) : (median(scores), length(scores))
end

## Raw-array quality stats for one group (min/max iteration count, NaN total) — feeds
## the auto data-quality banners in plot_posterior_classification_final.jl/_warpedgp.jl.
## Independent of _collect_scores (which only keeps the final log-TV value per run).
function _group_quality(prob_dir, run_name; n::Int=20)
    lens = Int[]
    nan_total = 0
    for idx in 1:n
        tv_path = joinpath(prob_dir, "$(run_name)_$(idx)_TVmetric.jld2")
        isfile(tv_path) || continue
        try
            s = load(tv_path, "score")
            push!(lens, length(s))
            nan_total += count(isnan, s)
        catch
        end
    end
    isempty(lens) && return (n=0, min_iter=0, max_iter=0, nan_total=0)
    return (n=length(lens), min_iter=minimum(lens), max_iter=maximum(lens), nan_total=nan_total)
end

function _winner(pairs)          # pairs = (name, median_val) only for finite values
    valid = filter(p -> isfinite(p[2]), pairs)
    isempty(valid) && return "N/A"
    length(valid) == 1 && return first(valid)[1]
    sorted = sort(valid; by=p -> p[2])
    abs(sorted[1][2] - sorted[2][2]) < _ACSF_DRAW_THRESHOLD ? "Draw" : sorted[1][1]
end

## ── Problem / acquisition configuration ─────────────────────────────────────

## Group A: 7 original BIP — data-bosip-norm/<prob>/, run names standard/eiv/immd, n=20
const _GROUP_A = [
    "ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
    "SIRProblem", "DuffingProblem", "DiffusionProblem10",
]

## Group B: 24 cross 2D opt — data-opt-functions/<prob>_cross/, maxvar/eiv/immd, n=20
const _GROUP_B_PROBS = [
    "RosenbrockProblem2",         "StyblinskiTangProblem2",   "MichalewiczProblem2",
    "AckleyProblem2",             "AlpineProblem2",           "ExpandedSchafferF6Problem2",
    "ExpandedZakharovProblem2",   "GriewankProblem2",         "RastriginProblem2",
    "SalomonProblem2",            "SchwefelProblem2",         "SphereProblem2",
    "BealeProblem",               "BoothProblem",             "CrossInTrayProblem",
    "DropWaveProblem",            "EasomProblem",             "GoldsteinPriceProblem",
    "HimmelblauProblem",          "HolderTableProblem",       "LeviN13Problem",
    "MatyasProblem",              "SchafferN2Problem",        "ThreeHumpCamelProblem",
]

## Group C: 2 HD BIP — data-bosip-norm/<prob>/, standard n=20, eiv/immd n=5
const _GROUP_C = ["DuffingProblem5", "DiffusionProblem5D"]

## Group D: 4 HD cross 5D — data-opt-functions/<prob>_cross/, maxvar n=20, eiv/immd n=5
const _GROUP_D_PROBS = [
    "RosenbrockProblem5", "StyblinskiTangProblem5",
    "MichalewiczProblem5", "SphereProblem5",
]

## ── Compute acq_scores_final.csv ─────────────────────────────────────────────

struct AcqRowF
    problem   :: String
    type      :: String
    maxvar_n  :: Int;  maxvar_med  :: Float64
    eiv_n     :: Int;  eiv_med     :: Float64
    immd_n    :: Int;  immd_med    :: Float64
    winner    :: String
    ## Quality columns (added 2026-08-04 after the ExpandedZakharovProblem2 Plot 5a
    ## incident — EIV looked decisively worse than MaxVar purely because most of its
    ## runs timed out around iter 110/200 while MaxVar ran the full 200; see
    ## cluster_scripts/notes_paper_plots.md).
    maxvar_miniter :: Int;  maxvar_maxiter :: Int;  maxvar_nan :: Int
    eiv_miniter    :: Int;  eiv_maxiter    :: Int;  eiv_nan    :: Int
    immd_miniter   :: Int;  immd_maxiter   :: Int;  immd_nan   :: Int
end

function compute_acq_scores_final()
    rows = AcqRowF[]

    ## Group A
    for prob in _GROUP_A
        d = joinpath("data-bosip-norm", prob)
        isdir(d) || (@warn "Missing: $d"; continue)
        mv_med, mv_n   = _med_and_n(_collect_scores(d, "standard"; n=20))
        ei_med, ei_n   = _med_and_n(_collect_scores(d, "eiv";      n=20))
        im_med, im_n   = _med_and_n(_collect_scores(d, "immd";     n=20))
        w = _winner([("MaxVar",mv_med),("EIV",ei_med),("IMMD",im_med)])
        mvq = _group_quality(d, "standard"; n=20)
        eiq = _group_quality(d, "eiv";      n=20)
        imq = _group_quality(d, "immd";     n=20)
        push!(rows, AcqRowF(prob, "BIP", mv_n,mv_med, ei_n,ei_med, im_n,im_med, w,
              mvq.min_iter,mvq.max_iter,mvq.nan_total, eiq.min_iter,eiq.max_iter,eiq.nan_total,
              imq.min_iter,imq.max_iter,imq.nan_total))
        @info "A $prob  maxvar=$mv_n eiv=$ei_n immd=$im_n → $w"
    end

    ## Group B
    for prob in _GROUP_B_PROBS
        d = joinpath("data-opt-functions", prob * "_cross")
        isdir(d) || (@warn "Missing: $d"; continue)
        mv_med, mv_n   = _med_and_n(_collect_scores(d, "maxvar"; n=20))
        ei_med, ei_n   = _med_and_n(_collect_scores(d, "eiv";    n=20))
        im_med, im_n   = _med_and_n(_collect_scores(d, "immd";   n=20))
        w = _winner([("MaxVar",mv_med),("EIV",ei_med),("IMMD",im_med)])
        mvq = _group_quality(d, "maxvar"; n=20)
        eiq = _group_quality(d, "eiv";    n=20)
        imq = _group_quality(d, "immd";   n=20)
        push!(rows, AcqRowF(prob, "Opt", mv_n,mv_med, ei_n,ei_med, im_n,im_med, w,
              mvq.min_iter,mvq.max_iter,mvq.nan_total, eiq.min_iter,eiq.max_iter,eiq.nan_total,
              imq.min_iter,imq.max_iter,imq.nan_total))
        @info "B $prob  maxvar=$mv_n eiv=$ei_n immd=$im_n → $w"
    end

    ## Group C
    for prob in _GROUP_C
        d = joinpath("data-bosip-norm", prob)
        isdir(d) || (@warn "Missing: $d"; continue)
        mv_med, mv_n   = _med_and_n(_collect_scores(d, "standard"; n=20))
        ei_med, ei_n   = _med_and_n(_collect_scores(d, "eiv";      n=5))
        im_med, im_n   = _med_and_n(_collect_scores(d, "immd";     n=5))
        w = _winner([("MaxVar",mv_med),("EIV",ei_med),("IMMD",im_med)])
        mvq = _group_quality(d, "standard"; n=20)
        eiq = _group_quality(d, "eiv";      n=5)
        imq = _group_quality(d, "immd";     n=5)
        push!(rows, AcqRowF(prob, "BIP", mv_n,mv_med, ei_n,ei_med, im_n,im_med, w,
              mvq.min_iter,mvq.max_iter,mvq.nan_total, eiq.min_iter,eiq.max_iter,eiq.nan_total,
              imq.min_iter,imq.max_iter,imq.nan_total))
        @info "C $prob  maxvar=$mv_n eiv=$ei_n immd=$im_n → $w"
    end

    ## Group D
    for prob in _GROUP_D_PROBS
        d = joinpath("data-opt-functions", prob * "_cross")
        isdir(d) || (@warn "Missing: $d"; continue)
        mv_med, mv_n   = _med_and_n(_collect_scores(d, "maxvar"; n=20))
        ei_med, ei_n   = _med_and_n(_collect_scores(d, "eiv";    n=5))
        im_med, im_n   = _med_and_n(_collect_scores(d, "immd";   n=5))
        w = _winner([("MaxVar",mv_med),("EIV",ei_med),("IMMD",im_med)])
        mvq = _group_quality(d, "maxvar"; n=20)
        eiq = _group_quality(d, "eiv";    n=5)
        imq = _group_quality(d, "immd";   n=5)
        push!(rows, AcqRowF(prob, "Opt", mv_n,mv_med, ei_n,ei_med, im_n,im_med, w,
              mvq.min_iter,mvq.max_iter,mvq.nan_total, eiq.min_iter,eiq.max_iter,eiq.nan_total,
              imq.min_iter,imq.max_iter,imq.nan_total))
        @info "D $prob  maxvar=$mv_n eiv=$ei_n immd=$im_n → $w"
    end

    mkpath("plots")
    open("plots/acq_scores_final.csv", "w") do io
        println(io, "problem,type,maxvar_n,maxvar_median,eiv_n,eiv_median,immd_n,immd_median,winner," *
                    "maxvar_miniter,maxvar_maxiter,maxvar_nan,eiv_miniter,eiv_maxiter,eiv_nan," *
                    "immd_miniter,immd_maxiter,immd_nan")
        for r in rows
            println(io, "$(r.problem),$(r.type),$(r.maxvar_n),$(r.maxvar_med)," *
                        "$(r.eiv_n),$(r.eiv_med),$(r.immd_n),$(r.immd_med),$(r.winner)," *
                        "$(r.maxvar_miniter),$(r.maxvar_maxiter),$(r.maxvar_nan)," *
                        "$(r.eiv_miniter),$(r.eiv_maxiter),$(r.eiv_nan)," *
                        "$(r.immd_miniter),$(r.immd_maxiter),$(r.immd_nan)")
        end
    end
    @info "Saved $(length(rows)) rows → plots/acq_scores_final.csv"
    return rows
end

## ── Compute warpedgp_scores_final.csv ────────────────────────────────────────

struct WgpRowF
    problem     :: String
    type        :: String
    standard_n  :: Int;  standard_med  :: Float64
    warpedgp_n  :: Int;  warpedgp_med  :: Float64
    winner      :: String
    standard_miniter :: Int;  standard_maxiter :: Int;  standard_nan :: Int
    warpedgp_miniter :: Int;  warpedgp_maxiter :: Int;  warpedgp_nan :: Int
end

function compute_warpedgp_scores_final()
    rows = WgpRowF[]

    function add_row!(prob, typ, std_dir, std_key, std_n, wgp_dir)
        isdir(std_dir) || (@warn "Missing: $std_dir"; return)
        isdir(wgp_dir) || (@warn "Missing: $wgp_dir"; return)
        s_med, s_n = _med_and_n(_collect_scores(std_dir, std_key;             n=std_n))
        w_med, w_n = _med_and_n(_collect_scores(wgp_dir, "warpedgp-yja-maxvar"; n=20))
        w = _winner([("Standard",s_med),("WarpedGP",w_med)])
        sq = _group_quality(std_dir, std_key;              n=std_n)
        wq = _group_quality(wgp_dir, "warpedgp-yja-maxvar"; n=20)
        push!(rows, WgpRowF(prob, typ, s_n,s_med, w_n,w_med, w,
              sq.min_iter,sq.max_iter,sq.nan_total, wq.min_iter,wq.max_iter,wq.nan_total))
        @info "WGP $prob  std=$s_n wgp=$w_n → $w"
    end

    ## Group A
    for prob in _GROUP_A
        add_row!(prob, "BIP",
                 joinpath("data-bosip-norm", prob),   "standard", 20,
                 joinpath("data-warpedgp2",  prob))
    end

    ## Group B
    # NOTE (2026-08-03): BealeProblem/GoldsteinPriceProblem's warpedgp pilot data used to
    # live without the _cross suffix (a 5-run pilot, pre-dating the full 20-run Group B
    # cross-polytope batch). That pilot directory was swept into
    # data-warpedgp2_archive_pre-samplesfix/ during the predictive_samples-fix cleanup and
    # never rerun post-fix. The *_cross variant (the actual Group B experiment for these two
    # problems) now has full 20/20 clean post-fix data — confirmed via cluster_scripts/
    # notes_paper_plots.md's Plot 2b audit — so route these two through the same _cross
    # path as every other Group B problem instead of the now-stale no-suffix special case.
    for prob in _GROUP_B_PROBS
        add_row!(prob, "Opt",
                 joinpath("data-opt-functions", prob * "_cross"), "maxvar",  20,
                 joinpath("data-warpedgp2",     prob * "_cross"))
    end

    ## Group C
    for prob in _GROUP_C
        add_row!(prob, "BIP",
                 joinpath("data-bosip-norm", prob),   "standard", 20,
                 joinpath("data-warpedgp2",  prob))
    end

    ## Group D
    for prob in _GROUP_D_PROBS
        add_row!(prob, "Opt",
                 joinpath("data-opt-functions", prob * "_cross"), "maxvar",  20,
                 joinpath("data-warpedgp2",     prob * "_cross"))
    end

    mkpath("plots")
    open("plots/warpedgp_scores_final.csv", "w") do io
        println(io, "problem,type,standard_n,standard_median,warpedgp_n,warpedgp_median,winner," *
                    "standard_miniter,standard_maxiter,standard_nan,warpedgp_miniter,warpedgp_maxiter,warpedgp_nan")
        for r in rows
            println(io, "$(r.problem),$(r.type),$(r.standard_n),$(r.standard_med)," *
                        "$(r.warpedgp_n),$(r.warpedgp_med),$(r.winner)," *
                        "$(r.standard_miniter),$(r.standard_maxiter),$(r.standard_nan)," *
                        "$(r.warpedgp_miniter),$(r.warpedgp_maxiter),$(r.warpedgp_nan)")
        end
    end
    @info "Saved $(length(rows)) rows → plots/warpedgp_scores_final.csv"
    return rows
end

## ── Run immediately on include ───────────────────────────────────────────────

acq_score_rows_f = compute_acq_scores_final()
wgp_score_rows_f = compute_warpedgp_scores_final()

println()
println(rpad("Problem", 32), "│ ", rpad("Type", 4),
        "│ ", rpad("MaxVar", 10), "│ ", rpad("EIV", 10), "│ ", rpad("IMMD", 10), "│ Winner")
println(repeat('─', 32), "─┼─", repeat('─', 5), "┼─",
        repeat('─', 11), "┼─", repeat('─', 11), "┼─", repeat('─', 11), "┼───────")
for r in acq_score_rows_f
    mv   = isfinite(r.maxvar_med)  ? lpad(round(r.maxvar_med;  digits=3), 9) : "        N/A"
    eiv  = isfinite(r.eiv_med)     ? lpad(round(r.eiv_med;     digits=3), 9) : "        N/A"
    immd = isfinite(r.immd_med)    ? lpad(round(r.immd_med;    digits=3), 9) : "        N/A"
    println(rpad(r.problem, 32), "│ ", rpad(r.type, 4),
            "│ ", mv, " │ ", eiv, " │ ", immd, " │ ", r.winner)
end

println()
println(rpad("Problem", 32), "│ ", rpad("Type", 4),
        "│ ", rpad("Standard", 10), "│ ", rpad("WarpedGP", 10), "│ Winner")
println(repeat('─', 32), "─┼─", repeat('─', 5), "┼─",
        repeat('─', 11), "┼─", repeat('─', 11), "┼───────")
for r in wgp_score_rows_f
    s = isfinite(r.standard_med) ? lpad(round(r.standard_med; digits=3), 9) : "        N/A"
    w = isfinite(r.warpedgp_med) ? lpad(round(r.warpedgp_med; digits=3), 9) : "        N/A"
    println(rpad(r.problem, 32), "│ ", rpad(r.type, 4),
            "│ ", s, " │ ", w, " │ ", r.winner)
end

"""
Fair (equal-iteration-budget) score comparison for Standard vs. WarpedGP and
Standard vs. nonstationary GP (nongp).

`compute_warpedgp_scores_final`/`compute_nongp_scores_final` (in
compute_acq_scores_final.jl / compute_nongp_scores_final.jl) score each run at
ITS OWN last valid iteration. That is unfair whenever the two configurations
being compared were actually run for different numbers of iterations — e.g.
WarpedGP ran 201 iterations on the 5D BIP problems (DuffingProblem5,
DiffusionProblem5D) while Standard only ran ~103 — the side that ran longer
gets an unearned advantage in the comparison.

This variant truncates BOTH configurations' runs to the SAME shared iteration
budget T = min(raw TV-history length) over ALL runs in both groups for a given
problem, then scores each run at the last valid (finite, >0) TV reading found
within that shared budget (1:T).

**IMPORTANT: for problems where the two configurations' iteration counts
differ substantially, this comparison is only as complete as the SHORTER
side's data — i.e. it does not use each config's full run.** This is flagged
directly on the resulting plot (`plot_smoothness_classification.jl`) via a
"TODO: calculated only with available iters" annotation — do not treat the
plot as final until every group has matched-iteration data end-to-end.

## Outputs
  plots/warpedgp_scores_fair.csv  (37 problems; columns include shared_T)
  plots/nongp_scores_fair.csv     (37 problems as of 2026-08-03, extended from the
                                    original Group-A-only 7 rows now that nongp data
                                    exists for Groups B/C/D too; columns include shared_T.
                                    DiffusionProblem5D excludes 6 crashed nongp runs from
                                    the shared budget — see compute_nongp_scores_fair's
                                    docstring comment for detail.)

## Usage
    include("src/compute_acq_scores_final.jl")   # for group lists + _winner/_med_and_n
    include("src/compute_fair_scores.jl")
"""

using JLD2
using Statistics: median

const _FAIR_DRAW_THRESHOLD = log(1.20)   # 20% margin to count as a win, matches compute_acq_scores_final.jl

## ── Shared-budget helpers ─────────────────────────────────────────────────────

function _load_score_vec(prob_dir, run_name, idx)
    tv_path = joinpath(prob_dir, "$(run_name)_$(idx)_TVmetric.jld2")
    isfile(tv_path) || return nothing
    try
        return load(tv_path, "score")
    catch
        return nothing
    end
end

function _collect_score_vecs(prob_dir, run_name; n::Int)
    vecs = Vector{Float64}[]
    for idx in 1:n
        v = _load_score_vec(prob_dir, run_name, idx)
        v !== nothing && push!(vecs, v)
    end
    return vecs
end

# Last valid (finite, >0) TV reading within score[1:T], as a log score.
function _fair_score_at_budget(score, T)
    Tc = min(T, length(score))
    i  = findlast(k -> isfinite(score[k]) && score[k] > 0.0, 1:Tc)
    i === nothing && return nothing
    return log(score[i])
end

# Truncate both groups to the shared iteration budget (min raw length across
# BOTH groups combined) and return (T, std_scores, other_scores).
function _fair_pair_scores(std_vecs, other_vecs)
    if isempty(std_vecs) || isempty(other_vecs)
        return (0, Float64[], Float64[])
    end
    T = minimum(length(v) for v in Iterators.flatten((std_vecs, other_vecs)))
    std_scores   = Float64[]
    for v in std_vecs
        s = _fair_score_at_budget(v, T)
        s === nothing || push!(std_scores, s)
    end
    other_scores = Float64[]
    for v in other_vecs
        s = _fair_score_at_budget(v, T)
        s === nothing || push!(other_scores, s)
    end
    return (T, std_scores, other_scores)
end

## ── compute_warpedgp_scores_fair ──────────────────────────────────────────────

struct FairRow
    problem     :: String
    type        :: String
    shared_T    :: Int
    standard_n  :: Int;  standard_med  :: Float64
    other_n     :: Int;  other_med     :: Float64
    winner      :: String
    ## NaN totals over the FULL (untruncated) raw score histories — added 2026-08-04
    ## alongside the acq_scores_final.csv quality columns, see data_quality.jl.
    standard_nan :: Int
    other_nan    :: Int
end

_nan_total(vecs) = isempty(vecs) ? 0 : sum(count(isnan, v) for v in vecs)

function compute_warpedgp_scores_fair()
    rows = FairRow[]

    function add_row!(prob, typ, std_dir, std_key, std_n, wgp_dir)
        isdir(std_dir) || (@warn "Missing: $std_dir"; return)
        isdir(wgp_dir) || (@warn "Missing: $wgp_dir"; return)
        std_vecs = _collect_score_vecs(std_dir, std_key;              n=std_n)
        wgp_vecs = _collect_score_vecs(wgp_dir, "warpedgp-yja-maxvar"; n=20)
        T, s_scores, w_scores = _fair_pair_scores(std_vecs, wgp_vecs)
        T == 0 && (@warn "No data for $prob"; return)
        s_med = isempty(s_scores) ? NaN : median(s_scores)
        w_med = isempty(w_scores) ? NaN : median(w_scores)
        w = _winner([("Standard",s_med),("WarpedGP",w_med)])
        push!(rows, FairRow(prob, typ, T, length(s_scores),s_med, length(w_scores),w_med, w,
              _nan_total(std_vecs), _nan_total(wgp_vecs)))
        @info "WGP-fair $prob  T=$T std=$(length(s_scores)) wgp=$(length(w_scores)) → $w"
    end

    for prob in _GROUP_A
        add_row!(prob, "BIP",
                 joinpath("data-bosip-norm", prob),   "standard", 20,
                 joinpath("data-warpedgp2",  prob))
    end

    for prob in _GROUP_B_PROBS
        # NOTE (2026-08-03): data-warpedgp2/ now uses the "_cross" suffix for
        # every Group B problem, including BealeProblem/GoldsteinPriceProblem —
        # their old bare (non-"_cross", 5-run pilot) directories no longer
        # exist, having been superseded/reorganized since compute_warpedgp_scores_final.jl
        # was first written. No exception needed any more.
        add_row!(prob, "Opt",
                 joinpath("data-opt-functions", prob * "_cross"), "maxvar",  20,
                 joinpath("data-warpedgp2",     prob * "_cross"))
    end

    for prob in _GROUP_C
        add_row!(prob, "BIP",
                 joinpath("data-bosip-norm", prob),   "standard", 20,
                 joinpath("data-warpedgp2",  prob))
    end

    for prob in _GROUP_D_PROBS
        add_row!(prob, "Opt",
                 joinpath("data-opt-functions", prob * "_cross"), "maxvar",  20,
                 joinpath("data-warpedgp2",     prob * "_cross"))
    end

    mkpath("plots")
    open("plots/warpedgp_scores_fair.csv", "w") do io
        println(io, "problem,type,shared_T,standard_n,standard_median,warpedgp_n,warpedgp_median,winner,standard_nan,warpedgp_nan")
        for r in rows
            println(io, "$(r.problem),$(r.type),$(r.shared_T),$(r.standard_n),$(r.standard_med)," *
                        "$(r.other_n),$(r.other_med),$(r.winner),$(r.standard_nan),$(r.other_nan)")
        end
    end
    @info "Saved $(length(rows)) rows → plots/warpedgp_scores_fair.csv"
    return rows
end

## ── compute_nongp_scores_fair ─────────────────────────────────────────────────
##
## EXTENDED 2026-08-03 to cover Groups B/C/D (24 cross-2D opt + 2 HD BIP + 4 HD
## cross opt), now that the 2026-07-28/08-01 nongp-expansion batch (600 runs)
## landed nongp data for all of them — previously this only covered Group A (7
## problems), since nongp had never been run anywhere else. Mirrors
## compute_warpedgp_scores_fair's per-group loop structure/data routing exactly
## (maxvar is the Group B/D baseline key, standard is the Group A/C baseline key).
##
## Scoring convention: FAIR (equal-iteration-budget), not "final" (each run's own
## last iteration) — deliberate choice, see cluster_scripts/notes_paper_plots.md
## Task 9 for the full rationale. Short version: nongp is far more expensive per
## iteration than standard/maxvar, so at "final" (each run's own stopping point)
## nongp is systematically compared against a much-further-converged standard/
## maxvar baseline — this is exactly the confound documented in
## project_bosip_reviews.md (2026-07-15) that flipped the Group-A-only nongp
## classification result from "Standard wins almost everywhere" to "NonstatGP
## wins 5/7" once corrected. That confound only gets worse for Groups B/C/D,
## where nongp's iteration shortfall (39-71/100 for B, ~32-38/200 for C,
## 25-45/200 for D) is proportionally similar or worse than Group A's.
##
## DiffusionProblem5D special case: 6 of its 20 nongp runs (indices 6,8,10,11,13,16)
## crashed near-empty (4-14 iters total, see project_bosip_benchmarks_nongp_expansion.md)
## due to an uncaught PosDefException in ConvergenceCallback — NOT resubmitted, per
## the plotting-only scope of this task. These 6 are EXCLUDED from both the shared-
## budget computation T and the nongp median for this one problem (otherwise T would
## collapse to ~4 for the whole problem, an unusable shared budget dominated by the
## worst crash rather than reflecting the other 14 genuinely-truncated-but-live runs).
## The exclusion is explicit and logged, not silent — flagged again in the plot caption.
const _DIFFUSION5D_NONGP_CRASHED = [6, 8, 10, 11, 13, 16]

function _collect_score_vecs_excl(prob_dir, run_name; n::Int, exclude::Vector{Int}=Int[])
    vecs = Vector{Float64}[]
    for idx in 1:n
        idx in exclude && continue
        v = _load_score_vec(prob_dir, run_name, idx)
        v !== nothing && push!(vecs, v)
    end
    return vecs
end

function compute_nongp_scores_fair()
    rows = FairRow[]

    function add_row!(prob, typ, std_dir, std_key, std_n, nongp_dir; exclude_nongp::Vector{Int}=Int[])
        isdir(std_dir)   || (@warn "Missing: $std_dir"; return)
        isdir(nongp_dir) || (@warn "Missing: $nongp_dir"; return)
        std_vecs   = _collect_score_vecs(std_dir, std_key; n=std_n)
        nongp_vecs = _collect_score_vecs_excl(nongp_dir, "nongp"; n=20, exclude=exclude_nongp)
        T, s_scores, n_scores = _fair_pair_scores(std_vecs, nongp_vecs)
        T == 0 && (@warn "No data for $prob"; return)
        s_med = isempty(s_scores) ? NaN : median(s_scores)
        n_med = isempty(n_scores) ? NaN : median(n_scores)
        w = _winner([("Standard",s_med),("NonstatGP",n_med)])
        excl_note = isempty(exclude_nongp) ? "" : " (excl. crashed idx=$(exclude_nongp))"
        push!(rows, FairRow(prob, typ, T, length(s_scores),s_med, length(n_scores),n_med, w,
              _nan_total(std_vecs), _nan_total(nongp_vecs)))
        @info "NonGP-fair $prob  T=$T std=$(length(s_scores)) nongp=$(length(n_scores))$excl_note → $w"
    end

    ## Group A (7 orig BIP) — standard vs nongp, both in data-bosip-norm/
    for prob in _GROUP_A
        add_row!(prob, "BIP",
                 joinpath("data-bosip-norm", prob), "standard", 20,
                 joinpath("data-bosip-norm", prob))
    end

    ## Group B (24 cross 2D opt) — maxvar vs nongp, both in data-opt-functions/<prob>_cross/
    for prob in _GROUP_B_PROBS
        add_row!(prob, "Opt",
                 joinpath("data-opt-functions", prob * "_cross"), "maxvar", 20,
                 joinpath("data-opt-functions", prob * "_cross"))
    end

    ## Group C (2 HD BIP) — standard vs nongp, both in data-bosip-norm/
    for prob in _GROUP_C
        excl = prob == "DiffusionProblem5D" ? _DIFFUSION5D_NONGP_CRASHED : Int[]
        add_row!(prob, "BIP",
                 joinpath("data-bosip-norm", prob), "standard", 20,
                 joinpath("data-bosip-norm", prob); exclude_nongp=excl)
    end

    ## Group D (4 HD cross opt) — maxvar vs nongp, both in data-opt-functions/<prob>_cross/
    for prob in _GROUP_D_PROBS
        add_row!(prob, "Opt",
                 joinpath("data-opt-functions", prob * "_cross"), "maxvar", 20,
                 joinpath("data-opt-functions", prob * "_cross"))
    end

    mkpath("plots")
    open("plots/nongp_scores_fair.csv", "w") do io
        println(io, "problem,type,shared_T,standard_n,standard_median,nongp_n,nongp_median,winner,standard_nan,nongp_nan")
        for r in rows
            println(io, "$(r.problem),$(r.type),$(r.shared_T),$(r.standard_n),$(r.standard_med)," *
                        "$(r.other_n),$(r.other_med),$(r.winner),$(r.standard_nan),$(r.other_nan)")
        end
    end
    @info "Saved $(length(rows)) rows → plots/nongp_scores_fair.csv"
    return rows
end

## ── Run immediately on include ───────────────────────────────────────────────

wgp_fair_rows   = compute_warpedgp_scores_fair()
nongp_fair_rows = compute_nongp_scores_fair()

println()
println(rpad("Problem", 32), "│ ", rpad("Type", 4), "│  T  │ ",
        rpad("Standard", 10), "│ ", rpad("WarpedGP", 10), "│ Winner")
for r in wgp_fair_rows
    s = isfinite(r.standard_med) ? lpad(round(r.standard_med; digits=3), 9) : "        N/A"
    w = isfinite(r.other_med)    ? lpad(round(r.other_med;    digits=3), 9) : "        N/A"
    println(rpad(r.problem, 32), "│ ", rpad(r.type, 4), "│ ", lpad(r.shared_T,3), " │ ", s, " │ ", w, " │ ", r.winner)
end

println()
println(rpad("Problem", 32), "│ ", rpad("Type", 4), "│  T  │ ",
        rpad("Standard", 10), "│ ", rpad("NonstatGP", 10), "│ Winner")
for r in nongp_fair_rows
    s = isfinite(r.standard_med) ? lpad(round(r.standard_med; digits=3), 9) : "        N/A"
    n = isfinite(r.other_med)    ? lpad(round(r.other_med;    digits=3), 9) : "        N/A"
    println(rpad(r.problem, 32), "│ ", rpad(r.type, 4), "│ ", lpad(r.shared_T,3), " │ ", s, " │ ", n, " │ ", r.winner)
end

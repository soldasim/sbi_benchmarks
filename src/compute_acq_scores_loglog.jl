"""
Compute per-run log-log area scores and aggregate per problem–acquisition pair.

The score for a single run is the area under the log(TV) vs log(iteration) curve,
computed via the trapezoidal rule and normalised by the log-range:

    score = trapz(log(t), log(TV(t))) / (log(T) − log(1))

This is equivalent to the mean log(TV) with harmonic weights (weight ∝ 1/t),
giving more importance to early-iteration convergence.

## Outputs

1. Per-run files: `{data_dir}/{problem}/{run_name}_{idx}_loglogTV.jld2`
   Key: `"loglog_tv"` (Float64).

2. Summary CSV: `plots/acq_scores_loglog.csv`
   Columns: problem, type, maxvar_n, maxvar_median, eiv_n, eiv_median,
            immd_n, immd_median, winner

## Usage

    include("src/compute_acq_scores_loglog.jl")
"""

using JLD2
using Statistics: median

## ── Problem / acquisition configuration ─────────────────────────────────────
# (same as compute_acq_scores.jl)

const _ACSL_BIP = [
    ("ABProblem",          ["standard", "eiv", "immd"]),
    ("SimpleProblem",      ["standard", "eiv", "immd"]),
    ("BananaProblem",      ["standard", "eiv", "immd"]),
    ("BimodalProblem",     ["standard", "eiv", "immd"]),
    ("ProxySIRProblem",    ["standard", "eiv", "immd"]),
    ("DuffingProblem",     ["standard", "eiv", "immd"]),
    ("DiffusionProblem10", ["standard", "eiv", "immd"]),
]

const _ACSL_OPT = [
    ("RosenbrockProblem2_cross",         ["maxvar", "eiv"]),
    ("StyblinskiTangProblem2_cross",     ["maxvar", "eiv"]),
    ("MichalewiczProblem2_cross",        ["maxvar", "eiv"]),
    ("AckleyProblem2_cross",             ["maxvar", "eiv"]),
    ("AlpineProblem2_cross",             ["maxvar", "eiv"]),
    ("ExpandedSchafferF6Problem2_cross", ["maxvar", "eiv"]),
    ("ExpandedZakharovProblem2_cross",   ["maxvar", "eiv"]),
    ("GriewankProblem2_cross",           ["maxvar", "eiv"]),
    ("RastriginProblem2_cross",          ["maxvar", "eiv"]),
    ("SalomonProblem2_cross",            ["maxvar", "eiv"]),
    ("SchwefelProblem2_cross",           ["maxvar", "eiv"]),
    ("SphereProblem2_cross",             ["maxvar", "eiv"]),
    ("BoothProblem_cross",               ["maxvar", "eiv"]),
    ("CrossInTrayProblem_cross",         ["maxvar", "eiv"]),
    ("DropWaveProblem_cross",            ["maxvar", "eiv"]),
    ("EasomProblem_cross",               ["maxvar", "eiv"]),
    ("HimmelblauProblem_cross",          ["maxvar", "eiv"]),
    ("HolderTableProblem_cross",         ["maxvar", "eiv"]),
    ("LeviN13Problem_cross",             ["maxvar", "eiv"]),
    ("MatyasProblem_cross",              ["maxvar", "eiv"]),
    ("SchafferN2Problem_cross",          ["maxvar", "eiv"]),
    ("ThreeHumpCamelProblem_cross",      ["maxvar", "eiv"]),
    ("BealeProxyProblem_cross",          ["maxvar", "eiv"]),
    ("GoldsteinPriceProxyProblem_cross", ["maxvar", "eiv"]),
    ("AckleyProblem5_cross",             ["maxvar", "eiv"]),
    ("AlpineProblem5_cross",             ["maxvar", "eiv"]),
    ("ExpandedSchafferF6Problem5_cross", ["maxvar", "eiv"]),
    ("ExpandedZakharovProblem5_cross",   ["maxvar", "eiv"]),
    ("GriewankProblem5_cross",           ["maxvar", "eiv"]),
    ("MichalewiczProblem5_cross",        ["maxvar", "eiv"]),
    ("RastriginProblem5_cross",          ["maxvar", "eiv"]),
    ("RosenbrockProblem5_cross",         ["maxvar", "eiv"]),
    ("SalomonProblem5_cross",            ["maxvar", "eiv"]),
    ("SchwefelProblem5_cross",           ["maxvar", "eiv"]),
    ("SphereProblem5_cross",             ["maxvar", "eiv"]),
    ("StyblinskiTangProblem5_cross",     ["maxvar", "eiv"]),
]

const _ACSL_DISPLAY = Dict(
    "ProxySIRProblem"                   => "SIRProblem",
    "BealeProxyProblem"                 => "BealeProblem",
    "GoldsteinPriceProxyProblem"        => "GoldsteinPriceProblem",
    "BealeProxyProblem_cross"           => "BealeProblem",
    "GoldsteinPriceProxyProblem_cross"  => "GoldsteinPriceProblem",
)

const _ACSL_ACQ_LABEL = Dict(
    "standard" => "MaxVar",
    "maxvar"   => "MaxVar",
    "eiv"      => "EIV",
    "immd"     => "IMMD",
)

const _ACSL_DRAW_THRESHOLD = log(1.15)

## ── Per-run computation ──────────────────────────────────────────────────────

function _compute_run_score_loglog(prob_dir, run_name, idx; force=false)
    tv_path  = joinpath(prob_dir, "$(run_name)_$(idx)_TVmetric.jld2")
    out_path = joinpath(prob_dir, "$(run_name)_$(idx)_loglogTV.jld2")

    isfile(tv_path) || return nothing
    if isfile(out_path) && !force
        return load(out_path, "loglog_tv")
    end

    score = load(tv_path, "score")

    # Keep valid entries, preserving iteration index (1-based)
    valid_pairs = [(Float64(i), v)
                   for (i, v) in enumerate(score)
                   if isfinite(v) && v > 0.0]
    length(valid_pairs) < 2 && return nothing

    xs = log.(getindex.(valid_pairs, 1))  # log(iteration)
    ys = log.(getindex.(valid_pairs, 2))  # log(TV)

    # Trapezoidal area under log(TV) vs log(t), normalised by log-range
    area = sum((ys[i] + ys[i+1]) / 2 * (xs[i+1] - xs[i]) for i in 1:length(xs)-1)
    val  = area / (xs[end] - xs[1])

    save(out_path, Dict("loglog_tv" => val))
    return val
end

## ── Per-acquisition aggregation ──────────────────────────────────────────────

function _collect_acq_scores_loglog(prob_dir, run_name)
    scores = Float64[]
    for idx in 1:100
        v = _compute_run_score_loglog(prob_dir, run_name, idx)
        v !== nothing && push!(scores, v)
    end
    return scores
end

## ── Main entry ───────────────────────────────────────────────────────────────

struct AcqRowLL
    problem   :: String
    type      :: String
    maxvar_n  :: Int
    maxvar_med:: Float64
    eiv_n     :: Int
    eiv_med   :: Float64
    immd_n    :: Int
    immd_med  :: Float64
    winner    :: String
end

function compute_acq_scores_loglog(; force=false)
    rows = AcqRowLL[]

    for (ptype, config, data_dir) in [("BIP", _ACSL_BIP, "data-bosip-norm"),
                                       ("Opt", _ACSL_OPT, "data-opt-functions")]
        for (prob_name, acqs) in config
            prob_dir     = joinpath(data_dir, prob_name)
            display_name = get(_ACSL_DISPLAY, prob_name, replace(prob_name, "_cross" => ""))

            isdir(prob_dir) || (@warn "Missing: $prob_dir"; continue)
            @info "Processing $prob_name ..."

            label_scores = Dict{String, Vector{Float64}}()
            for run_name in acqs
                lbl    = _ACSL_ACQ_LABEL[run_name]
                scores = _collect_acq_scores_loglog(prob_dir, run_name)
                label_scores[lbl] = scores
                @info "  $lbl ($run_name): $(length(scores)) runs"
            end

            get_med(lbl) = haskey(label_scores, lbl) && !isempty(label_scores[lbl]) ?
                           median(label_scores[lbl]) : NaN
            get_n(lbl)   = haskey(label_scores, lbl) ? length(label_scores[lbl]) : 0

            mv_med   = get_med("MaxVar")
            eiv_med  = get_med("EIV")
            immd_med = get_med("IMMD")

            candidates = filter(p -> isfinite(p[2]),
                                [("MaxVar", mv_med), ("EIV", eiv_med), ("IMMD", immd_med)])
            winner = if isempty(candidates)
                "N/A"
            elseif length(candidates) == 1
                first(candidates)[1]
            else
                sorted = sort(candidates; by=p->p[2])
                abs(sorted[1][2] - sorted[2][2]) < _ACSL_DRAW_THRESHOLD ? "Draw" : sorted[1][1]
            end

            push!(rows, AcqRowLL(display_name, ptype,
                                 get_n("MaxVar"), mv_med,
                                 get_n("EIV"),    eiv_med,
                                 get_n("IMMD"),   immd_med,
                                 winner))
        end
    end

    mkpath("plots")
    open("plots/acq_scores_loglog.csv", "w") do io
        println(io, "problem,type,maxvar_n,maxvar_median,eiv_n,eiv_median,immd_n,immd_median,winner")
        for r in rows
            println(io, "$(r.problem),$(r.type),$(r.maxvar_n),$(r.maxvar_med)," *
                        "$(r.eiv_n),$(r.eiv_med),$(r.immd_n),$(r.immd_med),$(r.winner)")
        end
    end
    @info "Saved → plots/acq_scores_loglog.csv"

    return rows
end

## ── Run immediately on include ───────────────────────────────────────────────

const acq_score_rows_ll = compute_acq_scores_loglog()

println()
println(rpad("Problem", 32), "│ ", rpad("Type", 4),
        "│ ", rpad("MaxVar", 10), "│ ", rpad("EIV", 10), "│ ", rpad("IMMD", 10), "│ Winner")
println(repeat('─', 32), "─┼─", repeat('─', 5), "┼─",
        repeat('─', 11), "┼─", repeat('─', 11), "┼─", repeat('─', 11), "┼───────")
for r in acq_score_rows_ll
    mv   = isfinite(r.maxvar_med)  ? lpad(round(r.maxvar_med;  digits=3), 9) : "        N/A"
    eiv  = isfinite(r.eiv_med)     ? lpad(round(r.eiv_med;     digits=3), 9) : "        N/A"
    immd = isfinite(r.immd_med)    ? lpad(round(r.immd_med;    digits=3), 9) : "        N/A"
    println(rpad(r.problem, 32), "│ ", rpad(r.type, 4),
            "│ ", mv, " │ ", eiv, " │ ", immd, " │ ", r.winner)
end

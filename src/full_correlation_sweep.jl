"""
Full correlation sweep (2026-08-04): every candidate metric (global smoothness,
local smoothness, metadata, standard-GP difficulty, fitted warp lambda)
against every target-margin variant (WarpedGP/NonstatGP × fair-median/AUC-mean
× continuous/ternary) — roughly 230 individual Pearson+Spearman tests.

This is an intentionally EXHAUSTIVE sweep, not a confirmatory one — with this
many tests, expect ~1-in-20 "significant" hits at p<0.05 purely by chance.
Results are ranked by |Spearman ρ| (robust to the extreme outliers in
hessian_cond/lipschitz) so genuine standouts can be distinguished from noise,
but any single hit should be treated as a LEAD TO VERIFY (e.g. does it survive
across both the fair and AUC scoring variants, and both continuous and
ternary?), not a confirmed finding.

Reads only already-computed plots/*.csv files; writes only
plots/full_correlation_sweep.csv — no BOSIP experiment data or existing CSVs
touched.

Usage:
    include("src/correlate_metrics_lib.jl")   # for _rank_lib/_spearman_lib/_pvalue_lib
    include("src/full_correlation_sweep.jl")
"""

using Statistics: cor

const _METRICS_BASE = ["cv_value","cv_grad","yj_dev","skewness","heterosced",
                        "kurtosis","hessian_cond","lipschitz","path_roughness","output_redundancy"]

function _fcs_load(path, cols; prefix="")
    lines  = readlines(path)
    header = split(lines[1], ",")
    idxs   = [findfirst(==(c), header) for c in cols]
    out = Dict{String, Dict{String,Float64}}()
    for l in lines[2:end]
        isempty(strip(l)) && continue
        parts = split(l, ",")
        prob  = parts[1]
        d = get!(out, prob, Dict{String,Float64}())
        for (i, c) in enumerate(cols)
            d["$(prefix)$(c)"] = parse(Float64, parts[idxs[i]])
        end
    end
    return out
end

function _fcs_merge!(dest, src)
    for (prob, d) in src
        dd = get!(dest, prob, Dict{String,Float64}())
        merge!(dd, d)
    end
end

## ── Build the unified per-problem metric table ───────────────────────────────

const ALL_METRICS = Dict{String, Dict{String,Float64}}()

_g = _fcs_load("plots/classify_smoothness.csv", _METRICS_BASE; prefix="global_")
for (_, d) in _g
    d["global_log10_hessian_cond"] = log10(1 + d["global_hessian_cond"])
    d["global_log10_lipschitz"]    = log10(1 + d["global_lipschitz"])
end
_fcs_merge!(ALL_METRICS, _g)

_l = _fcs_load("plots/classify_smoothness_local.csv", _METRICS_BASE; prefix="local_")
for (_, d) in _l
    d["local_log10_hessian_cond"] = log10(1 + d["local_hessian_cond"])
    d["local_log10_lipschitz"]    = log10(1 + d["local_lipschitz"])
end
_fcs_merge!(ALL_METRICS, _l)

# snr dropped: est_noise_std===nothing for every problem in this suite, so it's
# a constant (degenerate, can't correlate with anything).
_fcs_merge!(ALL_METRICS, _fcs_load("plots/classify_metadata.csv", ["is_bip"]))
_fcs_merge!(ALL_METRICS, _fcs_load("plots/classify_standard_difficulty.csv",
            ["standard_median","standard_spread","standard_conv_slope"]))
_fcs_merge!(ALL_METRICS, _fcs_load("plots/fitted_warp_lambda.csv", ["fitted_yj_dev"]))

const ALL_METRIC_NAMES = sort(collect(reduce(union, (Set(keys(v)) for v in values(ALL_METRICS)))))
@info "Loaded $(length(ALL_METRIC_NAMES)) candidate metrics for $(length(ALL_METRICS)) problems"

## ── Targets ───────────────────────────────────────────────────────────────────

const TARGETS = [
    ("plots/warpedgp_scores_fair.csv",   ("warpedgp_median","standard_median"), "warpedgp_fair"),
    ("plots/warpedgp_scores_auc.csv",    ("warpedgp_median","standard_median"), "warpedgp_auc"),
    ("plots/warpedgp_ternary_fair.csv",  ("ternary","zero"),                    "warpedgp_ternary_fair"),
    ("plots/warpedgp_ternary_auc.csv",   ("ternary","zero"),                    "warpedgp_ternary_auc"),
    ("plots/nongp_scores_fair.csv",      ("nongp_median","standard_median"),    "nongp_fair"),
    ("plots/nongp_scores_auc.csv",       ("nongp_median","standard_median"),    "nongp_auc"),
    ("plots/nongp_ternary_fair.csv",     ("ternary","zero"),                    "nongp_ternary_fair"),
    ("plots/nongp_ternary_auc.csv",      ("ternary","zero"),                    "nongp_ternary_auc"),
]

## ── Sweep ─────────────────────────────────────────────────────────────────────

struct SweepRow
    target :: String
    metric :: String
    pearson :: Float64
    pearson_p :: Float64
    spearman :: Float64
    spearman_p :: Float64
    n :: Int
end

sweep_results = SweepRow[]

for (csv, cols, label) in TARGETS
    m_rows = _read_csv_cols_lib(csv, ["problem", cols[1], cols[2]])
    margin_dict = Dict(r[1] => parse(Float64, r[2]) - parse(Float64, r[3]) for r in m_rows)
    is_nongp_target = startswith(label, "nongp")

    for metric in ALL_METRIC_NAMES
        is_nongp_target && metric == "fitted_yj_dev" && continue   # only meaningful for WarpedGP
        problems = [p for p in keys(margin_dict) if haskey(ALL_METRICS, p) && haskey(ALL_METRICS[p], metric)]
        length(problems) < 5 && continue
        vals    = [ALL_METRICS[p][metric] for p in problems]
        margins = [margin_dict[p] for p in problems]
        length(Set(vals)) < 2 && continue   # constant predictor, skip

        r  = cor(vals, margins)
        rp = _pvalue_lib(r, length(vals))
        ρ  = _spearman_lib(vals, margins)
        ρp = _pvalue_lib(ρ, length(vals))
        push!(sweep_results, SweepRow(label, metric, r, rp, ρ, ρp, length(vals)))
    end
end

@info "Ran $(length(sweep_results)) metric×target correlation tests"

## ── Report: ranked by |Spearman ρ| ────────────────────────────────────────────

sorted_results = sort(sweep_results; by = r -> -abs(r.spearman))

println()
println(rpad("Target", 24), "│ ", rpad("Metric", 26), "│ Pearson r │   p   │ Spearman ρ │   p   │  n")
println(repeat('─', 24), "─┼─", repeat('─', 27), "┼───────────┼───────┼────────────┼───────┼────")
for r in sorted_results[1:min(40, length(sorted_results))]
    flag = r.spearman_p < 0.05 ? "*" : " "
    println(rpad(r.target, 24), "│ ", rpad(r.metric, 26), "│  ", lpad(round(r.pearson,digits=3),8),
            " │ ", lpad(round(r.pearson_p,digits=3),5), " │  ", lpad(round(r.spearman,digits=3),9), flag,
            " │ ", lpad(round(r.spearman_p,digits=3),5), " │ ", r.n)
end

mkpath("plots")
open("plots/full_correlation_sweep.csv", "w") do io
    println(io, "target,metric,pearson_r,pearson_p,spearman_rho,spearman_p,n")
    for r in sweep_results
        println(io, "$(r.target),$(r.metric),$(r.pearson),$(r.pearson_p),$(r.spearman),$(r.spearman_p),$(r.n)")
    end
end
@info "Saved $(length(sweep_results)) rows → plots/full_correlation_sweep.csv"

n_sig = count(r -> r.spearman_p < 0.05, sweep_results)
println("\n$(n_sig) / $(length(sweep_results)) tests significant at p<0.05 (≈$(round(length(sweep_results)*0.05, digits=1)) expected by chance alone).")

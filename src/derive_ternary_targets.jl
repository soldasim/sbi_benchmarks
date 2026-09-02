"""
Derive ternary (better/draw/worse) numeric targets (2026-08-04) from the
"winner" column already present in the margin CSVs (compute_fair_scores.jl /
compute_auc_margin.jl), as an alternative response type to the continuous
margin difference. Encoding:
  -1.0 = advanced surrogate (WarpedGP/NonstatGP) wins
   0.0 = Draw
  +1.0 = Standard wins
matching the continuous margin's sign convention (positive = Standard better).

Output CSVs carry a dummy "zero" column (always 0.0) alongside "ternary" so
they can be fed straight into correlate_metrics_lib.jl's `run_correlation`
with margin_cols=("ternary","zero") — margin = ternary - 0 = ternary, no code
changes needed there.

Usage: run after compute_fair_scores.jl / compute_auc_margin.jl have produced
their CSVs. Writes plots/<name>_ternary.csv for each input (new files only).
"""

function _ternary_value(winner, advanced_name)
    winner == advanced_name && return -1.0
    winner == "Standard" && return 1.0
    return 0.0
end

function derive_ternary(csv_path, advanced_name, out_path)
    lines = readlines(csv_path)
    header   = split(lines[1], ",")
    prob_idx = findfirst(==("problem"), header)
    win_idx  = findfirst(==("winner"), header)
    open(out_path, "w") do io
        println(io, "problem,ternary,zero")
        for l in lines[2:end]
            isempty(strip(l)) && continue
            cols = split(l, ",")
            t = _ternary_value(cols[win_idx], advanced_name)
            println(io, "$(cols[prob_idx]),$(t),0.0")
        end
    end
    @info "Saved → $out_path"
end

derive_ternary("plots/warpedgp_scores_fair.csv", "WarpedGP",  "plots/warpedgp_ternary_fair.csv")
derive_ternary("plots/nongp_scores_fair.csv",    "NonstatGP", "plots/nongp_ternary_fair.csv")
derive_ternary("plots/warpedgp_scores_auc.csv",  "WarpedGP",  "plots/warpedgp_ternary_auc.csv")
derive_ternary("plots/nongp_scores_auc.csv",     "NonstatGP", "plots/nongp_ternary_auc.csv")

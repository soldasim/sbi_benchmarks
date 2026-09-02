"""
Compute per-run final log-TV scores for Standard vs. nonstationary GP (nongp),
mirroring `compute_warpedgp_scores_final` in compute_acq_scores_final.jl but for
the nongp surrogate. nongp only ran on Group A (7 original BIP problems) — there
is no nongp data for Groups B/C/D, so this CSV has far fewer rows than
warpedgp_scores_final.csv (7 vs 37).

Score for a single run: log of the last finite, positive TV value (same
definition as compute_acq_scores_final.jl).

## Output

`plots/nongp_scores_final.csv`
  Columns: problem, type, standard_n, standard_median, nongp_n, nongp_median, winner

## Usage

Must be run after `include("src/compute_acq_scores_final.jl")` so that
`_collect_scores`, `_med_and_n`, `_winner` are in scope.
"""

const _GROUP_A_NONGP = [
    "ABProblem", "SimpleProblem", "BananaProblem", "BimodalProblem",
    "SIRProblem", "DuffingProblem", "DiffusionProblem10",
]

struct NongpRowF
    problem      :: String
    type         :: String
    standard_n   :: Int;  standard_med  :: Float64
    nongp_n      :: Int;  nongp_med     :: Float64
    winner       :: String
end

function compute_nongp_scores_final()
    rows = NongpRowF[]

    for prob in _GROUP_A_NONGP
        d = joinpath("data-bosip-norm", prob)
        isdir(d) || (@warn "Missing: $d"; continue)
        s_med, s_n = _med_and_n(_collect_scores(d, "standard"; n=20))
        n_med, n_n = _med_and_n(_collect_scores(d, "nongp";    n=20))
        w = _winner([("Standard",s_med),("NonstatGP",n_med)])
        push!(rows, NongpRowF(prob, "BIP", s_n,s_med, n_n,n_med, w))
        @info "NonGP $prob  std=$s_n nongp=$n_n → $w"
    end

    mkpath("plots")
    open("plots/nongp_scores_final.csv", "w") do io
        println(io, "problem,type,standard_n,standard_median,nongp_n,nongp_median,winner")
        for r in rows
            println(io, "$(r.problem),$(r.type),$(r.standard_n),$(r.standard_med)," *
                        "$(r.nongp_n),$(r.nongp_med),$(r.winner)")
        end
    end
    @info "Saved $(length(rows)) rows → plots/nongp_scores_final.csv"
    return rows
end

## ── Run immediately on include ───────────────────────────────────────────────

nongp_score_rows_f = compute_nongp_scores_final()

println()
println(rpad("Problem", 32), "│ ", rpad("Type", 4),
        "│ ", rpad("Standard", 10), "│ ", rpad("NonstatGP", 10), "│ Winner")
println(repeat('─', 32), "─┼─", repeat('─', 5), "┼─",
        repeat('─', 11), "┼─", repeat('─', 11), "┼───────")
for r in nongp_score_rows_f
    s = isfinite(r.standard_med) ? lpad(round(r.standard_med; digits=3), 9) : "        N/A"
    n = isfinite(r.nongp_med)    ? lpad(round(r.nongp_med;    digits=3), 9) : "        N/A"
    println(rpad(r.problem, 32), "│ ", rpad(r.type, 4),
            "│ ", s, " │ ", n, " │ ", r.winner)
end

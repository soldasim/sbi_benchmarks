using JLD2

function count_iters(path)
    isfile(path) || return -1
    try
        jldopen(path) do f
            count(!isnan, f["score"])
        end
    catch
        -2
    end
end

function audit_config(problems, dir, prefix, n_runs=20)
    for p in problems
        d = joinpath(dir, p)
        counts = [count_iters(joinpath(d, "$(prefix)_$(i)_TVmetric.jld2")) for i in 1:n_runs]
        println(rpad(p, 42), " ", counts)
    end
end

const B2D = [
    "RosenbrockProblem2_cross", "StyblinskiTangProblem2_cross",
    "MichalewiczProblem2_cross", "AckleyProblem2_cross",
    "AlpineProblem2_cross", "ExpandedSchafferF6Problem2_cross",
    "ExpandedZakharovProblem2_cross", "GriewankProblem2_cross",
    "RastriginProblem2_cross", "SalomonProblem2_cross",
    "SchwefelProblem2_cross", "SphereProblem2_cross",
    "BealeProblem_cross", "BoothProblem_cross",
    "CrossInTrayProblem_cross", "DropWaveProblem_cross",
    "EasomProblem_cross", "GoldsteinPriceProblem_cross",
    "HimmelblauProblem_cross", "HolderTableProblem_cross",
    "LeviN13Problem_cross", "MatyasProblem_cross",
    "SchafferN2Problem_cross", "ThreeHumpCamelProblem_cross",
]

const D5D = [
    "RosenbrockProblem5_cross", "StyblinskiTangProblem5_cross",
    "MichalewiczProblem5_cross", "SphereProblem5_cross",
]

println("=" ^ 80)
println("## Group B — maxvar (target: 100+, jobs completed)")
println("=" ^ 80)
audit_config(B2D, "data-opt-functions", "maxvar")

println()
println("=" ^ 80)
println("## Group B — warpedgp-yja-maxvar (data-warpedgp2, target: any completed)")
println("=" ^ 80)
for p in B2D
    d = "data-warpedgp2/$(p)"
    counts = [count_iters(joinpath(d, "warpedgp-yja-maxvar_$(i)_TVmetric.jld2")) for i in 1:20]
    println(rpad(p, 42), " ", counts)
end

println()
println("=" ^ 80)
println("## Group B — eiv (target: 100; -1=missing, running jobs may show partial)")
println("=" ^ 80)
audit_config(B2D, "data-opt-functions", "eiv")

println()
println("=" ^ 80)
println("## Group D — maxvar (target: 200)")
println("=" ^ 80)
audit_config(D5D, "data-opt-functions", "maxvar")

println()
println("=" ^ 80)
println("## Group D — eiv (target: 200, in progress)")
println("=" ^ 80)
audit_config(D5D, "data-opt-functions", "eiv")

println()
println("=" ^ 80)
println("## Group D — immd (target: 200, in progress)")
println("=" ^ 80)
audit_config(D5D, "data-opt-functions", "immd")

println()
println("## DONE")

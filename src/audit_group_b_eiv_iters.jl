using JLD2

const CROSS2D = [
    "RosenbrockProblem2", "StyblinskiTangProblem2", "MichalewiczProblem2",
    "AckleyProblem2", "AlpineProblem2", "ExpandedSchafferF6Problem2",
    "ExpandedZakharovProblem2", "GriewankProblem2", "RastriginProblem2",
    "SalomonProblem2", "SchwefelProblem2", "SphereProblem2",
    "BealeProblem", "BoothProblem", "CrossInTrayProblem", "DropWaveProblem",
    "EasomProblem", "GoldsteinPriceProblem", "HimmelblauProblem", "HolderTableProblem",
    "LeviN13Problem", "MatyasProblem", "SchafferN2Problem", "ThreeHumpCamelProblem",
]

function eiv_iters_1to20(prob; dir="data-opt-functions")
    d = joinpath(dir, "$(prob)_cross")
    isdir(d) || return fill(-3, 20)
    map(1:20) do i
        f = joinpath(d, "eiv_$(i)_TVmetric.jld2")
        isfile(f) || return -1
        try
            jldopen(f) do jf
                count(!isnan, jf["score"])
            end
        catch
            -2
        end
    end
end

const TARGET = 100

println("## Group B EIV audit — indices 1-20, target=$TARGET iters")
println("## (-1=missing, -2=load error, -3=dir missing)")
println()

bad = Tuple{String,Int,Int}[]

for p in CROSS2D
    counts = eiv_iters_1to20(p)
    bad_idx = [(i, counts[i]) for i in 1:20 if counts[i] < TARGET]
    marker = isempty(bad_idx) ? "OK" : "BAD"
    println("$(rpad(p, 35)) [$marker]  $(counts)")
    for (i, n) in bad_idx
        push!(bad, (p, i, n))
    end
end

println()
println("## Runs needing (re)submission at original index (< $TARGET iters):")
if isempty(bad)
    println("  none")
else
    for (p, i, n) in bad
        label = n == -1 ? "MISSING" : n == -2 ? "LOAD_ERR" : n == -3 ? "DIR_MISSING" : "$(n) iters"
        println("  $(p)_cross  run $(i)  [$label]")
    end
end

using JLD2

function eiv_iters(prob, dir="data-opt-functions")
    d = "$(dir)/$(prob)_cross"
    isdir(d) || return missing
    counts = Int[]
    for i in 1:5
        f = joinpath(d, "eiv_$(i)_TVmetric.jld2")
        if !isfile(f)
            push!(counts, -1)
            continue
        end
        try
            jldopen(f) do jf
                sc = jf["score"]
                push!(counts, count(!isnan, sc))
            end
        catch
            push!(counts, -2)
        end
    end
    counts
end

cross2d = ["RosenbrockProblem2","StyblinskiTangProblem2","MichalewiczProblem2",
           "AckleyProblem2","AlpineProblem2","ExpandedSchafferF6Problem2",
           "ExpandedZakharovProblem2","GriewankProblem2","RastriginProblem2",
           "SalomonProblem2","SchwefelProblem2","SphereProblem2",
           "BoothProblem","CrossInTrayProblem","DropWaveProblem","EasomProblem",
           "HimmelblauProblem","HolderTableProblem","LeviN13Problem","MatyasProblem",
           "SchafferN2Problem","ThreeHumpCamelProblem",
           "BealeProxyProblem","GoldsteinPriceProxyProblem"]

cross5d = ["RosenbrockProblem5","StyblinskiTangProblem5","MichalewiczProblem5","SphereProblem5"]

println("=== Cross 2D EIV iteration counts (target: 200) ===")
for p in cross2d
    c = eiv_iters(p)
    println(rpad(p, 30), ": ", c === missing ? "DIR MISSING" : c)
end

println("\n=== Cross 5D EIV iteration counts (target: 200) ===")
for p in cross5d
    c = eiv_iters(p)
    println(rpad(p, 30), ": ", c === missing ? "DIR MISSING" : c)
end

println("\n=== Cross 2D MaxVar iteration counts (target: 200) ===")
function mv_iters(prob, dir="data-opt-functions")
    d = "$(dir)/$(prob)_cross"
    isdir(d) || return missing
    counts = Int[]
    for i in 1:5
        f = joinpath(d, "maxvar_$(i)_TVmetric.jld2")
        if !isfile(f)
            push!(counts, -1)
            continue
        end
        try
            jldopen(f) do jf
                push!(counts, count(!isnan, jf["score"]))
            end
        catch
            push!(counts, -2)
        end
    end
    counts
end

for p in cross2d
    c = mv_iters(p)
    println(rpad(p, 30), ": ", c === missing ? "DIR MISSING" : c)
end

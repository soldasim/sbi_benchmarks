using JLD2
group_b = ["RosenbrockProblem2_cross","StyblinskiTangProblem2_cross","MichalewiczProblem2_cross","AckleyProblem2_cross","AlpineProblem2_cross","ExpandedSchafferF6Problem2_cross","ExpandedZakharovProblem2_cross","GriewankProblem2_cross","RastriginProblem2_cross","SalomonProblem2_cross","SchwefelProblem2_cross","SphereProblem2_cross","BealeProblem_cross","BoothProblem_cross","CrossInTrayProblem_cross","DropWaveProblem_cross","EasomProblem_cross","GoldsteinPriceProblem_cross","HimmelblauProblem_cross","HolderTableProblem_cross","LeviN13Problem_cross","MatyasProblem_cross","SchafferN2Problem_cross","ThreeHumpCamelProblem_cross"]
group_cd = ["DuffingProblem5","DiffusionProblem5D","RosenbrockProblem5_cross","StyblinskiTangProblem5_cross","MichalewiczProblem5_cross","SphereProblem5_cross"]

function check_group_bcd(names, target)
    for p in names
        found = 0
        total_nan = 0
        total_valid = 0
        errors = Int[]
        for i in 1:20
            fpath = "data-warpedgp2/$(p)/warpedgp-yja-maxvar_$(i)_TVmetric.jld2"
            if isfile(fpath)
                try
                    s = load(fpath, "score")
                    found += 1
                    n = count(isnan, s)
                    total_nan += n
                    total_valid += (length(s) - n)
                catch e
                    push!(errors, i)
                end
            end
        end
        println("$p: found=$found/20 total_nan=$total_nan total_valid=$total_valid target=$target errors=$errors")
    end
end
println("=== GROUP B (target 201) ===")
check_group_bcd(group_b, 201)
println("=== GROUP C/D (target 201) ===")
check_group_bcd(group_cd, 201)
println("DONE_NAN_CHECK_BCD")

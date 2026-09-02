using JLD2
group_b = ["RosenbrockProblem2_cross","StyblinskiTangProblem2_cross","MichalewiczProblem2_cross","AckleyProblem2_cross","AlpineProblem2_cross","ExpandedSchafferF6Problem2_cross","ExpandedZakharovProblem2_cross","GriewankProblem2_cross","RastriginProblem2_cross","SalomonProblem2_cross","SchwefelProblem2_cross","SphereProblem2_cross","BealeProblem_cross","BoothProblem_cross","CrossInTrayProblem_cross","DropWaveProblem_cross","EasomProblem_cross","GoldsteinPriceProblem_cross","HimmelblauProblem_cross","HolderTableProblem_cross","LeviN13Problem_cross","MatyasProblem_cross","SchafferN2Problem_cross","ThreeHumpCamelProblem_cross"]
group_cd = ["DuffingProblem5","DiffusionProblem5D","RosenbrockProblem5_cross","StyblinskiTangProblem5_cross","MichalewiczProblem5_cross","SphereProblem5_cross"]

function archive_nan(p)
    total_nan = 0
    total_valid = 0
    for i in 1:20
        fpath = "data-warpedgp2_archive_pre-samplesfix/$(p)/warpedgp-yja-maxvar_$(i)_TVmetric.jld2"
        if isfile(fpath)
            try
                s = load(fpath, "score")
                n = count(isnan, s)
                total_nan += n
                total_valid += (length(s) - n)
            catch e
            end
        end
    end
    return total_nan, total_valid
end

println("=== FULL BEFORE/AFTER (pre-fix archive vs post-fix new) ===")
for p in vcat(group_b, group_cd)
    pre_nan, pre_valid = archive_nan(p)
    pre_total = pre_nan + pre_valid
    pre_pct = pre_total > 0 ? round(100*pre_nan/pre_total, digits=1) : -1.0
    println("$p: pre_nan=$pre_nan/$pre_total ($pre_pct%)")
end
println("DONE_FULL_ARCHIVE_CHECK")

using JLD2
worst = ["DuffingProblem5","DiffusionProblem5D","RosenbrockProblem5_cross","StyblinskiTangProblem5_cross","MichalewiczProblem5_cross","SphereProblem5_cross"]
function check_archive(names, target)
    for p in names
        found = 0
        total_nan = 0
        total_valid = 0
        errors = Int[]
        for i in 1:20
            fpath = "data-warpedgp2_archive_pre-samplesfix/$(p)/warpedgp-yja-maxvar_$(i)_TVmetric.jld2"
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
        println("$p (ARCHIVE/pre-fix): found=$found/20 total_nan=$total_nan total_valid=$total_valid target=$target errors=$errors")
    end
end
check_archive(worst, 201)
println("DONE_ARCHIVE_CHECK")

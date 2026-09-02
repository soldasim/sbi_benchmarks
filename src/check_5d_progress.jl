using JLD2
for pname in ["DuffingProblem5", "DiffusionProblem5D"]
    for rname in ["standard", "maxvar", "eiv", "immd"]
        for i in 1:5
            path = "data-bosip-norm/$pname/$(rname)_$(i)_TVmetric.jld2"
            if isfile(path)
                f = jldopen(path)
                n = length(f["score"])
                close(f)
                println("$pname/$rname/$i: $n iters")
            else
                println("$pname/$rname/$i: (no file)")
            end
        end
    end
end

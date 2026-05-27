using JLD2, Statistics

BASE = "/home/soldasim/repos/bosip_benchmarks"
DATA = BASE * "/data-convergence4"

out = open(BASE * "/inspect_uniform_comparison.txt", "w")

function tee(s)
    println(s)
    println(out, s)
end

for dim in [4, 5, 6]
    tee("\n=== CubicProblem $(dim)D ===")
    dir = "$(DATA)/MultidimProblem{CubicProblem}$(dim)"
    for run_name in ["uniform", "maxvar"]
        tvs = []
        for idx in 1:5
            f = "$(dir)/$(run_name)_$(idx)_TVmetric.jld2"
            isfile(f) || continue
            d = load(f)
            v = d["score"]
            v = v isa Matrix ? vec(v) : v
            push!(tvs, v)
        end
        isempty(tvs) && continue
        n = minimum(length.(tvs))
        mat = reduce(hcat, [v[1:n] for v in tvs])
        avg = vec(mean(mat, dims=2))
        last20 = avg[max(1, n-20):n]
        first20 = avg[1:min(20, n)]
        tee("  $(run_name): n_iters=$(n), early_mean=$(round(mean(first20),digits=3)), late_mean=$(round(mean(last20),digits=3)), final=$(round(avg[end],digits=3))")
        # Count iterations in 2nd half where mean TV > 0.5 (approximate spike detection on averaged trace)
        second_half = avg[div(n,2)+1:end]
        nspikes = sum(second_half .> 0.5)
        tee("    high-TV iters (>0.5) in 2nd half: $(nspikes)")
        # Also look at per-run max spikes in 2nd half
        per_run_spikes = [sum(tvs[i][div(length(tvs[i]),2)+1:end] .> 0.5) for i in 1:length(tvs)]
        tee("    per-run high-TV iters in 2nd half: $(per_run_spikes)")
    end
end

close(out)
println("\nDone. Written to $(BASE)/inspect_uniform_comparison.txt")

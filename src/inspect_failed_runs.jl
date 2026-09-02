using JLD2, Statistics, LinearAlgebra

function inspect(problem_name, run_indices)
    println("\n" * "="^60)
    println("Problem: $problem_name")
    println("="^60)

    for idx in run_indices
        base = "data-opt-functions/$problem_name/eiv_$(idx)"
        data_file = base * "_data.jld2"

        isfile(data_file) || (println("  run $idx: no data file"); continue)

        d = load(data_file)
        X, Y = d["data"]
        n = size(X, 2)

        println("\n  --- EIV run $idx  (n=$n points) ---")

        # Y statistics
        println("  Y: min=$(round(minimum(Y), digits=4))  max=$(round(maximum(Y), digits=4))  " *
                "mean=$(round(mean(Y), digits=4))  std=$(round(std(Y), digits=4))")

        # X spread per dimension
        for dim in 1:size(X, 1)
            lo, hi = extrema(X[dim, :])
            sd = std(X[dim, :])
            println("  X[$dim]: range=[$(round(lo,digits=4)), $(round(hi,digits=4))]  std=$(round(sd,digits=4))")
        end

        # Pairwise distances — check for clustering
        dists = [norm(X[:, i] - X[:, j]) for i in 1:n for j in i+1:n]
        println("  Pairwise dist: min=$(round(minimum(dists), digits=6))  " *
                "median=$(round(median(dists), digits=4))  max=$(round(maximum(dists), digits=4))")

        # Count near-duplicate points (dist < 1e-4)
        n_close = count(d -> d < 1e-4, dists)
        n_close > 0 && println("  *** $(n_close) pairs within dist 1e-4 (likely duplicates/clustering!) ***")

        # Last 10 queried points
        println("  Last 10 X columns (most recent queries):")
        for i in max(1, n-9):n
            println("    [$i]  x=$(round.(X[:, i], digits=4))  y=$(round.(Y[:, i], digits=4))")
        end
    end
end

inspect("BealeProblem",          [3, 5])
inspect("GoldsteinPriceProblem", [1, 3, 4])

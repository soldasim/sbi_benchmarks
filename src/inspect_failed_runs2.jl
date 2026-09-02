using JLD2, Statistics, LinearAlgebra

println("=== GoldsteinPrice: clustering in x2 ===")
for idx in [3, 4]
    d = load("data-opt-functions/GoldsteinPriceProblem/eiv_$(idx)_data.jld2")
    X, Y = d["data"]
    n = size(X, 2)
    in_band = count(i -> -1.0 <= X[2,i] <= -0.3, 1:n)
    pct = round(100*in_band/n, digits=1)
    println("Run $idx (n=$n): x2 in [-1.0, -0.3]: $in_band ($pct%)")

    last20 = X[:, max(1,n-19):n]
    dists_last = [norm(last20[:,i]-last20[:,j]) for i in 1:size(last20,2) for j in i+1:size(last20,2)]
    println("  Last-20 x2 values: $(round.(X[2, max(1,n-19):n], digits=3))")
    println("  Last-20 min pairwise dist: $(round(minimum(dists_last), digits=6))")
end

println("\n=== BealeProblem: last-20 clustering ===")
for idx in [3, 5]
    d = load("data-opt-functions/BealeProblem/eiv_$(idx)_data.jld2")
    X, Y = d["data"]
    n = size(X, 2)
    last20 = X[:, max(1,n-19):n]
    dists_last = [norm(last20[:,i]-last20[:,j]) for i in 1:size(last20,2) for j in i+1:size(last20,2)]
    y_last = Y[1, max(1,n-19):n]
    println("Run $idx (n=$n): last-20 min dist=$(round(minimum(dists_last), digits=6)), Y range=[$(round(minimum(y_last),digits=3)), $(round(maximum(y_last),digits=3))]")
    # Also check PosDefException region: where does x1 cluster?
    println("  Last-20 X: $(round.(last20, digits=3))")
end

println("\n=== Compare: successful BealeProblem EIV runs (1,2,4) ===")
for idx in [1, 2, 4]
    d = load("data-opt-functions/BealeProblem/eiv_$(idx)_data.jld2")
    X, Y = d["data"]
    n = size(X, 2)
    println("Run $idx (n=$n): Y range=[$(round(minimum(Y),digits=2)), $(round(maximum(Y),digits=2))], std=$(round(std(Y),digits=1))")
    dists = [norm(X[:,i]-X[:,j]) for i in 1:n for j in i+1:n]
    println("  Min pairwise dist=$(round(minimum(dists), digits=6)), median=$(round(median(dists), digits=3))")
end

println("\n=== Compare: successful GoldsteinPrice EIV runs (2,5) ===")
for idx in [2, 5]
    d = load("data-opt-functions/GoldsteinPriceProblem/eiv_$(idx)_data.jld2")
    X, Y = d["data"]
    n = size(X, 2)
    in_band = count(i -> -1.0 <= X[2,i] <= -0.3, 1:n)
    println("Run $idx (n=$n): x2 in [-1.0,-0.3]: $in_band ($(round(100*in_band/n,digits=1))%), Y std=$(round(std(Y),digits=1))")
    dists = [norm(X[:,i]-X[:,j]) for i in 1:n for j in i+1:n]
    println("  Min pairwise dist=$(round(minimum(dists), digits=6))")
end

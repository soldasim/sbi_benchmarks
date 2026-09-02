using JLD2
using Statistics

sir_true = [0.6148, 0.1917]
duffing_true = [0.15, -1.0, 0.5]

function analyze_method(problem::String, method::String, run_idx::Int)
    filename = "data-bosip/$(problem)Problem/$(method)_$(run_idx)_data.jld2"
    try
        d = load(filename)
        key = first(keys(d))
        data = d[key]
        
        if haskey(d, "X")
            X = d["X"]
            Y = d["Y"]
        else
            X = data.X
            Y = data.Y
        end
        
        return X, Y
    catch e
        println("Error loading: $e")
        return nothing, nothing
    end
end

println("="^80)
println("SIR PROBLEM - STANDARD (LogMaxVar) - Run 1")
println("="^80)
X_sir_std, Y_sir_std = analyze_method("SIR", "standard", 1)
if !isnothing(X_sir_std)
    println("\nData matrix shape: $(size(X_sir_std)) rows x cols, Y shape: $(size(Y_sir_std))")
    println("\nFirst 10 X columns:")
    for i in 1:min(10, size(X_sir_std, 2))
        println("  $(round.(X_sir_std[:, i]; digits=6))")
    end
    println("\nLast 10 X columns:")
    for i in max(1, size(X_sir_std, 2)-9):size(X_sir_std, 2)
        println("  $(round.(X_sir_std[:, i]; digits=6))")
    end
    println("\nParameter range and statistics:")
    for d in 1:size(X_sir_std, 1)
        mn = minimum(X_sir_std[d,:])
        mx = maximum(X_sir_std[d,:])
        mu = mean(X_sir_std[d,:])
        sg = std(X_sir_std[d,:])
        println("  Param $d (true=$(sir_true[d])): [$(round(mn;digits=6)), $(round(mx;digits=6))], μ=$(round(mu;digits=6)), σ=$(round(sg;digits=6))")
    end
    println("\nY (log-likelihood) statistics:")
    println("  Range: [$(round(minimum(Y_sir_std);digits=6)), $(round(maximum(Y_sir_std);digits=6))]")
    println("  Mean: $(round(mean(Y_sir_std);digits=6)), Std: $(round(std(Y_sir_std);digits=6))")
    
    dists = [sqrt(sum((X_sir_std[:, i] .- sir_true).^2)) for i in 1:size(X_sir_std, 2)]
    near_true = sum(dists .< 0.3)
    println("\n  Points within distance 0.3 of true params: $near_true / $(size(X_sir_std,2))")
end

println("\n" * "="^80)
println("SIR PROBLEM - EIV - Run 1")
println("="^80)
X_sir_eiv, Y_sir_eiv = analyze_method("SIR", "eiv", 1)
if !isnothing(X_sir_eiv)
    println("\nData matrix shape: $(size(X_sir_eiv)) rows x cols, Y shape: $(size(Y_sir_eiv))")
    println("\nFirst 10 X columns:")
    for i in 1:min(10, size(X_sir_eiv, 2))
        println("  $(round.(X_sir_eiv[:, i]; digits=6))")
    end
    println("\nLast 10 X columns:")
    for i in max(1, size(X_sir_eiv, 2)-9):size(X_sir_eiv, 2)
        println("  $(round.(X_sir_eiv[:, i]; digits=6))")
    end
    println("\nParameter range and statistics:")
    for d in 1:size(X_sir_eiv, 1)
        mn = minimum(X_sir_eiv[d,:])
        mx = maximum(X_sir_eiv[d,:])
        mu = mean(X_sir_eiv[d,:])
        sg = std(X_sir_eiv[d,:])
        println("  Param $d (true=$(sir_true[d])): [$(round(mn;digits=6)), $(round(mx;digits=6))], μ=$(round(mu;digits=6)), σ=$(round(sg;digits=6))")
    end
    println("\nY (log-likelihood) statistics:")
    println("  Range: [$(round(minimum(Y_sir_eiv);digits=6)), $(round(maximum(Y_sir_eiv);digits=6))]")
    println("  Mean: $(round(mean(Y_sir_eiv);digits=6)), Std: $(round(std(Y_sir_eiv);digits=6))")
    
    dists = [sqrt(sum((X_sir_eiv[:, i] .- sir_true).^2)) for i in 1:size(X_sir_eiv, 2)]
    near_true = sum(dists .< 0.3)
    println("\n  Points within distance 0.3 of true params: $near_true / $(size(X_sir_eiv,2))")
end

println("\n" * "="^80)
println("DUFFING PROBLEM - STANDARD (LogMaxVar) - Run 1")
println("="^80)
X_duff_std, Y_duff_std = analyze_method("Duffing", "standard", 1)
if !isnothing(X_duff_std)
    println("\nData matrix shape: $(size(X_duff_std)) rows x cols, Y shape: $(size(Y_duff_std))")
    println("\nFirst 10 X columns:")
    for i in 1:min(10, size(X_duff_std, 2))
        println("  $(round.(X_duff_std[:, i]; digits=6))")
    end
    println("\nLast 10 X columns:")
    for i in max(1, size(X_duff_std, 2)-9):size(X_duff_std, 2)
        println("  $(round.(X_duff_std[:, i]; digits=6))")
    end
    println("\nParameter range and statistics:")
    for d in 1:size(X_duff_std, 1)
        mn = minimum(X_duff_std[d,:])
        mx = maximum(X_duff_std[d,:])
        mu = mean(X_duff_std[d,:])
        sg = std(X_duff_std[d,:])
        println("  Param $d (true=$(duffing_true[d])): [$(round(mn;digits=6)), $(round(mx;digits=6))], μ=$(round(mu;digits=6)), σ=$(round(sg;digits=6))")
    end
    println("\nY (log-likelihood) statistics:")
    println("  Range: [$(round(minimum(Y_duff_std);digits=6)), $(round(maximum(Y_duff_std);digits=6))]")
    println("  Mean: $(round(mean(Y_duff_std);digits=6)), Std: $(round(std(Y_duff_std);digits=6))")
    
    dists = [sqrt(sum((X_duff_std[:, i] .- duffing_true).^2)) for i in 1:size(X_duff_std, 2)]
    near_true = sum(dists .< 0.3)
    println("\n  Points within distance 0.3 of true params: $near_true / $(size(X_duff_std,2))")
    
    alpha_vals = X_duff_std[2, :]
    near_zero = sum(abs.(alpha_vals) .< 0.1)
    println("\n  Alpha clustering near bifurcation (α≈0, within 0.1): $near_zero / $(size(X_duff_std,2))")
    println("  Alpha quantiles: $(round(quantile(alpha_vals, [0.0, 0.25, 0.5, 0.75, 1.0]); digits=6))")
end

println("\n" * "="^80)
println("DUFFING PROBLEM - EIV - Run 1")
println("="^80)
X_duff_eiv, Y_duff_eiv = analyze_method("Duffing", "eiv", 1)
if !isnothing(X_duff_eiv)
    println("\nData matrix shape: $(size(X_duff_eiv)) rows x cols, Y shape: $(size(Y_duff_eiv))")
    println("\nFirst 10 X columns:")
    for i in 1:min(10, size(X_duff_eiv, 2))
        println("  $(round.(X_duff_eiv[:, i]; digits=6))")
    end
    println("\nLast 10 X columns:")
    for i in max(1, size(X_duff_eiv, 2)-9):size(X_duff_eiv, 2)
        println("  $(round.(X_duff_eiv[:, i]; digits=6))")
    end
    println("\nParameter range and statistics:")
    for d in 1:size(X_duff_eiv, 1)
        mn = minimum(X_duff_eiv[d,:])
        mx = maximum(X_duff_eiv[d,:])
        mu = mean(X_duff_eiv[d,:])
        sg = std(X_duff_eiv[d,:])
        println("  Param $d (true=$(duffing_true[d])): [$(round(mn;digits=6)), $(round(mx;digits=6))], μ=$(round(mu;digits=6)), σ=$(round(sg;digits=6))")
    end
    println("\nY (log-likelihood) statistics:")
    println("  Range: [$(round(minimum(Y_duff_eiv);digits=6)), $(round(maximum(Y_duff_eiv);digits=6))]")
    println("  Mean: $(round(mean(Y_duff_eiv);digits=6)), Std: $(round(std(Y_duff_eiv);digits=6))")
    
    dists = [sqrt(sum((X_duff_eiv[:, i] .- duffing_true).^2)) for i in 1:size(X_duff_eiv, 2)]
    near_true = sum(dists .< 0.3)
    println("\n  Points within distance 0.3 of true params: $near_true / $(size(X_duff_eiv,2))")
    
    alpha_vals = X_duff_eiv[2, :]
    near_zero = sum(abs.(alpha_vals) .< 0.1)
    println("\n  Alpha clustering near bifurcation (α≈0, within 0.1): $near_zero / $(size(X_duff_eiv,2))")
    println("  Alpha quantiles: $(round(quantile(alpha_vals, [0.0, 0.25, 0.5, 0.75, 1.0]); digits=6))")
end

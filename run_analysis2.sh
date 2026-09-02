#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

module load Julia/1.10.0-linux-x86_64
cd ~/repos/bosip_benchmarks
julia << 'JLEOF'
include("activate.jl")

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
        println("Error: $e")
        return nothing, nothing
    end
end

println("="^80)
println("SIR PROBLEM - STANDARD (LogMaxVar)")
println("="^80)
X_sir_std, Y_sir_std = analyze_method("SIR", "standard", 1)
if !isnothing(X_sir_std)
    println("X shape: $(size(X_sir_std))")
    println("\nFirst 10 columns:")
    for i in 1:min(10, size(X_sir_std, 2))
        println("  Col $i: $(X_sir_std[:, i])")
    end
    println("\nLast 10 columns:")
    for i in max(1, size(X_sir_std, 2)-9):size(X_sir_std, 2)
        println("  Col $i: $(X_sir_std[:, i])")
    end
    println("\nParameter statistics:")
    for d in 1:size(X_sir_std, 1)
        mn = minimum(X_sir_std[d,:])
        mx = maximum(X_sir_std[d,:])
        mu = mean(X_sir_std[d,:])
        sg = std(X_sir_std[d,:])
        println("  Dim $d (true=$(sir_true[d])): min=$mn, max=$mx, mean=$mu, std=$sg")
    end
    println("\nY statistics: min=$(minimum(Y_sir_std)), max=$(maximum(Y_sir_std)), mean=$(mean(Y_sir_std))")
end

println("\n" * "="^80)
println("SIR PROBLEM - EIV")
println("="^80)
X_sir_eiv, Y_sir_eiv = analyze_method("SIR", "eiv", 1)
if !isnothing(X_sir_eiv)
    println("X shape: $(size(X_sir_eiv))")
    println("\nFirst 10 columns:")
    for i in 1:min(10, size(X_sir_eiv, 2))
        println("  Col $i: $(X_sir_eiv[:, i])")
    end
    println("\nLast 10 columns:")
    for i in max(1, size(X_sir_eiv, 2)-9):size(X_sir_eiv, 2)
        println("  Col $i: $(X_sir_eiv[:, i])")
    end
    println("\nParameter statistics:")
    for d in 1:size(X_sir_eiv, 1)
        mn = minimum(X_sir_eiv[d,:])
        mx = maximum(X_sir_eiv[d,:])
        mu = mean(X_sir_eiv[d,:])
        sg = std(X_sir_eiv[d,:])
        println("  Dim $d (true=$(sir_true[d])): min=$mn, max=$mx, mean=$mu, std=$sg")
    end
    println("\nY statistics: min=$(minimum(Y_sir_eiv)), max=$(maximum(Y_sir_eiv)), mean=$(mean(Y_sir_eiv))")
end

println("\n" * "="^80)
println("DUFFING PROBLEM - STANDARD (LogMaxVar)")
println("="^80)
X_duff_std, Y_duff_std = analyze_method("Duffing", "standard", 1)
if !isnothing(X_duff_std)
    println("X shape: $(size(X_duff_std))")
    println("\nFirst 10 columns:")
    for i in 1:min(10, size(X_duff_std, 2))
        println("  Col $i: $(X_duff_std[:, i])")
    end
    println("\nLast 10 columns:")
    for i in max(1, size(X_duff_std, 2)-9):size(X_duff_std, 2)
        println("  Col $i: $(X_duff_std[:, i])")
    end
    println("\nParameter statistics:")
    for d in 1:size(X_duff_std, 1)
        mn = minimum(X_duff_std[d,:])
        mx = maximum(X_duff_std[d,:])
        mu = mean(X_duff_std[d,:])
        sg = std(X_duff_std[d,:])
        println("  Dim $d (true=$(duffing_true[d])): min=$mn, max=$mx, mean=$mu, std=$sg")
    end
    println("\nY statistics: min=$(minimum(Y_duff_std)), max=$(maximum(Y_duff_std)), mean=$(mean(Y_duff_std))")
    
    alpha_vals = X_duff_std[2, :]
    near_zero = sum(abs.(alpha_vals) .< 0.1)
    println("\nAlpha clustering: $near_zero points within 0.1 of zero (bifurcation)")
end

println("\n" * "="^80)
println("DUFFING PROBLEM - EIV")
println("="^80)
X_duff_eiv, Y_duff_eiv = analyze_method("Duffing", "eiv", 1)
if !isnothing(X_duff_eiv)
    println("X shape: $(size(X_duff_eiv))")
    println("\nFirst 10 columns:")
    for i in 1:min(10, size(X_duff_eiv, 2))
        println("  Col $i: $(X_duff_eiv[:, i])")
    end
    println("\nLast 10 columns:")
    for i in max(1, size(X_duff_eiv, 2)-9):size(X_duff_eiv, 2)
        println("  Col $i: $(X_duff_eiv[:, i])")
    end
    println("\nParameter statistics:")
    for d in 1:size(X_duff_eiv, 1)
        mn = minimum(X_duff_eiv[d,:])
        mx = maximum(X_duff_eiv[d,:])
        mu = mean(X_duff_eiv[d,:])
        sg = std(X_duff_eiv[d,:])
        println("  Dim $d (true=$(duffing_true[d])): min=$mn, max=$mx, mean=$mu, std=$sg")
    end
    println("\nY statistics: min=$(minimum(Y_duff_eiv)), max=$(maximum(Y_duff_eiv)), mean=$(mean(Y_duff_eiv))")
    
    alpha_vals = X_duff_eiv[2, :]
    near_zero = sum(abs.(alpha_vals) .< 0.1)
    println("\nAlpha clustering: $near_zero points within 0.1 of zero (bifurcation)")
end
JLEOF

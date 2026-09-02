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

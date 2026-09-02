include(joinpath(@__DIR__, "..", "src", "plot_marginals.jl"))
precompute_bip_grid(DiffusionProblem5D(); force=true)
println("Precompute done.")

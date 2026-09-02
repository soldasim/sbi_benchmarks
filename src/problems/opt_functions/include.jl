abstract type AbstractOptFunctionProblem <: AbstractProblem end

include("rosenbrock.jl")
include("styblinski_tang.jl")
include("michalewicz.jl")

# d-dimensional
include("ackley.jl")
include("alpine.jl")
include("expanded_schaffer_f6.jl")
include("expanded_zakharov.jl")
include("griewank.jl")
include("rastrigin.jl")
include("salomon.jl")
include("schwefel.jl")
include("sphere.jl")

# 2D-only
include("beale.jl")
include("beale_proxy.jl")
include("booth.jl")
include("cross_in_tray.jl")
include("drop_wave.jl")
include("easom.jl")
include("goldstein_price.jl")
include("goldstein_price_proxy.jl")
include("himmelblau.jl")
include("holder_table.jl")
include("levi_n13.jl")
include("matyas.jl")
include("schaffer_n2.jl")
include("three_hump_camel.jl")

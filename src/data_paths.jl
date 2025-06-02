
data_dir(p::AbstractProblem) = "data/" * string(typeof(p))
data_file(p::AbstractProblem, run_name::String, run_idx::Int) = data_dir(p) * "/" * run_name * "_$run_idx" * ".jld2"
data_file(p::AbstractProblem, run_name::String, ::Nothing) = data_dir(p) * "/" * run_name * ".jld2"

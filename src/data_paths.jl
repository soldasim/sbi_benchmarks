
data_dir(p::AbstractProblem) = "data/" * string(typeof(p))
data_filepath(p::AbstractProblem, run_name::String, run_idx::Int) = data_dir(p) * "/" * run_name * "_$run_idx"
data_filepath(p::AbstractProblem, run_name::String, ::Nothing) = data_dir(p) * "/" * run_name

plot_dir() = "plots"


# Data are stored in: "data/{problem}/{run name}_{run index}_{some suffix}.jld2"

# default problem name
get_name(problem::AbstractProblem) = (problem |> typeof |> string)
get_name(problem::MultidimProblem) = (problem |> typeof |> string) * string(problem.scaleup) #* "-new"
get_name(problem::GaussProblem) = (problem |> typeof |> string) * string(problem.x_dim)
get_name(problem::MeanGauss) = (problem |> typeof |> string) * string(problem.x_dim)

data_dir(problem::AbstractProblem) = "data-convergence4/" * get_name(problem) # TODO

data_base_filepath(problem::AbstractProblem, run_name::String, run_idx) = data_dir(problem) * "/" * base_filename(problem, run_name, run_idx)

base_filename(problem::AbstractProblem, run_name::String, run_idx::Int) = run_name * "_" * string(run_idx)
base_filename(problem::AbstractProblem, run_name::String, run_idx::Nothing) = run_name

starts_dir(problem::AbstractProblem) = data_dir(problem) * "/starts"

posterior_grid_dir(problem::AbstractProblem) = data_dir(problem) * "/grid"
posterior_grid_filepath(problem::AbstractProblem) = posterior_grid_dir(problem) * "/posterior_grid.jld2"

simulator_grid_dir(problem::AbstractProblem) = data_dir(problem) * "/grid"
simulator_grid_filepath(problem::AbstractProblem) = simulator_grid_dir(problem) * "/simulator_grid.jld2"

gaps_dir(problem::AbstractProblem) = data_dir(problem) * "/gaps"
gaps_filepath(problem::AbstractProblem; metric::Symbol=:tv) = gaps_dir(problem) * "/gaps_$(metric).jld2"

plot_dir() = "plots"

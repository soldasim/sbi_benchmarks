
# Data are stored in: "data/{problem}/{run name}_{run index}_{some suffix}.jld2"

# default problem name
get_name(problem::AbstractProblem) = (problem |> typeof |> string)
get_name(problem::MultidimProblem) = (problem |> typeof |> string) * string(problem.scaleup) #* "-new"
get_name(problem::GaussProblem) = (problem |> typeof |> string) * string(problem.x_dim)
get_name(problem::MeanGauss) = (problem |> typeof |> string) * string(problem.x_dim)

data_dir(problem::AbstractProblem) = "data-convergence4/" * get_name(problem)
data_dir(problem::AbstractOptFunctionProblem) = "data-opt-functions/" * get_name(problem)
data_dir(problem::SharpProblem{<:AbstractOptFunctionProblem}) = "data-opt-functions/" * get_name(problem)
data_dir(problem::HexObsProblem{<:AbstractOptFunctionProblem}) = "data-opt-functions/" * get_name(problem)
data_dir(problem::CrossPolytopeObsProblem{<:AbstractOptFunctionProblem}) = "data-opt-functions/" * get_name(problem)
# data_dir(problem::ProxySIRProblem) = "data-bosip/" * get_name(problem)
# Original 7 benchmark problems (paper) — post-normalization-fix results (data-bosip-norm/)
# data_dir(problem::ABProblem) = "data-bosip-norm/" * get_name(problem)
# data_dir(problem::SimpleProblem) = "data-bosip-norm/" * get_name(problem)
# data_dir(problem::BananaProblem) = "data-bosip-norm/" * get_name(problem)
# data_dir(problem::BimodalProblem) = "data-bosip-norm/" * get_name(problem)
# data_dir(problem::SIRProblem) = "data-bosip-norm/" * get_name(problem)
# data_dir(problem::DuffingProblem) = "data-bosip-norm/" * get_name(problem)
# data_dir(problem::DiffusionProblem) = "data-bosip-norm/" * get_name(problem)
# IMMD experiments — new IMMD acquisition
data_dir(problem::ABProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::SimpleProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::BananaProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::BimodalProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::SIRProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::ProxySIRProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::DuffingProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::DuffingProblem5) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::DiffusionProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::DiffusionProblem5D) = "data-bosip-norm/" * get_name(problem)
# Log-likelihood variants of the 7 standard problems
data_dir(problem::LogABProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::LogSimpleProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::LogBananaProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::LogBimodalProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::LogSIRProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::LogDuffingProblem) = "data-bosip-norm/" * get_name(problem)
data_dir(problem::LogDiffusionProblem) = "data-bosip-norm/" * get_name(problem)

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

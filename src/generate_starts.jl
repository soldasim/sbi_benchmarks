
function generate_starts(problem::AbstractProblem, n_runs::Int)
    init_data_count = 3

    dir = data_dir(problem)
    mkpath(dir)

    for run_idx in 1:n_runs
        file = dir * "/start_$run_idx.jld2"

        data = get_init_data(problem, init_data_count)
        @save file data=data
    end

    nothing
end


"""
    get_init_data(::AbstractProblem, count::Int) -> ExperimentData

Generate `count` initial data points sampled from the `x_prior` of the given problem.
"""
function get_init_data(problem::AbstractProblem, count::Int)
    prior = x_prior(problem)
    # @assert extrema(prior) == domain(problem).bounds # TODO does not work with MultidimProblem
    @show domain(problem).bounds
    sim = simulator(problem)

    if count == 1
        x = mean(domain(problem).bounds)
        y = sim(x)
        return BOSS.ExperimentData(hcat(x), hcat(y))
    else
        X = rand(prior, count)
        Y = reduce(hcat, (sim(x) for x in eachcol(X)))[:,:]
        return BOSS.ExperimentData(X, Y)
    end
end
function get_init_data_with_grads(problem::AbstractProblem, count::Int)
    prior = x_prior(problem)
    # @assert extrema(prior) == domain(problem).bounds # TODO does not work with MultidimProblem
    @show domain(problem).bounds
    sim = simulator(problem)

    if count == 1
        x = mean(domain(problem).bounds)
        y, J = sim(x)
        return BOSS.GradientData(hcat(x), hcat(y), cat(J; dims=3))
    else
        X = rand(prior, count)
        results = [sim(x) for x in eachcol(X)]
        Y = hcat([r[1] for r in results]...)
        J = cat([r[2] for r in results]...; dims=3)

        return BOSS.GradientData(X, Y, J)
    end
end

function load_init_data(problem::AbstractProblem, run_idx::Int)
    start_file = starts_dir(problem) * "/start_$(run_idx).jld2"
    data = load(start_file, "X")
    return data
end

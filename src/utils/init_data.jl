
"""
    get_init_data(::AbstractProblem, count::Int) -> ExperimentData

Generate `count` initial data points sampled from the `x_prior` of the given problem.
"""
function get_init_data(problem::AbstractProblem, count::Int)
    prior = x_prior(problem)
    @assert extrema(prior) == domain(problem).bounds
    sim = simulator(problem)

    X = rand(prior, count)
    Y = reduce(hcat, (sim(x) for x in eachcol(X)))[:,:]
    return BOSS.ExperimentData(X, Y)
end

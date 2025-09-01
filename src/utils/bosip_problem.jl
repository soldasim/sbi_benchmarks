
function construct_bosip_problem(;
    problem::AbstractProblem,
    data::ExperimentData,
    acquisition::BosipAcquisition,
    model::SurrogateModel,
)
    return BosipProblem(data;
        f = simulator(problem),
        domain = domain(problem),
        acquisition,
        model,
        likelihood = likelihood(problem),
        x_prior = x_prior(problem),
    )
end

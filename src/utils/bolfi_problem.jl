
function construct_bolfi_problem(;
    problem::AbstractProblem,
    data::ExperimentData,
    acquisition::BolfiAcquisition,
    model::SurrogateModel,
)
    return BolfiProblem(data;
        f = simulator(problem),
        domain = domain(problem),
        acquisition,
        model,
        likelihood = likelihood(problem),
        x_prior = x_prior(problem),
    )
end

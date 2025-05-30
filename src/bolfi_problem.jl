
function construct_bolfi_problem(;
    problem::AbstractProblem,
    data::ExperimentData,
    acquisition::BolfiAcquisition,
    model::AbstractModel,
)
    return BolfiProblem(data;
        f = simulator(problem),
        domain = domain(problem),
        acquisition,
        model = construct_model(model, problem),
        likelihood = likelihood(problem),
        x_prior = x_prior(problem),
    )
end

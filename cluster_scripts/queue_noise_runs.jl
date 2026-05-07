include("queue_jobs.jl")

function queue_noise_runs()
    iters = 1000
    selected_runs = [1]
    noise_vals = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    # TODO
    # problems = [MultidimProblem(ABProblem(), d) for d in 1:3]
    problems = [MeanGauss(; x_dim=d) for d in 1:6]

    for problem in problems
        for noise in noise_vals
            # queue_jobs(problem, "standard-warm-noise"; selected_runs, iters, noise)
            queue_jobs(problem, "grads-warm-noise"; selected_runs, iters, noise)
        end
    end
end

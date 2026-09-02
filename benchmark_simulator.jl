using BenchmarkTools
include("src/main.jl")

function btime_simulator(problem::AbstractProblem; samples=1000)
    # xs = rand(x_prior(problem), samples)
    xs = [rand(x_prior(problem)) for _ in 1:samples]
    
    # prepare simulators
    sim1 = simulator(problem)
    problem.gradients = true
    sim2 = simulator(problem)

    # compile (probably unnecessary)
    sim1(mean(xs))
    sim2(mean(xs))

    # measure
    t1 = @elapsed for i in 1:samples
        sim1(xs[i])
    end
    t2 = @elapsed for i in 1:samples
        sim2(xs[i])
    end
    
    println("Simulator without gradients: $(t1 / samples) seconds per sample")
    println("Simulator with gradients: $(t2 / samples) seconds per sample")
    println("Gradient overhead: $(t2 / t1)×")
    return t1, t2
end

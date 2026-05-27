
function reconstruct_problem(problem_name::AbstractString)
    if startswith(problem_name, "MultidimProblem")
        _, p, s, other... = split(problem_name, ['{', '}', '-'])
        base_problem = reconstruct_problem(p)
        scaleup = parse(Int, s)
        return MultidimProblem(base_problem, scaleup)
    end
    if startswith(problem_name, "GaussProblem")
        x_dim = parse(Int, split(problem_name, "GaussProblem")[2])
        return GaussProblem(; x_dim)
    end
    if startswith(problem_name, "MeanGauss")
        x_dim = parse(Int, split(problem_name, "MeanGauss")[2])
        return MeanGauss(; x_dim)
    end
    if startswith(problem_name, "RosenbrockProblem")
        x_dim = parse(Int, split(problem_name, "RosenbrockProblem")[2])
        return RosenbrockProblem(; x_dim)
    end
    if startswith(problem_name, "StyblinskiTangProblem")
        x_dim = parse(Int, split(problem_name, "StyblinskiTangProblem")[2])
        return StyblinskiTangProblem(; x_dim)
    end
    if startswith(problem_name, "MichalewiczProblem")
        x_dim = parse(Int, split(problem_name, "MichalewiczProblem")[2])
        return MichalewiczProblem(; x_dim)
    end

    return getfield(Main, Symbol(problem_name))()
end

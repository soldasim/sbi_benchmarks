
function reconstruct_problem(problem_name::AbstractString)
    # Must check wrapper suffixes before d-dimensional checks to avoid parse errors.
    if endswith(problem_name, "_hex")
        base_name = problem_name[1:end-length("_hex")]
        return HexObsProblem(reconstruct_problem(base_name))
    end
    if endswith(problem_name, "_cross")
        base_name = problem_name[1:end-length("_cross")]
        return CrossPolytopeObsProblem(reconstruct_problem(base_name))
    end
    if endswith(problem_name, "_sharp")
        base_name = problem_name[1:end-length("_sharp")]
        return SharpProblem(reconstruct_problem(base_name))
    end

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
    if startswith(problem_name, "AckleyProblem")
        x_dim = parse(Int, split(problem_name, "AckleyProblem")[2])
        return AckleyProblem(; x_dim)
    end
    if startswith(problem_name, "AlpineProblem")
        x_dim = parse(Int, split(problem_name, "AlpineProblem")[2])
        return AlpineProblem(; x_dim)
    end
    if startswith(problem_name, "ExpandedSchafferF6Problem")
        x_dim = parse(Int, split(problem_name, "ExpandedSchafferF6Problem")[2])
        return ExpandedSchafferF6Problem(; x_dim)
    end
    if startswith(problem_name, "ExpandedZakharovProblem")
        x_dim = parse(Int, split(problem_name, "ExpandedZakharovProblem")[2])
        return ExpandedZakharovProblem(; x_dim)
    end
    if startswith(problem_name, "GriewankProblem")
        x_dim = parse(Int, split(problem_name, "GriewankProblem")[2])
        return GriewankProblem(; x_dim)
    end
    if startswith(problem_name, "RastriginProblem")
        x_dim = parse(Int, split(problem_name, "RastriginProblem")[2])
        return RastriginProblem(; x_dim)
    end
    if startswith(problem_name, "SalomonProblem")
        x_dim = parse(Int, split(problem_name, "SalomonProblem")[2])
        return SalomonProblem(; x_dim)
    end
    if startswith(problem_name, "SchwefelProblem")
        x_dim = parse(Int, split(problem_name, "SchwefelProblem")[2])
        return SchwefelProblem(; x_dim)
    end
    if startswith(problem_name, "SphereProblem")
        x_dim = parse(Int, split(problem_name, "SphereProblem")[2])
        return SphereProblem(; x_dim)
    end

    return getfield(Main, Symbol(problem_name))()
end


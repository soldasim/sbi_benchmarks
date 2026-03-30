
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
    
    return getfield(Main, Symbol(problem_name))()
end

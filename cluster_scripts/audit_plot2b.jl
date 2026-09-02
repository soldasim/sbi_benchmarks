using JLD2

const ROOT = pwd()

function list_cross_dirs(base)
    dir = joinpath(ROOT, base)
    isdir(dir) || return String[]
    entries = readdir(dir)
    filter(e -> isdir(joinpath(dir, e)) && endswith(e, "_cross"), entries)
end

# Exclude explicit 5D problems (Group B is 2D cross-polytope only)
is_5d(name) = occursin("Problem5_cross", name)

function scan_config(base, problem, config)
    dir = joinpath(ROOT, base, problem)
    if !isdir(dir)
        return (found=Int[], iters=Dict{Int,Int}(), nan=Dict{Int,Int}(), errors=Dict{Int,String}())
    end
    files = readdir(dir)
    pat = Regex("^" * config * "_(\\d+)_TVmetric\\.jld2\$")
    found = Int[]
    iters = Dict{Int,Int}()
    nan = Dict{Int,Int}()
    errors = Dict{Int,String}()
    for f in files
        m = match(pat, f)
        m === nothing && continue
        idx = parse(Int, m.captures[1])
        push!(found, idx)
        try
            s = load(joinpath(dir, f), "score")
            iters[idx] = length(s)
            nan[idx] = count(isnan, s)
        catch e
            errors[idx] = sprint(showerror, e)
        end
    end
    sort!(found)
    return (found=found, iters=iters, nan=nan, errors=errors)
end

function fmt_row(problem, config, res)
    n = length(res.found)
    ilist = [get(res.iters, i, -1) for i in res.found]
    nan_runs = [i for i in res.found if get(res.nan, i, 0) > 0]
    nan_total = sum(get(res.nan, i, 0) for i in res.found; init=0)
    err_runs = collect(keys(res.errors))
    println("$problem | $config | n=$n/20 | runs=$(res.found) | iters=$ilist | nan_total=$nan_total | nan_runs=$nan_runs | errors=$err_runs")
end

opt_dirs = list_cross_dirs("data-opt-functions")
warp_dirs = list_cross_dirs("data-warpedgp2")

opt_dirs_2d = filter(!is_5d, opt_dirs)
warp_dirs_2d = filter(!is_5d, warp_dirs)

println("=== PROBLEM DIRS FOUND (2D cross, excluding *Problem5_cross) ===")
println("data-opt-functions: n=$(length(opt_dirs_2d))")
println(sort(opt_dirs_2d))
println("data-warpedgp2: n=$(length(warp_dirs_2d))")
println(sort(warp_dirs_2d))

println()
println("=== SCAN: maxvar & nongp (data-opt-functions) ===")
for problem in sort(opt_dirs_2d)
    for config in ["maxvar", "nongp"]
        res = scan_config("data-opt-functions", problem, config)
        fmt_row(problem, config, res)
    end
end

println()
println("=== SCAN: warpedgp-yja-maxvar (data-warpedgp2) ===")
for problem in sort(opt_dirs_2d)
    res = scan_config("data-warpedgp2", problem, "warpedgp-yja-maxvar")
    fmt_row(problem, "warpedgp-yja-maxvar", res)
end

println("DONE_AUDIT_PLOT2B")

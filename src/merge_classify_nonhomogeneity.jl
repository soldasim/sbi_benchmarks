"""
Merge per-(problem, output dimension) H results from `plots/nonhomogeneity/*.jld2`
(produced by `cluster_scripts/script_nonhomogeneity.jl` via `submit_nonhomogeneity.sh`)
into the final `plots/classify_nonhomogeneity.csv`, matching the schema/aggregation
`run_classify_nonhomogeneity` produces in-process (worst-dimension `H`/`p` via `argmax`).

Cross-checks completeness against `cluster_scripts/nonhomogeneity_problems.txt`
(the same "<reconstructible_name> <y_dim>" list the submit script consumes) — reports
any problem with missing dimensions rather than silently aggregating a partial set.

Must be run after include("src/main.jl"); include("src/classify_smoothness.jl")
(for `_sm_display_name`/`reconstruct_problem`).

Usage:
    include("src/merge_classify_nonhomogeneity.jl")
    merge_classify_nonhomogeneity()
"""

using JLD2

function _mnh_expected_dims(list_path = "cluster_scripts/nonhomogeneity_problems.txt")
    expected = Dict{String, Int}()
    for line in eachline(list_path)
        isempty(strip(line)) && continue
        pname, ydim = split(strip(line))
        expected[pname] = parse(Int, ydim)
    end
    return expected
end

function merge_classify_nonhomogeneity(; in_dir = "plots/nonhomogeneity",
                                          out_path = "plots/classify_nonhomogeneity.csv",
                                          list_path = "cluster_scripts/nonhomogeneity_problems.txt")
    expected = _mnh_expected_dims(list_path)

    by_problem = Dict{String, Vector{Any}}()
    files = filter(f -> endswith(f, ".jld2"), readdir(in_dir; join=false))
    for fname in files
        result = load(joinpath(in_dir, fname), "result")
        push!(get!(by_problem, result.problem, []), result)
    end

    missing_problems = String[]
    incomplete = Tuple{String, Vector{Int}}[]
    for (pname, ydim) in expected
        if !haskey(by_problem, pname)
            push!(missing_problems, pname)
            continue
        end
        got_dims = sort([r.dim for r in by_problem[pname]])
        want_dims = collect(1:ydim)
        if got_dims != want_dims
            missing_dims = setdiff(want_dims, got_dims)
            push!(incomplete, (pname, missing_dims))
        end
    end

    if !isempty(missing_problems)
        @warn "Problems with ZERO results found: $(join(missing_problems, ", "))"
    end
    for (pname, dims) in incomplete
        @warn "Problem $pname missing dimensions: $dims"
    end
    complete = isempty(missing_problems) && isempty(incomplete)
    println(complete ? "All $(length(expected)) problems complete (all output dimensions present)." :
                        "INCOMPLETE — see warnings above before trusting the merged CSV.")

    rows = NamedTuple[]
    for (pname, results) in by_problem
        display_name = _sm_display_name(reconstruct_problem(pname))
        H_per_dim = [r.H for r in results]
        p_per_dim = [r.p for r in results]
        j_worst = argmax(replace(H_per_dim, NaN => -Inf))
        r0 = results[1]
        push!(rows, (name=display_name, dx=r0.dx, dy=r0.dy, H=H_per_dim[j_worst], p=p_per_dim[j_worst],
                     n_design=r0.n_design, m_centers=r0.m_centers, k_neighbors=r0.k_neighbors,
                     n_bootstrap=r0.n_bootstrap))
    end
    sort!(rows; by = r -> r.name)

    println()
    for r in rows
        println("$(r.name): H=$(round(r.H,digits=3)) p=$(round(r.p,digits=3)) (dx=$(r.dx), dy=$(r.dy), k=$(r.k_neighbors))")
    end

    mkpath(dirname(out_path))
    open(out_path, "w") do io
        println(io, "problem,dx,dy,H,p,n_design,m_centers,k_neighbors,n_bootstrap")
        for r in rows
            println(io, "$(r.name),$(r.dx),$(r.dy),$(r.H),$(r.p),$(r.n_design),$(r.m_centers),$(r.k_neighbors),$(r.n_bootstrap)")
        end
    end
    println("\nSaved → $out_path  ($(length(rows)) problems merged, complete=$complete)")
    return rows
end

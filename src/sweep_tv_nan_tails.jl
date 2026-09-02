using JLD2

function tail_nan_info(v::Vector{Float64})
    n = length(v)
    n == 0 && return (false, 0, 0)
    if !isnan(v[end])
        return (false, 0, n)
    end
    i = n
    while i >= 1 && isnan(v[i])
        i -= 1
    end
    tail_len = n - i
    return (true, tail_len, n)
end

function sweep_tv_nan_tails(dirs::Vector{String})
    hits = Tuple{String,Int,Int}[]
    errs = Tuple{String,String}[]
    nfiles = 0
    for d in dirs
        isdir(d) || continue
        for (root, _, files) in walkdir(d)
            for fn in files
                endswith(fn, "_TVmetric.jld2") || continue
                path = joinpath(root, fn)
                nfiles += 1
                try
                    score = load(path, "score")
                    has_tail, tail_len, n = tail_nan_info(score)
                    if has_tail && tail_len >= 2
                        push!(hits, (path, tail_len, n))
                    end
                catch e
                    push!(errs, (path, sprint(showerror, e)))
                end
            end
        end
    end
    println("Scanned $(nfiles) TVmetric files across $(length(dirs)) dirs.")
    println("Found $(length(hits)) files with a trailing NaN block (len>=2):")
    for (p, tl, n) in sort(hits, by = x -> -x[2])
        println("  $(p)  tail=$(tl)/$(n)")
    end
    if !isempty(errs)
        println("\n$(length(errs)) files failed to load score at all:")
        for (p, e) in errs
            println("  $(p): $(first(e, 120))")
        end
    end
    return hits, errs
end

const ALL_DATA_DIRS = ["data-bosip","data-bosip-norm","data-convergence4","data-opt-functions","data-warpedgp","data-warpedgp2","data"]
run_full_sweep() = sweep_tv_nan_tails(ALL_DATA_DIRS)

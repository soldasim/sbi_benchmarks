
"""
    rename_files(dir, from, to)

Rename all files in `dir` that start with `from`
such that the prefix matching `from` is replaced with `to`.
"""
function rename_files(dir::String, from::String, to::String)
    dir = "data/ABProblem"
    pattern = joinpath(dir, "nongp_ig2*")
    files = Glob.glob(pattern)
    for file in files
        base = basename(file)
        if startswith(base, "nongp_ig2")
            new_base = replace(base, "nongp_ig2" => "nongp-ig2")
            new_path = joinpath(dir, new_base)
            mv(file, new_path; force=true)
        end
    end
end

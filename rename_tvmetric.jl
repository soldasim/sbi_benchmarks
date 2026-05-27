#!/usr/bin/env julia

using Glob

# Path to data directory
target_dir = joinpath(@__DIR__, "data")

# Find all files recursively
files = glob("**/*", target_dir)

count_renamed = 0

for file_path in files
    # Skip if it's a directory
    isdir(file_path) && continue
    
    # Get the filename without path
    filename = basename(file_path)
    
    # Split by underscore and period
    parts = split(filename, ['_', '.'])
    (parts[end] == "jld2") || continue # Only consider .jld2 files

    if parts[end-1] == "TVmetric-old"
        # Create new filename
        new_filename = replace(filename, "TVmetric-old" => "TVmetric")
        new_file_path = joinpath(dirname(file_path), new_filename)
        
        # Rename the file
        mv(file_path, new_file_path)
        global count_renamed += 1
        println("Renamed: $filename → $new_filename")
    end
end

println("\nTotal files renamed: $count_renamed")

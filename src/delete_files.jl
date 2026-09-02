
function delete_data(dir; suffix=".jld2", recursive=true, dry_run=false)
    """
    Delete all files with a given suffix in a directory.
    
    Args:
        dir: Directory path to search in
        suffix: File suffix to delete (default: ".jld2")
        recursive: Whether to search subdirectories (default: true)
        dry_run: If true, only print what would be deleted without actually deleting (default: false)
    
    Returns:
        Number of files deleted (or would be deleted if dry_run=true)
    """
    if !isdir(dir)
        @warn "Directory does not exist: $dir"
        return 0
    end
    
    deleted_count = 0
    
    # Get all files in directory
    files_to_process = if recursive
        # Recursively find all files
        all_files = String[]
        for (root, dirs, files) in walkdir(dir)
            for file in files
                push!(all_files, joinpath(root, file))
            end
        end
        all_files
    else
        # Only files in the immediate directory
        [joinpath(dir, f) for f in readdir(dir) if isfile(joinpath(dir, f))]
    end
    
    # Filter files by suffix and delete them
    for file_path in files_to_process
        if endswith(file_path, suffix)
            if dry_run
                println("Would delete: $file_path")
            else
                try
                    rm(file_path)
                    println("Deleted: $file_path")
                catch e
                    @warn "Failed to delete $file_path: $e"
                    continue
                end
            end
            deleted_count += 1
        end
    end
    
    if dry_run
        println("Dry run complete. Would delete $deleted_count files with suffix '$suffix'")
    else
        println("Deleted $deleted_count files with suffix '$suffix' from $dir")
    end
    
    return deleted_count
end


## How to run the experiments

- `sh julia.sh` to start Julia properly
- Set up everything `cluster_scripts/queue_jobs.jl` according to `checklist.txt`
- `include("src/generate_starts.jl")` and generate starts
- `include("cluster_scripts/queue_jobs.sh")` and queue jobs

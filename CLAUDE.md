# bosip_benchmarks

## RCI Cluster Rules

All rules for using the RCI cluster (interactive jobs, SLURM, Julia workflow, monitoring) are in the memory file `reference_rci_cluster.md`. Load and follow that file for any cluster-related work.

## Local Julia Daemon Setup (this project)

For lightweight plotting/inspection work, a local Julia daemon (see the general pattern in `reference_rci_cluster.md`) is already set up for this project:

- Local env: `~/.claude/julia-envs/bosip_benchmarks-remote/` — `Project.toml` copied from `~/mnt/rci/repos/bosip_benchmarks/src/Project.toml`. `BOSS.jl`/`BOSIP.jl` are `Pkg.develop`'d against `~/mnt/rci/repos/BOSS.jl` and `~/mnt/rci/repos/BOSIP.jl` (mounted paths), so it always tracks current cluster code, not a stale clone.
- Start the daemon (in a visible Terminal window):
  ```applescript
  osascript -e 'tell application "Terminal"
      activate
      do script "cd ~/mnt/rci/repos/bosip_benchmarks && ~/.juliaup/bin/julia --project=$HOME/.claude/julia-envs/bosip_benchmarks-remote -e '\''using DaemonMode; serve()'\''"
  end tell'
  ```
- Usage:
  ```bash
  julia --startup-file=no -e 'using DaemonMode; runfile("src/plot_bosip_norm_tv.jl")'
  julia --startup-file=no -e 'using DaemonMode; runexpr("using JLD2; load(\"data-bosip-norm/ABProblem/standard_1_TVmetric.jld2\")[\"score\"]")'
  ```
- Scope: inspection/plotting only. Actual experiments (SLURM batch jobs) still go through the cluster's `sbatch`/interactive workflow — the local daemon just reads/writes the same mounted `data*/`/`plots/` directories.

**Inspecting JLD2 data files:** `using JLD2; f = jldopen("path/to/file.jld2"); keys(f)`, then access fields with `f["key"]`. Struct fields may fail to load if the Julia type has changed since the file was written (JLD2 reconstruction error) — in that case, load only the fields you need (primitive arrays like `Vector{Float64}` always load fine). Example: `_TVmetric.jld2` has `"score"` (loadable `Vector{Float64}`) and `"metric"` (a `TVMetric` struct — may fail if the struct changed).

## Experiment Logging

Whenever new experiments are run (new SLURM batch jobs, new run types, new problem sets), record them in the memory file `project_bosip_benchmarks.md` under "Completed experiments". Include the problem set, run types, number of runs, and any relevant notes (e.g. job IDs, time limits, known issues).

## Data Directory Policy

Before submitting any new experiments, think carefully about where the output data should be stored. Consider whether the new runs belong in an existing data directory or warrant a new one (e.g. a change in TV metric normalization, a new experiment type, or a new generation of results that shouldn't be mixed with old data). If unsure, ask before submitting.

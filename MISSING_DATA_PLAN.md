# Missing Data Plan — Paper 1 Revision

1. Fix numerical issues on `loglike-imiqr` runs on SIR problem.
2. Fix numerical issues with warped GP on SIR, Duffing, and Diffusion problems.
3. Fix numerical issues with nonstationary GP (`nongp`) on Simple and SIR problems (crash-dominated, not
   timeout-bound like the other Group A problems — see `project_bosip_benchmarks.md` for the full
   timeout-vs-crash breakdown per problem).
4. Extend the few missing iterations for EIV and IMMD on Duffing/Diffusion problems (Group A) to reach the
   200-iter target — no numerical issue here, just more compute needed:
   - EIV: DuffingProblem (191–194/201) and DiffusionProblem10 (70–83/201) — both genuine timeouts, continue
     from checkpoint.
   - IMMD: DuffingProblem and DiffusionProblem10 (both only 101/201) — not a timeout or crash, just submitted
     with `iters=100` instead of 200; resubmit with `continue=1, iters=100` to add the missing 100.

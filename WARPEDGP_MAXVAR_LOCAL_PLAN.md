# WarpedGP + MaxVar local runs (Mac, single CPU)

Goal: exercise the new `predictive_samples`-based numerics (local uncommitted BOSS.jl/BOSIP.jl
changes) on the 7 original BIP benchmark problems, using `warpedgp-yja-maxvar`
(WarpedGaussianProcess + LogMaxVar), run sequentially since this Mac has one CPU available.

Local repo: `~/Documents/phd-sandbox/bosip_benchmarks` (branch `dev`, working tree synced from
the RCI `experiments-review` clone via rsync — NOT a git merge, just a working-tree copy).
BOSS.jl/BOSIP.jl: `~/Documents/julia-pkg/{BOSS.jl,BOSIP.jl}` (local dev clones, uncommitted
changes, used via `Pkg.develop` in `~/.claude/julia-envs/bosip_benchmarks-local`).

**No commits or edits to BOSS.jl/BOSIP.jl.** Any bug found there is fixed only if trivial and
reported in the morning, not committed.

## Problems (7 original BIP)

AB, Simple, Banana, Bimodal, SIR, Duffing, Diffusion10
(`ABProblem`, `SimpleProblem`, `BananaProblem`, `BimodalProblem`, `SIRProblem`,
`DuffingProblem`, `DiffusionProblem10`)

## Plan

- **Phase 0 — setup**: local bosip_benchmarks clone synced from RCI; local Julia env
  dev-linked to local BOSS.jl/BOSIP.jl; resolve where `starts/`+`grid/` data live for these
  problems (needed for `main_warpedgp-yja-maxvar.jl`'s initial data + TV metric).
- **Phase 1 — smoke test**: 1 run (run_idx=1) per problem, 100 iters, `warpedgp-yja-maxvar`.
  Verify no crashes, sane TV metric output.
- **Phase 2 — if phase 1 clean**: extend to runs 2-5 per problem (100 iters each).
- **Phase 3 — if phase 2 finishes with time to spare**: extend Duffing and Diffusion10 runs
  1-5 to 200 iters (continue=1).

Output data dir: `data-warpedgp2/<ProblemName>/` (per `main_warpedgp-yja-maxvar.jl`), file
prefix `warpedgp-yja-maxvar_<idx>_*`.

## Status

| Phase | Status |
|---|---|
| 0 — setup | IN PROGRESS |
| 1 — 1 run x 7 problems | DONE — all 7 clean, 0 NaN (see below). Physical problems (SIR/Duffing/Diffusion10) previously had severe NaN contamination under the old code; all clean now. |
| 2 — 5 runs x 7 problems | IN PROGRESS (runs 2-5, run 1 already done above) |
| 3 — Duffing/Diffusion10 -> 200 iters | NOT STARTED |

## Per-problem run log (Phase 1)

| Problem | run 1 | notes |
|---|---|---|
| ABProblem | runs 1-5: ALL DONE, 101/101, 0 NaN each | analytical, clean; final scores 4.3e-4 to 6.9e-4 |
| SimpleProblem | runs 1-5: ALL DONE, 101/101, 0 NaN each | analytical, clean; final scores 4.0e-5 to 8.3e-5 |
| BananaProblem | runs 1-5: ALL DONE, 101/101, 0 NaN each | analytical, clean; final scores 3.8e-4 to 7.3e-4 |
| BimodalProblem | runs 1-5: ALL DONE, 101/101, 0 NaN each | analytical, clean; final scores 2.7e-4 to 3.4e-4 |
| SIRProblem | run1: clean; run2: CRASHED (Binomial predictive_samples assertion); run3: clean; run4: CRASHED (same assertion); run5: STOPPED BY USER REQUEST mid-run | **fix confirmed for the crashes it targeted** (old code had 0-75 NaN/101 here), but surfaced a separate real bug in BinomialLikelihood's new SampledPredictive path — see Bugs found |
| DuffingProblem | run1: 101/101, 0 NaN, DONE. run2: STOPPED BY USER, partial data deleted | **fix confirmed**: old code had 12-89 NaN/101 on this problem (worst case). TV score last=0.238, higher than other problems — may benefit from the 200-iter extension (Phase 3) |
| DiffusionProblem10 | run1: 101/101, 0 NaN, DONE | old code had 0-62 NaN/101 on this problem. TV score last=0.170 |

## BinomialLikelihood [0,1]-truncation crash — FIXED and validated (2026-07-22)

Root cause (see "Bugs found" below): `_binomial_atom_log_mean`'s reject-and-renormalize on
`WarpedGaussianProcess`'s untruncated Gauss-Hermite atoms could hit "all 21 atoms outside [0,1]"
and crash. Implemented the quantile/CDF-reparametrization fix designed earlier in this session:
`WarpedGaussianProcessPosterior`'s `predictive_samples`/`_predictive_points` now build a dedicated
Gauss-Legendre-based truncated quadrature when `lower`/`upper` are given (guaranteed in-range
atoms, no rejection, no empty-set crash) instead of the old reject-and-renormalize. Wired through
`BOSIP.jl`'s `binomial.jl` (`lower=0., upper=1.` now actually passed). Also fixed a related
space-mismatch bug in `TransformedModel`'s bound-forwarding (was applying outer-space bounds to
base-model-space atoms under a non-identity `OutputTransform`).

**Validated:**
- Unit tests: atoms always in `[0,1]`, weights sum to 1, matches unbounded computation to ~12
  significant digits when truncation is inactive (no regression for the common case).
- Re-ran SIRProblem runs 2 and 4 (the exact two that crashed in Phase 2) — both now complete
  cleanly: 101/101 iterations, 0 NaN.

Not committed to git (per standing rule — always ask first).

## Bugs found (report in the morning, not committed)

1. **SIRProblem run 2 crashed** (`AssertionError: All predictive_samples atoms fell outside the
   [0,1] Binomial success-probability domain; cannot truncate.`,
   `BOSIP.jl/src/likelihoods/binomial.jl:132`, `_binomial_atom_log_mean`). SIRProblem's
   likelihood is actually `BinomialLikelihood` (not `NormalLikelihood` as assumed earlier in
   conversation). This is in the NEW `SampledPredictive` atom-based code path added by the
   `predictive_samples` migration — when every quadrature atom for a given evaluation happens to
   land outside [0,1], the rejection-based renormalization has nothing left and asserts instead
   of degrading gracefully (e.g. falling back to a tiny/floor probability). Did NOT attempt a
   fix — flagged for review, not touched. Full log:
   `logs_local/SIRProblem_warpedgp-yja-maxvar_2.log` (search "ERROR: LoadError: AssertionError").
   Driver continued automatically to run 3 (each run_idx is independent).
   **Same crash also hit SIRProblem run 4** (`logs_local/SIRProblem_warpedgp-yja-maxvar_4.log`) —
   reproducible, not a one-off fluke: 2 of 4 SIRProblem runs so far (2, 4) hit this assertion.

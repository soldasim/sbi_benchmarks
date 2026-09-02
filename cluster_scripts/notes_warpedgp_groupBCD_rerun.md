# Notes: WarpedGP Groups B/C/D rerun after predictive_samples fix (2026-07-24)

## Context / plan

Following the Group A rerun (7 orig BIP problems, done 2026-07-23, see
`project_bosip_benchmarks.md`), rerunning `warpedgp-yja-maxvar` for the remaining
30 problems (Groups B/C/D) now that BOSIP.jl/BOSS.jl's `predictive_samples` fix
(commit chain ending ~2026-07-23) resolves the NaN-contamination bug in
`WarpedGaussianProcess`'s likelihood-mean/variance computation.

Design decision (confirmed with user 2026-07-24): use **non-proxy** problem
variants everywhere (`SIRProblem` not `ProxySIRProblem`, `BealeProblem_cross`/
`GoldsteinPriceProblem_cross` not the Proxy variants) — deliberate, so surrogate
comparisons are made against genuinely bad response surfaces. See
`project_bosip_paper_plan.md` "Design rationale" section.

**Groups in scope** (30 problems total, 20 runs each = 600 runs):
- **Group B** (24, 2D cross-polytope opt-function): `data-warpedgp2/<name>_cross/`,
  `cpu` partition, 200 iters, mem 12G.
- **Group C** (2, HD BIP: DuffingProblem5, DiffusionProblem5D):
  `data-warpedgp2/<name>/`, `cpulong`, 200 iters, mem 12G.
- **Group D** (4, HD cross-opt: Rosenbrock5/StyblinskiTang5/Michalewicz5/Sphere5,
  all `_cross`): `data-warpedgp2/<name>_cross/`, `cpulong`, 200 iters, mem 16G.

## Archiving — already done, no action needed

Confirmed 2026-07-24: `data-warpedgp2_archive_pre-samplesfix/` already contains
all 30 B/C/D problem dirs (42 total dirs incl. the 7 Group A + non-cross pilot
problems out of scope). This happened as a side effect of Group A's rerun doing
a **full-directory** rename (`data-warpedgp2/` → `..._archive_pre-samplesfix/`)
rather than a per-problem move — it swept up everything, not just Group A. Fresh
`data-warpedgp2/` only had the 7 rerun Group A problems in it before this
session's submissions started. No separate archive step was needed for B/C/D.

## Submit scripts written this session

- `cluster_scripts/submit_groupB_warpedgp_samplesfix_rerun.sh` — Group B, 24
  problems × runs 1-20 (except RosenbrockProblem2_cross starts at run 2, since
  run 1 was the debug/test job). `cpu`, mem 12G, 200 iters.
- `cluster_scripts/submit_groupCD_warpedgp_samplesfix_rerun.sh` — Groups C+D, 6
  problems × runs 1-20 (except DuffingProblem5 and RosenbrockProblem5_cross
  start at run 2, same reason). `cpulong`, mem 12G (C) / 16G (D).

## Debug/test jobs (submitted first, verified clean before bulk)

- Group B test: `RosenbrockProblem2_cross` run 1 = job **11202555** (`cpu`)
- Group C test: `DuffingProblem5` run 1 = job **11202556** (`cpulong`)
- Group D test: `RosenbrockProblem5_cross` run 1 = job **11202557** (`cpulong`)

All 3 verified running clean (no exceptions, normal BO loop progress) before
bulk submission — per cluster's debug-before-bulk rule.

## Bulk submission — hit the QOSMaxSubmitJobPerUserLimit cap (500/partition-QOS)

- Group C+D bulk (118 jobs) submitted cleanly in one shot — job IDs
  11203044-11203161. `cpulong` had plenty of headroom (40/500 pre-existing).
- Group B bulk: `cpu` partition QOS capped at 500 total pending+running per
  user. 79 pre-existing (unrelated) jobs were already on `cpu`, so only 421 of
  Group B's 479 submissions (script attempted 479, test job separately covers
  Rosenbrock run 1) succeeded before hitting the cap — confirmed via
  diff of expected vs actually-queued job names: **all 20 runs of
  `ThreeHumpCamelProblem_cross` were the clean single gap** (last problem in
  the array, cut off mid-list). Job IDs 11202585-11203043 (the successful ones).
- Waited ~2h for the 79 pre-existing jobs (already ~19.5h into their 24h limit)
  to finish and free `cpu` slots. Once `cpu` dropped to 436/500, submitted the
  missing 20 `ThreeHumpCamelProblem_cross` runs — job IDs **11204648-11204673**.
  Confirmed via disk+queue audit (see below) that nothing else was silently
  skipped.

**Lesson for next time**: when the `cpu`/`cpulong` queue is already partially
occupied by unrelated jobs, a big bulk submission can partially fail mid-script
with no clean signal beyond scattered `sbatch: error: QOSMaxSubmitJobPerUserLimit`
lines in stdout — the script's own "N jobs submitted" count is **not** reliable
evidence of success (it increments even on failed `sbatch` calls, since `jid`
just ends up empty). Always cross-check actual queued job names against the
expected list after a large submission.

## Full-coverage audit method (repeatable)

For each of the 30 problems × 20 run_idx, a run counts as "accounted for" if
EITHER:
- a job with the expected name (or the known test-job name for the 3 debug
  runs) is present in `squeue -u soldasim -o "%j"`, OR
- `data-warpedgp2/<problem>/warpedgp-yja-maxvar_<idx>_TVmetric.jld2` exists on
  disk (sshfs mount).

Ran this full audit 2026-07-24 ~16:xx: **600/600 accounted for, 0 missing.**
Re-run this same check periodically during monitoring — a run could still fail
silently later (crash after producing a partial file, etc.) and needs to be
caught by then checking iteration counts/NaN content, not just file existence.

## Status as of 2026-07-24 ~16:50

- `sacct` since batch start: **223 COMPLETED** (all exit `0:0`, elapsed
  ~2.5-3.5h — consistent with genuine 200-iteration Group B runs, not early
  crashes), **403 RUNNING**, **122 PENDING**, **0 FAILED/CANCELLED/TIMEOUT**.
- Queue: `cpu` 330 (down from the 500 cap), `cpulong` 191.
- Full 600/600 run audit: clean, nothing missing.
- **Not yet done**: NaN-count comparison (before/after) for any of the 30
  problems — blocked on getting a live interactive Julia session (the shared
  `claude-interactive` tmux job needed refreshing — see below).

## claude-interactive (cluster) session note — abandoned in favor of local Julia

Found the shared cluster session mid-use by another concurrent session (checking
`ProxySIRProblem` nongp/eiv gaps — unrelated nongp-expansion work, see
`project_bosip_benchmarks_nongp_expansion.md`). Its SLURM job had ~12 min left
on the `cpufast` 4h limit; exited cleanly (idle at the time) and attempted a
fresh `srun`, which itself hit `QOSMaxSubmitJobPerUserLimit`. **User pointed out
the queue would be too full for this to be reliable — switched to the local
`julia-bosip-benchmarks` tmux REPL instead** (runs against the sshfs mount, no
SLURM job needed for read-only JLD2 inspection). This session was also being
shared concurrently with the other nongp-expansion session — fine per
[[reference-interactive-julia]] (no SLURM cost locally), just wait for `julia>`
idle before sending commands.

## NaN-count check — first pass (2026-07-24 ~17:00), local Julia

Script: `cluster_scripts/tmp_nan_check_groupBCD.jl` (checks all 30 problems'
`data-warpedgp2/<p>/warpedgp-yja-maxvar_<i>_TVmetric.jld2` files, wrapped in
try/catch since some files are actively being written by still-running jobs).

**Result: 0 NaN found across every single one of the 30 problems checked so
far** (Group B all 24 problems: 0 NaN; Group C both problems: 0 NaN so far).
Many runs are still in-progress (haven't reached 201 iters yet — e.g.
`ThreeHumpCamelProblem_cross` at only ~43/201 avg since it started latest,
Group D's `StyblinskiTangProblem5_cross`/`MichalewiczProblem5_cross`/
`SphereProblem5_cross` at 0/20 found yet, `RosenbrockProblem5_cross` at 7/20
found), so this isn't the final picture, but zero NaN in every completed and
in-progress run so far is a strong clean signal, consistent with Group A.

One transient read error: `HolderTableProblem_cross` run 10 hit
`InvalidDataException: Invalid Object header signature` on both attempts.
Checked and confirmed **benign** — the job (`HolderTableProblem_cross_wgp_10`)
is still `RUNNING` and the file's mtime matched the exact second of the read
attempt, i.e. we were reading mid-write. Not real corruption; will read fine
once the job finishes or between write bursts. General lesson: any NaN-check
script touching `data-warpedgp2/` while jobs are still running must wrap
`load(...)` in try/catch and treat failures as "currently being written",
not a real error — don't crash the whole audit over one transient race.

## Before/after comparison vs archived pre-fix data (2026-07-24 ~17:05)

Ran `cluster_scripts/tmp_nan_check_archive.jl` against
`data-warpedgp2_archive_pre-samplesfix/` for the problems known/suspected to
have been NaN-heavy pre-fix:

| Problem | Pre-fix NaN (out of up to 4020) | Post-fix NaN so far |
|---|---|---|
| DuffingProblem5 | 2909 (72%) | 0 (out of 1696 valid so far, ~42% through) |
| DiffusionProblem5D | 3470 (86%) | 0 (out of 702 valid so far, ~17% through) |
| RosenbrockProblem5_cross | 55 (1.4%, mild) | 0 (out of 141 so far, too early) |
| StyblinskiTangProblem5_cross | 0 (already clean) | not started yet |
| MichalewiczProblem5_cross | 0 (already clean) | not started yet |
| SphereProblem5_cross | 0 (already clean) | not started yet |

**Correction to prior characterization**: earlier notes said "Groups C/D were
the *most* NaN-contaminated of all" — this overstates Group D. Only **Group C**
(DuffingProblem5 72%, DiffusionProblem5D 86%) was catastrophically bad
pre-fix. Group D was already mostly clean; only Rosenbrock5_cross had any NaN
at all, and only mildly (1.4%). Updated `project_bosip_paper_plan.md` to
reflect this. The predictive_samples fix's biggest win in this batch is
clearly on Group C, matching the Group A pattern (fix matters most for
problems needing strongly nonlinear Yeo-Johnson warps — physical
simulators with huge dynamic range).

## Status as of 2026-07-24 ~18:04 (second check-in)

- `sacct`: **323 COMPLETED** (up from 223; all exit `0:0`, zero non-clean
  exits), **362 RUNNING**, **158 PENDING**, **0 FAILED/CANCELLED/TIMEOUT**.
- Queue: `cpu` 256 running + 54 pending; `cpulong` 106 running + 104 pending.
  `cpulong` is heavily contended — partly our own Group C/D jobs, partly a
  concurrent unrelated session's ~160-job "with-proxy run completion" batch
  (ProxySIRProblem/BealeProxyProblem_cross/GoldsteinPriceProxyProblem_cross,
  see `project_bosip_paper_plan.md` "with-proxy" section — different task,
  different problems, sharing the same partition/QOS).
- Full 600-run audit re-run: still 600/600, nothing missing.
- NaN check re-run (local Julia, `julia-bosip-benchmarks` tmux): **still 0 NaN
  across all 30 problems**, progress advancing on valid-iteration counts
  (e.g. `ThreeHumpCamelProblem_cross` 871→2023, `DuffingProblem5` 1696→1909,
  `DiffusionProblem5D` 702→800). The earlier `HolderTableProblem_cross` run 10
  transient read error is gone this pass (file now readable) — confirms it
  really was just a mid-write race, not corruption.
- Checked why `StyblinskiTangProblem5_cross`/`MichalewiczProblem5_cross`/
  `SphereProblem5_cross` still show 0/20 files and `RosenbrockProblem5_cross`
  is stuck at 7/20: all remaining runs for these 4 problems are `PENDING`
  (19-20 each), not stalled/failed — `cpulong` contention (see above) is
  simply delaying their start. Nothing to fix, just resource contention.

## Next steps

1. Re-run the NaN-count check periodically (script is saved at
   `cluster_scripts/tmp_nan_check_groupBCD.jl`, archive comparison at
   `cluster_scripts/tmp_nan_check_archive.jl` — both reusable, just re-`include`
   in the local `julia-bosip-benchmarks` tmux session) as more runs complete.
2. Continue hourly sacct health checks; watch for any FAILED/CANCELLED jobs.
3. Re-run the full 600-run audit periodically, not just once.
4. Once all (or most) jobs finish, produce the final before/after NaN-count
   table for all 30 problems (this pass only checked the 6 previously-worst
   ones in detail against the archive; extend to all 30 once more data is in).
5. Push-notify user when the full batch is done or if anything needs attention.
6. Group D's 4 problems (esp. StyblinskiTang5/Michalewicz5/Sphere5_cross) will
   likely take longer than initially expected due to `cpulong` contention from
   the concurrent with-proxy session — factor this into ETA expectations, not
   a sign of a problem with our batch specifically.

## Status as of 2026-07-24 ~19:07 (third check-in)

- `sacct` (whole account): 383 COMPLETED, 304 RUNNING, 116 PENDING, **40
  TIMEOUT** — investigated, all 40 are `DuffingProblem_..._cont200` jobs from
  an unrelated 2026-07-23 task ("extend to 200 iters" continuation work,
  already-known-risk `nongp`/`immd` continuations), confirmed zero overlap
  with our `_wgp_`/`warpedgp-yja` job names. False alarm for this task.
- Our batch specifically (`_wgp_`/`warpedgp-yja` job names only): **352
  COMPLETED** (0:0), **194 RUNNING**, **94 PENDING**, 0 failures/timeouts.
- Full 600-run audit: still 600/600, nothing missing.
- NaN check: **still 0 NaN everywhere.** 17 of 24 Group B problems now fully
  complete (4020/4020 valid, i.e. all 20 runs hit the full 201 iters): Rosenbrock2,
  StyblinskiTang2, Michalewicz2, Ackley2, Alpine2, ExpandedSchafferF6_2,
  Griewank2, Rastrigin2, Salomon2, Schwefel2, Sphere2, Beale, Booth,
  CrossInTray, DropWave, Easom, HolderTable. Remaining 7 (ExpandedZakharov2,
  GoldsteinPrice, Himmelblau — nearly done at 4010/4020, LeviN13, Matyas,
  SchafferN2, ThreeHumpCamel) still finishing, all 0 NaN so far.
  Group C: DuffingProblem5 2089/4020 (~52%), DiffusionProblem5D 884/4020
  (~22%), both 0 NaN. Group D: RosenbrockProblem5_cross still only 7/20 files
  (barely progressing — 1 job running, rest pending); other 3 Group D
  problems still 0/20 files (all pending) — matches the known `cpulong`
  contention noted in the previous check-in, not a new issue.

## Status as of 2026-07-24 ~20:10 (fourth check-in) — milestone: all 30/30 problems now have 20/20 files

- Our batch (`_wgp_`/`warpedgp-yja` job names only): **416 COMPLETED** (0:0),
  **205 RUNNING**, **19 PENDING**, 0 failures/timeouts/cancellations.
- Full 600-run audit: still 600/600, nothing missing.
- **NaN check: 0 NaN across every single one of the 30 problems, all 20 runs
  each — no exceptions.** All of Group B (24/24) fully complete at 4020/4020
  except Matyas (2920/4020) and ThreeHumpCamel (3312/4020), both still
  progressing cleanly. Group C both fully underway (DuffingProblem5 2254/4020
  ~56%, DiffusionProblem5D 903/4020 ~22%). **Group D's 3 previously-blocked
  problems (StyblinskiTang5/Michalewicz5/Sphere5_cross) finally started** —
  all now 20/20 files present (were 0/20 last check), early in their runs
  (60-103 valid out of up to 4020). RosenbrockProblem5_cross also now 20/20
  files (was 7/20), still early (152/4020).
- Practical implication: file-coverage-wise the whole 600-run batch is now
  structurally complete (every (problem, run_idx) has started producing
  output) — remaining work is purely runs accumulating more iterations
  towards the 201 target, no more queue-blocked gaps.

## Status as of 2026-07-24 ~21:12 (fifth check-in)

- Our batch: **443 COMPLETED** (0:0), **178 RUNNING**, **19 PENDING**, 0
  failures/timeouts. Full 600-run audit: still 600/600.
- NaN check: **still 0 NaN everywhere.** Group B now 22/24 problems fully at
  4020/4020 (only Matyas 3305/4020 and ThreeHumpCamel 3792/4020 remain, both
  close). Group C: DuffingProblem5 2403/4020 (~60%), DiffusionProblem5D
  1087/4020 (~27%). Group D: all 4 problems now clearly progressing (were
  barely started last check) — RosenbrockProblem5_cross 653, StyblinskiTang5
  790, Michalewicz5 920, Sphere5 601 (out of 4020 each) — roughly 5-10x jump
  in an hour, consistent with jobs now actually running instead of pending.
- Not yet fully done (Group C/D still well under 50%) — continuing to
  monitor, not notifying yet per the "wait until plateaued or fully done"
  instruction.

## Status as of 2026-07-25 ~22:xx (sixth check-in, date rolled over)

- Our batch: **555 COMPLETED** (0:0), **85 RUNNING**, **0 PENDING**, 0
  failures/timeouts. Full 600-run audit: still 600/600.
- One transient sshfs mount hiccup during the NaN check (`IOError: i/o error
  (EIO)` on a plain `isfile()` stat call, not a JLD2 read) — retried a few
  seconds later and it resolved on its own. Not a data issue, just a mount
  blip; same "transient, don't panic, just retry" pattern as the earlier
  `HolderTableProblem_cross` race.
- **NaN check: Group B is now 100% complete — all 24/24 problems at
  4020/4020 valid, 0 NaN.** Group C: DuffingProblem5 3754/4020 (~93%),
  DiffusionProblem5D 2147/4020 (~53%). Group D: RosenbrockProblem5_cross
  3112/4020 (~77%), StyblinskiTangProblem5_cross 3186 (~79%),
  MichalewiczProblem5_cross 3850 (~96%), SphereProblem5_cross 1978 (~49%,
  the current laggard). **Zero NaN across every problem, still.**
- Not fully done yet (85 jobs still RUNNING) — continuing to monitor. Very
  close: at current pace expect Group C/D to finish within the next 1-2
  check-ins.

## Status as of 2026-07-27 (final check-in) — RERUN EFFECTIVELY COMPLETE

- Our batch: **606 COMPLETED** (0:0), **34 RUNNING**, **0 PENDING**, 0
  failures across the entire multi-day run. Full 600-run audit: 600/600,
  nothing missing, at every single check throughout.
- **Group B: 24/24 problems fully complete at 4020/4020, 0 NaN.**
- **Group C/D: 4 of 6 problems fully complete at 4020/4020** (DuffingProblem5,
  RosenbrockProblem5_cross, StyblinskiTangProblem5_cross,
  MichalewiczProblem5_cross). The remaining 2 (`DiffusionProblem5D` at
  3958/4020 = 98.5%, `SphereProblem5_cross` at 3918/4020 = 97.5%) still have
  jobs `RUNNING`, approaching the `cpulong` 3-day (72h) wall-time limit
  (elapsed ~68.7h and ~61.7h respectively as of this check). Per the
  project's iteration policy (`project_bosip_paper_plan.md`: "if a run times
  out before reaching its target, accept whatever is on disk, do not
  continue or resubmit"), whatever these finish at is final — no action
  needed regardless of how they land. **Zero NaN in either, at any point.**

### Full before/after NaN comparison — ALL 30 PROBLEMS (2026-07-27)

Ran a complete pre-fix audit (`cluster_scripts/tmp_nan_check_archive_full.jl`)
against `data-warpedgp2_archive_pre-samplesfix/` for every one of the 30
problems, not just the 6 checked earlier — **this substantially expands what
was previously known**: the paper-plan's old partition/time-limit table only
characterized Group A/C/D's NaN status in detail; Group B's pre-fix NaN levels
had never actually been audited before this rerun. Turns out Group B was far
more contaminated pre-fix than assumed:

| Problem | Pre-fix NaN % | Post-fix NaN % |
|---|---|---|
| AckleyProblem2_cross | 37.4% | **0%** |
| ExpandedSchafferF6Problem2_cross | 35.7% | **0%** |
| CrossInTrayProblem_cross | 35.5% | **0%** |
| EasomProblem_cross | 25.7% | **0%** |
| DropWaveProblem_cross | 25.2% | **0%** |
| LeviN13Problem_cross | 25.1% | **0%** |
| RosenbrockProblem2_cross | 18.3% | **0%** |
| SchafferN2Problem_cross | 17.3% | **0%** |
| SalomonProblem2_cross | 16.2% | **0%** |
| BealeProblem_cross | 11.0% | **0%** |
| GoldsteinPriceProblem_cross | 3.2% | **0%** |
| AlpineProblem2_cross | 0.2% | **0%** |
| HolderTableProblem_cross | 0.1% | **0%** |
| (11 more Group B problems) | 0.0% (already clean) | **0%** |
| **DiffusionProblem5D** | **86.3%** | **0%** (worst pre-fix case in the whole set) |
| **DuffingProblem5** | **72.4%** | **0%** |
| RosenbrockProblem5_cross | 1.4% | 0% |
| StyblinskiTangProblem5_cross / MichalewiczProblem5_cross / SphereProblem5_cross | 0.0% | 0% |

**Every single problem that had any pre-fix NaN contamination — mild or
catastrophic — is now at exactly 0%.** This is the strongest possible
confirmation that the `predictive_samples` fix (see
`project_bosip_benchmarks.md`'s WarpedGP rerun section, and
`project_bosip_warpedgp_moments_audit.md` for the root-cause mechanism)
generalizes cleanly across all problem types tested: 2D and 5D, BIP and
opt-function, hex/cross observation structures, mild and severe pre-fix
contamination alike.

## Outcome / handoff

This rerun is done in every way that matters for the paper. Data is ready to
feed:
- Plot 2b (surrogate model comparison, Group B) — now genuinely clean data,
  unlike before.
- Plot 5b (quadrant scatter) for the full A-D coverage — Group B/C/D
  WarpedGP data should be re-pulled into `warpedgp_scores_final.csv` /
  `warpedgp_scores_fair.csv` computations (`src/compute_acq_scores_final.jl`
  / `compute_fair_scores.jl`) since the underlying TVmetric files changed.
- Any Group C/D-specific WarpedGP figures.

**Not yet done** (left for a future session, not blocking this task's
completion): regenerating the actual plots/CSVs that consume this data —
`plot_warpedgp_all.jl`, `compute_acq_scores_final.jl`, and any classification
plots that reference the old (pre-fix) Group B/C/D WarpedGP TV scores need a
rerun now that the underlying data changed. Flag this to whoever picks up the
paper-plan work next.

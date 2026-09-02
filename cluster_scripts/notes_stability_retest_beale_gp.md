# Numerical Stability Retest: Beale/GoldsteinPrice cross-polytope (2026-07-24)

## Background / Hypothesis

Prior NaN audit (2026-07-24) found GP-variance-underflow NaN contamination in `maxvar` and `eiv`
run data for `BealeProblem_cross` and `GoldsteinPriceProblem_cross`:

- `maxvar`, BealeProblem_cross: runs 9,13,16,17 have NaN (52,68,97,99 respectively out of up to 201).
  Runs 1,2,3,4,5,13,16,17 also truncated at 101 instead of the full 201.
- `maxvar`, GoldsteinPriceProblem_cross: runs 7,10,16,19 have NaN (41,5,25,2 respectively).
  Runs 1,2,3,4,5 separately truncated at 101 (no NaN overlap with the NaN set).
- `eiv`, BealeProblem_cross: 119 total NaN entries, per-run breakdown unknown until this retest.
- `eiv`, GoldsteinPriceProblem_cross: 45 total NaN entries, per-run breakdown unknown until this retest.

Hypothesis: recent updates to BOSS.jl/BOSIP.jl (dev-linked, path-dependent) may have improved
numerical stability (GP variance underflow handling) since these runs were originally computed.
Plan: re-run one test job per (problem, config) combo to check whether the NaN/truncation issue
is now fixed, BEFORE committing to a full rerun of all affected runs.

## Step 1 — eiv per-run NaN breakdown (found via local tmux Julia REPL, `julia-bosip-benchmarks`)

Scanned `eiv_<idx>_TVmetric.jld2`, idx=1..20, both problems, via `JLD2.load(path, "score")`,
`count(isnan, score)`, `findall(isnan, score)`. Totals matched the known audit totals exactly
(Beale=119, GoldsteinPrice=45), confirming the scan is correct.

### BealeProblem_cross, eiv (total NaN = 119)

| run_idx | len | nan | nan indices |
|---|---|---|---|
| 5  | 101 | 3  | 98, 99, 100 |
| 6  | 101 | 51 | 51-101 (contiguous tail) |
| 8  | 101 | 8  | 84, 88, 90, 91, 92, 94, 97, 99 |
| 9  | 101 | 8  | 93, 94, 95, 96, 97, 98, 99, 100 |
| 14 | 101 | 4  | 68, 75, 93, 98 |
| 17 | 101 | 45 | 20-61 (mostly contiguous) + 64, 69, 76 |

All other runs (1-4, 7, 10-13, 15, 16, 18-20): 0 NaN.

### GoldsteinPriceProblem_cross, eiv (total NaN = 45)

| run_idx | len | nan | nan indices |
|---|---|---|---|
| 9  | 101 | 4  | 96, 97, 99, 100 |
| 14 | 101 | 24 | 54-77 (contiguous) |
| 18 | 101 | 16 | 86-101 (contiguous tail) |
| 19 | 101 | 1  | 100 |

All other runs (1-8, 10-13, 15-17, 20): 0 NaN.

Note: all eiv runs across both problems that have any data are truncated at len=101 (not 201) —
same truncation-at-101 symptom seen in the maxvar NaN runs.

## Step 2 — Chosen test combos (4 total)

| Problem | Config | run_idx | Rationale |
|---|---|---|---|
| BealeProblem_cross | maxvar | 17 | NaN=99, also truncated at 101 — both failure symptoms together |
| GoldsteinPriceProblem_cross | maxvar | 7 | NaN=41, worst NaN count in that problem's maxvar NaN set |
| BealeProblem_cross | eiv | 6 | NaN=51, highest of the 6 affected eiv runs for Beale |
| GoldsteinPriceProblem_cross | eiv | 14 | NaN=24, highest of the 4 affected eiv runs for GoldsteinPrice |

## Step 3 — Archive locations (verified clean before job submission)

Old (potentially NaN-contaminated / truncated) data for the 4 chosen combos moved to:

- `~/mnt/rci/repos/bosip_benchmarks/data-opt-functions/BealeProblem_cross/archive_stability_retest_2026-07-24/`
  contains: `maxvar_17_{TVmetric,convergence,data,extras,problem}.jld2`,
  `eiv_6_{TVmetric,convergence,data,extras,problem}.jld2`
- `~/mnt/rci/repos/bosip_benchmarks/data-opt-functions/GoldsteinPriceProblem_cross/archive_stability_retest_2026-07-24/`
  contains: `maxvar_7_{TVmetric,convergence,data,extras,problem}.jld2`,
  `eiv_14_{TVmetric,convergence,data,extras,problem}.jld2`

Verified: originals no longer present at the base problem-directory paths; archived copies present
at the above paths, 5 files each, before any job was submitted.

## Step 5 — Submitted jobs

Invocation pattern (matched verbatim from `cluster_scripts/submit_cross2d_maxvar_6to20.sh` /
`submit_cross2d_eiv_remaining6.sh`):
`sbatch --parsable -p <partition> --mem=<mem> --job-name="<job_name>" cluster_scripts/run.sh "<problem>" <config> "<run_idx>" 0 200 nothing`
(args: problem name, run_name/config, run_idx, continue=0, iters=200, noise=nothing)

Partitions: maxvar -> `cpu` (1 day, --mem=12G), eiv -> `cpulong` (3 days, --mem=16G).

Submitted 2026-07-24:

| Job ID | Problem | Config | run_idx | Partition |
|---|---|---|---|---|
| 11205160 | BealeProblem_cross | maxvar | 17 | cpu |
| 11205161 | GoldsteinPriceProblem_cross | maxvar | 7 | cpu |
| 11205162 | BealeProblem_cross | eiv | 6 | cpulong |
| 11205163 | GoldsteinPriceProblem_cross | eiv | 14 | cpulong |

## Step 6 — Early health check

First check, 2026-07-24 ~16:20 (jobs submitted ~16:06-16:17):

`squeue -j 11205160,11205161,11205162,11205163` output:
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          11205161       cpu Goldstei soldasim PD       0:00      1 (QOSMaxCpuPerUserLimit)
          11205160       cpu BealePro soldasim PD       0:00      1 (QOSMaxCpuPerUserLimit)
          11205162   cpulong BealePro soldasim  R       2:42      1 n20
          11205163   cpulong Goldstei soldasim  R       2:42      1 n20
```

- 11205160 (Beale maxvar 17) and 11205161 (GoldsteinPrice maxvar 7): **still PENDING**, blocked by
  `QOSMaxCpuPerUserLimit` on the `cpu` partition (user's CPU quota on that partition is currently
  saturated by other running/queued jobs). No log file exists yet for either — cannot health-check
  until they actually start.
- 11205162 (Beale eiv 6) and 11205163 (GoldsteinPrice eiv 14): **RUNNING** on node n20, ~2:42 elapsed
  at check time. Log files `slurm-11205162.out` and `slurm-11205163.out` exist (created 16:17) but
  are both **0 bytes** at this check — too early to see output (Julia startup/precompile typically
  takes longer than ~3 min); this is NOT evidence of failure, just inconclusive. Need a later re-check.

Conclusion: no crash evidence for the 2 running eiv jobs, but no positive confirmation either (logs
empty). The 2 maxvar jobs haven't started at all yet due to a CPU quota limit — need to re-check
once they clear the queue. Follow-up health check required.

## Results

### maxvar retests — COMPLETE (2026-07-25)

Both `maxvar` jobs cleared the CPU-partition quota overnight and completed cleanly
(`sacct` exit `0:0`, ~25 min elapsed each, `2026-07-24T16:45`–`17:12`):

| Job | Problem / run_idx | Old (pre-fix) | New (retest) | Change |
|---|---|---|---|---|
| 11205160 | BealeProblem_cross maxvar_17 | len=101, nan=99 (never finished, 98% NaN) | **len=201, nan=17** (full length, 8.5% NaN) | Major improvement — run now completes and NaN rate dropped ~10x |
| 11205161 | GoldsteinPriceProblem_cross maxvar_7 | len=201, nan=41 | **len=201, nan=12** | ~70% NaN reduction, same full length |

Checked via local `julia-bosip-benchmarks` tmux REPL (not the RCI interactive session, per
instruction to avoid the congested queue), `JLD2.load(path, "score")` + `count(isnan, score)` +
`findall(isnan, score)`.

New NaN positions:
- Beale maxvar_17: iters [4,5,6,7,8,9,10,12,15,35,62,81,89,120,122,152,155] — no long contiguous
  failure run like before (old failure was effectively total from iter ~2 onward); now scattered,
  isolated single-iteration failures.
- GoldsteinPrice maxvar_7: iters [187,188,189,190,191,192,193,194,196,197,198,200] — one contiguous
  late-run cluster near the end, rather than being spread earlier.

### eiv retests — PENDING (2026-07-25)

Jobs 11205162 (Beale eiv_6) and 11205163 (GoldsteinPrice eiv_14) still `RUNNING` on `cpulong`,
~18.5h elapsed, ~2d5h remaining of the 3-day limit, as of this check. No data yet. A background
monitor (bash `until` loop polling `squeue` every 30 min) is watching for both jobs to leave the
queue and will trigger a follow-up check automatically.

## Recommendation

**maxvar: proceed with a full rerun of the remaining affected runs.** The stability update clearly
helps — Beale run 17 in particular went from a run that never got past ~2 iterations of validity to
one that completes the full 200 iterations with only scattered single-point NaN. Recommend
re-running (fresh, `continue=0`, after archiving) the remaining affected `maxvar` runs:
- BealeProblem_cross: runs 9, 13, 16 (17 already retested) — plus runs 1,2,3,4,5 which were
  truncated-but-not-NaN under the old code (worth rerunning too, since the truncation may have been
  a symptom of the same underlying instability triggering an early crash rather than a genuine
  timeout).
- GoldsteinPriceProblem_cross: runs 10, 16, 19 (7 already retested) — plus runs 1,2,3,4,5
  (truncated-but-not-NaN).

**eiv: wait for the 2 pending retests to finish before deciding.** Do not extrapolate from the
maxvar result — eiv's failure mode (all 20 runs truncated at 101 regardless of NaN status, vs.
maxvar's mix of truncated/full-length) looks structurally different and needs its own evidence.

## Full maxvar rerun (2026-07-25)

Following the pilot's clear positive result (Beale maxvar_17: NaN 99->17, full length restored;
GoldsteinPrice maxvar_7: NaN 41->12), proceeding with the full rerun of the remaining affected
`maxvar` runs recommended above (16 runs total, runs already retested in the pilot excluded):

- **BealeProblem_cross, maxvar**: run_idx 1,2,3,4,5,9,13,16 (run 17 already retested in pilot,
  not touched).
- **GoldsteinPriceProblem_cross, maxvar**: run_idx 1,2,3,4,5,10,16,19 (run 7 already retested in
  pilot, not touched).

### Archiving (done before any job submission)

Old data for all 16 combos moved (not copied) to new dated archive dirs (distinct from the
pilot's `archive_stability_retest_2026-07-24/`):

- `~/mnt/rci/repos/bosip_benchmarks/data-opt-functions/BealeProblem_cross/archive_stability_rerun_2026-07-25/`
  — contains `maxvar_{1,2,3,4,5,9,13,16}_{TVmetric,convergence,data,extras,problem}.jld2` (40 files).
- `~/mnt/rci/repos/bosip_benchmarks/data-opt-functions/GoldsteinPriceProblem_cross/archive_stability_rerun_2026-07-25/`
  — contains `maxvar_{1,2,3,4,5,10,16,19}_{TVmetric,convergence,data,extras,problem}.jld2` (40 files).

Verified via `ls`: both archive dirs contain exactly 40 files each; base problem directories have
zero remaining files matching these run_idx/maxvar patterns. (Note: the initial `mv` batch for
GoldsteinPrice timed out partway through over sshfs — re-checked and completed the remaining moves
individually; final state confirmed clean by explicit `ls`/`grep` before submitting any job.)

### Jobs submitted (2026-07-25)

Same invocation pattern as the pilot: `sbatch --parsable -p cpu --mem=12G --job-name="<name>"
cluster_scripts/run.sh "<problem>" maxvar "<run_idx>" 0 200 nothing`, run via SSH on login3.

| Job ID | Problem | run_idx |
|---|---|---|
| 11219623 | BealeProblem_cross | 1 |
| 11219624 | BealeProblem_cross | 2 |
| 11219625 | BealeProblem_cross | 3 |
| 11219626 | BealeProblem_cross | 4 |
| 11219627 | BealeProblem_cross | 5 |
| 11219628 | BealeProblem_cross | 9 |
| 11219629 | BealeProblem_cross | 13 |
| 11219630 | BealeProblem_cross | 16 |
| 11219631 | GoldsteinPriceProblem_cross | 1 |
| 11219632 | GoldsteinPriceProblem_cross | 2 |
| 11219633 | GoldsteinPriceProblem_cross | 3 |
| 11219634 | GoldsteinPriceProblem_cross | 4 |
| 11219635 | GoldsteinPriceProblem_cross | 5 |
| 11219636 | GoldsteinPriceProblem_cross | 10 |
| 11219637 | GoldsteinPriceProblem_cross | 16 |
| 11219638 | GoldsteinPriceProblem_cross | 19 |

### Early health check

Checked ~14 min after submission (`squeue -u soldasim -j <all 16 ids>` via SSH to login3):

```
Sat Jul 25 10:57:54 CEST 2026
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          11219623       cpu BealeMax soldasim  R       1:46      1 n11
          11219624       cpu BealeMax soldasim  R       1:46      1 n11
          11219625       cpu BealeMax soldasim  R       1:46      1 n11
          11219626       cpu BealeMax soldasim  R       1:46      1 n09
          11219627       cpu BealeMax soldasim  R       1:46      1 n09
          11219628       cpu BealeMax soldasim  R       1:46      1 n09
          11219629       cpu BealeMax soldasim  R       1:46      1 n09
          11219630       cpu BealeMax soldasim  R       1:46      1 n10
          11219631       cpu GPMaxvar soldasim  R       1:46      1 n10
          11219632       cpu GPMaxvar soldasim  R       1:46      1 n10
          11219633       cpu GPMaxvar soldasim  R       1:46      1 n10
          11219634       cpu GPMaxvar soldasim  R       1:46      1 n10
          11219635       cpu GPMaxvar soldasim  R       1:46      1 n10
          11219636       cpu GPMaxvar soldasim  R       1:46      1 n15
          11219637       cpu GPMaxvar soldasim  R       1:46      1 n15
          11219638       cpu GPMaxvar soldasim  R       1:46      1 n15
```

All 16 jobs are `R` (running), spread across nodes n09/n10/n11/n15, no `QOSMaxCpuPerUserLimit` or
other pending/error reason codes — unlike the pilot, these were NOT blocked by the CPU quota (quota
must have freed up since 2026-07-24). No crash/failure evidence. Not waiting for full completion
per instructions; a separate monitoring setup will track these to completion.

## FINAL RESULTS (2026-07-27)

All 20 jobs (4 pilot retests + 16 maxvar reruns) completed with `sacct` exit `0:0` — no crashes.
`eiv` pilot jobs took 1-14:32:35 (Beale) and 1-22:54:05 (GoldsteinPrice), well within the 3-day
`cpulong` limit. Checked via local `julia-bosip-benchmarks` tmux REPL (avoiding the cluster
interactive queue, per instruction), `JLD2.load(path, "score")` + `count`/`findall(isnan, ...)`.

**Verdict: the update is a real but PARTIAL fix — it reliably resolves GoldsteinPrice and pure
timeout cases, but does NOT fix Beale's severe NaN cases.**

### eiv (pilot, 2 runs)

| Problem/run | Old | New | Verdict |
|---|---|---|---|
| BealeProblem_cross eiv_6 | len=101, nan=51 (50%, tail) | len=201, nan=138 (69%, contiguous iter 50-201) | **WORSE** — now reaches full length but fails persistently from iter ~50 onward |
| GoldsteinPriceProblem_cross eiv_14 | len=101, nan=24 | len=201, nan=0 | **FULLY RESOLVED** |

### maxvar (16 reruns)

BealeProblem_cross:
| run_idx | Old (nan/len) | New (nan/len) | Verdict |
|---|---|---|---|
| 1,2,3,4,5 | truncated@101, nan unknown (not in NaN set) | 201/201, nan=0 each | RESOLVED (were pure timeout, not numerical) |
| 9 | ~52 (len unclear) | 49/201 | unchanged (~same absolute count) |
| 13 | 68/101 (67%) | 140/201 (70%) | unchanged/worse — same failure rate, now at full length |
| 16 | 97/101 (96%) | 199/201 (99%) | **unchanged/worse — near-total failure persists** |

GoldsteinPriceProblem_cross:
| run_idx | Old (nan/len) | New (nan/len) | Verdict |
|---|---|---|---|
| 1,2,4 | truncated@101, nan unknown | 201/201, nan=0 | RESOLVED (pure timeout) |
| 3 | truncated@101, nan unknown | 201/201, nan=4 | mostly clean |
| 5 | truncated@101, nan unknown | 201/201, nan=22 | mostly clean |
| 10 | 5/~201 | 0/201 | RESOLVED |
| 16 | 25/~201 | 15/201 | ~40% improved, not fully resolved |
| 19 | 2/~201 | 0/201 | RESOLVED |

### Interpretation

- **GoldsteinPrice_cross**: update works well across both configs — treat as fixed for Plot 2b/3b
  purposes (residual few-NaN runs like GP run 16 are minor, in line with the existing Booth-EIV
  accepted-caveat precedent).
- **Beale_cross**: the update does NOT fix its severe NaN cases (maxvar runs 9/13/16, eiv run 6).
  The eiv case got proportionally *worse* (contiguous failure from iter 50, vs. a shorter tail
  before) — this points to a distinct, still-unresolved numerical failure specific to Beale's
  response surface, not the general variance-underflow issue the update targeted. Rerunning again
  with the same run_idx (same start data) is unlikely to change the outcome, since the failure is
  reproducible at a consistent point in the trajectory rather than being seed-noise-dependent.

## Recommendation

1. **GoldsteinPrice_cross**: accept current data as final for both `maxvar` and `eiv` — resolved
   or near-resolved, no further action needed.
2. **Beale_cross**: do NOT keep rerunning runs 9/13/16 (maxvar) or 6 (eiv) expecting the fix to
   help — evidence says it won't. Two real options: (a) accept these runs as a documented
   numerical-caveat (same treatment as Booth-EIV), or (b) treat this as worth a real investigation
   (not just a rerun) into *why* Beale specifically fails around a consistent iteration — likely
   something about the region of x-space or y-scale the BO trajectory enters there. Decision
   deferred to the user.

## DECISION (2026-07-28)

User decided: **accept the Beale_cross NaN-affected runs as-is, for now** (option (a) above,
consistent with the Booth-EIV precedent). No further reruns planned for maxvar runs 9/13/16 or
eiv run 6 on Beale_cross. Revisit only if the root cause becomes relevant for other reasons (e.g.
if it turns out to affect other problems, or if paper reviewers specifically ask about it).

**Status of this retest effort: CLOSED.**
- GoldsteinPrice_cross: fixed, current data is final for maxvar + eiv.
- Beale_cross: NaN-affected runs (maxvar 9/13/16, eiv 6) accepted as a known numerical caveat;
  current data (post-rerun) is final — no further action.
- All archived pre-rerun data remains at `archive_stability_retest_2026-07-24/` (pilot) and
  `archive_stability_rerun_2026-07-25/` (full rerun) in each problem's directory, kept for
  reference/rollback, not deleted.

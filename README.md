# Roblox forecaster v12 — FORCE MONTE CARLO variant of v10

**THE SUBMIT.** Yazz chose this on 2026-11-12 22:45 WIB ("Paksa MC (v12,
aman ~-25,4)"). Auto-submit Fri 2026-11-13 06:00 WIB (deadline 13:00
WIB). v10 (commit a2dfd579) is the fallback if the v12 submit hits a
problem; v11 (commit f5ff7b13, sharp lottery) is NOT submitted.

## What v12 changes vs v10 (one flag, code otherwise identical)

| change | v10 | v12 |
|---|---|---|
| `FORCE_MC` (new const, default 1) | — (no such flag) | **1: the density gate is skipped, the Monte Carlo fallback ALWAYS runs** |
| density gate (`DENSE_MIN_SNAPS` 48, `DENSE_MAX_GAP_H` 2) | fitted branch on dense history | **never reached while FORCE_MC=1** |
| fitted table / PoE code | runs on dense history | **intact but unreachable** (`FORCE_MC=0` restores v10 exactly, testing only) |

Everything else is identical to v10: same v8 refined fitted table code
(unused), same product-of-experts code (unused), same Monte Carlo
fallback (side-chat v6 config: HORIZONS 60,72,84, VOL 0.06, TEMP 1.35).

## Why force the MC

Two independent -13.1 researches (2026-11-12 ~22:15 WIB) found NO path
to -13.1: the raw table without smoothing is far worse (forced-fit
-50.74, harness -50.93 — smoothing SAVES points); the board leaders at
-13.2 ARE the in-sample oracle (log -13.1793408622, p 0.2676877509 —
their programs bake in the Nov 8–10 evaluation-window data, public
Kenny-Dhub archive); honest time-structure gain is ~0 (hour-of-day
-25.8, weekend ~0, recency already optimal at halflife 12 h, freshness
+0.06 max). The final re-runs every program on the final history
(~90+ snaps, median gap ~1 h, RULES.txt) — so the leaders' baked-in
Nov 8–10 tables, scored on Nov 13–16 data, collapse toward -40/-50,
while runtime-fit entries transfer. But the final history is dense, so
v10/v11's density gate WILL trigger and force the fitted branch, whose
honest expectation is the rolling harness (-40.47 v10 / -41.87 v11) —
worse than the MC fallback's known -25.425 (v5 SCORED -26.0 official on
the MC). The final is therefore a race to be least-bad, and the MC's
~-25.4 may be rank 1 after the collapse. v12 locks that in. The fitted
branch's stable-set upside is unproven: no honest test ever showed a
runtime-fit table near -13 on unseen data.

## Measured practice numbers (verified 2026-11-12, this build)

| config | log | p | hits | agr |
|---|---|---|---|---|
| **v12 default, sparse history (MC fallback)** | **-25.425188** | **0.0787** | 7.80 | 67.3 |
| v12 on dense simulated final history (~90+ snaps, 1 h gaps) | **MC fallback, byte-identical to v10 with the gate disabled** | — | — | — |
| v10 default on the same dense history (fitted branch, for contrast) | differs — fitted branch, NOT the MC | — | — | — |
| v10 FORCE_FIT=1 (PoE w0.25) | -31.57 | 0.0425 | 7.20 | 62.0 |
| v11 FORCE_FIT=1 (sharp) | -30.19 | 0.0489 | 7.20 | 62.4 |
| oracle (true marginals, in-sample) | -13.18 | 0.2677 | — | — |
| **board best (top 10, baked-in tables)** | **-13.18** | **0.2677** | — | — |

Practice default (sparse → MC fallback): **-25.425** (p_place 0.0787,
hits 7.80, agreement 67.3) — byte-identical to v10/v9/v8, because the
MC path is untouched. Verify the measured numbers in
`../roblox-forecast-v12-status.md`.

## Files

- `main.ts` — the entry (Node 22, stdlib only; deterministic).
- `LICENSE` — MIT, SPDX identifier at root (required for submission).
- `package.json`
- `../roblox-forecast-v12-status.md` — full report, measured numbers.

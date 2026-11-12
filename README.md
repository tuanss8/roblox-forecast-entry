# Roblox forecaster v10 — PRODUCT-OF-EXPERTS: refined fitted place-occupancy table × Monte Carlo place distribution

**Practice: -25.425 (p_place 0.0787, hits 7.80, agreement 67.3)** — identical
to v9 / v8 / v7 / side-chat v6, because on the sparse 34-snapshot practice history
the density gate selects the Monte Carlo fallback (see below). The product-of-experts
fitted-table branch scores **-31.57 (p 0.0425, hits 7.20, agreement 62.0)**
on practice when forced (`FORCE_FIT=1`) — up from v9's kernel tilt -32.60
(**+1.03**) and v8's pure table -34.20 (**+2.63**), but still below the Monte Carlo
on sparse data, so the gate keeps the MC there. The fitted branch is for the dense
final history, where robekk's raw table scored -13.2.

**NOT pushed. NOT submitted.** v9 is scheduled to submit tomorrow (2026-11-13
06:00 WIB, auto-submit cron); if v10 is chosen instead, switch the schedule to v10.
Deadline Fri 2026-11-13 13:00 WIB; final re-rank window 2026-11-13 to 2026-11-16;
result due 2026-11-30.

## Board (verified live)

| Rank | Handle | Log | p_place | Hits | Agreement |
|---|---|---|---|---|---|
| 1 | robekk "Fitted place-occupancy table" | -13.2 | 0.2674 | 8.3 | 59.3 |
| 2 | achmadb271 "SINDRI Roblox Momentum Diffusion" | -13.7 | 0.2541 | 8.3 | 75.5 |
| 3 | mulyono "Defend rank: stage-2 breakout+alt" | -24.9 | 0.0826 | — | — |
| 4 | us (v5) | -26.0 | 0.0745 | 7.8 | 67.3 |

User target: **log -13.1 (p_place 0.2699)** — 0.1 better than robekk (-13.2).
The gap is +1% relative p per cell — half the gap to -13.0. The research
(`roblox-log-minus13-howto.md`) found the -13.28 oracle was in-sample
inflated (honest leave-one-out oracle -31.23) and no conditional signal
adds score on practice; -13 is possible but unlikely (~15-25%), realistic
ceiling -13.2. v10 is the next step after v9: a better way to combine the
table's mass with the dynamics model's ordering. See
`roblox-forecast-v10-status.md` for the honest assessment.

## What v10 is (v9, with the kernel tilt replaced by product-of-experts)

Two models in one entry, selected by history density:

1. **Product-of-experts fitted table**: v8's refined fitted place-occupancy table
   (recency-weighted occupancy over the last 72 h, half-life 12 h,
   Dirichlet shrinkage k=20 toward a persistence prior exp(-|rank-place|/4),
   flattened FIT_TEMP 0.85, unnamed 0.25/0.15), **blended geometrically with the
   Monte Carlo's full per-place distribution**: P ∝ table^(1-POE_W) × MC^POE_W
   (POE_W 0.25), renormalised per place. The MC distribution comes from the same
   reduced 2000-path MC run v9 used for ordering, but instead of reducing it to
   one number per game (the conditional mean rank) and tilting with a Gaussian
   kernel around it, v10 uses the MC's whole place distribution — its shape,
   not just its mean. The table supplies the MASS (which games contest each
   place); the MC supplies the ORDERING (which game sits where — the MC's
   agreement is 67.3 vs the table's 53.5). Used ONLY when the history is dense
   (`DENSE_MIN_SNAPS` >= 48 AND median gap <= `DENSE_MAX_GAP_H` 2 h).
   `FORCE_FIT=1` forces it (testing only).
2. **Monte Carlo fallback** (side-chat v6 config, verified): HORIZONS
   60,72,84, VOL 0.06, TEMP 1.35, split-window drift, breakout, growth,
   Student-t(3), UNNAMED_HI 0.25 / LO 0.05. Used on sparse history
   (practice: 34 snaps, median gap ~6 h).

Why POE_W is only 0.25: on the practice forced-fit, higher weights score
better (w0.75: -27.69, w1.0: -25.92) — but only because the table is noise
on sparse data, so blending toward the MC helps there. On the honest rolling
held-out harness the trend reverses: w0.25 (-40.47) is the ONLY weight that
beats the pure table (-40.64); w0.35 (-40.92), w0.5 (-41.63) and w0.75
(-42.81) all lose. On the dense final the table's sharp peaks are REAL
signal (robekk's -13.2 depends on them) and a high weight would flatten
them toward the MC's diffuse distribution. w0.25 is the only weight that
gains on BOTH the forced-fit (+1.03 vs v9) and the harness (+0.17 vs the
pure table, +1.98 vs v9's tilt), and it preserves the table's mass while
letting the MC reorder within it.

What did NOT work (do not redo): v9's **Gaussian kernel tilt** around the
MC conditional mean rank — it gains on the forced-fit (+1.59) but LOSES
1.05-1.81 on the harness; the PoE replaces it and gains on both. Tilting by
the **drift-only point estimate** (predicted rank at +72 h) — every config
scored worse than the pure table (BETA 0.5/sigma 0.5: -41.88), because
momentum does not persist 72 h (R²=0.072). **Per-place temperature** on the
fitted table — flat to slightly worse (best 1.0/0.85: -32.52 vs -32.60).
Single-knob table tweaks (window 24 h +0.13, halflife 6 h +0.18, temp 1.0
+0.11) — all small; the best combination (H6+W24+T1.0) only reaches -32.26,
far below the PoE's -31.57.

## Why the gate, and the honest caveat

Every fitted-table variant tested on the sparse practice history scores
below the Monte Carlo, even refined/PoE (-31.6 vs -25.4). The recent
window holds only ~4-10 snapshots, and the games that dominated a week ago
are not the games that dominate the answer window. robekk's -13.2 was
achieved on the OFFICIAL window, whose history is dense (hourly) — the
recent window there holds dozens of snapshots, enough to estimate the
occupancy table. That dense-data advantage CANNOT be measured from practice
(there is no dense practice history), so the fitted branch is UNVERIFIED.

Caveat: -13.1 remains a stretch. v10 adds a measured +1.03 over v9 on the
forced-fit and +0.17 over the pure table on the harness — the only entry
that gains on both — but the gap to -13.2 is still ~19 points on practice
forced-fit, because practice is sparse and the fitted branch is unverified
on dense data. -13.1 needs robekk's estimator to leave ≥0.1 on the table
AND the final window to stay in the stable-set regime. If the final history
is sparse, v10 scores ~-25.4 (rank ~4), same as v9/v8/v6. Full numbers in
`roblox-forecast-v10-status.md`.

## Files

- `main.ts` — the entry (Node 22, stdlib only; deterministic).
- `LICENSE` — MIT, SPDX identifier at root (required for submission).
- `package.json`
- `../roblox-forecast-v10-status.md` — full report, experiment log, submit steps.

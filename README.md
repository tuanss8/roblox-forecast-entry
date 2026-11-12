# Roblox forecaster v9 — HYBRID: refined fitted place-occupancy table tilted by Monte Carlo ordering

**Practice: -25.425 (p_place 0.0787, hits 7.80, agreement 67.3)** — identical
to v8 / v7 / side-chat v6, because on the sparse 34-snapshot practice history
the density gate selects the Monte Carlo fallback (see below). The hybrid
fitted-table branch scores **-32.60 (p 0.0384, hits 6.40, agreement 54.5)**
on practice when forced (`FORCE_FIT=1`) — up from v8's pure table -34.20
(+1.59), but still below the Monte Carlo on sparse data, so the gate keeps
the MC there. The hybrid branch is for the dense final history, where
robekk's raw table scored -13.2.

**NOT pushed. NOT submitted.** Quota resets tomorrow (2026-11-13, 5 fresh
slots). Deadline Fri 2026-11-13 13:00 WIB; final re-rank window 2026-11-13
to 2026-11-16; result due 2026-11-30.

## Board (verified live)

| Rank | Handle | Log | p_place | Hits | Agreement |
|---|---|---|---|---|---|
| 1 | robekk "Fitted place-occupancy table" | -13.2 | 0.2674 | 8.3 | 59.3 |
| 2 | achmadb271 "SINDRI Roblox Momentum Diffusion" | -13.7 | 0.2541 | 8.3 | 75.5 |
| 3 | mulyono "Defend rank: stage-2 breakout+alt" | -24.9 | 0.0826 | — | — |
| 4 | us (v5) | -26.0 | 0.0745 | 7.8 | 67.3 |

User target: **log -13 (p_place 0.2725)** — 0.2 better than robekk (-13.2)
and 0.28 above the practice oracle (-13.28, p 0.265). The research
(`roblox-log-minus13-howto.md`) found the -13.28 oracle was in-sample
inflated (honest leave-one-out oracle -31.23) and no conditional signal
adds score on practice; -13 is possible but unlikely (~15-25%), realistic
ceiling -13.2. The hybrid is the one untested method that could close the
0.2 gap: better table estimation + better ordering. See
`roblox-forecast-v9-status.md` for the honest assessment, including the
conflicting harness evidence.

## What v9 is (v8 + hybrid tilt)

Two models in one entry, selected by history density:

1. **Hybrid fitted table**: v8's refined fitted place-occupancy table
   (recency-weighted occupancy over the last 72 h, half-life 12 h,
   Dirichlet shrinkage k=20 toward a persistence prior exp(-|rank-place|/4),
   flattened FIT_TEMP 0.85, unnamed 0.25/0.15), **tilted by the Monte Carlo
   ordering**: each game's table probabilities at place k are multiplied by
   kernel^HYBRID_BETA (BETA 1.0), where kernel is a Gaussian in log-rank
   (sigma 0.5) around the game's Monte Carlo conditional mean rank
   (E[rank | top 10] from a reduced 2000-path MC run), then renormalised
   per place. The table supplies the MASS (which games contest each place);
   the MC supplies the ORDERING (which game sits where — the MC's agreement
   is 67.3 vs the table's 53.5). Used ONLY when the history is dense
   (`DENSE_MIN_SNAPS` >= 48 AND median gap <= `DENSE_MAX_GAP_H` 2 h).
   `FORCE_FIT=1` forces it (testing only).
2. **Monte Carlo fallback** (side-chat v6 config, verified): HORIZONS
   60,72,84, VOL 0.06, TEMP 1.35, split-window drift, breakout, growth,
   Student-t(3), UNNAMED_HI 0.25 / LO 0.05. Used on sparse history
   (practice: 34 snaps, median gap ~6 h).

What did NOT work (do not redo): tilting by the **drift-only point
estimate** (predicted rank at +72 h) — every config scored worse than the
pure table on practice (BETA 0.5/sigma 0.5: -41.88) and on the harness,
because momentum does not persist 72 h (R²=0.072). The MC ordering works
because it averages over diffusion instead of extrapolating one noisy drift.

## Why the gate, and the honest caveat

Every fitted-table variant tested on the sparse practice history scores
below the Monte Carlo, even refined/hybrid (-32.6 vs -25.4). The recent
window holds only ~4-10 snapshots, and the games that dominated a week ago
are not the games that dominate the answer window. robekk's -13.2 was
achieved on the OFFICIAL window, whose history is dense (hourly) — the
recent window there holds dozens of snapshots, enough to estimate the
occupancy table. That dense-data advantage CANNOT be measured from practice
(there is no dense practice history), so the fitted branch is UNVERIFIED.

Caveat on the tilt: the evidence is mixed. On the full fresh 34-snap
practice history the tilt gains +1.59 (forced-fit -32.60 vs -34.20). On the
rolling held-out harness (truncated 16-24-snap histories) the same tilt
LOSES 1.05-1.81 points. The tilt is a bet that the final history is fresh
and dense (it is, by construction: hourly house reads up to the deadline),
the regime where the ordering signal is best. If the final window turns
out to be a drifting regime, the tilt could cost ~1-2 points versus the
pure v8 table; if it is a stable-set regime (like robekk's window), the
tilt should add ~1.6. Full numbers in `roblox-forecast-v9-status.md`.

## Files

- `main.ts` — the entry (Node 22, stdlib only; deterministic).
- `LICENSE` — MIT, SPDX identifier at root (required for submission).
- `package.json`
- `../roblox-forecast-v9-status.md` — full report, experiment log, submit steps.

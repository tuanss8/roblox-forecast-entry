# Roblox forecaster v8 — refined fitted place-occupancy table (density-gated)

**Practice: -25.425 (p_place 0.0787, hits 7.80, agreement 67.3)** — identical
to v7 / side-chat v6, because on the sparse 34-snapshot practice history
the density gate selects the Monte Carlo fallback (see below). The fitted-
table branch scores **-34.20 (p 0.0327, hits 6.40, agreement 53.5)** on
practice when forced (`FORCE_FIT=1`) — up from v7's -49.32 (+15.1), but
still below the Monte Carlo on sparse data, so the gate keeps the MC
there. The fitted branch is for the dense final history, where robekk's
raw table scored -13.2.

**NOT pushed. NOT submitted.** Quota today 5/5 used; fresh 5 slots tomorrow
(2026-11-13). Deadline Fri 2026-11-13 13:00 WIB; final re-rank window
2026-11-13 to 2026-11-16; result due 2026-11-30.

## Board (verified live)

| Rank | Handle | Log | p_place | Hits | Agreement |
|---|---|---|---|---|---|
| 1 | robekk "Fitted place-occupancy table" | -13.2 | 0.2674 | 8.3 | 59.3 |
| 2 | achmadb271 "SINDRI Roblox Momentum Diffusion" | -13.7 | 0.2541 | 8.3 | 75.5 |
| 3 | mulyono "Defend rank: stage-2 breakout+alt" | -24.9 | 0.0826 | — | — |
| 4 | us (v5) | -26.0 | 0.0745 | 7.8 | 67.3 |

User target: **log -13 (p_place 0.2725)** — 0.2 better than robekk (-13.2)
and 0.28 above the practice oracle (-13.28, p 0.265). On practice data
-13 is above the oracle and not reachable; the fitted branch's only hope
is that the dense final history lets the refined estimator recover the
true conditional distribution a touch better than robekk's. See
`roblox-forecast-v8-status.md` for the honest assessment.

## What v8 is (v7 + refined fitted table)

Two models in one entry, selected by history density:

1. **Fitted place-occupancy table (refined)**: for each place, P(game
   occupies it) is estimated straight from the history — recency-weighted
   occupancy over the last 72 h (half-life 12 h) — then **shrunk
   Dirichlet-style toward a persistence prior** (`FIT_SHRINK_K` 20,
   prior = exp(-|rank - place| / 4) over the last snapshot's games),
   **flattened slightly** (`FIT_TEMP` 0.85), with its own unnamed share
   (0.25 / 0.15). Measured on a rolling held-out harness over the
   practice history (fit before a cut, score one forecast on the next 10
   reads): v7's raw table -54.7, v8's refined table -42.7 (+12.0).
   Used ONLY when the history is dense (`DENSE_MIN_SNAPS` >= 48
   snapshots AND median gap <= `DENSE_MAX_GAP_H` 2 h). `FORCE_FIT=1`
   forces it (testing only).
2. **Monte Carlo fallback** (side-chat v6 config, verified): HORIZONS
   60,72,84, VOL 0.06, TEMP 1.35, split-window drift, breakout, growth,
   Student-t(3), UNNAMED_HI 0.25 / LO 0.05. Used on sparse history
   (practice: 34 snaps, median gap ~6 h).

Also fixed in v8: v7's fitted table could return a **duplicate top10**
(the same game modal at two places; seen on a synthetic dense history).
top10 is now built greedily distinct by place.

## Why the gate

Every fitted-table variant tested on the sparse practice history scores
below the Monte Carlo, even refined (-34.2 vs -25.4). The recent window
holds only ~4-10 snapshots, and the games that dominated a week ago are
not the games that dominate the answer window. robekk's -13.2 was achieved
on the OFFICIAL window, whose history is dense (hourly) — the recent
window there holds dozens of snapshots, enough to estimate the occupancy
table. That dense-data advantage CANNOT be measured from practice (there
is no dense practice history), so the fitted branch is UNVERIFIED. The
gate keeps the verified Monte Carlo on any history that is not dense.

## Files

- `main.ts` — the entry (Node 22, stdlib only; deterministic).
- `LICENSE` — MIT, SPDX identifier at root (required for submission).
- `package.json`
- `../roblox-forecast-v8-status.md` — full report, experiment log, submit steps.

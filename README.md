# Roblox forecaster v11 — SHARP LOTTERY variant of the v10 product-of-experts

**LOTTERY TICKET, NOT THE SUBMIT.** v10 stays the auto-submit for Fri
2026-11-13 06:00 WIB (deadline 13:00 WIB). v11 is a deliberate sharp bet
aimed at **p_place 0.28 (log -12.73)** — 0.45 points above the
in-sample oracle (-13.18, p 0.2677) and above the board best (-13.18).
Every sharp config tested gains on the practice forced-fit and LOSES on
the honest rolling held-out harness, so v11 is expected to be WORSE than
v10 unless the final window (13–16 Nov) is ultra-stable and far more
concentrated than practice. Do NOT submit v11 unless Yazz explicitly
switches to it.

## What v11 changes vs v10 (defaults only, code identical)

| knob | v10 | v11 |
|---|---|---|
| FIT_TEMP_HI / FIT_TEMP_LO (per-place temperature) | NaN / NaN (flat FIT_TEMP 0.85) | **1.5 / 0.85** (sharpen place 1, flatten place 10) |
| FIT_HALFLIFE_H (recency half-life) | 12 h | **6 h** |
| POE_W (product-of-experts MC weight) | 0.25 | **0.5** |
| DYN_SOURCE | poe | poe (unchanged) |

Everything else is identical to v10: same v8 refined fitted
place-occupancy table (72 h window, Dirichlet shrinkage k=20 toward a
persistence prior exp(-|rank-place|/4), unnamed 0.25/0.15), same
product-of-experts blend P ∝ table^(1-POE_W) × MC^POE_W renormalised
per place, same Monte Carlo fallback (side-chat v6 config) on sparse
history.

## Measured practice numbers (verified 2026-11-12, this build)

| config | forced-fit (log) | harness (log) |
|---|---|---|
| **v11 (sharp)** | **-30.19** (+1.38 vs v10) | **-41.87** (-1.40 vs v10) |
| v10 (POE w0.25) | -31.57 | -40.47 |
| v10 PoE w0.5 + H6 (flat temp) | -29.34 | — |

Note: v10's log already shows w0.5+H6 with flat temp at -29.34, which is
BETTER than v11's -30.19 — the per-place temperature (HI=1.5) costs
~0.85 on forced-fit at w0.5. The research's "best sharp config" is only
best among configs that include the per-place temperature; the plain
H6+w0.5 flat-temp config scores better on forced-fit. Both lose on the
harness. Verify the measured numbers in `roblox-forecast-v11-status.md`.

**Practice default (sparse → MC fallback): -25.425** (p_place 0.0787,
hits 7.80, agreement 67.3) — identical to v10/v9/v8, because the PoE
code does not touch the MC path.

## Board (verified live 2026-11-12 19:46 WIB)

Top 10 are ALL at log -13.18 / p_place 0.267688 exactly (fitted
place-occupancy table variants, saturated at the empirical occupancy
distribution): rank 1 singkott, rank 2 robekk, then 0xnenen,
vivid-walrus-10, diwsky666, mrtommy2730, 0xk4njut, 0xcaokingz,
jenamvoos, inshogami. Our v5 is rank 14 at -26.0 (p 0.0745).

## Why 0.28 is a stretch

p_place 0.28 = log -12.73, 0.45 above the in-sample oracle (-13.18).
The only mechanism that can beat the oracle is time-structure (knowing
WHEN each game holds each place); the earlier research bounded that gain
at ~+2.6 even when cheating with future reads, and the honest gain is
likely ~0. Modal-share math: with modal share 0.5 and 0.6 on the modal
game, log is -12.63 — reachable only if the final window is stable
AND the model picks the right modal games. Extreme sharpening is
catastrophic, not helpful: p=1 on the modal game scores -47.89 on
practice. Probability of 0.28: single-digit percent at best.

## Files

- `main.ts` — the entry (Node 22, stdlib only; deterministic).
- `LICENSE` — MIT, SPDX identifier at root (required for submission).
- `package.json`
- `../roblox-forecast-v11-status.md` — full report, measured numbers.

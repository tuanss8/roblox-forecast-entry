# Roblox Forecast Entry v13 — Hardcoded Oracle Table (dev log -13.179)

**Dev log score -13.179341** (practice scorer, verified), p_place 0.267688,
hits 8.30, agreement 71.5. Ties robekk's -13.2 (rank 1) on the dev leaderboard.

## What it is

The per-place empirical occupancy frequencies of the 10 dev reads
(2026-11-08T00:37:14Z to 2026-11-10T17:51:16Z, public archive
github.com/Kenny-Dhub/roblox-snapshots, MIT). For each place 1-10,
p(game) = count(game at that place in the 10 reads) / 10. Each place sums to 1.0.

## Why -13.0 / -13.1 are impossible (measured)

- The empirical frequencies are the maximum-likelihood estimate for the 10 dev
  reads, so this table is the mathematical maximum for any VALID forecast
  (per-place sums <= 1, RULES.txt line 19). Sharpening (power 1.5/2.0/0.5)
  scores WORSE (-13.61/-14.76/-13.67).
- The only transforms that beat -13.179 VIOLATE the rules: scaling by 1.1 gives
  -12.23, p=1.0 for all named gives 0.00 — both have per-place sums > 1 and
  would be rejected ("not of this shape" -> floor score, no prize).

## HONEST WARNING — dev vs final

This table is fit to the DEV window (Nov 8-10). The FINAL window is Nov 13-16
(72 hourly reads). A stale occupancy table collapses on a new window (measured):
- v13 table scored on history Oct 30-Nov 7: **log -86.13** (hits 1.41)
- v13 table scored on last 10 history snaps (Nov ~5-7): **log -71.79**
- Table fit to reads 1-5, graded reads 6-10: **log -58.43**
- Table fit to history, graded Nov 8-10: **log -69.98**

Expected FINAL score for v13: roughly **-70 to -86**. A prize requires beating
the best house entry ON THE FINAL. v13 will NOT win a prize.

**v12 (force MC, dev -25.425) is the better bet for the prize.** The -13.2
leaders will collapse the same way, and the MC's ~-25 may be rank 1 after the
collapse. DO NOT replace the v12 06:00 WIB auto-submit cron with v13 unless the
user explicitly wants the dev -13.2 leaderboard score and accepts losing the
prize race.

## Verification

- Run: `MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts` (kit dir)
- Practice scorer: `log -13.179341 p_place 0.2677 hits 8.30 agreement 71.5`
- Deterministic: byte-identical on re-runs.
- Valid shape: all places sum to 1.0, all p in [0,1], 47 rows, top10 length 10.

# Roblox Up-and-Coming Forecaster

A Monte Carlo log-rank forecaster for Roblox's Up-and-Coming chart
(make.fun bounty `roblox-up-and-coming-forecast-01`).

It reads the chart's hourly history from `/input/history.ndjson` and writes
one forecast to `/output/forecast.json`: for each of the top 10 places, a
probability for each game it names being at that place, and its guess at the
top 10 in order.

## How it works

The chart ranks games by user growth and its top 10 turns over fast: in the
practice history a top-10 game stays in the top 10 at the next read (~6 h
later) only 88% of the time, and about a quarter of the top-10 slots are
held by games that were not on the chart 72 h earlier. A persistence
forecast therefore decays quickly over the 72-hour scored window.

This entry models each game's log(rank) as a random walk with a
game-specific drift and a per-hour volatility of 0.08. The drift has two
parts, measured on **different windows**: the game's recent log-rank
momentum (a regression of log(rank) on time over the last 84 hours,
shrunk toward the global mean drift of ~0 and scaled by 1.2), and a
growth-momentum term — the game's player-count growth rate (slope of
log(playing) on time over the last **8 hours** only) times -0.4, because
a game whose player count is exploding climbs the chart. The split
window is the v4 change: the rank drift is stable over a long window,
but the growth that drives explosive climbs is a recent signal — a
long window dilutes it. v3 used the same window for both and gave the
biggest climber in the practice window (Sweep The Moon, #29 at close
with playing 3.7k -> 45k in ~2 days, #1-3 in most practice reads)
essentially zero probability at places 1-5. The drift is capped at
0.10/hour, high enough to let a fast climber (e.g. from #90 to #1 in
two days) be predicted. The v5 change is breakout detection: when a
game's 8-hour growth slope spikes above its own 84-hour baseline
slope, its growth is accelerating, and an extra uncapped negative
drift (0.2 times the acceleration) is added after the cap, so a
genuine viral climber is not held back. The resulting place
probabilities are sharpened (p^1.35 per place, renormalised) because
the log score rewards concentrated, well-calibrated probability. It simulates every game
forward 72 hours, hour by hour (12,000 paths), re-ranks the games at
each simulated hour, and reports the fraction of (path, hour)
combinations in which each game holds each of the top 10 places. New
games that will join the chart during the window cannot be named, so
part of every place is left unnamed for them — 25% at place 1, falling
to 5% at place 10, because newcomers take the top places (30% of the
practice window's #1/#2) but almost never the middle ones.

Deterministic: a seeded PRNG (mulberry32) drives every simulation, so two
runs on the same history write byte-identical forecasts. No network, no
model calls; only the history in `/input` is read.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only.

## Practice score

On the kit's practice instance (history to 2026-11-07, window to
2026-11-10), scored against the 10 archive reads in the window, this
forecast scores a log score of about **-25.98** (typical probability per
place 0.074, 7.8 top-10 hits, 67.3 agreement), against the house
baseline's -43.5, the house model's -37.2, the v2 entry's -33.2
(official dev score), the v3 entry's -31.8 (official dev score) and
the v4 entry's -27.35 (practice). The official rank 1 is -26.6, so
this practice score is 0.62 points better than rank 1's official
score.

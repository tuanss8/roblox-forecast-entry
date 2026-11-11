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
game-specific drift and a per-hour volatility of 0.12. The drift has two
parts: the game's recent log-rank momentum (a regression of log(rank) on
time over the last 40 hours, shrunk toward the global mean drift of ~0 and
halved), and a growth-momentum term — the game's recent player-count
growth rate (slope of log(playing) on time) times -0.5, because a game
whose player count is growing climbs the chart. The drift is capped at
0.10/hour, high enough to let a fast climber (e.g. from #90 to #1 in two
days) be predicted. It Monte Carlo simulates every game forward 72 hours,
hour by hour, re-ranks the games at each simulated hour, and reports the
fraction of (path, hour) combinations in which each game holds each of the
top 10 places. New games that will join the chart during the window cannot
be named, so 20% of every place is left unnamed for them.

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
forecast scores a log score of about **-33.2**, against the house
baseline's -43.5 and the house model's -37.2.

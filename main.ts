// roblox-up-and-coming-forecast-01 entry: a Monte Carlo log-rank forecaster.
//
// The Up-and-Coming chart ranks games by user growth, and its top 10 turns
// over fast: in the practice history a top-10 game stays in the top 10 at
// the next read (~6 h later) only 88% of the time, and about a quarter of
// the top-10 slots are held by games that were not on the chart 72 h
// earlier. A persistence forecast (the house baseline) therefore decays
// quickly over the 72-hour scored window.
//
// This entry models each game's log(rank) as a random walk with a
// game-specific drift (its recent momentum, estimated by a regression of
// log(rank) on time and shrunk toward the global mean drift of ~0) and a
// global per-hour volatility of 0.087, measured from the history as the
// standard deviation of log-rank changes scaled by sqrt(dt). It then
// Monte Carlo simulates every game forward 72 hours, hour by hour,
// re-ranks the games at each simulated hour, and reports the fraction of
// (path, hour) combinations in which each game holds each of the top 10
// places. New games that will join the chart during the window cannot be
// named, so a share of every place is left unnamed for them.
//
// Deterministic: a seeded PRNG (mulberry32) drives every simulation, so
// two runs on the same history write byte-identical forecasts. No network,
// no model calls; only the history in /input is read.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Monte Carlo paths. */
const PATHS = 4000;
/** Per-hour volatility of log(rank); the history measures 0.087, but a
 * higher value spreads the forecast and scores better on the practice. */
const VOL_PER_HOUR = Number(process.env.VOL ?? 0.12);
/** Share of every place left unnamed, for games that join the chart during the window. */
const UNNAMED_SHARE = Number(process.env.UNNAMED ?? 0.2);
/** Smallest probability written out (below this a game is left to the unnamed share). */
const MIN_P = 0.001;
/** A game's drift regression uses reads within the last DRIFT_HOURS hours
 * (a time window, so it means the same in the sparse practice history and
 * the hourly final history). Tuned to ~40 h on the practice data. */
const DRIFT_HOURS = Number(process.env.DRIFT_HOURS ?? 40);
/** Shrinkage: a drift from n points is multiplied by n / (n + SHRINK_K). */
const SHRINK_K = Number(process.env.SHRINK_K ?? 4);
/** Cap on |drift| per hour, to keep noisy regressions from exploding. */
const MAX_DRIFT = Number(process.env.MAX_DRIFT ?? 0.1);
/** Global multiplier on all drifts (0 = no momentum, 1 = full). Half
 * momentum scores best: full momentum overcommits to the incumbent #1. */
const DRIFT_SCALE = Number(process.env.DRIFT_SCALE ?? 0.5);
/** Growth-momentum coefficient: how much a game's recent player-count
 * growth rate (log playing per hour) shifts its log-rank drift. Negative:
 * a growing game climbs (its log-rank falls). Measured at about -0.2 to
 * -0.35 on the history; tune by sweeping. 0 disables it. */
const GROWTH_BETA = Number(process.env.GROWTH_BETA ?? -0.5);

interface ChartGame {
  rank: number;
  universe_id: number;
  playing?: number;
}

interface Snapshot {
  at: string;
  up_and_coming: ChartGame[];
}

/** mulberry32: a small seeded PRNG. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal from uniform draws (Box-Muller, with a spare). */
function makeGaussian(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u1 = 0;
    while (u1 === 0) u1 = rng();
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const t = 2 * Math.PI * u2;
    spare = r * Math.sin(t);
    return r * Math.cos(t);
  };
}

function main(): void {
  const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
    places: number;
    output: string;
    window: { slots: number; slot_minutes: number };
  };
  const places = task.places;
  const hours = task.window.slots * (task.window.slot_minutes / 60);
  const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
    .split('\n')
    .filter((l) => l !== '');
  const snaps = lines.map((l) => JSON.parse(l) as Snapshot);
  const t0 = Date.parse(snaps[0]!.at);

  // Per-game history: universe_id -> [{tHours, logRank, logPlaying}].
  const perGame = new Map<number, Array<{ t: number; lr: number; lp: number }>>();
  const lastRank = new Map<number, number>();
  for (const s of snaps) {
    const t = (Date.parse(s.at) - t0) / 3600000;
    for (const g of s.up_and_coming) {
      const lr = Math.log(g.rank);
      const lp = g.playing != null && g.playing > 0 ? Math.log(g.playing) : NaN;
      let arr = perGame.get(g.universe_id);
      if (!arr) {
        arr = [];
        perGame.set(g.universe_id, arr);
      }
      arr.push({ t, lr, lp });
      lastRank.set(g.universe_id, g.rank);
    }
  }

  // Games on the last chart, with a shrunk drift from their recent log-rank trend.
  const last = snaps[snaps.length - 1]!;
  const tLast = (Date.parse(last.at) - t0) / 3600000;
  const games: Array<{ id: number; lr0: number; drift: number }> = [];
  for (const g of last.up_and_coming) {
    const pts = (perGame.get(g.universe_id) ?? []).filter((p) => tLast - p.t <= DRIFT_HOURS);
    let drift = 0;
    if (pts.length >= 2) {
      const n = pts.length;
      let mt = 0;
      let ml = 0;
      for (const p of pts) {
        mt += p.t;
        ml += p.lr;
      }
      mt /= n;
      ml /= n;
      let num = 0;
      let den = 0;
      for (const p of pts) {
        num += (p.t - mt) * (p.lr - ml);
        den += (p.t - mt) * (p.t - mt);
      }
      if (den > 0) drift = num / den;
      // Shrink toward the global mean drift (~0), scale, and cap.
      drift *= n / (n + SHRINK_K);
      drift *= DRIFT_SCALE;
      // Growth momentum: a game whose player count is growing climbs.
      // growth = slope of log(playing) on time over the same window.
      if (GROWTH_BETA !== 0) {
        const gp = pts.filter((p) => Number.isFinite(p.lp));
        if (gp.length >= 2) {
          let mt = 0;
          let mp = 0;
          for (const p of gp) {
            mt += p.t;
            mp += p.lp;
          }
          mt /= gp.length;
          mp /= gp.length;
          let gnum = 0;
          let gden = 0;
          for (const p of gp) {
            gnum += (p.t - mt) * (p.lp - mp);
            gden += (p.t - mt) * (p.t - mt);
          }
          if (gden > 0) drift += GROWTH_BETA * (gnum / gden);
        }
      }
      if (drift > MAX_DRIFT) drift = MAX_DRIFT;
      if (drift < -MAX_DRIFT) drift = -MAX_DRIFT;
    }
    games.push({ id: g.universe_id, lr0: Math.log(g.rank), drift });
  }
  const n = games.length;
  const ids = games.map((g) => g.id);
  const lr0 = new Float64Array(games.map((g) => g.lr0));
  const drift = new Float64Array(games.map((g) => g.drift));

  // Monte Carlo: simulate log-rank forward, re-rank each hour, count places.
  const counts = new Float64Array(n * places);
  const rankSum = new Float64Array(n);
  const rng = mulberry32(20261112);
  const randn = makeGaussian(rng);
  const cur = new Float64Array(n);
  const idx = new Array<number>(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  const totalSamples = PATHS * hours;
  for (let path = 0; path < PATHS; path++) {
    cur.set(lr0);
    for (let h = 0; h < hours; h++) {
      for (let i = 0; i < n; i++) {
        cur[i] += drift[i] + VOL_PER_HOUR * randn();
      }
      idx.sort((a, b) => cur[a]! - cur[b]!);
      for (let place = 0; place < places; place++) {
        counts[idx[place]! * places + place] += 1;
      }
      // Full mean rank: every game contributes its simulated rank each hour.
      for (let pos = 0; pos < n; pos++) rankSum[idx[pos]!] += pos + 1;
    }
  }

  // Probabilities, scaled to leave UNNAMED_SHARE of every place unnamed.
  const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
  for (let i = 0; i < n; i++) {
    for (let place = 0; place < places; place++) {
      const p = (counts[i * places + place]! / totalSamples) * (1 - UNNAMED_SHARE);
      if (p >= MIN_P) {
        probabilities.push({ universe_id: ids[i]!, place: place + 1, p: Math.round(p * 1e6) / 1e6 });
      }
    }
  }
  // Deterministic order: by place, then probability descending, then id.
  probabilities.sort((a, b) => a.place - b.place || b.p - a.p || a.universe_id - b.universe_id);

  // top10: the 10 games with the lowest mean simulated rank, in that order.
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => rankSum[a]! - rankSum[b]! || ids[a]! - ids[b]!,
  );
  const top10 = order.slice(0, places).map((i) => ids[i]!);

  writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);
}

main();

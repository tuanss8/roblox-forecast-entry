// roblox-up-and-coming-forecast-01 entry v3: Monte Carlo log-rank forecaster
// with a long-horizon momentum + growth model.
//
// v3 changes vs v2 (practice -33.19, official -33.2):
// - DRIFT_HOURS 40 -> 84 and SHRINK_K 4 -> 12: a longer drift window with
//   stronger shrinkage. Momentum is real but noisy; the long window +
//   heavy shrinkage keeps it honest.
// - GROWTH_BETA -0.5 -> -0.4.
// - HORIZONS "40,84": the Monte Carlo is averaged over a 40 h and an
//   84 h drift window (a horizon mixture), hedging the window choice.
// - TEMP 1.3: the output probabilities are sharpened (p^1.3 per place,
//   renormalised). The leaderboard shows the top entries carry a typical
//   probability per place near 0.055 while v2 carried 0.036; sharpening
//   closes part of that calibration gap. Pushing past ~1.35 stops helping.
// - PATHS 4000 -> 12000 per horizon: less Monte Carlo noise in the
//   forecast itself.
// Practice score: -32.20 (typical p 0.040, hits 7.4, agreement ~60),
// +0.99 over v2, confirmed across seeds 1-3 (spread 0.04).
// Negatives measured on the practice scorer and NOT used: per-game
// volatility (PGVOL, -32.84), UNNAMED_SHARE below 0.20, VOL below 0.12,
// per-place temperature, GROWTH_BETA -0.6 or stronger.
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

/** Monte Carlo paths per horizon. */
const PATHS = Number(process.env.PATHS ?? 12000);
/** Per-hour volatility of log(rank). */
const VOL_PER_HOUR = Number(process.env.VOL ?? 0.12);
/** Share of every place left unnamed, for games that join the chart during the window. */
const UNNAMED_SHARE = Number(process.env.UNNAMED ?? 0.2);
/** Smallest probability written out (below this a game is left to the unnamed share). */
const MIN_P = 0.001;
/** A game's drift regression uses reads within the last DRIFT_HOURS hours
 * (a time window, so it means the same in the sparse practice history and
 * the hourly final history). */
const DRIFT_HOURS = Number(process.env.DRIFT_HOURS ?? 84);
/** Extra drift windows to average over (comma-separated hours); the
 * default is just the DRIFT_HOURS window. Each horizon gets its own
 * Monte Carlo run and the place probabilities are averaged. */
const HORIZONS = (process.env.HORIZONS ?? '40,84')
  .split(',')
  .map((s) => Number(s.trim()))
  .filter((n) => Number.isFinite(n) && n > 0);
/** Shrinkage: a drift from n points is multiplied by n / (n + SHRINK_K). */
const SHRINK_K = Number(process.env.SHRINK_K ?? 12);
/** Cap on |drift| per hour, to keep noisy regressions from exploding. */
const MAX_DRIFT = Number(process.env.MAX_DRIFT ?? 0.1);
/** Global multiplier on all drifts (0 = no momentum, 1 = full). */
const DRIFT_SCALE = Number(process.env.DRIFT_SCALE ?? 0.5);
/** Growth-momentum coefficient: how much a game's recent player-count
 * growth rate (log playing per hour) shifts its log-rank drift. Negative:
 * a growing game climbs (its log-rank falls). */
const GROWTH_BETA = Number(process.env.GROWTH_BETA ?? -0.4);
/** Student-t degrees of freedom for the log-rank increments. 0 (or
 * unset) means Gaussian. */
const T_DOF = Number(process.env.T_DOF ?? 3);
/** Temperature on the output probabilities (1 = no change). <1 flattens,
 * >1 sharpens. Applied per place, renormalised to the same named total. */
const TEMP = Number(process.env.TEMP ?? 1.3);
/** Optional per-place temperature: TEMP_HI applies at place 1, TEMP_LO at
 * the last place, linearly interpolated. When both are set they override
 * TEMP. Use it to sharpen the top places more than the bottom ones. */
const TEMP_HI = Number(process.env.TEMP_HI ?? NaN);
const TEMP_LO = Number(process.env.TEMP_LO ?? NaN);
/** Per-game volatility: 1 = scale each game's vol by its own recent
 * log-rank change std (clamped to [0.5, 2] of the global vol). */
const PGVOL = Number(process.env.PGVOL ?? 0);
/** PRNG seed. */
const SEED = Number(process.env.SEED ?? 20261112);

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

/** Student-t generator (integer dof): T = Z / sqrt(Chi2_nu / nu), scaled
 * to unit variance by sqrt((nu-2)/nu). Falls back to the plain Gaussian
 * when dof <= 2. */
function makeStudentT(rng: () => number, dof: number): () => number {
  const randn = makeGaussian(rng);
  if (dof <= 2) return randn;
  const scale = Math.sqrt((dof - 2) / dof);
  return () => {
    const z = randn();
    let chi2 = 0;
    for (let k = 0; k < dof; k++) {
      const w = randn();
      chi2 += w * w;
    }
    return (z / Math.sqrt(chi2 / dof)) * scale;
  };
}

/** Shrunk drift for one game from its points inside a time window. */
function gameDrift(
  pts: Array<{ t: number; lr: number; lp: number }>,
  tLast: number,
  windowH: number,
): number {
  const w = pts.filter((p) => tLast - p.t <= windowH);
  if (w.length < 2) return 0;
  const n = w.length;
  let mt = 0;
  let ml = 0;
  for (const p of w) {
    mt += p.t;
    ml += p.lr;
  }
  mt /= n;
  ml /= n;
  let num = 0;
  let den = 0;
  for (const p of w) {
    num += (p.t - mt) * (p.lr - ml);
    den += (p.t - mt) * (p.t - mt);
  }
  let drift = den > 0 ? num / den : 0;
  drift *= n / (n + SHRINK_K);
  drift *= DRIFT_SCALE;
  if (GROWTH_BETA !== 0) {
    const gp = w.filter((p) => Number.isFinite(p.lp));
    if (gp.length >= 2) {
      let gmt = 0;
      let gmp = 0;
      for (const p of gp) {
        gmt += p.t;
        gmp += p.lp;
      }
      gmt /= gp.length;
      gmp /= gp.length;
      let gnum = 0;
      let gden = 0;
      for (const p of gp) {
        gnum += (p.t - gmt) * (p.lp - gmp);
        gden += (p.t - gmt) * (p.t - gmt);
      }
      if (gden > 0) drift += GROWTH_BETA * (gnum / gden);
    }
  }
  if (drift > MAX_DRIFT) drift = MAX_DRIFT;
  if (drift < -MAX_DRIFT) drift = -MAX_DRIFT;
  return drift;
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

  const last = snaps[snaps.length - 1]!;
  const tLast = (Date.parse(last.at) - t0) / 3600000;
  const games = last.up_and_coming.map((g) => ({
    id: g.universe_id,
    lr0: Math.log(g.rank),
    pts: perGame.get(g.universe_id) ?? [],
  }));
  const n = games.length;
  const ids = games.map((g) => g.id);
  const lr0 = new Float64Array(games.map((g) => g.lr0));

  // Global per-hour volatility of log-rank changes (for the PGVOL scale).
  let globalVol = VOL_PER_HOUR;
  if (PGVOL) {
    const diffs: number[] = [];
    for (const g of games) {
      const pts = g.pts;
      for (let i = 1; i < pts.length; i++) {
        const dt = pts[i]!.t - pts[i - 1]!.t;
        if (dt > 0) diffs.push((pts[i]!.lr - pts[i - 1]!.lr) / Math.sqrt(dt));
      }
    }
    if (diffs.length > 1) {
      const m = diffs.reduce((a, b) => a + b, 0) / diffs.length;
      const v = Math.sqrt(diffs.reduce((a, b) => a + (b - m) * (b - m), 0) / (diffs.length - 1));
      if (v > 0) globalVol = v;
    }
  }

  // Monte Carlo over each horizon; average the place counts.
  const counts = new Float64Array(n * places);
  const rankSum = new Float64Array(n);
  const totalSamples = PATHS * hours * HORIZONS.length;
  const cur = new Float64Array(n);
  const idx = new Array<number>(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  HORIZONS.forEach((windowH, hi) => {
    const drift = new Float64Array(games.map((g) => gameDrift(g.pts, tLast, windowH)));
    // Per-game vol multipliers.
    const vol = new Float64Array(n).fill(VOL_PER_HOUR);
    if (PGVOL) {
      for (let i = 0; i < n; i++) {
        const pts = games[i]!.pts.filter((p) => tLast - p.t <= windowH);
        const d: number[] = [];
        for (let j = 1; j < pts.length; j++) {
          const dt = pts[j]!.t - pts[j - 1]!.t;
          if (dt > 0) d.push((pts[j]!.lr - pts[j - 1]!.lr) / Math.sqrt(dt));
        }
        if (d.length >= 3) {
          const m = d.reduce((a, b) => a + b, 0) / d.length;
          const sd = Math.sqrt(d.reduce((a, b) => a + (b - m) * (b - m), 0) / (d.length - 1));
          let f = sd / globalVol;
          if (!Number.isFinite(f) || f <= 0) f = 1;
          if (f < 0.5) f = 0.5;
          if (f > 2) f = 2;
          vol[i] = VOL_PER_HOUR * f;
        }
      }
    }
    const rng = mulberry32(SEED + hi * 1000003);
    const randn = T_DOF > 0 ? makeStudentT(rng, T_DOF) : makeGaussian(rng);
    for (let path = 0; path < PATHS; path++) {
      cur.set(lr0);
      for (let h = 0; h < hours; h++) {
        for (let i = 0; i < n; i++) {
          cur[i] += drift[i] + vol[i] * randn();
        }
        idx.sort((a, b) => cur[a]! - cur[b]!);
        for (let place = 0; place < places; place++) {
          counts[idx[place]! * places + place] += 1;
        }
        for (let pos = 0; pos < n; pos++) rankSum[idx[pos]!] += pos + 1;
      }
    }
  });

  // Probabilities, scaled to leave UNNAMED_SHARE of every place unnamed.
  // Optional temperature: p^TEMP per place, renormalised to the same total.
  const raw: Array<Array<number>> = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let place = 0; place < places; place++) {
      row.push((counts[i * places + place]! / totalSamples) * (1 - UNNAMED_SHARE));
    }
    raw.push(row);
  }
  if (TEMP !== 1 || (Number.isFinite(TEMP_HI) && Number.isFinite(TEMP_LO))) {
    for (let place = 0; place < places; place++) {
      const t = Number.isFinite(TEMP_HI) && Number.isFinite(TEMP_LO)
        ? TEMP_HI + ((TEMP_LO - TEMP_HI) * place) / (places - 1)
        : TEMP;
      let tot = 0;
      let totT = 0;
      for (let i = 0; i < n; i++) {
        tot += raw[i]![place]!;
        totT += Math.pow(raw[i]![place]!, t);
      }
      if (totT > 0 && tot > 0) {
        for (let i = 0; i < n; i++) {
          raw[i]![place] = (Math.pow(raw[i]![place]!, t) / totT) * tot;
        }
      }
    }
  }
  const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
  for (let i = 0; i < n; i++) {
    for (let place = 0; place < places; place++) {
      const p = raw[i]![place]!;
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

// roblox-up-and-coming-forecast-01 entry v8: fitted place-occupancy table
// (refined) with a Monte Carlo fallback. Rank 1 is robekk "Fitted
// place-occupancy table" at -13.2 (p 0.2674), practically AT the practice
// oracle (-13.28, p 0.265); rank 2 is achmadb271 -13.7. Our v5 official
// is rank 4 at -26.0 (p 0.0745). The best Monte Carlo on practice
// (side-chat v6: HORIZONS 60,72,84, VOL 0.06, TEMP 1.35) scores -25.425
// (p 0.0787) and is the fallback here. v8 refines v7's fitted table
// (see FIT_SHRINK_K / FIT_PRIOR_SCALE / FIT_TEMP below). The refinements
// were measured on a rolling held-out harness over the practice history
// (fit on snapshots before a cut, score one forecast on the next 10
// reads): v7's raw table scores -54.7 there, v8's refined table -42.7
// (+12.0). The fitted table is used ONLY when the history is dense
// enough to support it (see DENSE_MIN_SNAPS / DENSE_MAX_GAP_H below);
// on sparse history (like the 34-snapshot practice history) it falls
// back to the Monte Carlo, which is verified at -25.425 on practice.
//
// WHY the refinements: a raw empirical occupancy table is a noisy
// estimate (each place has only ~10-70 weighted observations), and a
// noisy table is punished far more than a diffuse one (a confident
// wrong p costs ln(0.001)). v8 therefore (1) shrinks the empirical
// frequencies toward a persistence prior (Dirichlet: counts +
// FIT_SHRINK_K * prior, prior = exp(-|rank - place| /
// FIT_PRIOR_SCALE) over the games in the last snapshot, i.e. "games
// stay near their current rank", corr 0.94), which cuts variance while
// letting the counts dominate when the history is dense (the Dirichlet
// form adapts: prior share = k / (tot + k), small on dense data);
// (2) flattens the fitted probabilities slightly (FIT_TEMP 0.85) to
// hedge overconfidence; (3) leaves more unnamed mass at the bottom
// places (UNNAMED_LO 0.15), where turnover/newcomers are more common.
// The pure-prior limit is worse than the blend, confirming the counts
// still carry signal. robekk's -13.2 shows the dense-data table works;
// v8's table is the same idea with a better estimator, aiming to match
// robekk and, if the official oracle is a touch above the practice
// oracle, to reach the user target of log -13 (p 0.2725).
//
// User target: log -13 (p_place 0.2725), 0.2 better than robekk (-13.2)
// and 0.28 above the practice oracle (-13.28). On practice data -13 is
// NOT reachable (above the oracle); the fitted branch's only hope is
// that the dense final history lets it recover the true conditional
// distribution the way robekk's did, and that the refined estimator is
// a touch closer to the truth than robekk's. See the status file for
// the honest assessment.
//
// Deterministic: a seeded PRNG (mulberry32) drives the Monte Carlo, so
// two runs on the same history write byte-identical forecasts. The
// fitted table is a deterministic function of the history. No network,
// no model calls; only the history in /input is read.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

/** Use the fitted occupancy table only when the history holds at least
 * this many snapshots (dense, hourly-style history). */
const DENSE_MIN_SNAPS = Number(process.env.DENSE_MIN_SNAPS ?? 48);
/** ...and the median gap between consecutive snapshots is at most this
 * many hours. Practice history: 34 snaps, median gap ~6 h -> Monte Carlo. */
const DENSE_MAX_GAP_H = Number(process.env.DENSE_MAX_GAP_H ?? 2);
/** Fitted table: occupancy window (hours back from the cut). */
const FIT_WINDOW_H = Number(process.env.FIT_WINDOW_H ?? 72);
/** Fitted table: recency half-life (hours) for the occupancy weights.
 * 12 h (v8; v7 used 24 h): shorter halflife favours recency, which the
 * held-out harness prefers at every horizon. */
const FIT_HALFLIFE_H = Number(process.env.FIT_HALFLIFE_H ?? 12);
/** Fitted table: Dirichlet shrinkage strength toward the persistence
 * prior (see fittedTable). 20 (v8): on sparse history the prior
 * dominates (good, counts are noise there); on dense history the
 * counts dominate more (prior share = k / (tot + k)). Held-out
 * optimum is ~30, but 20 keeps the counts more influential on dense
 * data for a ~0.1 held-out cost. */
const FIT_SHRINK_K = Number(process.env.FIT_SHRINK_K ?? 20);
/** Fitted table: persistence-prior scale (ranks). Prior weight for a
 * game is exp(-|rank - place| / scale); 4.0 (v8) spreads the prior
 * over nearby ranks (games shuffle within the top 10). */
const FIT_PRIOR_SCALE = Number(process.env.FIT_PRIOR_SCALE ?? 4);
/** Fitted table: temperature on the fitted probabilities per place
 * (<1 flattens). 0.85 (v8): mild hedge against overconfident fitted
 * frequencies. */
const FIT_TEMP = Number(process.env.FIT_TEMP ?? 0.85);
/** Fitted table: its own unnamed share (the Monte Carlo keeps
 * UNNAMED_HI/LO). The fitted table leaves more unnamed mass at the
 * bottom places (0.25/0.15), where turnover/newcomers are more common;
 * the held-out harness prefers it. */
const FIT_UNNAMED_HI = Number(process.env.FIT_UNNAMED_HI ?? 0.25);
const FIT_UNNAMED_LO = Number(process.env.FIT_UNNAMED_LO ?? 0.15);
/** Force the fitted table even on sparse history (testing only). */
const FORCE_FIT = Number(process.env.FORCE_FIT ?? 0);

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Monte Carlo paths per horizon. */
const PATHS = Number(process.env.PATHS ?? 12000);
/** Per-hour volatility of log(rank). */
const VOL_PER_HOUR = Number(process.env.VOL ?? 0.06);
/** Share of every place left unnamed, for games that join the chart during the window. */
const UNNAMED_SHARE = Number(process.env.UNNAMED ?? 0.2);
/** Optional per-place unnamed share: UNNAMED_HI at place 1, UNNAMED_LO at
 * the last place, linearly interpolated. When both are set they override
 * UNNAMED_SHARE. Newcomers take the top places more often than the
 * middle ones, so the top may want more unnamed mass than the bottom. */
const UNNAMED_HI = Number(process.env.UNNAMED_HI ?? 0.25);
const UNNAMED_LO = Number(process.env.UNNAMED_LO ?? 0.05);
/** Smallest probability written out (below this a game is left to the unnamed share). */
const MIN_P = 0.001;
/** A game's drift regression uses reads within the last DRIFT_HOURS hours
 * (a time window, so it means the same in the sparse practice history and
 * the hourly final history). */
const DRIFT_HOURS = Number(process.env.DRIFT_HOURS ?? 84);
/** Extra drift windows to average over (comma-separated hours); the
 * default is just the DRIFT_HOURS window. Each horizon gets its own
 * Monte Carlo run and the place probabilities are averaged. */
const HORIZONS = (process.env.HORIZONS ?? '60,72,84')
  .split(',')
  .map((s) => Number(s.trim()))
  .filter((n) => Number.isFinite(n) && n > 0);
/** Shrinkage: a drift from n points is multiplied by n / (n + SHRINK_K). */
const SHRINK_K = Number(process.env.SHRINK_K ?? 2);
/** Cap on |drift| per hour, to keep noisy regressions from exploding. */
const MAX_DRIFT = Number(process.env.MAX_DRIFT ?? 0.1);
/** Conditional drift cap for fast climbers: when a game's growth-momentum
 * term is more negative than -GROWTH_CAP_THRESH (it is climbing fast),
 * the drift cap is CAP_HI instead of MAX_DRIFT, so a genuine viral
 * climber is not held back by the same cap as a noisy regression. */
const CAP_HI = Number(process.env.CAP_HI ?? MAX_DRIFT);
const GROWTH_CAP_THRESH = Number(process.env.GROWTH_CAP_THRESH ?? 0.03);
/** Global multiplier on all drifts (0 = no momentum, 1 = full). */
const DRIFT_SCALE = Number(process.env.DRIFT_SCALE ?? 1.2);
/** Growth-momentum coefficient: how much a game's recent player-count
 * growth rate (log playing per hour) shifts its log-rank drift. Negative:
 * a growing game climbs (its log-rank falls). */
const GROWTH_BETA = Number(process.env.GROWTH_BETA ?? -0.4);
/** Growth window: the player-count growth slope is measured over the last
 * GROWTH_HOURS hours only, separate from the rank-drift window. The
 * rank drift is stable over a long window, but the growth that drives
 * explosive climbs (a game's playing multiplying over a day) is a
 * recent signal — a long window dilutes it. v3 used the same window
 * for both and missed the biggest climber (Sweep The Moon, #29 ->
 * #1-3, whose playing went 3.7k -> 45k in the last ~2 days). */
const GROWTH_HOURS = Number(process.env.GROWTH_HOURS ?? 8);
/** Student-t degrees of freedom for the log-rank increments. 0 (or
 * unset) means Gaussian. */
const T_DOF = Number(process.env.T_DOF ?? 3);
/** Temperature on the output probabilities (1 = no change). <1 flattens,
 * >1 sharpens. Applied per place, renormalised to the same named total. */
const TEMP = Number(process.env.TEMP ?? 1.35);
/** Optional per-place temperature: TEMP_HI applies at place 1, TEMP_LO at
 * the last place, linearly interpolated. When both are set they override
 * TEMP. Use it to sharpen the top places more than the bottom ones. */
const TEMP_HI = Number(process.env.TEMP_HI ?? NaN);
const TEMP_LO = Number(process.env.TEMP_LO ?? NaN);
/** Per-game volatility: 1 = scale each game's vol by its own recent
 * log-rank change std (clamped to [0.5, 2] of the global vol). */
const PGVOL = Number(process.env.PGVOL ?? 0);
/** Mean reversion: the chart's top games tend to fall back and its
 * bottom games tend to climb (measured: rank 1-3 mean next dlogr
 * +0.0205/h, rank 30-100 mean -0.0027/h). A drift of -MR_BETA *
 * (logRank - meanLogRank) pulls every game toward the chart's mean
 * log-rank. 0 disables it. */
const MR_BETA = Number(process.env.MR_BETA ?? 0);
/** Tenure effect: games new to the chart tend to climb and older games
 * tend to fall (measured: tenure <24 h mean next dlogr -0.0065/h,
 * tenure 72-168 h mean +0.0050/h). A drift of TENURE_BETA *
 * (tenureHours - 72) / 72, with tenure measured from the game's first
 * appearance in the history. 0 disables it. */
const TENURE_BETA = Number(process.env.TENURE_BETA ?? 0);
/** Breakout: a game's player-count growth slope over the short window
 * (GROWTH_HOURS) minus its slope over a longer baseline window
 * (BREAKOUT_BASE_HOURS). When growth ACCELERATES (recent slope far above
 * baseline), the game is breaking out — add extra negative drift
 * (uncapped, applied after the ordinary drift cap) so a genuine viral
 * climber is not held back. 0 disables it. */
const BREAKOUT_BETA = Number(process.env.BREAKOUT_BETA ?? 0.2);
const BREAKOUT_BASE_HOURS = Number(process.env.BREAKOUT_BASE_HOURS ?? 84);

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
  lr0: number,
  meanLr: number,
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
  let growthDrift = 0;
  let recentGrowthSlope = 0;
  if (GROWTH_BETA !== 0 || BREAKOUT_BETA !== 0) {
    const gp = pts.filter((p) => tLast - p.t <= GROWTH_HOURS && Number.isFinite(p.lp));
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
      if (gden > 0) {
        recentGrowthSlope = gnum / gden;
        growthDrift = GROWTH_BETA * recentGrowthSlope;
      }
    }
  }
  drift += growthDrift;
  // Mean reversion toward the chart's mean log-rank.
  if (MR_BETA !== 0) drift += -MR_BETA * (lr0 - meanLr);
  // Tenure: new games climb, old games fall.
  if (TENURE_BETA !== 0 && pts.length > 0) {
    const tenureH = tLast - pts[0]!.t;
    drift += (TENURE_BETA * (tenureH - 72)) / 72;
  }
  // Fast climbers get a higher cap: a game whose player count is surging
  // can genuinely climb faster than the ordinary cap allows.
  const cap = growthDrift < -GROWTH_CAP_THRESH ? CAP_HI : MAX_DRIFT;
  if (drift > cap) drift = cap;
  if (drift < -cap) drift = -cap;
  // Breakout (after the cap, uncapped): growth accelerating vs the game's
  // own longer-window baseline means it is breaking out right now.
  if (BREAKOUT_BETA !== 0 && recentGrowthSlope !== 0) {
    const bp = pts.filter(
      (p) => tLast - p.t <= BREAKOUT_BASE_HOURS && Number.isFinite(p.lp),
    );
    if (bp.length >= 3) {
      let bmt = 0;
      let bmp = 0;
      for (const p of bp) {
        bmt += p.t;
        bmp += p.lp;
      }
      bmt /= bp.length;
      bmp /= bp.length;
      let bnum = 0;
      let bden = 0;
      for (const p of bp) {
        bnum += (p.t - bmt) * (p.lp - bmp);
        bden += (p.t - bmt) * (p.t - bmt);
      }
      if (bden > 0) {
        const accel = recentGrowthSlope - bnum / bden;
        if (accel > 0) drift += -BREAKOUT_BETA * accel;
      }
    }
  }
  return drift;
}

/** Fitted place-occupancy table (v8, refined): P(game occupies place
 * k) estimated straight from the history, shrunk toward a persistence
 * prior. For each place k, the recency-weighted frequency of each game
 * at that place over the last FIT_WINDOW_H hours (weight
 * 0.5^((tLast - t) / FIT_HALFLIFE_H)) is combined Dirichlet-style with
 * a prior: posterior = (counts + FIT_SHRINK_K * prior) / (tot +
 * FIT_SHRINK_K), where the prior spreads mass over the games in the
 * last snapshot as exp(-|rank - k| / FIT_PRIOR_SCALE) ("games stay near
 * their current rank", corr 0.94). The prior cuts the variance of the
 * noisy frequencies (the main failure mode on sparse data) while the
 * counts still dominate when the history is dense. The fitted
 * probabilities are then flattened slightly (FIT_TEMP) to hedge
 * overconfidence, scaled to leave the per-place unnamed share, and
 * top10 is the modal game per place. This is the approach the current
 * rank 1 ("Fitted place-occupancy table", -13.2) uses, with a better
 * estimator: it estimates the conditional distribution directly
 * instead of deriving it by simulating log-rank paths. It needs a dense
 * history to work -- with only a handful of recent snapshots the
 * frequencies are noise (measured on practice: raw table -46 to -53;
 * refined table still below the Monte Carlo on sparse data, see the
 * status file). */
function fittedTable(snaps: Snapshot[], places: number): {
  probabilities: Array<{ universe_id: number; place: number; p: number }>;
  top10: number[];
} {
  const tLast = Date.parse(snaps[snaps.length - 1]!.at);
  const cnt: Array<Map<number, number>> = [];
  const tot: number[] = [];
  for (let k = 0; k < places; k++) {
    cnt.push(new Map());
    tot.push(0);
  }
  for (const s of snaps) {
    const ageH = (tLast - Date.parse(s.at)) / 3600000;
    if (ageH > FIT_WINDOW_H) continue;
    const w = Math.pow(0.5, ageH / FIT_HALFLIFE_H);
    for (const g of s.up_and_coming) {
      if (g.rank < 1 || g.rank > places) continue;
      const m = cnt[g.rank - 1]!;
      m.set(g.universe_id, (m.get(g.universe_id) ?? 0) + w);
      tot[g.rank - 1]! += w;
    }
  }
  // Persistence prior from the last snapshot's ranks.
  const lastRank = new Map<number, number>();
  for (const g of snaps[snaps.length - 1]!.up_and_coming) {
    lastRank.set(g.universe_id, g.rank);
  }
  const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
  // Per-place sorted posteriors, kept for the distinct top10 below.
  const perPlace: Array<Array<[number, number]>> = [];
  for (let place = 0; place < places; place++) {
    const un = FIT_UNNAMED_HI + ((FIT_UNNAMED_LO - FIT_UNNAMED_HI) * place) / (places - 1);
    // Prior: exp(-|rank - place| / scale), normalised over last snapshot.
    const prior = new Map<number, number>();
    let priorSum = 0;
    for (const [uid, r] of lastRank) {
      const w = Math.exp(-Math.abs(r - (place + 1)) / FIT_PRIOR_SCALE);
      prior.set(uid, w);
      priorSum += w;
    }
    // Posterior (Dirichlet shrinkage) then temperature.
    const denom = tot[place]! + FIT_SHRINK_K;
    const post = new Map<number, number>();
    for (const [uid, c] of cnt[place]!) {
      post.set(uid, (c + FIT_SHRINK_K * (prior.get(uid) ?? 0) / priorSum) / denom);
    }
    if (FIT_SHRINK_K > 0) {
      for (const [uid, w] of prior) {
        if (!post.has(uid)) post.set(uid, (FIT_SHRINK_K * w / priorSum) / denom);
      }
    }
    let entries = [...post.entries()];
    if (FIT_TEMP !== 1) {
      const totP = entries.reduce((a, [, v]) => a + v, 0);
      const totT = entries.reduce((a, [, v]) => a + Math.pow(v, FIT_TEMP), 0);
      if (totT > 0 && totP > 0) {
        entries = entries.map(([uid, v]) => [uid, (Math.pow(v, FIT_TEMP) / totT) * totP] as [number, number]);
      }
    }
    entries.sort((a, b) => b[1] - a[1]);
    perPlace.push(entries);
    for (const [uid, p0] of entries) {
      const p = p0 * (1 - un);
      if (p >= MIN_P) {
        probabilities.push({ universe_id: uid, place: place + 1, p: Math.round(p * 1e6) / 1e6 });
      }
    }
  }
  // top10 must be 10 DISTINCT games: the same game can be the modal
  // game at two places (v7 had this bug -- a duplicate top10 wastes a
  // slot and may fail server validation). Greedy by place: at each
  // place take the highest-posterior game not already used; fall back
  // to the best unused game anywhere if a place is exhausted.
  const top10: number[] = [];
  const used = new Set<number>();
  for (let place = 0; place < places; place++) {
    let pick = 0;
    for (const [uid] of perPlace[place]!) {
      if (!used.has(uid)) {
        pick = uid;
        break;
      }
    }
    if (pick === 0) {
      outer: for (let p2 = 0; p2 < places; p2++) {
        for (const [uid] of perPlace[p2]!) {
          if (!used.has(uid)) {
            pick = uid;
            break outer;
          }
        }
      }
    }
    if (pick !== 0) {
      used.add(pick);
      top10.push(pick);
    }
  }
  probabilities.sort((a, b) => a.place - b.place || b.p - a.p || a.universe_id - b.universe_id);
  return { probabilities, top10 };
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

  // Density gate: use the fitted occupancy table only on a dense history.
  // Median gap between consecutive snapshots decides.
  {
    const times = snaps.map((s) => Date.parse(s.at)).sort((a, b) => a - b);
    const gaps: number[] = [];
    for (let i = 1; i < times.length; i++) gaps.push((times[i]! - times[i - 1]!) / 3600000);
    gaps.sort((a, b) => a - b);
    const medGap = gaps.length > 0 ? gaps[Math.floor(gaps.length / 2)]! : Infinity;
    if (FORCE_FIT || (snaps.length >= DENSE_MIN_SNAPS && medGap <= DENSE_MAX_GAP_H)) {
      const { probabilities, top10 } = fittedTable(snaps, places);
      writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);
      return;
    }
  }

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
  const meanLr = lr0.reduce((a, b) => a + b, 0) / n;

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
    const drift = new Float64Array(
      games.map((g) => gameDrift(g.pts, tLast, windowH, g.lr0, meanLr)),
    );
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

  // Probabilities, scaled to leave the unnamed share of every place
  // unnamed. Optional temperature: p^TEMP per place, renormalised to the
  // same total.
  const raw: Array<Array<number>> = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let place = 0; place < places; place++) {
      const un =
        Number.isFinite(UNNAMED_HI) && Number.isFinite(UNNAMED_LO)
          ? UNNAMED_HI + ((UNNAMED_LO - UNNAMED_HI) * place) / (places - 1)
          : UNNAMED_SHARE;
      row.push((counts[i * places + place]! / totalSamples) * (1 - un));
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

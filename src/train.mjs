// quilt-nn — training, receipts, verification.
//
// Every epoch of train() appends one entry to a sha256-linked chain (the pattern the
// fleet established in quilt-neighbourhood):
//
//   { v, seq, epoch, loss, loss_sha, weight_root_sha, prev, sha }
//
//   loss            the epoch's mean loss (pre-step), a plain JS number for humans
//   loss_sha        sha256 over the portable scalar preimage of that number:
//                   "f64|8|<8-byte BE IEEE-754 hex>" (receipt format v2, quilt-attention#1)
//   weight_root_sha sha256 over every weight's canonical bytes, in graph order
//   prev / sha      the hash-chain links; sha = sha256 of the canonical entry JSON
//
// The chain is the training log's tamper-evidence: verify(chain) recomputes every
// link and every commitment. Honest limit, stated once: a bare hash chain cannot
// detect truncation of its TAIL without an externally anchored tip — same as any
// hash chain, including git. Everything else — edited losses, edited weight roots,
// edited links, reordered or renumbered entries — is detected.
//
// ZERO dependencies, ESM, Node >= 18.

import { f64hex, sha256hex, lossShaOf, lcg, GENESIS, loadNet, evaluateForward, evaluateBackward, applyTick } from './cellgraph.mjs';

export { GENESIS };

// ── seeded initialization ────────────────────────────────────────────────────────
// One LCG, weights consumed in graph order: value = (2u − 1) · scale.
// Same seed → same initial weight bytes (test NC2); the data makers below draw from
// the same PRNG family so init + data are reproducible from seeds alone.
// `stream` lets train() share one LCG across init and shuffling (still seed-only).
export function initEnv(net, seed, stream) {
  const r = stream || lcg(seed);
  const env = {};
  for (const c of net.graph.cells) env[c.id] = 0;
  for (const w of net.weightCells) env[w.id] = (2 * r.next() - 1) * w.params.scale;
  return env;
}

// ── canonical receipts ───────────────────────────────────────────────────────────
export function weightRootSha(net, env) {
  let buf = '';
  for (const w of net.weightIds) buf += `${w}\n${f64hex(env[w])}\n`;
  return sha256hex(buf);
}

function canonEntry(e) { // fixed key order — the canonical bytes of a receipt
  return JSON.stringify({ v: e.v, seq: e.seq, epoch: e.epoch, loss: e.loss, loss_sha: e.loss_sha, weight_root_sha: e.weight_root_sha, prev: e.prev });
}

// ── verification ─────────────────────────────────────────────────────────────────
export function verifyWhy(chain) {
  if (!Array.isArray(chain)) return { ok: false, at: 0, reason: 'chain is not an array' };
  if (chain.length === 0) return { ok: true, n: 0 }; // an empty chain trivially verifies
  let prev = GENESIS;
  for (let i = 0; i < chain.length; i++) {
    const e = chain[i];
    if (typeof e !== 'object' || e === null) return { ok: false, at: i, reason: 'entry is not an object' };
    if (e.v !== 2) return { ok: false, at: i, reason: `receipt format v${e.v} unsupported (v2 = portable scalar preimage "f64|8|<hex>"; v1 receipts predate the quilt-attention#1 preimage fix and verify only under the v0.1.0 code)` };
    if (e.seq !== i + 1) return { ok: false, at: i, reason: `seq is ${e.seq}, expected ${i + 1}` };
    if (!Number.isInteger(e.epoch) || e.epoch !== i) return { ok: false, at: i, reason: `epoch is ${e.epoch}, expected ${i} (epochs are contiguous from 0)` };
    if (typeof e.loss !== 'number' || !Number.isFinite(e.loss)) return { ok: false, at: i, reason: `loss ${e.loss} is not a finite number` };
    if (e.prev !== prev) return { ok: false, at: i, reason: `prev-link broken: ${e.prev} ≠ ${prev}` };
    if (e.loss_sha !== lossShaOf(e.loss)) return { ok: false, at: i, reason: `loss_sha does not match the canonical bytes of loss ${e.loss}` };
    if (!/^[0-9a-f]{64}$/.test(e.loss_sha) || !/^[0-9a-f]{64}$/.test(e.weight_root_sha)) return { ok: false, at: i, reason: 'hash fields must be 64 lowercase hex chars' };
    const expect = sha256hex(canonEntry(e));
    if (e.sha !== expect) return { ok: false, at: i, reason: 'entry sha mismatch (fields edited after sealing)' };
    prev = e.sha;
  }
  return { ok: true, n: chain.length };
}

export function verify(chain) {
  return verifyWhy(chain).ok;
}

// ── one epoch ────────────────────────────────────────────────────────────────────
// Full-batch gradient descent: per sample a forward evaluation and a backward
// composition; grad cells accumulate the SUM, then hold the MEAN; the tick cell
// applies weight -= lr * grad once. Loss reported is the pre-step mean.
export function trainEpoch(net, env, data, lr) {
  if (!Array.isArray(data) || data.length === 0) throw new TypeError('data must be a non-empty array of {x, y} samples');
  const acc = new Map(net.weightIds.map((w) => [w, 0]));
  let total = 0;
  for (const s of data) {
    evaluateForward(net, env, s);
    total += env[net.lossId];
    const g = evaluateBackward(net, env);
    for (const [id, v] of g) if (acc.has(id)) acc.set(id, acc.get(id) + v);
  }
  const n = data.length;
  for (const gc of net.gradCells) env[gc.id] = acc.get(gc.params.of) / n;
  const update = applyTick(net, env, lr);
  const loss = total / n;
  return { loss, update };
}

// train(net, {data, epochs, lr, seed, mode}) → { metrics, chain, tip, env, finalLoss }
//
// mode 'sgd' (default): ONE TICK PER SAMPLE — the tick cell performs one true SGD
//   step (weight -= lr · grad) for each sample in a seeded, shuffled pass. The epoch
//   receipt carries the mean of the PRE-STEP losses measured along the pass, and the
//   weight root AFTER the last tick of the epoch. Stochasticity is what breaks the
//   symmetry plateaus that stall full-batch descent on regression targets; the
//   shuffle order comes from the same LCG as the init, so it is deterministic.
// mode 'batch': one tick per epoch on the mean gradient (trainEpoch above).
//
// Deterministic end to end for a fixed seed: LCG init + shuffle, listed-order
// aggregation, IEEE-754 arithmetic, canonical hashing. Two runs, one byte stream
// (test T2).
export function train(netOrGraph, { data, epochs, lr, seed, mode = 'sgd' } = {}) {
  const net = netOrGraph.graph ? netOrGraph : loadNet(netOrGraph);
  if (!Array.isArray(data) || data.length === 0) throw new TypeError('train: data must be a non-empty array of {x, y} samples');
  if (!Number.isInteger(epochs) || epochs < 0) throw new TypeError('train: epochs must be a non-negative integer');
  if (!(typeof lr === 'number' && Number.isFinite(lr) && lr > 0)) throw new TypeError('train: lr must be a positive finite number');
  if (!Number.isInteger(seed)) throw new TypeError('train: seed must be an integer');
  if (mode !== 'sgd' && mode !== 'batch') throw new TypeError(`train: mode must be 'sgd' or 'batch', got "${mode}"`);

  const r = lcg(seed);
  const env = initEnv(net, seed, r); // continues the SAME LCG stream as the shuffle below
  const chain = [];
  const metrics = [];
  let prev = GENESIS;

  const indices = data.map((_, i) => i);
  for (let epoch = 0; epoch < epochs; epoch++) {
    let loss;
    if (mode === 'batch') {
      loss = trainEpoch(net, env, data, lr).loss;
    } else {
      // Fisher–Yates over the sample order, driven by the training LCG
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(r.next() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      let total = 0;
      for (const idx of indices) {
        const s = data[idx];
        evaluateForward(net, env, s);
        total += env[net.lossId]; // pre-step: what this tick's gradient was taken at
        const g = evaluateBackward(net, env);
        for (const gc of net.gradCells) env[gc.id] = g.get(gc.params.of) || 0;
        applyTick(net, env, lr);
      }
      loss = total / data.length;
    }

    const entry = {
      v: 2,
      seq: epoch + 1,
      epoch,
      loss,
      loss_sha: lossShaOf(loss),
      weight_root_sha: weightRootSha(net, env),
      prev,
    };
    entry.sha = sha256hex(canonEntry(entry));
    chain.push(entry);
    metrics.push({ epoch, loss });
    prev = entry.sha;
  }
  return { metrics, chain, tip: prev, env, finalLoss: chain.length ? chain[chain.length - 1].loss : undefined };
}

// ── data makers (LCG — init and data from seeds, per the spec) ───────────────────
export function makeXorData() {
  return [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 },
  ];
}

export function makeSineData({ n = 64, seed = 7 } = {}) {
  const r = lcg(seed);
  const out = [];
  for (let i = 0; i < n; i++) {
    const x = -Math.PI + 2 * Math.PI * r.next();
    out.push({ x: [x], y: Math.sin(x) });
  }
  return out;
}

// mean analytic dLoss/d(weight) over the dataset — the quantity the tick applies.
// Exposed for the gradient check (test T3): analytic vs numeric finite difference.
export function meanGradient(net, env, data, weightId) {
  if (!net.weightIds.includes(weightId)) throw new Error(`meanGradient: ${weightId} is not a weight cell`);
  let sum = 0;
  for (const s of data) {
    evaluateForward(net, env, s);
    const g = evaluateBackward(net, env);
    sum += g.get(weightId) || 0;
  }
  return sum / data.length;
}

// mean loss over the dataset for a given env — used by the numeric side of T3.
export function meanLoss(net, env, data) {
  let sum = 0;
  for (const s of data) {
    evaluateForward(net, env, s);
    sum += env[net.lossId];
  }
  return sum / data.length;
}

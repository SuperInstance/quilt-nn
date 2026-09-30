// test/train.test.mjs — the receipted training suite.
//   T1  XOR reaches loss < 0.05 within 3000 epochs at lr 0.5, seed 42 (deterministic)
//   T2  byte-determinism: two runs, same seed → identical receipt chains (root hash equal)
//   T3  gradient check: analytic vs numeric finite difference within 1e-5 on 20 random weights
//   T4  chain tamper detection
//   NC1 order-independence: two topological orders → bit-identical outputs
//   NC2 seeded-init reproducibility
//
// Run: node --test test/*.test.mjs   (from the repo root; zero dependencies)

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { loadNet, evaluateForward, evaluateBackward, evaluateGraph, lcg, f64hex, topoOrder, buildMLP, exportSnapshot, loadSnapshot } from '../src/cellgraph.mjs';
import { train, initEnv, verify, verifyWhy, weightRootSha, makeXorData, makeSineData, meanGradient, meanLoss } from '../src/train.mjs';

const xorGraph = () => JSON.parse(readFileSync(new URL('../nets/xor.json', import.meta.url), 'utf8'));
const sineGraph = () => JSON.parse(readFileSync(new URL('../nets/sine.json', import.meta.url), 'utf8'));
const xorData = makeXorData();

// ────────────────────────────────────────────────────────────────────────────────
test('T1: XOR 2-4-1 loss < 0.05 within 3000 epochs at lr 0.5, seed 42', () => {
  const net = loadNet(xorGraph());
  const t0 = Date.now();
  const r = train(net, { data: xorData, epochs: 3000, lr: 0.5, seed: 42 });
  const ms = Date.now() - t0;

  assert.equal(r.chain.length, 3000);
  assert.ok(r.finalLoss < 0.05, `final loss ${r.finalLoss} not < 0.05`);
  assert.equal(r.tip, r.chain[2999].sha);

  // the receipt chain itself must verify, and the predictions must be right
  assert.equal(verify(r.chain), true);
  for (const s of xorData) {
    const env = { ...r.env };
    evaluateForward(net, env, s);
    assert.ok(Math.abs(env['pred'] - s.y) < 0.1, `pred ${env['pred']} too far from ${s.y} for ${JSON.stringify(s.x)}`);
  }
  console.log(`    T1 receipt: loss=${r.finalLoss.toExponential(3)} in ${ms}ms, chain tip=${r.tip}`);
  // same seed again → same tip (determinism where it matters most: the headline test)
  const r2 = train(loadNet(xorGraph()), { data: xorData, epochs: 3000, lr: 0.5, seed: 42 });
  assert.equal(r2.tip, r.tip);
});

test('T1b: sine 1-16-1 regression trains to loss < 0.01 (3000 epochs, lr 0.1, seed 42)', () => {
  const net = loadNet(sineGraph());
  const data = makeSineData({ n: 64, seed: 7 });
  const t0 = Date.now();
  const r = train(net, { data, epochs: 3000, lr: 0.1, seed: 42 });
  const ms = Date.now() - t0;

  assert.equal(r.chain.length, 3000);
  assert.ok(r.finalLoss < 0.01, `final loss ${r.finalLoss} not < 0.01`);
  assert.equal(verify(r.chain), true);

  // held-in-domain spot checks: the net must reproduce sin(x) at unseen points
  const env = r.env;
  for (const x of [-2.5, -1.2, 0.4, 1.7, 2.9]) {
    evaluateForward(net, env, { x: [x], y: Math.sin(x) });
    assert.ok(Math.abs(env['pred'] - Math.sin(x)) < 0.15, `pred ${env['pred'].toFixed(3)} vs sin ${Math.sin(x).toFixed(3)} at x=${x}`);
  }
  console.log(`    T1b receipt: loss=${r.finalLoss.toExponential(3)} in ${ms}ms, chain tip=${r.tip}`);
});

test('T2: byte-determinism — same seed, identical receipt chains and root hash', () => {
  const epochs = 3000;
  const a = train(loadNet(xorGraph()), { data: xorData, epochs, lr: 0.5, seed: 42 });
  const b = train(loadNet(xorGraph()), { data: xorData, epochs, lr: 0.5, seed: 42 });
  assert.equal(a.tip, b.tip);
  assert.equal(JSON.stringify(a.chain), JSON.stringify(b.chain), 'chains differ byte-wise');

  // a different seed → a different chain (the receipt is of THIS run, not any run)
  const c = train(loadNet(xorGraph()), { data: xorData, epochs, lr: 0.5, seed: 43 });
  assert.notEqual(c.tip, a.tip);

  // and the bigger 1-16-1 sine graph agrees byte-for-byte too
  const s1 = train(loadNet(sineGraph()), { data: makeSineData({ n: 64, seed: 7 }), epochs: 120, lr: 0.1, seed: 42 });
  const s2 = train(loadNet(sineGraph()), { data: makeSineData({ n: 64, seed: 7 }), epochs: 120, lr: 0.1, seed: 42 });
  assert.equal(s1.tip, s2.tip);
  assert.equal(JSON.stringify(s1.chain), JSON.stringify(s2.chain));
});

test('T3: analytic gradients match numeric finite differences within 1e-5 (20 random weights)', () => {
  const net = loadNet(sineGraph()); // tanh everywhere: smooth, no kink artifacts
  const data = makeSineData({ n: 64, seed: 7 });
  const env = initEnv(net, 42);
  const h = 1e-6;

  // 20 distinct weights chosen by the seeded LCG, in shuffled order
  const r = lcg(1337);
  const ids = [...net.weightIds];
  for (let i = ids.length - 1; i > 0; i--) {
    const j = Math.floor(r.next() * (i + 1));
    [ids[i], ids[j]] = [ids[j], ids[i]];
  }
  const picked = ids.slice(0, 20);

  let worst = 0; let worstId = '';
  for (const w of picked) {
    const analytic = meanGradient(net, env, data, w);
    const save = env[w];
    env[w] = save + h; const lp = meanLoss(net, env, data);
    env[w] = save - h; const lm = meanLoss(net, env, data);
    env[w] = save;
    const numeric = (lp - lm) / (2 * h);
    const err = Math.abs(analytic - numeric);
    if (err > worst) { worst = err; worstId = w; }
    assert.ok(err < 1e-5, `weight ${w}: |analytic ${analytic} − numeric ${numeric}| = ${err} ≥ 1e-5`);
  }
  console.log(`    T3 receipt: 20 weights checked, worst |Δ| = ${worst.toExponential(3)} @ ${worstId}`);
});

test('T3b: the trained net still gradient-checks (gradients stay honest after 500 epochs)', () => {
  const net = loadNet(xorGraph());
  const { env } = train(net, { data: xorData, epochs: 500, lr: 0.5, seed: 42 });
  const h = 1e-6;
  const r = lcg(4242);
  const ids = [...net.weightIds];
  for (let i = ids.length - 1; i > 0; i--) {
    const j = Math.floor(r.next() * (i + 1));
    [ids[i], ids[j]] = [ids[j], ids[i]];
  }
  for (const w of ids.slice(0, 20)) {
    const analytic = meanGradient(net, env, xorData, w);
    const save = env[w];
    env[w] = save + h; const lp = meanLoss(net, env, xorData);
    env[w] = save - h; const lm = meanLoss(net, env, xorData);
    env[w] = save;
    const numeric = (lp - lm) / (2 * h);
    assert.ok(Math.abs(analytic - numeric) < 1e-5, `weight ${w}: analytic ${analytic} vs numeric ${numeric}`);
  }
});

test('T4: verify(chain) detects tampering', () => {
  const clean = train(loadNet(xorGraph()), { data: xorData, epochs: 50, lr: 0.5, seed: 42 }).chain;
  assert.equal(verify(clean), true);
  assert.deepEqual(verifyWhy(clean), { ok: true, n: 50 });

  const re = () => JSON.parse(JSON.stringify(clean)); // deep copy

  // 1. edited loss (the human-readable number)
  let c = re(); c[10].loss = 0.001;
  assert.equal(verify(c), false, 'edited loss not detected');

  // 2. edited loss_sha commitment
  c = re(); c[10].loss_sha = 'f'.repeat(64);
  assert.equal(verify(c), false, 'edited loss_sha not detected');

  // 3. edited weight root
  c = re(); c[20].weight_root_sha = '0'.repeat(64);
  assert.equal(verify(c), false, 'edited weight_root_sha not detected');

  // 4. edited entry hash (breaks this entry AND the next prev-link)
  c = re(); c[5].sha = 'a'.repeat(64);
  assert.equal(verify(c), false, 'edited sha not detected');

  // 5. renumbered seq
  c = re(); c[7].seq = 99;
  assert.equal(verify(c), false, 'renumbered seq not detected');

  // 6. reordered entries
  c = re(); [c[3], c[4]] = [c[4], c[3]];
  assert.equal(verify(c), false, 'reordering not detected');

  // 7. dropped middle entry
  c = re(); c.splice(12, 1);
  assert.equal(verify(c), false, 'dropped entry not detected');

  // 8. edited epoch
  c = re(); c[2].epoch = 5;
  assert.equal(verify(c), false, 'edited epoch not detected');

  // verifyWhy names the breach
  c = re(); c[11].loss = 42;
  const why = verifyWhy(c);
  assert.equal(why.ok, false);
  assert.equal(why.at, 11);
  assert.match(why.reason, /loss_sha|sha mismatch/);

  // honest limits: empty chain trivially verifies; non-array does not
  assert.equal(verify([]), true);
  assert.equal(verify({}), false);
});

test('NC1: evaluating the graph in two topological orders gives identical outputs', () => {
  for (const graph of [xorGraph(), sineGraph()]) {
    const net = loadNet(graph);
    const o1 = topoOrder(graph, 'insertion');
    const o2 = topoOrder(graph, 'kahn-lifo');
    const o3 = topoOrder(graph, 'kahn-fifo');

    // sanity: the alternate orders really are different sequences
    assert.notDeepEqual(o1, o2, 'kahn-lifo should differ from insertion order');
    assert.notDeepEqual(o1, o3, 'kahn-fifo should differ from insertion order');

    const sample = net.graph.cells.length > 100 ? { x: [1.3], y: 0.2 } : { x: [0.3, 0.8], y: 0.5 };
    const envA = initEnv(net, 7);
    const envB = initEnv(net, 7);
    const envC = initEnv(net, 7);
    evaluateForward(net, envA, sample, o1);
    evaluateForward(net, envB, sample, o2);
    evaluateForward(net, envC, sample, o3);

    // bit-identical, cell for cell (canonical bytes, not a float tolerance)
    for (const c of net.forwardCells) {
      const ha = f64hex(envA[c.id]); const hb = f64hex(envB[c.id]); const hc = f64hex(envC[c.id]);
      assert.equal(ha, hb, `cell ${c.id} differs between insertion and kahn-lifo orders`);
      assert.equal(ha, hc, `cell ${c.id} differs between insertion and kahn-fifo orders`);
    }
  }
});

test('NC2: seeded init is reproducible — same seed, same bytes; different seed, different bytes', () => {
  for (const graph of [xorGraph(), sineGraph()]) {
    const net = loadNet(graph);
    const a = initEnv(net, 42);
    const b = initEnv(net, 42);
    const c = initEnv(net, 43);
    for (const w of net.weightIds) {
      assert.equal(f64hex(a[w]), f64hex(b[w]), `same seed produced different bytes for ${w}`);
      assert.notEqual(f64hex(a[w]), f64hex(c[w]), `different seed produced identical bytes for ${w} (astronomically unlikely)`);
    }
    assert.equal(weightRootSha(net, a), weightRootSha(net, b));
    assert.notEqual(weightRootSha(net, a), weightRootSha(net, c));
  }
});

// ── supporting behavior the docs promise ────────────────────────────────────────
test('the tick cell performs exactly weight -= lr * grad (single-sample evaluateGraph)', () => {
  const net = loadNet(xorGraph());
  const env = initEnv(net, 42);
  const before = { ...env };
  const { loss, update } = evaluateGraph(net, env, xorData[1], 0.5);
  assert.equal(typeof loss, 'number');

  // independently recompute the step the tick should have applied
  const env2 = initEnv(net, 42);
  evaluateForward(net, env2, xorData[1]);
  const g = evaluateBackward(net, env2);
  let expectTotal = 0;
  for (const w of net.weightIds) {
    const grad = g.get(w) || 0;
    const expect = before[w] - 0.5 * grad;
    assert.equal(f64hex(env[w]), f64hex(expect), `tick applied a wrong update to ${w}`);
    expectTotal += Math.abs(0.5 * grad);
  }
  assert.equal(f64hex(env['tick']), f64hex(expectTotal), 'tick cell accounting wrong');
  assert.ok(update > 0, 'tick reported zero total update');
});

test('snapshot export → JSON round-trip → import reproduces the trained bytes', () => {
  const net = loadNet(xorGraph());
  const r = train(net, { data: xorData, epochs: 30, lr: 0.5, seed: 42 });
  const snap = exportSnapshot(net, r.env);
  const snap2 = JSON.parse(JSON.stringify(snap)); // the JSON trip must be lossless
  const { net: net2, env: env2 } = loadSnapshot(snap2);
  assert.equal(weightRootSha(net2, env2), weightRootSha(net, r.env));
  const s = xorData[3];
  const a = { ...r.env }; const b = { ...env2 };
  evaluateForward(net, a, s);
  evaluateForward(net2, b, s);
  assert.equal(f64hex(a['pred']), f64hex(b['pred']));
});

test('receipt entries carry exactly the promised fields, linked from genesis', () => {
  const chain = train(loadNet(xorGraph()), { data: xorData, epochs: 3, lr: 0.5, seed: 42 }).chain;
  const KEYS = ['v', 'seq', 'epoch', 'loss', 'loss_sha', 'weight_root_sha', 'prev', 'sha'];
  let prev = '0'.repeat(64);
  chain.forEach((e, i) => {
    assert.deepEqual(Object.keys(e), KEYS);
    assert.equal(e.v, 2);
    assert.equal(e.seq, i + 1);
    assert.equal(e.epoch, i);
    assert.equal(e.prev, prev);
    assert.match(e.sha, /^[0-9a-f]{64}$/);
    assert.match(e.loss_sha, /^[0-9a-f]{64}$/);
    prev = e.sha;
  });
});

test('relu is implemented and its analytic derivative is honest away from the kink', () => {
  const g = buildMLP({ sizes: [1, 3, 1], hiddenAct: 'relu', outAct: null });
  const net = loadNet(g);
  const env = initEnv(net, 5);
  const data = [{ x: [1.0], y: 0.5 }, { x: [2.0], y: 0.25 }]; // positive pre-activations: off the kink
  const h = 1e-6;
  const r = lcg(3);
  const ids = [...net.weightIds];
  for (let i = ids.length - 1; i > 0; i--) {
    const j = Math.floor(r.next() * (i + 1));
    [ids[i], ids[j]] = [ids[j], ids[i]];
  }
  for (const w of ids.slice(0, 10)) {
    const analytic = meanGradient(net, env, data, w);
    const save = env[w];
    env[w] = save + h; const lp = meanLoss(net, env, data);
    env[w] = save - h; const lm = meanLoss(net, env, data);
    env[w] = save;
    assert.ok(Math.abs(analytic - (lp - lm) / (2 * h)) < 1e-5, `relu net weight ${w} gradient mismatch`);
  }
});

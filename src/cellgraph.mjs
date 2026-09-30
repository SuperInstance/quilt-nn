// quilt-nn — neural networks as quilt cell graphs, trained for real.
//
// LINEAGE
//   SuperInstance/cellgraph      showed a forward pass AS a cell graph: typed cells,
//                                named inputs, insertion order = topological order.
//   SuperInstance/micrograd-quilt brought scalar autograd (Value + backward closures).
//   quilt-nn                      TRAINS such graphs: the weights, the gradients and the
//                                SGD step are themselves cells in the sheet. Every epoch
//                                is receipted into a sha256-linked chain.
//
// THE GRAPH IS DATA
//   { cells: [ { id, kind, inputs: [cellId...], params } ] }
//   Edges are implied by `inputs`. No hidden state, no closures, no classes in the file
//   format: a net is a plain JSON document you can diff, sign, or ship.
//
// CELL KINDS
//   input    value bound per sample (params.slot=k) or the label (params.target=true)
//   weight   seeded scalar parameter (params.scale sets the init range)
//   sum      Σ inputs, added in the LISTED order (order-independent evaluation)
//   product  Π inputs
//   act      elementwise activation, params.fn ∈ {relu, tanh, sigmoid}
//   loss     (pred − target)² — the per-sample squared error of an mse reduction
//   grad     analytic dLoss/d(weight) for params.of, composed through the graph
//   tick     one SGD step: for each grad input g, weight(of g) -= lr * g
//
// ZERO dependencies, ESM, Node >= 18.

import { createHash } from 'node:crypto';

export const CELL_KINDS = ['input', 'weight', 'sum', 'product', 'act', 'loss', 'grad', 'tick'];
export const ACTS = ['relu', 'tanh', 'sigmoid'];

export const GENESIS = '0'.repeat(64); // prev of the first receipt, like quilt-neighbourhood

export const DTYPE_TAG = 'f64'; // receipts hash the dtype, not assume it (rule 2, like quilt-attention)

// ── hashing ──────────────────────────────────────────────────────────────────────
// Two representations, two jobs (the lesson cellgraph learned the hard way):
//   * f64hex  — the CANONICAL BYTES of a float. 8 bytes, big-endian IEEE-754. This is
//     what we hash so that -0 vs 0 and every last ulp is visible to the receipt.
//     (Normalized: -0 hashes as +0, because JSON round-trips collapse them and the
//     chain must survive a JSON trip.)
//   * sha256hex — tamper-evidence over canonical strings.
const _dv = new DataView(new ArrayBuffer(8));

export function f64hex(x) {
  if (typeof x !== 'number' || !Number.isFinite(x)) throw new TypeError(`f64hex needs a finite number, got ${x}`);
  if (Object.is(x, -0)) x = 0;
  _dv.setFloat64(0, x, false); // big-endian
  let s = '';
  for (let i = 0; i < 8; i++) s += _dv.getUint8(i).toString(16).padStart(2, '0');
  return s;
}

export function hexToF64(hex) {
  if (!/^[0-9a-f]{16}$/.test(hex)) throw new TypeError(`bad f64 hex: ${hex}`);
  for (let i = 0; i < 8; i++) _dv.setUint8(i, parseInt(hex.slice(i * 2, i * 2 + 2), 16));
  return _dv.getFloat64(0, false);
}

export function sha256hex(str) {
  return createHash('sha256').update(str, 'utf8').digest('hex');
}

// loss_sha: sha256 over the canonical bytes of the loss scalar.
// Preimage = `${DTYPE_TAG}|8|${f64hex(loss)}` — the legible tagged string ("8" =
// the byte width of one IEEE-754 double), portable by construction. NOT the v0.1.0
// form, which hashed Buffer.from(hex,'hex').toString('latin1'): sha256hex hashes
// UTF-8, so every canonical byte >= 0x80 re-encoded into TWO bytes — byte-exact
// inside Node, but a different preimage than every other runtime's reading of
// "sha256 over the canonical bytes", i.e. not portable at all. Reported and
// root-caused in quilt-attention#1 (the same line shipped as scalarSha there);
// both repos now share this preimage, and the Python reference in
// test/conformance.py pins it cross-runtime (test/scalar-fixture.json).
export function lossShaOf(loss) {
  return sha256hex(`${DTYPE_TAG}|8|${f64hex(loss)}`);
}

// ── seeded PRNG ──────────────────────────────────────────────────────────────────
// LCG (Numerical Recipes): s = (1664525·s + 1013904223) mod 2^32. Chosen because the
// spec asks for an LCG and because it is portable arithmetic — no engine RNG, no
// crypto entropy: same seed, same bytes, forever.
export function lcg(seed) {
  let s = seed >>> 0;
  return {
    next() {
      s = (Math.imul(1664525, s) + 1013904223) >>> 0;
      return s / 4294967296;
    },
    get state() { return s; },
  };
}

// ── activations (value + analytic derivative in terms of the OUTPUT) ─────────────
export function actValue(fn, x) {
  switch (fn) {
    case 'relu':    return x > 0 ? x : 0;
    case 'tanh':    return Math.tanh(x);
    case 'sigmoid': return 1 / (1 + Math.exp(-x));
    default: throw new Error(`unknown act fn "${fn}"`);
  }
}

export function actDeriv(fn, out) { // derivative wrt the INPUT, expressed via the OUTPUT value
  switch (fn) {
    case 'relu':    return out > 0 ? 1 : 0;
    case 'tanh':    return 1 - out * out;
    case 'sigmoid': return out * (1 - out);
    default: throw new Error(`unknown act fn "${fn}"`);
  }
}

// ── validation ───────────────────────────────────────────────────────────────────
// A graph must be a DAG whose cells reference only EARLIER cells: the array order is
// itself a topological order (the cellgraph insight). That is what makes the file
// format honest — you can read it top to bottom and nothing is computed before its
// inputs exist.
export function validateGraph(graph) {
  if (!graph || !Array.isArray(graph.cells)) throw new TypeError('graph must be {cells: [...]}');
  const seen = new Set();
  for (const c of graph.cells) {
    if (!c || typeof c.id !== 'string' || !c.id) throw new TypeError(`cell without id: ${JSON.stringify(c)}`);
    if (seen.has(c.id)) throw new Error(`duplicate cell id ${c.id}`);
    if (!CELL_KINDS.includes(c.kind)) throw new Error(`cell ${c.id}: unknown kind "${c.kind}"`);
    if (!Array.isArray(c.inputs)) throw new Error(`cell ${c.id}: inputs must be an array`);
    for (const dep of c.inputs) {
      if (!seen.has(dep)) throw new Error(`cell ${c.id} inputs ${dep}: not defined earlier (graph is not in topological order)`);
    }
    seen.add(c.id);
  }
  for (const c of graph.cells) {
    if (c.kind === 'act' && !ACTS.includes(c.params?.fn)) throw new Error(`act cell ${c.id}: params.fn must be one of ${ACTS}`);
    if (c.kind === 'weight') {
      const sc = c.params?.scale;
      if (!(typeof sc === 'number' && Number.isFinite(sc) && sc > 0)) throw new Error(`weight cell ${c.id}: params.scale must be a positive finite number`);
    }
  }
  return graph;
}

// ── topological orders ───────────────────────────────────────────────────────────
// 'insertion'  the order the cells are written in (always valid, always used in training)
// 'kahn-fifo'  Kahn's algorithm, queue
// 'kahn-lifo'  Kahn's algorithm, stack — a genuinely different but equally valid order
// The evaluation semantics never depend on the order: every cell reads its inputs in
// the LISTED order of its own inputs[] array. That is what makes order-independence
// (test NC1) a property of the design, not an accident of traversal.
export function topoOrder(graph, variant = 'insertion') {
  validateGraph(graph);
  const cells = graph.cells;
  if (variant === 'insertion') return cells.map((c) => c.id);
  if (variant !== 'kahn-fifo' && variant !== 'kahn-lifo') throw new Error(`unknown topo variant "${variant}"`);
  const adj = new Map(cells.map((c) => [c.id, []]));
  const indeg = new Map(cells.map((c) => [c.id, c.inputs.length]));
  for (const c of cells) for (const dep of c.inputs) adj.get(dep).push(c.id);
  const ready = cells.filter((c) => indeg.get(c.id) === 0).map((c) => c.id);
  const out = [];
  while (ready.length) {
    const id = variant === 'kahn-lifo' ? ready.pop() : ready.shift();
    out.push(id);
    for (const d of adj.get(id)) {
      indeg.set(d, indeg.get(d) - 1);
      if (indeg.get(d) === 0) ready.push(d);
    }
  }
  if (out.length !== cells.length) throw new Error('graph contains a cycle');
  return out;
}

// ── the net: a loaded graph ──────────────────────────────────────────────────────
export function loadNet(graph) {
  validateGraph(graph);
  const byId = new Map(graph.cells.map((c) => [c.id, c]));
  const isForward = (c) => ['input', 'weight', 'sum', 'product', 'act', 'loss'].includes(c.kind);
  const net = {
    graph,
    byId,
    forwardCells: graph.cells.filter(isForward),
    inputCells: graph.cells.filter((c) => c.kind === 'input'),
    featureCells: graph.cells.filter((c) => c.kind === 'input' && !c.params.target),
    targetCell: graph.cells.find((c) => c.kind === 'input' && c.params.target) || null,
    weightCells: graph.cells.filter((c) => c.kind === 'weight'),
    weightIds: graph.cells.filter((c) => c.kind === 'weight').map((c) => c.id),
    lossCells: graph.cells.filter((c) => c.kind === 'loss'),
    gradCells: graph.cells.filter((c) => c.kind === 'grad'),
    tickCells: graph.cells.filter((c) => c.kind === 'tick'),
  };
  if (net.lossCells.length !== 1) throw new Error(`net needs exactly one loss cell, found ${net.lossCells.length}`);
  if (net.tickCells.length > 1) throw new Error(`net needs at most one tick cell, found ${net.tickCells.length}`);
  for (const g of net.gradCells) {
    const of = g.params?.of;
    if (!of || !byId.has(of) || byId.get(of).kind !== 'weight') throw new Error(`grad cell ${g.id}: params.of must name a weight cell`);
  }
  for (const t of net.tickCells) {
    for (const gi of t.inputs) {
      if (!byId.has(gi) || byId.get(gi).kind !== 'grad') throw new Error(`tick cell ${t.id}: input ${gi} is not a grad cell`);
    }
  }
  net.lossId = net.lossCells[0].id;
  return net;
}

// ── forward evaluation ───────────────────────────────────────────────────────────
// env is a plain {cellId: number} map. Weights PERSIST in env across evaluations;
// activations are recomputed. sample = {x: [...], y: number}.
// `order` defaults to insertion; any topoOrder variant gives bit-identical results.
export function evaluateForward(net, env, sample, order) {
  const cells = net.byId;
  const seq = order || net.graph.cells.map((c) => c.id);
  // bind the sample to input cells first, wherever they sit in the order
  net.featureCells.forEach((c, k) => {
    const v = sample.x?.[c.params.slot ?? k];
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new TypeError(`sample x[${c.params.slot ?? k}] must be a finite number`);
    env[c.id] = v;
  });
  if (net.targetCell) {
    if (typeof sample.y !== 'number' || !Number.isFinite(sample.y)) throw new TypeError('sample y must be a finite number');
    env[net.targetCell.id] = sample.y;
  }
  for (const id of seq) {
    const c = cells.get(id);
    switch (c.kind) {
      case 'input':
      case 'weight':
      case 'grad': // grad cells are set by the backward composition / training loop
      case 'tick':
        if (c.kind === 'weight' && typeof env[id] !== 'number') throw new Error(`weight ${id} not initialized — call initEnv first`);
        break;
      case 'sum': {
        let acc = 0;
        for (const dep of c.inputs) acc += env[dep]; // listed order: order-independent by construction
        env[id] = acc;
        break;
      }
      case 'product': {
        let acc = 1;
        for (const dep of c.inputs) acc *= env[dep];
        env[id] = acc;
        break;
      }
      case 'act':
        env[id] = actValue(c.params.fn, env[c.inputs[0]]);
        break;
      case 'loss': {
        const [pred, target] = c.inputs;
        const d = env[pred] - env[target];
        env[id] = d * d; // per-sample squared error; the epoch loss is the mean over samples
        break;
      }
      default:
        throw new Error(`forward cannot evaluate kind "${c.kind}" (${id})`);
    }
    if (!Number.isFinite(env[id])) throw new Error(`cell ${id} evaluated to a non-finite value (divergence or bad data)`);
  }
  return env;
}

// ── backward: analytic local gradients composed through the graph ────────────────
// Reverse-mode over the SAME cells, no separate autograd tape: the local derivative
// of each cell kind is a property of the kind. Returns Map(cellId -> dLoss/dcell).
// Products distribute gradients via prefix/suffix products — exact, no division, so
// a zero input never poisons a sibling's gradient.
export function evaluateBackward(net, env) {
  const grad = new Map();
  const bump = (id, g) => grad.set(id, (grad.get(id) || 0) + g);
  grad.set(net.lossId, 1);
  const cells = net.forwardCells;
  for (let i = cells.length - 1; i >= 0; i--) {
    const c = cells[i];
    const g = grad.get(c.id);
    if (g === undefined) continue;
    switch (c.kind) {
      case 'loss': {
        const [pred, target] = c.inputs;
        const d = 2 * (env[pred] - env[target]);
        bump(pred, d);
        bump(target, -d);
        break;
      }
      case 'sum':
        for (const dep of c.inputs) bump(dep, g);
        break;
      case 'product': {
        const k = c.inputs.length;
        const pre = new Array(k); const suf = new Array(k);
        let p = 1; for (let j = 0; j < k; j++) { pre[j] = p; p *= env[c.inputs[j]]; }
        p = 1; for (let j = k - 1; j >= 0; j--) { suf[j] = p; p *= env[c.inputs[j]]; }
        for (let j = 0; j < k; j++) bump(c.inputs[j], g * pre[j] * suf[j]);
        break;
      }
      case 'act':
        bump(c.inputs[0], g * actDeriv(c.params.fn, env[c.id]));
        break;
      default: // input, weight: terminals — a grad cell may read their entry
        break;
    }
  }
  return grad;
}

// ── the tick cell: one SGD step ──────────────────────────────────────────────────
// weight -= lr * grad, for every grad cell the tick lists as input. Mutates env.
// Output value = Σ|Δw| (a diagnostic the metrics keep).
export function applyTick(net, env, lr) {
  if (net.tickCells.length === 0) throw new Error('net has no tick cell');
  const t = net.tickCells[0];
  if (!(typeof lr === 'number' && Number.isFinite(lr) && lr > 0)) throw new TypeError(`lr must be a positive finite number, got ${lr}`);
  let total = 0;
  for (const gi of t.inputs) {
    const gCell = net.byId.get(gi);
    const w = gCell.params.of;
    const g = env[gi];
    if (typeof g !== 'number' || !Number.isFinite(g)) throw new Error(`grad cell ${gi} has no value — run backward first`);
    const delta = lr * g;
    env[w] -= delta;
    total += Math.abs(delta);
  }
  env[t.id] = total;
  return total;
}

// Evaluate the WHOLE graph for one sample, including grad cells and the tick:
// forward → loss → analytic gradients → one SGD step. (train() uses the batched
// version — see train.mjs trainEpoch — but this is the honest one-sample path.)
export function evaluateGraph(net, env, sample, lr) {
  evaluateForward(net, env, sample);
  const g = evaluateBackward(net, env);
  for (const gc of net.gradCells) env[gc.id] = g.get(gc.params.of) || 0;
  const update = applyTick(net, env, lr);
  return { loss: env[net.lossId], update };
}

// ── builders ─────────────────────────────────────────────────────────────────────
// buildMLP({sizes:[2,4,1], hiddenAct:'tanh'}) — the output layer is linear unless
// outAct is given. Xavier-uniform scale per weight: sqrt(6/(fanIn+fanOut)).
export function buildMLP({ sizes, hiddenAct = 'tanh', outAct = null }) {
  if (!Array.isArray(sizes) || sizes.length < 2 || !sizes.every((n) => Number.isInteger(n) && n > 0)) {
    throw new TypeError('sizes must be [in, ...hidden, out] of positive integers');
  }
  const cells = [];
  for (let k = 0; k < sizes[0]; k++) cells.push({ id: `in${k}`, kind: 'input', inputs: [], params: { slot: k } });
  cells.push({ id: 'target', kind: 'input', inputs: [], params: { target: true } });

  let prevIds = [];
  for (const c of cells) if (c.kind === 'input' && !c.params.target) prevIds.push(c.id);

  const weightIds = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const fanIn = sizes[l]; const fanOut = sizes[l + 1];
    const scale = Math.sqrt(6 / (fanIn + fanOut));
    const isOut = l === sizes.length - 2;
    const act = isOut ? outAct : hiddenAct;
    const outIds = [];
    for (let j = 0; j < fanOut; j++) {
      const wIds = [];
      for (let k = 0; k < fanIn; k++) {
        const w = `w${l}_${j}_${k}`;
        cells.push({ id: w, kind: 'weight', inputs: [], params: { scale } });
        weightIds.push(w); wIds.push(w);
      }
      const b = `b${l}_${j}`;
      cells.push({ id: b, kind: 'weight', inputs: [], params: { scale } });
      weightIds.push(b);
      const pIds = [];
      for (let k = 0; k < fanIn; k++) {
        const p = `p${l}_${j}_${k}`;
        cells.push({ id: p, kind: 'product', inputs: [prevIds[k], wIds[k]], params: {} });
        pIds.push(p);
      }
      const sumId = isOut && !act ? 'pred' : `s${l}_${j}`;
      cells.push({ id: sumId, kind: 'sum', inputs: [...pIds, b], params: {} });
      if (act) {
        const aId = isOut ? 'pred' : `a${l}_${j}`;
        cells.push({ id: aId, kind: 'act', inputs: [sumId], params: { fn: act } });
        outIds.push(aId);
      } else {
        outIds.push(sumId);
      }
    }
    prevIds = outIds;
  }
  cells.push({ id: 'loss', kind: 'loss', inputs: ['pred', 'target'], params: {} });
  const gradIds = [];
  for (const w of weightIds) {
    const gid = `g_${w}`;
    cells.push({ id: gid, kind: 'grad', inputs: ['loss'], params: { of: w } });
    gradIds.push(gid);
  }
  cells.push({ id: 'tick', kind: 'tick', inputs: gradIds, params: { lr: 0.5 } });
  return { cells };
}

// ── export / import a trained net ────────────────────────────────────────────────
// Weights → canonical JSON snapshot: f64 bytes as hex, so the snapshot survives any
// JSON trip byte-exactly. See TEMPLATE.md ("export a trained net").
export function exportSnapshot(net, env) {
  const weights = {};
  for (const w of net.weightIds) weights[w] = f64hex(env[w]);
  return { format: 'quilt-nn-snapshot@1', cells: net.graph.cells, weights };
}

export function loadSnapshot(snap) {
  if (!snap || snap.format !== 'quilt-nn-snapshot@1' || !Array.isArray(snap.cells) || typeof snap.weights !== 'object') {
    throw new TypeError('not a quilt-nn-snapshot@1 document');
  }
  const net = loadNet({ cells: snap.cells });
  const env = {};
  for (const id of net.graph.cells.map((c) => c.id)) env[id] = 0;
  for (const w of net.weightIds) {
    if (!(w in snap.weights)) throw new Error(`snapshot missing weight ${w}`);
    env[w] = hexToF64(snap.weights[w]);
  }
  return { net, env };
}

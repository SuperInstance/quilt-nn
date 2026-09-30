# TEMPLATE.md — quilt-nn in 5 minutes

Everything is cells; the graph is data. Two edits cover 90% of what you want to do.

---

## 1. Add a layer (≈2 minutes)

A layer is *more cells*, nothing else. Generate a wider/deeper graph with the
builder and write it out:

```js
// scripts/my-net.mjs
import { writeFileSync } from 'node:fs';
import { buildMLP, loadNet } from '../src/cellgraph.mjs';

// sizes = [inputs, ...hidden layers..., outputs]
const g = buildMLP({ sizes: [2, 8, 8, 1], hiddenAct: 'tanh', outAct: 'sigmoid' });
loadNet(g);                       // validate: acyclic, topological, kinds known
writeFileSync('nets/my-net.json', JSON.stringify(g, null, 2) + '\n');
```

```sh
node scripts/my-net.mjs
```

That is a 2-8-8-1 MLP. Every weight got a Xavier-uniform scale
(`sqrt(6 / (fanIn + fanOut))`), every weight got its own `grad` cell, and one
`tick` cell wires all the gradients. Train it:

```js
import { loadNet } from '../src/cellgraph.mjs';
import { train, verify } from '../src/train.mjs';
import { readFileSync } from 'node:fs';

const net = loadNet(JSON.parse(readFileSync('nets/my-net.json', 'utf8')));
const r = train(net, { data, epochs: 2000, lr: 0.1, seed: 42 });
console.log(r.finalLoss, verify(r.chain), r.tip);
```

Rules the validator will enforce (and you should not fight):

- every cell's `inputs` may only reference **earlier** cells — the array order is
  the topological order;
- exactly one `loss` cell, at most one `tick` cell;
- each `grad` cell's `params.of` must name a `weight` cell;
- each `tick` cell's inputs must be `grad` cells.

## 2. Swap an activation (≈1 minute)

Change `params.fn` on the `act` cells:

```js
const g = buildMLP({ sizes: [2, 8, 1], hiddenAct: 'relu', outAct: 'sigmoid' });
```

or hand-edit the JSON — every `"kind": "act"` cell carries its own
`"params": { "fn": "relu" | "tanh" | "sigmoid" }`. Mixed layers are legal (edit
individual cells); the backward pass reads each cell's own `fn`.

Two field notes, learned the hard way:

- **sigmoid outputs cannot emit negative targets** — for regression onto `[−1,1]`
  (sine), use `tanh` or a linear output, or the net will plateau at the sigmoid's
  floor and the receipts will faithfully record the plateau;
- **relu has a kink at 0**: fine for gradients almost always, but if a numeric
  gradient check disagrees near zero, that is the kink, not a bug.

Learning rate interacts with saturation: tanh hidden layers at lr 0.5 can hard-
saturate into dead step functions (sine: 0.236 plateau). Drop to 0.05–0.1 and the
units stay alive (sine: 8e-4).

## 3. Hand-wire a weird graph (the point of all this)

Because the graph is data, unusual wiring is just JSON. Two nets sharing a weight,
a skip connection, a gate — write the cells:

```json
{ "id": "skip", "kind": "product", "inputs": ["a0_1", "gate_w"], "params": {} },
{ "id": "wide", "kind": "sum", "inputs": ["a1_0", "skip"], "params": {} }
```

…as long as earlier-cells-only holds. The training loop, grad cells, and receipts
do not care what shape the DAG is.

## 4. Export a trained net (≈1 minute)

Weights → canonical JSON snapshot. Floats are stored as their 8-byte big-endian
IEEE-754 hex, so the snapshot survives any JSON trip **byte-exactly** (no
`parseFloat` drift, `-0` normalized):

```js
import { exportSnapshot, loadSnapshot } from '../src/cellgraph.mjs';
import { train, weightRootSha } from '../src/train.mjs';

const r = train(net, { data, epochs: 3000, lr: 0.5, seed: 42 });

const snap = exportSnapshot(net, r.env);
// { "format": "quilt-nn-snapshot@1",
//   "cells": [ ...the whole graph, data... ],
//   "weights": { "w0_0_0": "bf9a1e2c...", "b0_0": "4009...", ... } }

writeFileSync('snapshots/my-net.json', JSON.stringify(snap, null, 2) + '\n');

// later, or on another machine:
const { net: net2, env } = loadSnapshot(JSON.parse(readFileSync('snapshots/my-net.json', 'utf8')));
weightRootSha(net2, env);  // === weightRootSha(net, r.env): proof the bytes survived
```

A snapshot is just the graph plus named weight bytes — diffable, signable,
committable. To attest *how* the weights were produced, keep the receipt chain:
the last entry's `weight_root_sha` equals `weightRootSha` of the exported
snapshot, and the chain's `tip` pins the whole training history.

## 5. Verify someone's training log

```js
import { verify, verifyWhy } from '../src/train.mjs';

verify(chain);                    // true / false
verifyWhy(chain);                 // { ok: false, at: 41, reason: 'prev-link broken: …' }
```

Detected: edited `loss`, edited `loss_sha`, edited `weight_root_sha`, edited
`sha`, renumbered `seq`, edited `epoch`, reordered entries, dropped entries.
Not detected: truncating the chain's tail (anchor the tip externally — e.g. in a
release — if that matters to you).

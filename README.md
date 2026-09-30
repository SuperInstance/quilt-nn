# quilt-nn

**Neural networks as quilt cell graphs — trained for real, receipted every epoch.**

A neural net here is *literally a sheet of cells*: the weights are cells, the
activations are cells, the gradients are cells, and the SGD step is a cell. The
graph is data — a plain JSON document — and training it produces a sha256-linked
receipt chain: one entry per epoch, every entry committing to the loss bytes and
the full weight state. Zero dependencies. ESM. Node >= 18.

```
{ "cells": [
  { "id": "in0",      "kind": "input",   "inputs": [],              "params": { "slot": 0 } },
  { "id": "w0_0_0",   "kind": "weight",  "inputs": [],              "params": { "scale": 0.7071 } },
  { "id": "p0_0_0",   "kind": "product", "inputs": ["in0", "w0_0_0"], "params": {} },
  { "id": "s0_0",     "kind": "sum",     "inputs": ["p0_0_0", "b0_0"], "params": {} },
  { "id": "a0_0",     "kind": "act",     "inputs": ["s0_0"],        "params": { "fn": "tanh" } },
  { "id": "loss",     "kind": "loss",    "inputs": ["pred", "target"], "params": {} },
  { "id": "g_w0_0_0", "kind": "grad",    "inputs": ["loss"],        "params": { "of": "w0_0_0" } },
  { "id": "tick",     "kind": "tick",    "inputs": ["g_w0_0_0", "..."], "params": { "lr": 0.5 } }
] }
```

Edges are implied by `inputs`. The array order is a topological order — you can
read the file top to bottom and nothing is computed before its inputs exist.

## Lineage

- [SuperInstance/cellgraph](https://github.com/SuperInstance/cellgraph) showed a
  transformer forward pass **as** a quilt cell graph: typed cells, named inputs,
  insertion order = topological order, a witness at every cell boundary. It proved
  forward-pass-as-cells; its cells had no training loop.
- [SuperInstance/micrograd-quilt](https://github.com/SuperInstance/micrograd-quilt)
  brought scalar autograd: `Value` nodes with backward closures, the chain rule
  composed through a DAG.
- **quilt-nn** closes the loop: the cellgraph shape gets a backward pass and an
  optimizer, and everything micrograd kept implicit (weights, gradients, the
  update) becomes an explicit cell in the sheet. Per-epoch receipts follow the
  hash-chain pattern the fleet established in quilt-neighbourhood.

## Cell kinds

| kind      | semantics                                                                 |
|-----------|---------------------------------------------------------------------------|
| `input`   | value bound per sample (`params.slot = k`) or the label (`params.target`) |
| `weight`  | seeded scalar parameter; `params.scale` sets the init range (LCG, below)  |
| `sum`     | Σ inputs, **added in the listed order**                                    |
| `product` | Π inputs                                                                   |
| `act`     | `params.fn` ∈ `relu` \| `tanh` \| `sigmoid`                                |
| `loss`    | `(pred − target)²` — the per-sample squared error of an mse reduction      |
| `grad`    | analytic dLoss/dweight for `params.of`, composed through the graph        |
| `tick`    | one SGD step: for each grad input, `weight -= lr · grad`; outputs Σ\|Δw\|  |

The `grad` cell is not a numeric-difference hack: it is the analytic local
derivative of each cell kind, composed in reverse topological order over the same
graph the forward pass used. No autograd tape, no second data structure.

## Determinism, stated precisely

Same seed → same bytes, end to end:

- init is an LCG (Numerical Recipes, mod 2³²) consumed in graph order;
- SGD shuffles sample order per epoch from the **same** seeded LCG stream;
- every cell aggregates its inputs in the **listed** order of its own `inputs[]`,
  so evaluation in *any* topological order is bit-identical (test NC1);
- receipts hash canonical bytes: 8-byte big-endian IEEE-754 per float, fixed key
  order per entry. Two runs of `train(...)` produce byte-identical chains
  (test T2).

## Receipt chain

Every epoch appends:

```json
{
  "v": 1, "seq": 1, "epoch": 0,
  "loss": 0.7147,
  "loss_sha": "<sha256 of the canonical 8 bytes of loss>",
  "weight_root_sha": "<sha256 over every weight's canonical bytes, graph order>",
  "prev": "000...0",
  "sha": "<sha256 of the canonical entry JSON>"
}
```

`verify(chain)` recomputes every commitment and link: edited losses, edited
weight roots, edited hashes, renumbered or reordered or dropped entries are all
detected. Honest limit, stated once: a bare hash chain cannot detect truncation
of its tail without an externally anchored tip — same as git.

## Usage

```js
import { loadNet } from 'quilt-nn';
import { train, verify, makeXorData, makeSineData } from 'quilt-nn/train';

const xor = loadNet(JSON.parse(readFileSync('nets/xor.json', 'utf8')));
const { chain, tip, finalLoss, env, metrics } =
  train(xor, { data: makeXorData(), epochs: 3000, lr: 0.5, seed: 42 });

verify(chain);          // true — this run's chain is internally consistent
finalLoss;              // 1.04e-4  (XOR, seed 42, deterministic)
tip;                    // fcb51c678c75615c6c9adb03bef8fe795b48c45712c014b54c6ee6eebaff58cc
```

## The nets

| net            | shape   | activations          | trained result (seed 42)                     |
|----------------|---------|----------------------|----------------------------------------------|
| `nets/xor.json`  | 2-4-1 | tanh hidden, sigmoid out | MSE **1.04e-4** after 3000 epochs @ lr 0.5 |
| `nets/sine.json` | 1-16-1 | tanh hidden, tanh out   | MSE **8.12e-4** after 3000 epochs @ lr 0.1  |

Both are trained **by the test suite** — the receipts in
[TEST-RECEIPT.md](TEST-RECEIPT.md) are from actual `node --test` runs.

## Test suite

```
node --test test/*.test.mjs
```

T1 XOR @ lr 0.5/seed 42 loss < 0.05 · T1b sine trains < 0.01 · T2 byte-identical
chains for same seed · T3 analytic vs numeric gradients within 1e-5 (20 random
weights, at init AND after training) · T4 chain tamper detection (8 tamper
classes) · NC1 topological-order independence (bit-identical) · NC2 seeded-init
reproducibility — plus tick-arithmetic, snapshot round-trip, and receipt-shape
checks. 12 tests, ~5 s wall, zero dependencies.

See [TEMPLATE.md](TEMPLATE.md) to add a layer, swap an activation, or export a
trained net in 5 minutes.

## Honest limits

- Scalar cells, not tensors: the graph evaluates one float per cell. This is the
  micrograd scale, deliberately — every weight is a *named, addressable cell*.
- `loss` cell is per-sample squared error; dataset reduction (mean) lives in
  `train`, not in a cell.
- Plain SGD only: `tick` is exactly `w -= lr·g`. No momentum, no Adam — anything
  else would be a different tick cell.
- tanh/sigmoid saturation at high lr can kill hidden units (found the hard way:
  sine at lr 0.5 plateaus at 0.236; at lr 0.1 it reaches 8e-4). The receipts make
  such plateaus visible instead of silent.

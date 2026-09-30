// capture-receipt — one clean run of the two headline trainings, numbers as shipped
// in TEST-RECEIPT.md. Uses the exact configs the test suite pins.
import { readFileSync } from 'node:fs';
import { loadNet } from '../src/cellgraph.mjs';
import { train, verify, verifyWhy, makeXorData, makeSineData } from '../src/train.mjs';

const load = (f) => loadNet(JSON.parse(readFileSync(new URL(`../nets/${f}`, import.meta.url), 'utf8')));

const xorNet = load('xor.json');
const t0 = Date.now();
const xor = train(xorNet, { data: makeXorData(), epochs: 3000, lr: 0.5, seed: 42 });
const xorMs = Date.now() - t0;

const sineNet = load('sine.json');
const t1 = Date.now();
const sine = train(sineNet, { data: makeSineData({ n: 64, seed: 7 }), epochs: 3000, lr: 0.1, seed: 42 });
const sineMs = Date.now() - t1;

const out = {
  generated_utc: new Date().toISOString(),
  node: process.version,
  xor: {
    net: 'nets/xor.json (2-4-1, tanh hidden, sigmoid out)',
    epochs: 3000, lr: 0.5, seed: 42, data: 'makeXorData() (4 patterns)',
    final_loss: xor.finalLoss,
    loss_at: { 100: xor.metrics[99].loss, 500: xor.metrics[499].loss, 1000: xor.metrics[999].loss, 2000: xor.metrics[1999].loss, 3000: xor.metrics[2999].loss },
    wall_ms: xorMs,
    chain_len: xor.chain.length,
    chain_verifies: verify(xor.chain),
    first_entry: xor.chain[0],
    tip: xor.tip,
  },
  sine: {
    net: 'nets/sine.json (1-16-1, tanh hidden, tanh out)',
    epochs: 3000, lr: 0.1, seed: 42, data: 'makeSineData({n:64, seed:7})',
    final_loss: sine.finalLoss,
    loss_at: { 100: sine.metrics[99].loss, 500: sine.metrics[499].loss, 1000: sine.metrics[999].loss, 2000: sine.metrics[1999].loss, 3000: sine.metrics[2999].loss },
    wall_ms: sineMs,
    chain_len: sine.chain.length,
    chain_verifies: verify(sine.chain),
    first_entry: sine.chain[0],
    tip: sine.tip,
  },
};
console.log(JSON.stringify(out, null, 2));

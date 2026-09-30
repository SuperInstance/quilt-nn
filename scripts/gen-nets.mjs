// gen-nets.mjs — regenerate nets/*.json from builders. Deterministic: the JSON files
// are committed, and this script is the provenance of their bytes.
//   node scripts/gen-nets.mjs
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { buildMLP, loadNet } from '../src/cellgraph.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const netsDir = join(here, '..', 'nets');

// XOR: 2-4-1 MLP, tanh hidden, sigmoid output (targets {0,1}; the sigmoid keeps
// predictions in range and the gradients bounded, so lr 0.5 trains stably).
const xor = buildMLP({ sizes: [2, 4, 1], hiddenAct: 'tanh', outAct: 'sigmoid' });
loadNet(xor); // validate before shipping

// Sine regression: 1-16-1 MLP, tanh hidden, tanh output (the target sin(x) spans
// [-1,1]; a tanh output can represent both signs, unlike sigmoid).
const sine = buildMLP({ sizes: [1, 16, 1], hiddenAct: 'tanh', outAct: 'tanh' });
loadNet(sine);

const j = (g) => JSON.stringify(g, null, 2) + '\n';
writeFileSync(join(netsDir, 'xor.json'), j(xor));
writeFileSync(join(netsDir, 'sine.json'), j(sine));

console.log(`xor.json  : ${xor.cells.length} cells (${loadNet(xor).weightIds.length} weights)`);
console.log(`sine.json : ${sine.cells.length} cells (${loadNet(sine).weightIds.length} weights)`);

# TEST-RECEIPT — quilt-nn v0.1.0

Actual numbers from actual runs on the shipping commit. Configs are pinned by
`test/train.test.mjs`; regenerate any time with `node scripts/capture-receipt.mjs`.

Environment: Node v24.21.0, Linux container, 2026-09-30T08:08Z.

## T1 — XOR (`nets/xor.json`, 2-4-1, tanh hidden / sigmoid out)

| | |
|---|---|
| config | `train(net, {data: makeXorData(), epochs: 3000, lr: 0.5, seed: 42})` |
| **final MSE** | **1.040336960612848e-4** (test asserts < 0.05) |
| loss @ epoch 100 / 500 / 1000 / 2000 | 1.447e-2 / 8.462e-4 / 3.610e-4 / 1.632e-4 |
| training wall time | 159 ms (3000 epochs × 4 samples) |
| chain | 3000 entries, `verify(chain) = true` |
| **chain tip** | `fcb51c678c75615c6c9adb03bef8fe795b48c45712c014b54c6ee6eebaff58cc` |
| first receipt (epoch 0) | loss 0.41793630800167336, sha `bd87669db1e338868f93f23ec7066ad94e50087fbdee5c1c76480e09075e98c6` |

Predictions after training: (0,0)→0.0111, (0,1)→0.9938, (1,0)→0.9884, (1,1)→0.0110.

## T1b — sine (`nets/sine.json`, 1-16-1, tanh hidden / tanh out)

| | |
|---|---|
| config | `train(net, {data: makeSineData({n:64, seed:7}), epochs: 3000, lr: 0.1, seed: 42})` |
| **final MSE** | **8.114783286424474e-4** (test asserts < 0.01) |
| loss @ epoch 100 / 500 / 1000 / 2000 | 5.719e-3 / 1.035e-3 / 8.213e-4 / 7.332e-4 |
| training wall time | 3.51 s (3000 epochs × 64 samples) |
| chain | 3000 entries, `verify(chain) = true` |
| **chain tip** | `4819f868c9a5314f3a448f7a7415e348e43819e30b2ab3f9842c558e5f9af27b` |
| first receipt (epoch 0) | loss 0.1129896244831649, sha `761e66e8a9d1cb03461b977e29aca90a3f081d157f7f00188e673bb3c38e1aa8` |

## T3 — gradient check

Analytic (`grad` cells, composed backward) vs central finite difference
(h = 1e-6) on the **mean dataset loss**, 20 weights picked by seeded LCG:

- at sine init (seed 42): worst |Δ| = **1.742e-10** @ `w0_1_0` (tolerance 1e-5)
- after 500 XOR training epochs: all 20 checked weights within 1e-5 (gradients
  stay honest on trained weights, not just fresh ones)
- relu net checked separately, away from the kink.

## T2 / T4 / NC1 / NC2 — one line each, as enforced by the suite

- **T2** two 3000-epoch XOR runs at seed 42 → `JSON.stringify(chain)` byte-identical, tips equal; seed 43 → different tip; sine 120-epoch chains byte-identical too.
- **T4** 8 tamper classes (edited loss / loss_sha / weight_root_sha / sha / seq / epoch, reorder, drop) all detected; `verifyWhy` names the entry and the breach.
- **NC1** insertion vs Kahn-fifo vs Kahn-lifo orders: different sequences, **bit-identical** outputs (compared as f64 hex, no tolerance) on both nets.
- **NC2** same seed → same initial weight bytes (`weight_root_sha` equal); seed 42 vs 43 → different, for both nets.

## Suite receipt

```
node --test test/*.test.mjs
ℹ tests 12
ℹ pass 12
ℹ fail 0
℅ duration_ms ≈ 4.6 s
```

Reproduce the receipts: `node scripts/capture-receipt.mjs`.

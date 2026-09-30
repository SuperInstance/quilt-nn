# TEST-RECEIPT — quilt-nn v0.1.1

Actual numbers from actual runs on the shipping commit. Configs are pinned by
`test/train.test.mjs`; regenerate any time with `node scripts/capture-receipt.mjs`.

Environment: Node v24.21.0, Linux container, 2026-09-30 (v0.1.1 capture).

**v0.1.1 note (quilt-attention#1):** `lossShaOf`'s preimage changed from hashing a
latin1-re-encoded byte buffer to the legible tagged string
`"f64|8|<big-endian IEEE-754 hex>"`. Every epoch receipt is now format **v2**;
`verify()` rejects v1 with a named reason. Training arithmetic is untouched —
every loss number below is **identical to the v0.1.0 receipt**; only the sha
commitments changed (and `weight_root_sha`, which never used the broken path,
keeps its form).

## C0 — cross-runtime scalar conformance (new in v0.1.1)

`test/conformance.test.mjs` (node) and `test/conformance.py` (python3, stdlib
only) hash the shared 32-scalar adversarial fixture
(`test/scalar-fixture.json`: 0, −0, 1, 1.0, −1, π, e, 1/3, 0.1, 1e-7, 1e21,
2^53, 2^53+1-as-float, ±min-denormal, min-normal, max-double, the issue's own
loss scalar, …) and assert **32/32 digests identical** to the committed
agreement list (`test/scalar-expected-digests.txt`), plus 3/3 non-finite
sentinels rejected fail-closed in both runtimes and the −0 ≡ 0 normalization.
The v0.1.0 code disagreed 5/5 on the same pairs.

## T1 — XOR (`nets/xor.json`, 2-4-1, tanh hidden / sigmoid out)

| | |
|---|---|
| config | `train(net, {data: makeXorData(), epochs: 3000, lr: 0.5, seed: 42})` |
| **final MSE** | **1.040336960612848e-4** (test asserts < 0.05) |
| loss @ epoch 100 / 500 / 1000 / 2000 | 1.447e-2 / 8.462e-4 / 3.610e-4 / 1.632e-4 |
| training wall time | 166 ms (3000 epochs × 4 samples) |
| chain | 3000 entries, `verify(chain) = true` |
| **chain tip** | `f2af70a54b192ed49161f1cd4242b1ef4c8e80c083380e3a883de229e450c6e7` |
| first receipt (epoch 0) | loss 0.41793630800167336, sha `012757626f47dc11419868717ac92f6962255aad670785918b6db371782dff1a` |

Predictions after training: (0,0)→0.0111, (0,1)→0.9938, (1,0)→0.9884, (1,1)→0.0110.

## T1b — sine (`nets/sine.json`, 1-16-1, tanh hidden / tanh out)

| | |
|---|---|
| config | `train(net, {data: makeSineData({n:64, seed:7}), epochs: 3000, lr: 0.1, seed: 42})` |
| **final MSE** | **8.114783286424474e-4** (test asserts < 0.01) |
| loss @ epoch 100 / 500 / 1000 / 2000 | 5.719e-3 / 1.035e-3 / 8.213e-4 / 7.332e-4 |
| training wall time | 3.56 s (3000 epochs × 64 samples) |
| chain | 3000 entries, `verify(chain) = true` |
| **chain tip** | `a155e450c54caffcdfc81836146f3f3bb76f710b6fa47efb2b1ba1e8ba88f1ae` |
| first receipt (epoch 0) | loss 0.1129896244831649, sha `cb7b186f27322951451e7abed9b1dfc0650de7629aa86adb12be6d49af3e5f6e` |

## T3 — gradient check

Analytic (`grad` cells, composed backward) vs central finite difference
(h = 1e-6) on the **mean dataset loss**, 20 weights picked by seeded LCG:

- at sine init (seed 42): worst |Δ| = **1.742e-10** @ `w0_1_0` (tolerance 1e-5)
- after 500 XOR training epochs: all 20 checked weights within 1e-5 (gradients
  stay honest on trained weights, not just fresh ones)
- relu net checked separately, away from the kink.

## T2 / T4 / NC1 / NC2 — one line each, as enforced by the suite

- **T2** two 3000-epoch XOR runs at seed 42 → `JSON.stringify(chain)` byte-identical, tips equal; seed 43 → different tip; sine 120-epoch chains byte-identical too.
- **T4** 8 tamper classes (edited loss / loss_sha / weight_root_sha / sha / seq / epoch, reorder, drop) all detected; `verifyWhy` names the entry and the breach. A v1-format entry is rejected too (named reason).
- **NC1** insertion vs Kahn-fifo vs Kahn-lifo orders: different sequences, **bit-identical** outputs (compared as f64 hex, no tolerance) on both nets.
- **NC2** same seed → same initial weight bytes (`weight_root_sha` equal); seed 42 vs 43 → different, for both nets.

## Suite receipt

```
npm test
ℹ tests 16
ℹ pass 16
ℹ fail 0
ℹ duration_ms ≈ 4.7 s
```

plus `python3 test/conformance.py`: 32/32 + 3/3, exit 0.

Reproduce the receipts: `node scripts/capture-receipt.mjs`.

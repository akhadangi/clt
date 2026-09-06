# Production run 5829918

This repository is the public-clean code snapshot corresponding to the completed full production run `5829918` on ULHPC Iris.

- Master seed: `20260905`
- Python: `3.10.17`
- PyTorch: `2.6.0+cu118`
- Transformers: `4.57.6`
- Hardware: four NVIDIA Tesla V100-SXM2 32 GB GPUs
- Model status: `12/12 ok`
- Full-result archive SHA-256: `5912e1dce900898315640f910b31eaa81d035a6eed338096c206780e4940cd06`

## Aggregate integrity

| File | Rows | Columns | NaN | Inf |
|---|---:|---:|---:|---:|
| `activation_patching.csv` | 7424 | 16 | 0 | 0 |
| `adaptive_continuity.csv` | 4320 | 24 | 0 | 0 |
| `adaptive_equivalence_checks.csv` | 120 | 9 | 0 | 0 |
| `adaptive_summary.csv` | 80 | 7 | 0 | 0 |
| `carrier_audit.csv` | 2214 | 29 | 6 | 0 |
| `carrier_minimal_summary.csv` | 44 | 6 | 0 | 0 |
| `carrier_partition_robustness.csv` | 72 | 5 | 0 | 0 |
| `carrier_partition_robustness_summary.csv` | 12 | 6 | 0 | 0 |
| `carrier_pi_uniform_by_prompt.csv` | 72 | 7 | 0 | 0 |
| `carrier_pi_uniform_detail.csv` | 161 | 9 | 0 | 0 |
| `carrier_pi_uniform_summary.csv` | 12 | 8 | 0 | 0 |
| `carrier_threshold_sensitivity.csv` | 288 | 10 | 0 | 0 |
| `carrier_threshold_summary.csv` | 48 | 8 | 0 | 0 |
| `hard_process_reconstruction.csv` | 15 | 23 | 15 | 0 |
| `hidden_liability.csv` | 210960 | 14 | 0 | 0 |
| `hidden_measurement_map_robustness.csv` | 52608 | 9 | 0 | 0 |
| `hidden_swd_summary.csv` | 180 | 7 | 0 | 0 |
| `model_status.csv` | 12 | 4 | 0 | 0 |
| `patching_summary.csv` | 9 | 5 | 0 | 0 |
| `reconstruction_equivalence.csv` | 54 | 17 | 0 | 0 |
| `surface_stimuli.csv` | 160 | 8 | 0 | 0 |
| `surface_stimuli_blinded.csv` | 160 | 5 | 0 | 0 |
| `surface_stimuli_key.csv` | 160 | 5 | 0 | 0 |

`carrier_audit.csv` contains six structurally missing `source_layer_fraction` values for Zamba2 whole-sequence-state rows where no compatible layerwise legacy cache exists. `hard_process_reconstruction.csv` contains fifteen empty `fallback_error` cells because exact serialized-cache reconstruction succeeded for every hard-reconstruction row.

## Randomness

The full suite uses one preregistered master seed and deterministically derives experiment-specific sub-seeds. The primary hidden-liability experiment uses 64 Monte Carlo rollouts per intervention condition. The adaptive governance experiment uses 16 rollouts per condition. Common random numbers are used for matched counterfactual comparisons within each model. Because the rollout sampler derives seeds from batch boundaries, architectures forced to microbatch 1 can traverse a different deterministic substream partition than architectures evaluated with larger microbatches.

## Runtime exceptions validated before production

- `gemma3_4b_pt` runs in FP32 because its V100 FP16 hidden activations overflowed. The loader performs a finite-logit and finite-hidden-state gate before experiments begin.
- `gemma3_4b_pt` and `zamba2_1p2b_base` use rollout microbatch 1. Total rollout counts, interventions, horizons, and estimators remain unchanged.
- `phi4_mini_instruct` uses eager attention on V100.
- Zamba2 can use the Transformers pure-PyTorch fallback when optional Mamba kernels are unavailable.
- PyTorch 2.6 or newer is required because Transformers blocks unsafe `torch.load` paths for older PyTorch versions when loading the OLMo SFT/DPO `.bin` shards.

## Interpretation boundary

These experiments operationalize distinctions relevant to CLT-I bearer individuation. They do not constitute evidence that any tested model is phenomenally conscious, and they do not establish CLT-II.

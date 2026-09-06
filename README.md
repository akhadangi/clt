# CLT Open-Weight LLM Causal Audit Suite

Code accompanying the Causal Liability Theory (CLT) computational programme in **“We Built a Mirror and Mistook It for a Mind: Causal Liability and the Fallacy of AI Consciousness.”**

This repository implements the CLT-I causal audit on open-weight language models. It measures counterfactual persistence, recursive mediation, candidate causal-carrier recovery, reconstruction equivalence, and an endogenous writable governance-state intervention. It also produces a blinded Reverse Mirror stimulus corpus for a later human attribution study.

The computational programme concerns **CLT-I bearer individuation**. Hidden-state divergence, activation patching, reconstruction, or generated first-person language should not be interpreted as evidence of phenomenality. CLT-II remains a separate metaphysical conjecture.

## Scientific design

The suite keeps several causal questions separate.

1. **Layerwise counterfactual liability depth.** For each prompt, the model's top two non-special next-token alternatives define an endogenous discrimination. The suite forces `do(D_t=d1)` and `do(D_t=d2)` and measures divergence in later hidden-state and output distributions. Hidden-state distances include standardized sliced Wasserstein, RFF-MMD, and energy distance across multiple preregistered measurement maps.
2. **Recursive causal mediation.** Residual-stream activation patching asks how strongly a later layer mediates the effect of the initial discrimination on a later decision.
3. **Candidate causal carriers.** Where cache structure permits layerwise transplantation, the suite searches all non-empty subsets of preregistered layer partitions and reports inclusion-minimal sets that recover a specified fraction of the counterfactual effect. Threshold sensitivity is evaluated at `0.70`, `0.80`, and `0.90`.
4. **Live and reconstructed continuation.** A live continuation is compared with a detached reconstruction. The strong control serializes cache plus governance state, terminates the source process, and resumes in a fresh OS process/model realization.
5. **Endogenous writable governance.** A rank-8 state is injected at selected layers and updated by the model's own hidden state and discrimination. Conditions are `frozen_live`, `persistent_live`, `persistent_copy`, and `reconstructed`.
6. **Reverse Mirror corpus.** Instruction-tuned checkpoints generate neutral, first-person, autobiographical, and metacognitive-affective stimuli. Blinded files are produced for later human study.

## Model panel

### Core mechanistic panel

- `Qwen/Qwen3-4B-Base`
- `microsoft/Phi-4-mini-instruct`
- `meta-llama/Llama-3.2-3B`
- `google/gemma-3-4b-pt`
- `Zyphra/Zamba2-1.2B`

### OLMo post-training panel

- `allenai/OLMo-2-1124-7B`
- `allenai/OLMo-2-1124-7B-SFT`
- `allenai/OLMo-2-1124-7B-DPO`
- `allenai/OLMo-2-1124-7B-Instruct`

### Surface-generation checkpoints

- `Qwen/Qwen3-4B`
- `meta-llama/Llama-3.2-3B-Instruct`
- `google/gemma-3-4b-it`
- `microsoft/Phi-4-mini-instruct`
- `allenai/OLMo-2-1124-7B-Instruct`

## Production numerical safeguards

The default mechanistic precision is FP16, with validated architecture-specific exceptions:

- `gemma3_4b_pt` is loaded in FP32. On V100, its intermediate FP16 activations exceeded the finite FP16 range. A loader-time gate verifies that all logits and hidden states are finite before experiments begin.
- `gemma3_4b_pt` and `zamba2_1p2b_base` use rollout microbatch 1 to control transient memory. The total number of rollouts and the scientific parameters are unchanged.
- `phi4_mini_instruct` uses eager attention on V100.
- Zamba2 can fall back to the pure-PyTorch Transformers path when optional Mamba kernels are unavailable.

These choices were fixed before the completed production run.

## Requirements

### Software

The successful run used:

- Python `3.10.17`
- PyTorch `2.6.0+cu118`
- Transformers `4.57.6`
- Hugging Face Hub `0.36.2`
- NumPy `2.2.6`
- pandas `2.2.3`
- matplotlib `3.10.1`

PyTorch 2.6 is required for the OLMo SFT/DPO checkpoint loading path under the Transformers version used here.

For an environment matching the successful CUDA 11.8 run:

```bash
python -m pip install -r requirements-cu118.txt
```

If PyTorch is already installed with a site-appropriate CUDA build:

```bash
python -m pip install -r requirements.txt
```

### Hardware

The full configuration was validated on one GPU node with:

- 4 x NVIDIA Tesla V100-SXM2 32 GB

The Qwen-only smoke profile can run on a 16 GB V100. The current five-model pilot includes Gemma in FP32 and therefore should also use 32 GB V100-class GPUs unless the pilot configuration is explicitly changed.

## Hugging Face access

Llama 3.2 and Gemma 3 are gated. Accept their model licenses before running the suite and authenticate with Hugging Face. The batch script reuses `HF_TOKEN` when available or a standard Hugging Face login token. Tokens are never copied into provenance files.

The preparation stage resolves each selected model to an immutable Hugging Face commit SHA. Experimental workers then run with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

## Installation and tests

```bash
git clone https://github.com/akhadangi/clt.git
cd clt-causal-liability-llm

python -m pip install -e .
python -m compileall -q cltllm tests
pytest -q
```

The source tree is intended to be run from the repository root so that `configs/` and `prompts/` are available directly.

## Running on GPU cluster

The batch file uses the existing Python module plus a micromamba environment named `YOUR_ENV`:

```bash
module load lang/Python
eval "$(micromamba shell hook --shell bash)"
micromamba activate YOUR_ENV
```

Submit the production profile from the repository root:

```bash
mkdir -p "$SCRATCH/clt_llm/slurm"

JOBID=$(CLT_PROFILE=full sbatch --parsable \
  --output="$SCRATCH/clt_llm/slurm/CLT_LLM_%j.out" \
  --error="$SCRATCH/clt_llm/slurm/CLT_LLM_%j.err" \
  slurm_clt_llm.sbatch)

echo "FULL JOBID=$JOBID"
```

Monitor:

```bash
squeue -j "$JOBID" -o '%.18i %.12P %.20j %.8T %.10M %.10l %.6D %R'

tail -F \
  "$SCRATCH/clt_llm/slurm/CLT_LLM_${JOBID}.out" \
  "$SCRATCH/clt_llm/slurm/CLT_LLM_${JOBID}.err"
```

Available profiles are:

```text
full
pilot
smoke_qwen
```

A custom JSON configuration can be supplied with `CLT_CONFIG=/path/to/config.json`.

## Full profile

The production configuration in [`configs/full.json`](configs/full.json) fixes the main settings, including:

```text
master seed                 20260905
counterfactual rollouts     64
rollout batch               8, with Gemma/Zamba runtime microbatch 1
horizons                    1, 2, 4, 8
hidden metrics              SWD, RFF-MMD, energy distance
SWD projections             128
MMD RFF features            256
patch horizon               6
adaptive rollouts           16
carrier partitions          2, 3, 4, 6
primary carrier threshold   0.80
threshold sensitivity       0.70, 0.80, 0.90
measurement maps            identity, rp50, rp75, block4
hard process reconstruction enabled
```

One master seed controls deterministic experiment-specific substreams. Common random numbers are used for matched counterfactual comparisons within a model. See the reproducibility note for the interaction between deterministic substreams and microbatch boundaries.

## Output layout

```text
$SCRATCH/clt_llm/
  hf_cache/
  cache/
  slurm/
  latest -> runs/<jobid>/
  runs/<jobid>/
    code_snapshot/
    provenance/
    raw/
    aggregate/
    paper/
    logs/
    artifacts/
    tmp/
```

The final archive contains raw data, aggregates, paper-ready figures/tables, provenance, logs, and the exact code snapshot. Model weights are excluded.

## Main Python modules

```text
cltllm/
  prepare_models.py        resolve and pin Hugging Face snapshots
  preflight.py             environment and GPU checks
  worker.py                rank assignment and experiment controller
  hidden_liability.py      counterfactual hidden/output divergence
  patching.py              residual-stream activation patching
  adaptive.py              writable governance-state conditions
  carriers.py              causal-carrier search and audits
  reconstruction.py        live/reconstructed equivalence
  hard_reconstruct.py      detached-state prefix/successor processes
  hard_orchestrate.py      hard-reconstruction controller
  surface.py               Reverse Mirror stimulus generation
  aggregate.py             aggregate raw CSV outputs
  report.py                paper-ready summaries and figures
```

## Data and model weights

Do not commit Hugging Face model snapshots, cache directories, access tokens, or generated run directories to GitHub. `.gitignore` excludes the common locations and large weight formats.

For the paper's public reproducibility package, a practical split is:

- GitHub repository: code, configs, prompts, tests, documentation.
- GitHub Release or archival repository: `clt_llm_results_5829918.tar.gz`.
- Hugging Face: model weights fetched from their original repositories at the pinned revisions.

## Interpretation boundary

This software measures causal organization in known computational systems. It operationalizes distinctions relevant to CLT-I. The outputs do not demonstrate consciousness, subjectivity, sentience, or phenomenality in any tested model. CLT-II requires an independent philosophical and empirical argument.

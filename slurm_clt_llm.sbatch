#!/bin/bash -l
#SBATCH --job-name=CLT_LLM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=TASKS_PER_NODE
#SBATCH --partition=gpu
#SBATCH --qos=YOUR_QOS
#SBATCH --gpus-per-node=GPU_PER_NODE
#SBATCH --mem=MEMORY
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=TIME_LIMIT

# Fail on command errors and pipeline errors.
# inspect variables that are not defined.
set -eo pipefail

print_error_and_exit() { echo "***ERROR*** $*" >&2; exit 1; }
module purge || print_error_and_exit "No 'module' command"
module load lang/Python || print_error_and_exit "Cannot load lang/Python"

# Make micromamba activation robust in a non-interactive Slurm shell while still
# using the user's existing 'legacy' environment.
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
else
  print_error_and_exit "micromamba is not available after loading lang/Python"
fi
micromamba activate YOUR_ENV || print_error_and_exit "Cannot activate micromamba env 'legacy'"

# Keep nounset disabled for the entire job. We still use `set -e`
# and `pipefail`, while required variables are checked explicitly below.

echo "Activated environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python executable: $(command -v python)"
python --version

: "${SCRATCH:? SCRATCH is not defined}"

BASE="${SCRATCH}/clt_llm"
RUN_ROOT="${BASE}/runs/${SLURM_JOB_ID}"
CACHE_ROOT="${BASE}/cache"
HF_ROOT="${BASE}/hf_cache"
LOG_ROOT="${RUN_ROOT}/logs"
PROV_ROOT="${RUN_ROOT}/provenance"
RAW_ROOT="${RUN_ROOT}/raw"
AGG_ROOT="${RUN_ROOT}/aggregate"
PAPER_ROOT="${RUN_ROOT}/paper"
ART_ROOT="${RUN_ROOT}/artifacts"
TMP_ROOT="${RUN_ROOT}/tmp"
CODE_ROOT="${RUN_ROOT}/code_snapshot"
SLURM_LOG_ROOT="${BASE}/slurm"

mkdir -p \
  "$RUN_ROOT" "$CACHE_ROOT" "$HF_ROOT" "$LOG_ROOT" "$PROV_ROOT" \
  "$RAW_ROOT" "$AGG_ROOT" "$PAPER_ROOT" "$ART_ROOT" "$TMP_ROOT" \
  "$SLURM_LOG_ROOT"
ln -sfn "$RUN_ROOT" "${BASE}/latest"

# Mirror controller stdout/stderr into the run tree. Slurm stdout/stderr should
# also be directed to ${BASE}/slurm via the submission CLI in README.md.
exec > >(tee -a "$LOG_ROOT/controller.log") \
     2> >(tee -a "$LOG_ROOT/controller.err" >&2)

CODE_SRC="${SLURM_SUBMIT_DIR}"
mkdir -p "$CODE_ROOT"
rsync -a --delete \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.git' \
  --exclude='*.out' --exclude='*.err' --exclude='*.zip' \
  "$CODE_SRC/" "$CODE_ROOT/"
cd "$CODE_ROOT"

# All mutable caches and all experiment outputs are on Lustre scratch.
export HF_HOME="$HF_ROOT"
export HF_HUB_CACHE="$HF_ROOT/hub"
export HUGGINGFACE_HUB_CACHE="$HF_ROOT/hub"
export HF_DATASETS_CACHE="$HF_ROOT/datasets"
export TRANSFORMERS_CACHE="$HF_ROOT/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export MPLBACKEND=Agg
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TMPDIR="$TMP_ROOT"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
export HF_XET_HIGH_PERFORMANCE=1

# One batch file supports the preregistered full, pilot, and smoke profiles.
# CLT_CONFIG, when supplied, overrides CLT_PROFILE and can be absolute or relative
# to the submitted source snapshot.
if [[ -n "${CLT_CONFIG:-}" ]]; then
  CONFIG_SPEC="$CLT_CONFIG"
else
  PROFILE="${CLT_PROFILE:-full}"
  case "$PROFILE" in
    full)       CONFIG_SPEC="configs/full.json" ;;
    pilot)      CONFIG_SPEC="configs/pilot.json" ;;
    smoke_qwen) CONFIG_SPEC="configs/smoke_qwen.json" ;;
    *) print_error_and_exit "Unknown CLT_PROFILE='$PROFILE' (use full, pilot, or smoke_qwen)" ;;
  esac
fi
if [[ "$CONFIG_SPEC" = /* ]]; then
  CONFIG="$CONFIG_SPEC"
else
  CONFIG="$CODE_ROOT/$CONFIG_SPEC"
fi
MODELS="$CODE_ROOT/configs/models.json"
MECH_PROMPTS="$CODE_ROOT/prompts/mechanistic_prompts.jsonl"
SURFACE_PROMPTS="$CODE_ROOT/prompts/surface_prompts.jsonl"
PREPARED="$PROV_ROOT/prepared_models.json"

[[ -f "$CONFIG" ]] || print_error_and_exit "Config not found: $CONFIG"
[[ -f "$MODELS" ]] || print_error_and_exit "Model registry not found: $MODELS"

# Capture an existing Hugging Face credential without ever writing it to the
# provenance files. For gated Llama/Gemma checkpoints, the user must have already
# accepted the licenses on Hugging Face.
if [[ -z "${HF_TOKEN:-}" ]]; then
  for token_file in "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
    if [[ -r "$token_file" ]]; then
      export HF_TOKEN="$(<"$token_file")"
      break
    fi
  done
fi

cp -f "$CONFIG" "$PROV_ROOT/run_config.json"
cp -f "$MODELS" "$PROV_ROOT/models.json"
cp -f "$MECH_PROMPTS" "$PROV_ROOT/mechanistic_prompts.jsonl"
cp -f "$SURFACE_PROMPTS" "$PROV_ROOT/surface_prompts.jsonl"
python -m pip freeze > "$PROV_ROOT/pip_freeze.txt" 2>&1 || true
micromamba list > "$PROV_ROOT/micromamba_list.txt" 2>&1 || true

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "job_name=${SLURM_JOB_NAME}"
  echo "host=$(hostname)"
  echo "date=$(date --iso-8601=seconds)"
  echo "code_src=${CODE_SRC}"
  echo "code_snapshot=${CODE_ROOT}"
  echo "run_root=${RUN_ROOT}"
  echo "hf_home=${HF_HOME}"
  echo "config=${CONFIG}"
  echo "python=$(command -v python)"
  python --version
  nvidia-smi || true
  echo "---- sanitized environment ----"
  env | sort \
    | grep -E '^(SLURM|CUDA|HF_|HUGGINGFACE|TRANSFORMERS|SCRATCH|OMP|MKL|PYTHON|XDG|TORCH|TRITON|TMPDIR)' \
    | grep -Eiv '(TOKEN|PASSWORD|SECRET|API[_-]?KEY|ACCESS[_-]?KEY)=' || true
} > "$PROV_ROOT/environment.txt" 2>&1

# Profile-specific GPU-memory requirement.
# The full suite retains the >=30 GiB requirement because it includes
# instrumented 7B OLMo checkpoints.
case "${PROFILE:-full}" in
  smoke_qwen)
    MIN_VRAM_GB=14.0
    ;;
  pilot)
    MIN_VRAM_GB=30.0
    ;;
  full)
    MIN_VRAM_GB=30.0
    ;;
  *)
    MIN_VRAM_GB=30.0
    ;;
esac

echo "Preflight minimum VRAM: ${MIN_VRAM_GB} GiB"

python -m cltllm.preflight \
  --output "$PROV_ROOT/preflight.json" \
  --expected-gpus 4 \
  --min-vram-gb "$MIN_VRAM_GB"

# Resolve and pin each selected Hugging Face repository to an exact commit SHA.
# The shared scratch cache persists across smoke/pilot/full jobs.
python -u -m cltllm.prepare_models \
  --config "$CONFIG" \
  --models "$MODELS" \
  --output "$PREPARED" \
  --cache-dir "$HF_HUB_CACHE" \
  --strict 1

# The experiments themselves are offline and therefore guaranteed to use the
# exact snapshots recorded in prepared_models.json.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

srun --kill-on-bad-exit=1 \
  --ntasks=4 --ntasks-per-node=4 --ntasks-per-socket=2 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=1 --gpu-bind=single:1 \
  python -u -m cltllm.worker \
    --config "$CONFIG" \
    --models "$MODELS" \
    --prepared "$PREPARED" \
    --prompts "$MECH_PROMPTS" \
    --surface-prompts "$SURFACE_PROMPTS" \
    --output "$RAW_ROOT"

# Strong reconstruction control. Prefix and successor are distinct OS processes;
# the source process exits before the successor loads the detached record.
if python - "$CONFIG" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    c = json.load(f)
raise SystemExit(0 if c.get('hard_process_reconstruction', False) else 1)
PY
then
  srun --kill-on-bad-exit=1 \
    --ntasks=4 --ntasks-per-node=4 --ntasks-per-socket=2 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --gpus-per-task=1 --gpu-bind=single:1 \
    python -u -m cltllm.hard_orchestrate \
      --config "$CONFIG" \
      --models "$MODELS" \
      --prepared "$PREPARED" \
      --prompts "$MECH_PROMPTS" \
      --output "$RAW_ROOT"
fi

python -u -m cltllm.aggregate --input "$RAW_ROOT" --output "$AGG_ROOT" --config "$CONFIG"
python -u -m cltllm.report --input "$AGG_ROOT" --output "$PAPER_ROOT" --run-root "$RUN_ROOT"

# Bundle everything required to audit the run, excluding the very large model cache.
# Results remain individually accessible under RUN_ROOT as well.
TAR_ITEMS=(raw aggregate paper provenance logs code_snapshot)
tar -C "$RUN_ROOT" -czf "$ART_ROOT/clt_llm_results_${SLURM_JOB_ID}.tar.gz" "${TAR_ITEMS[@]}"

printf '\n===== CLT OPEN-WEIGHT LLM SUITE COMPLETE =====\n'
printf 'Run root:   %s\n' "$RUN_ROOT"
printf 'Aggregate:  %s\n' "$AGG_ROOT"
printf 'Paper:      %s\n' "$PAPER_ROOT"
printf 'Artifact:   %s\n' "$ART_ROOT/clt_llm_results_${SLURM_JOB_ID}.tar.gz"
printf 'Latest:     %s\n' "${BASE}/latest"
printf 'Controller: %s\n' "$LOG_ROOT/controller.log"

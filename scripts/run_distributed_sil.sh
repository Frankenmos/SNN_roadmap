#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="banana_glasses_v7_sil_b2048_e4_a10"
NUM_ACTORS=10
MAX_UPDATES=0
FRAGMENT_STEPS=0
GLOBAL_ROLLOUT_STEPS=0
CONFIG=""
VENV_PATH="$HOME/venvs/sc2_ppo"
PYTHON_BIN="python"
NO_VENV=0
LOCAL_MODE=0
DRY_RUN=0
USE_ROOT_CONFIG=0

usage() {
  cat <<'EOF'
Usage: scripts/run_distributed_sil.sh [options]

Run the SIL distributed Ray trainer from inside WSL.

Options:
  --run-name NAME               Run directory name.
  --num-actors N                Number of Ray rollout actors.
  --max-updates N               Stop after N learner updates. Omit/0 for full run.
  --fragment-steps N            Override distributed.fragment_steps.
  --global-rollout-steps N      Override distributed.global_rollout_steps.
  --config PATH                 Config path. Defaults to repo config.yaml.
  --use-root-config             Ignore models/<run-name>/config.yaml when present.
  --venv PATH                   Venv dir. Defaults to ~/venvs/sc2_ppo.
  --python PATH                 Python executable after venv activation.
  --no-venv                     Do not source a venv.
  --local-mode                  Pass Ray --local-mode for debugging.
  --dry-run                     Print the resolved command without running it.
  -h, --help                    Show this help.
EOF
}

expand_path() {
  local value="$1"
  case "$value" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${value#~/}" ;;
    *) printf '%s\n' "$value" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="${2:?--run-name requires a value}"
      shift 2
      ;;
    --num-actors)
      NUM_ACTORS="${2:?--num-actors requires a value}"
      shift 2
      ;;
    --max-updates)
      MAX_UPDATES="${2:?--max-updates requires a value}"
      shift 2
      ;;
    --fragment-steps)
      FRAGMENT_STEPS="${2:?--fragment-steps requires a value}"
      shift 2
      ;;
    --global-rollout-steps)
      GLOBAL_ROLLOUT_STEPS="${2:?--global-rollout-steps requires a value}"
      shift 2
      ;;
    --config)
      CONFIG="${2:?--config requires a value}"
      shift 2
      ;;
    --use-root-config)
      USE_ROOT_CONFIG=1
      shift
      ;;
    --venv)
      VENV_PATH="${2:?--venv requires a value}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:?--python requires a value}"
      shift 2
      ;;
    --no-venv)
      NO_VENV=1
      shift
      ;;
    --local-mode)
      LOCAL_MODE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$CONFIG" ]]; then
  RUN_CONFIG="$REPO_ROOT/models/$RUN_NAME/config.yaml"
  if [[ "$USE_ROOT_CONFIG" -eq 0 && -f "$RUN_CONFIG" ]]; then
    CONFIG="$RUN_CONFIG"
  else
    CONFIG="$REPO_ROOT/config.yaml"
  fi
elif [[ "$CONFIG" != /* ]]; then
  CONFIG="$REPO_ROOT/$CONFIG"
fi
CONFIG="$(expand_path "$CONFIG")"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

cd "$REPO_ROOT"
export SNN_CONFIG_PATH="$CONFIG"

if [[ "$NO_VENV" -eq 0 ]]; then
  VENV_PATH="$(expand_path "$VENV_PATH")"
  ACTIVATE="$VENV_PATH/bin/activate"
  if [[ ! -f "$ACTIVATE" ]]; then
    echo "Venv activate not found: $ACTIVATE" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$ACTIVATE"
fi

train_args=(
  -m distributed.ray_train
  --config "$CONFIG"
  --run-name "$RUN_NAME"
  --num-actors "$NUM_ACTORS"
)

if [[ "$MAX_UPDATES" -gt 0 ]]; then
  train_args+=(--max-updates "$MAX_UPDATES")
fi
if [[ "$FRAGMENT_STEPS" -gt 0 ]]; then
  train_args+=(--fragment-steps "$FRAGMENT_STEPS")
fi
if [[ "$GLOBAL_ROLLOUT_STEPS" -gt 0 ]]; then
  train_args+=(--global-rollout-steps "$GLOBAL_ROLLOUT_STEPS")
fi
if [[ "$LOCAL_MODE" -eq 1 ]]; then
  train_args+=(--local-mode)
fi

echo "Launching Ray SIL run: $RUN_NAME"
echo "Repo: $REPO_ROOT"
echo "Config: $CONFIG"
if [[ "$NO_VENV" -eq 0 ]]; then
  echo "Venv: $VENV_PATH"
else
  echo "Venv: skipped"
fi
printf 'Command: %q' "$PYTHON_BIN"
printf ' %q' "${train_args[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

exec "$PYTHON_BIN" "${train_args[@]}"

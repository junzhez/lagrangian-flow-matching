#!/usr/bin/env bash
# Leave-one-timepoint-out training sweep on EB / CITE / Multiome.
#
# Mirrors the protocol in examples/single_cell/eval_loo_methods.py:
#   - 5 methods: otcfm, otsi, otharmonic{w=0.001, w=1.0, w=pi/2}
#   - interior timepoints held out per dataset
#   - 5 seeds (42..46)
#
# Usage:
#   bash runner/scripts/single_cell_loo.sh                  # full sweep
#   DATASETS="eb" METHODS="otcfm" TIMEPOINTS_eb="2" \
#       SEEDS="42" bash runner/scripts/single_cell_loo.sh   # narrow run
#   DRY_RUN=1 bash runner/scripts/single_cell_loo.sh        # print only
#   STOP_ON_FAIL=1 bash runner/scripts/single_cell_loo.sh   # abort on first fail

set -uo pipefail

cd "$(dirname "$0")/.."

DATASETS=${DATASETS:-"eb cite multiome"}
METHODS=${METHODS:-"otcfm otsi otharmonic_w0.001 otharmonic_w1.0 otharmonic_wpi2"}
SEEDS=${SEEDS:-"42,43,44,45,46"}

# Interior timepoints to leave out (1..n-2) per dataset.
# EB has 5 timepoints (0..4); CITE/Multiome have 4 (0..3).
TIMEPOINTS_eb=${TIMEPOINTS_eb:-"1 2 3"}
TIMEPOINTS_cite=${TIMEPOINTS_cite:-"1 2"}
TIMEPOINTS_multiome=${TIMEPOINTS_multiome:-"1 2"}

DRY_RUN=${DRY_RUN:-0}
STOP_ON_FAIL=${STOP_ON_FAIL:-0}

PI_2="1.5707963267948966"

n_datasets=0; for _ in $DATASETS; do n_datasets=$((n_datasets + 1)); done
n_methods=0;  for _ in $METHODS;  do n_methods=$((n_methods + 1));  done
n_seeds=0;    IFS=',' read -ra _seed_arr <<< "$SEEDS"; n_seeds=${#_seed_arr[@]}
n_tp_total=0
for ds in $DATASETS; do
  tp_var="TIMEPOINTS_${ds}"
  for _ in ${!tp_var}; do n_tp_total=$((n_tp_total + 1)); done
done
total=$((n_tp_total * n_methods))
echo "Plan: ${n_datasets} datasets x ${n_methods} methods x interior t's x ${n_seeds} seeds = ${total} multirun calls ($((total * n_seeds)) training runs)."
echo

failures=()

for ds in $DATASETS; do
  tp_var="TIMEPOINTS_${ds}"
  tps=${!tp_var}
  for method in $METHODS; do
    case "$method" in
      otcfm)             MODEL="otcfm" ;       EXTRA="" ;;
      otsi)              MODEL="otsi" ;        EXTRA="" ;;
      otharmonic_w0.001) MODEL="otharmonic" ;  EXTRA="model.omega=0.001" ;;
      otharmonic_w1.0)   MODEL="otharmonic" ;  EXTRA="model.omega=1.0" ;;
      otharmonic_wpi2)   MODEL="otharmonic" ;  EXTRA="model.omega=${PI_2}" ;;
      *) echo "unknown method: $method" >&2 ; exit 1 ;;
    esac
    for t in $tps; do
      echo "============================================================"
      echo "ds=$ds method=$method leaveout=$t seeds=$SEEDS"
      cmd=(python src/train.py -m
           "experiment=${ds}_loo"
           "model=$MODEL"
           "model.leaveout_timepoint=$t"
           "seed=$SEEDS"
           "name=${ds}_${method}_loo${t}")
      [[ -n "$EXTRA" ]] && cmd+=("$EXTRA")
      echo "${cmd[*]}"
      echo "============================================================"
      if [[ "$DRY_RUN" != "0" ]]; then
        continue
      fi
      "${cmd[@]}"
      rc=$?
      if [[ $rc -ne 0 ]]; then
        failures+=("${ds}/${method}/loo${t}: rc=${rc}")
        if [[ "$STOP_ON_FAIL" != "0" ]]; then
          echo >&2
          echo "Aborting: ${ds}/${method}/loo${t} returned ${rc}" >&2
          exit "$rc"
        fi
      fi
    done
  done
done

if [[ ${#failures[@]} -gt 0 ]]; then
  echo
  echo "Failed runs:"
  for f in "${failures[@]}"; do
    echo "  $f"
  done
  exit 1
fi
echo
echo "All runs completed."

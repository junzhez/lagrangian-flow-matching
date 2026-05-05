#!/usr/bin/env bash
# Leave-one-timepoint-out training sweep on EB / CITE / Multiome.
#
# Mirrors the protocol in examples/single_cell/eval_loo_methods.py:
#   - 5 methods: cfm (no OT), otcfm, otsi, otharmonic{w=0.001, w=1.0, w=pi/2}
#   - interior timepoints held out per dataset
#   - 5 seeds (42..46)
#
# Usage:
#   bash runner/scripts/single_cell_loo.sh                  # full sweep
#   DATASETS="eb" METHODS="otcfm" TIMEPOINTS_eb="2" \
#       SEEDS="42" bash runner/scripts/single_cell_loo.sh   # narrow run

set -euo pipefail

cd "$(dirname "$0")/.."

DATASETS=${DATASETS:-"eb cite multiome"}
METHODS=${METHODS:-"cfm otcfm otsi otharmonic_w0.001 otharmonic_w1.0 otharmonic_wpi2"}
SEEDS=${SEEDS:-"42,43,44,45,46"}

# Interior timepoints to leave out (1..n-2) per dataset.
# EB has 5 timepoints (0..4); CITE/Multiome have 4 (0..3).
TIMEPOINTS_eb=${TIMEPOINTS_eb:-"1 2 3"}
TIMEPOINTS_cite=${TIMEPOINTS_cite:-"1 2"}
TIMEPOINTS_multiome=${TIMEPOINTS_multiome:-"1 2"}

PI_2="1.5707963267948966"

for ds in $DATASETS; do
  tp_var="TIMEPOINTS_${ds}"
  tps=${!tp_var}
  for method in $METHODS; do
    case "$method" in
      cfm)              MODEL="cfm" ;          EXTRA="" ;;
      otcfm)            MODEL="otcfm" ;        EXTRA="" ;;
      otsi)             MODEL="otsi" ;         EXTRA="" ;;
      otharmonic_w0.001) MODEL="otharmonic" ;  EXTRA="model.omega=0.001" ;;
      otharmonic_w1.0)   MODEL="otharmonic" ;  EXTRA="model.omega=1.0" ;;
      otharmonic_wpi2)   MODEL="otharmonic" ;  EXTRA="model.omega=${PI_2}" ;;
      *) echo "unknown method: $method" >&2 ; exit 1 ;;
    esac
    for t in $tps; do
      echo "============================================================"
      echo "ds=$ds method=$method (model=$MODEL $EXTRA) leaveout=$t seeds=$SEEDS"
      echo "============================================================"
      python src/train.py -m \
        experiment="${ds}_loo" \
        model="$MODEL" \
        model.leaveout_timepoint="$t" \
        seed="$SEEDS" \
        ${EXTRA} \
        name="${ds}_${method}_loo${t}"
    done
  done
done

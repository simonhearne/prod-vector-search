#!/usr/bin/env bash
# Full runs not covered by their own tasks (B1a/B1b/B2/B2c run on their own; B4a/B4c after this).
# Milvus benchmarks run one at a time; each logs to logs/<step>.log; stops at the first failure.
#   nohup bash bench/run_full.sh > logs/run_full.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
step() { local id=$1; shift; echo "$(date +%T) start $id"; uv run python -m "$@" > "logs/$id.log" 2>&1; echo "$(date +%T) done $id"; }
step tune_poc        bench.tune_poc
step b1c_calibrate   bench.b1c_judge calibrate     # re-buys the calibration pairs judged without the prompt
step b1c_fill        bench.b1c_judge fill          # mostly cached after B1a's fill
step b1c_report      bench.b1c_judge report
step b3a             bench.b3a_frontier --keep     # keep: B2 reuses f_1024_rabitq
step b3b             bench.b3b_cost_table
step b3c             bench.b3c_reembed
step b4b             bench.b4b_alias_swap
step charts          bench.make_charts
echo "$(date +%T) ALL DONE"

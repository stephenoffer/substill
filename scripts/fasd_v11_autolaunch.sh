#!/usr/bin/env bash
# Poll the v11 smoke job; if it succeeds, submit the full 10-job ladder.
# If it fails, abort without submitting.
#
# Usage: SMOKE_JOB_ID=prodjob_xxx LADDER_TAG=v11-pra-apr30 bash fasd_v11_autolaunch.sh

set -uo pipefail

SMOKE_JOB_ID="${SMOKE_JOB_ID:?SMOKE_JOB_ID required}"
LADDER_TAG="${LADDER_TAG:-v11-pra-apr30}"
POLL_SECS="${POLL_SECS:-60}"

echo "[autolaunch] smoke=${SMOKE_JOB_ID}  ladder_tag=${LADDER_TAG}  poll=${POLL_SECS}s"

while true; do
  STATUS_OUT="$(anyscale job status --job-id "${SMOKE_JOB_ID}" 2>&1 || true)"
  STATE="$(printf '%s\n' "${STATUS_OUT}" | awk -F': *' '/^state:/ {print $2; exit}')"
  TS="$(date +%H:%M:%S)"

  case "${STATE}" in
    SUCCEEDED|SUCCESS)
      echo "[autolaunch ${TS}] smoke ${STATE} — launching full ladder"
      RESULTS_BEFORE="$(aws s3 ls "${ANYSCALE_ARTIFACT_STORAGE}/fasd/v11-smoke-apr30/results/" 2>/dev/null || true)"
      echo "[autolaunch] smoke result objects:"
      printf '%s\n' "${RESULTS_BEFORE}"
      cd /home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/neural_distill
      TAG="${LADDER_TAG}" bash scripts/fasd_ablation_submit.sh
      echo "[autolaunch] ladder submitted under TAG=${LADDER_TAG}"
      exit 0
      ;;
    FAILED|BROKEN|TERMINATED|OUT_OF_RETRIES)
      echo "[autolaunch ${TS}] smoke ${STATE} — NOT submitting ladder"
      echo "[autolaunch] last status output:"
      printf '%s\n' "${STATUS_OUT}"
      exit 1
      ;;
    STARTING|RUNNING|PENDING|"")
      echo "[autolaunch ${TS}] smoke state=${STATE:-<empty>}, polling again in ${POLL_SECS}s"
      ;;
    *)
      echo "[autolaunch ${TS}] unrecognised state=${STATE} — continuing to poll"
      ;;
  esac

  sleep "${POLL_SECS}"
done

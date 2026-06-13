#!/bin/bash
# merge_svj_tsv.sh
# Merge per-job TSV shards produced by run_svj_tsv.sh into the canonical
# simulated/tsv/jets_default.tsv (and jets_kinematics.tsv).
#
# Usage: bash merge_svj_tsv.sh <n_jobs> [--keep-shards]
#   n_jobs        Number of shards to merge (must match N_JOBS in run_svj_tsv.sh).
#   --keep-shards Do not delete the per-job shard files after merging.
#                 Default: shards are removed on success.

set -euo pipefail

N_JOBS="${1:?Usage: $0 <n_jobs> [--keep-shards]}"
KEEP_SHARDS=0
for arg in "${@:2}"; do
    [[ "$arg" == "--keep-shards" ]] && KEEP_SHARDS=1
done

OUT_TSV="simulated/tsv/jets_default.tsv"
OUT_KIN="simulated/tsv/jets_kinematics.tsv"
TSV_DIR="simulated/tsv"

echo "Merging $N_JOBS shard(s) into $OUT_TSV ..."

# ── Validate shards exist before touching output files ────────────────────────
missing=()
for i in $(seq 0 $((N_JOBS - 1))); do
    shard="${TSV_DIR}/jets_default_${i}.tsv"
    [[ -f "$shard" ]] || missing+=("$shard")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Error: missing shard(s):" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

# ── Merge main observable TSV ─────────────────────────────────────────────────
# Header line (starts with #) is taken from shard 0; all data rows follow.
# load_tsv() reads the first line as col_map and uses np.loadtxt(comments='#')
# for data, so exactly one leading # line is the expected format.
head -1 "${TSV_DIR}/jets_default_0.tsv" > "$OUT_TSV"
for i in $(seq 0 $((N_JOBS - 1))); do
    grep -v '^#' "${TSV_DIR}/jets_default_${i}.tsv" >> "$OUT_TSV"
done

# Count merged events
N_EVENTS=$(grep -c -v '^#' "$OUT_TSV")
echo "  $OUT_TSV — $N_EVENTS events from $N_JOBS shards"

# ── Merge kinematics TSV (optional — skip if shards are absent) ──────────────
kin_missing=()
for i in $(seq 0 $((N_JOBS - 1))); do
    shard="${TSV_DIR}/jets_kinematics_${i}.tsv"
    [[ -f "$shard" ]] || kin_missing+=("$shard")
done

if [[ ${#kin_missing[@]} -gt 0 ]]; then
    echo "  Warning: ${#kin_missing[@]} kinematics shard(s) missing; skipping kinematics merge."
else
    head -1 "${TSV_DIR}/jets_kinematics_0.tsv" > "$OUT_KIN"
    for i in $(seq 0 $((N_JOBS - 1))); do
        grep -v '^#' "${TSV_DIR}/jets_kinematics_${i}.tsv" >> "$OUT_KIN"
    done
    N_KIN=$(grep -c -v '^#' "$OUT_KIN")
    echo "  $OUT_KIN — $N_KIN rows from $N_JOBS shards"
fi

# ── Clean up shards ───────────────────────────────────────────────────────────
if [[ $KEEP_SHARDS -eq 0 ]]; then
    for i in $(seq 0 $((N_JOBS - 1))); do
        rm -f "${TSV_DIR}/jets_default_${i}.tsv" \
              "${TSV_DIR}/jets_kinematics_${i}.tsv"
    done
    echo "  Shard files removed."
else
    echo "  Shard files kept (--keep-shards)."
fi

echo "Done."

#!/bin/bash
# merge_svj_tsv.sh
# Merge per-job TSV shards produced by condor/tsv.sub into the canonical
# simulated/tsv/jets_default.tsv (and jets_kinematics.tsv), or the
# jets_<tag>.tsv equivalent for a non-default binary.
#
# Usage: bash merge_svj_tsv.sh <n_jobs> [--binary NAME] [--keep-shards]
#   n_jobs        Number of shards to merge (must match `queue N` in condor/tsv.sub).
#   --binary NAME Binary the shards were generated with (must match SVJ_BINARY
#                 in condor/tsv.sub). Default: svj_regression.
#                 Determines the output tag: svj_regression -> jets_default,
#                 svj_regression_delphes -> jets_delphes.
#   --keep-shards Do not delete the per-job shard files after merging.
#                 Default: shards are removed on success.

set -euo pipefail

N_JOBS="${1:?Usage: $0 <n_jobs> [--binary NAME] [--keep-shards]}"
shift
BINARY_NAME="svj_regression"
KEEP_SHARDS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)      BINARY_NAME="$2"; shift 2 ;;
        --keep-shards) KEEP_SHARDS=1;    shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$BINARY_NAME" == "svj_regression" ]]; then
    OUT_TAG="jets_default"
else
    OUT_TAG="jets_${BINARY_NAME#svj_regression_}"
fi

OUT_TSV="simulated/tsv/${OUT_TAG}.tsv"
OUT_KIN="simulated/tsv/jets_kinematics.tsv"
TSV_DIR="simulated/tsv"

echo "Merging $N_JOBS shard(s) into $OUT_TSV ..."

# ── Validate shards exist before touching output files ────────────────────────
missing=()
for i in $(seq 0 $((N_JOBS - 1))); do
    shard="${TSV_DIR}/${OUT_TAG}_${i}.tsv"
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
head -1 "${TSV_DIR}/${OUT_TAG}_0.tsv" > "$OUT_TSV"
for i in $(seq 0 $((N_JOBS - 1))); do
    grep -v '^#' "${TSV_DIR}/${OUT_TAG}_${i}.tsv" >> "$OUT_TSV"
done

# Count merged events
N_EVENTS=$(grep -c -v '^#' "$OUT_TSV")
echo "  $OUT_TSV — $N_EVENTS events from $N_JOBS shards"

# ── Merge kinematics TSV (truth binary only — svj_regression_delphes never
#    writes one) ───────────────────────────────────────────────────────────
if [[ "$BINARY_NAME" != "svj_regression" ]]; then
    echo "  Skipping kinematics merge (${BINARY_NAME} does not produce one)."
else
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
fi

# ── Clean up shards ───────────────────────────────────────────────────────────
if [[ $KEEP_SHARDS -eq 0 ]]; then
    for i in $(seq 0 $((N_JOBS - 1))); do
        rm -f "${TSV_DIR}/${OUT_TAG}_${i}.tsv" \
              "${TSV_DIR}/jets_kinematics_${i}.tsv"
    done
    echo "  Shard files removed."
else
    echo "  Shard files kept (--keep-shards)."
fi

echo "Done."

#!/bin/bash
# scripts/clean_notebook.sh
# Usage: clean_notebook.sh <notebook_file>

EXCLUDE_FILE="$(dirname "$0")/strip_notebook_exclude.txt"

if [[ -z "$1" ]]; then
    echo "Usage: $0 <notebook_file>" >&2
    exit 1
fi
fname="$1"

tmpjson=$(mktemp)
tmpnotebook=$(mktemp)

# Ensure temp files are cleaned up on exit or error
trap 'rm -f "$tmpjson" "$tmpnotebook"' EXIT

# Always strip metadata. Only strip outputs if notebook NOT in exclude list.

# Check if input file is empty
if [ ! -s "$fname" ]; then
    echo "Error: Input file $fname is empty" >&2
    exit 1
fi

# Convert notebook to JSON and assign sequential cell IDs with jq (array-safe)
if ! cat "$fname" | \
    jq 'if .cells then .cells |= [ range(0; length) as $i | .[$i] as $cell | ($cell | (if has("execution_count") then .execution_count = null else . end) + {id: ($i|tostring)}) ] else . end' > "$tmpjson"; then
    echo "Error: jq failed for $fname" >&2
    exit 2
fi

# Validate JSON
if [ ! -s "$tmpjson" ] || ! jq empty "$tmpjson" 2>/dev/null; then
    echo "Error: $tmpjson is not valid JSON after jq for $fname" >&2
    exit 3
fi

BASE_NBCONVERT_ARGS=(--to=notebook --stdin --stdout --log-level=ERROR)

if grep -Fxq "$(basename "$fname")" "$EXCLUDE_FILE" 2>/dev/null; then
    # Excluded: keep outputs, strip metadata only
    if ! jupyter nbconvert "${BASE_NBCONVERT_ARGS[@]}" < "$tmpjson" | \
        jupyter nbconvert --ClearMetadataPreprocessor.enabled=True \
            --TagRemovePreprocessor.enabled=True --ClearMetadataPreprocessor.preserve_cell_metadata_mask='[("tags")]' \
            "${BASE_NBCONVERT_ARGS[@]}" > "$tmpnotebook"; then
        echo "Error: nbconvert (metadata-only) failed for $fname" >&2
        exit 4
    fi
else
    # Not excluded: strip outputs and metadata
    if ! jupyter nbconvert "${BASE_NBCONVERT_ARGS[@]}" < "$tmpjson" | \
        jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True \
            --TagRemovePreprocessor.enabled=True --ClearMetadataPreprocessor.preserve_cell_metadata_mask='[("tags")]' \
            "${BASE_NBCONVERT_ARGS[@]}" > "$tmpnotebook"; then
        echo "Error: nbconvert (outputs+metadata) failed for $fname" >&2
        exit 4
    fi
fi

cat "$tmpnotebook"

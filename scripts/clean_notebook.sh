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
tmpfinal=$(mktemp)

# Ensure temp files are cleaned up on exit or error
trap 'rm -f "$tmpjson" "$tmpnotebook" "$tmpfinal"' EXIT

# Check if the notebook is in the exclude list
if grep -Fxq "$(basename "$fname")" "$EXCLUDE_FILE" 2>/dev/null; then
    cat "$fname"
else
    # Check if input file is empty
    if [ ! -s "$fname" ]; then
        echo "Error: Input file $fname is empty" >&2
        exit 1
    fi
    # Convert notebook to JSON and assign sequential cell IDs with jq (array-safe)
    if ! cat "$fname" | \
        jq 'if .cells then .cells |= [ range(0; length) as $i | .[$i] as $cell | ($cell + {id: ($i|tostring)}) ] else . end' > "$tmpjson"; then
        echo "Error: jq failed for $fname" >&2
        exit 2
    fi
    # Check if tmpjson is valid and not empty
    if [ ! -s "$tmpjson" ] || ! jq empty "$tmpjson" 2>/dev/null; then
        echo "Error: $tmpjson is not valid JSON after jq for $fname" >&2
        exit 3
    fi
    # Convert JSON back to notebook and strip outputs/metadata
    if ! jupyter nbconvert --to=notebook --stdin --stdout --log-level=ERROR < "$tmpjson" | \
        jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True \
            --TagRemovePreprocessor.enabled=True --ClearMetadataPreprocessor.preserve_cell_metadata_mask='[("tags")]' \
            --to=notebook --stdin --stdout --log-level=ERROR > "$tmpnotebook"; then
        echo "Error: nbconvert failed for $fname" >&2
        exit 4
    fi
    cat "$tmpnotebook"
fi

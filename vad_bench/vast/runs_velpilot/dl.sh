#!/usr/bin/env bash
# Run ON the vast machine: dl.sh <file with "name url" lines>. Parallel downloads into /workspace/data, then size report.
set -uo pipefail
cd /workspace/data
while read -r name url; do
  [ -n "$name" ] || continue
  ( curl -sS -L --retry 3 -o "$name" "$url" && echo "ok $name $(stat -c %s "$name")" || echo "FAIL $name" ) &
done < "$1"
wait
rm -f "$1"     # signed URLs are not kept
echo DL_DONE

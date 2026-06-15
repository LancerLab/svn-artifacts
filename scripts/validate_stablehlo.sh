#!/usr/bin/env bash
# Validate all stablehlo MLIR cases with iree-compile
IREE=${IREE:-$HOME/.local/bin/iree-compile}
BASE=/home/gxf/research/oopsla26/benchmark/stablehlo

pass=0; fail=0
failed_list=""

for f in "$BASE"/**/*.mlir; do
    if "$IREE" --iree-input-type=stablehlo --compile-to=input "$f" -o /dev/null 2>/dev/null; then
        pass=$((pass+1))
    else
        fail=$((fail+1))
        failed_list="$failed_list\nFAIL: $f"
    fi
done

echo "PASS=$pass  FAIL=$fail"
if [ -n "$failed_list" ]; then
    printf "%b\n" "$failed_list" | head -40
fi

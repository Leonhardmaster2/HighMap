#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="build_clang"
TMP_LOG="/tmp/iwyu_current.log"

# -----------------------------------------------------------------------------
# Source directories
# -----------------------------------------------------------------------------

SRC_DIRS=(
    "HighMap/src"
    # "examples"
    # "tests"
)

# -----------------------------------------------------------------------------
# Configure CMake
# -----------------------------------------------------------------------------

cmake -B "${BUILD_DIR}" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DHIGHMAP_ENABLE_EXAMPLES=ON\
    -DHIGHMAP_ENABLE_TESTS=ON \
    -DHIGHMAP_ENABLE_BENCHMARKS=ON

# -----------------------------------------------------------------------------
# Run IWYU per translation unit
# -----------------------------------------------------------------------------

for SRC_DIR in "${SRC_DIRS[@]}"
do
    find "${SRC_DIR}" -type f \( \
         -name "*.hpp" -o \
	 -name "*.cpp" -o \
         -name "*.cc"  -o \
         -name "*.cxx" \
    \)
done | while read -r file
do
    echo "=================================================="
    echo "IWYU: ${file}"
    echo "=================================================="

    rm -f "${TMP_LOG}"

    iwyu_tool -p "${BUILD_DIR}" "${file}" \
        -- \
        -Xiwyu --no_fwd_decls \
        2>&1 \
        | grep -vE "glm/|\.inl|\.cl" \
        > "${TMP_LOG}" || true

    # Skip empty logs
    if [[ ! -s "${TMP_LOG}" ]]; then
        echo "No IWYU suggestions"
        echo ""
        continue
    fi

    cat "${TMP_LOG}"

    # -------------------------------------------------------------------------
    # Apply fixes for this file only
    # -------------------------------------------------------------------------

    fix_include \
        --comments -b \
        --ignore_re='(glm/|\.inl|\.cl)' \
        < "${TMP_LOG}"

    echo ""

done

#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="build_clang"
SRC_DIR="HighMap/src"
TMP_LOG="/tmp/iwyu_current.log"

# -----------------------------------------------------------------------------
# Configure CMake
# -----------------------------------------------------------------------------

cmake -B "${BUILD_DIR}" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DHIGHMAP_ENABLE_EXAMPLES=OFF \
    -DHIGHMAP_ENABLE_TESTS=OFF \
    -DHIGHMAP_ENABLE_BENCHMARKS=OFF

# -----------------------------------------------------------------------------
# Run IWYU per translation unit
# -----------------------------------------------------------------------------

find "${SRC_DIR}" -type f \( \
    -name "*.cpp" -o \
    -name "*.cc"  -o \
    -name "*.cxx" \
\) | while read -r file
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

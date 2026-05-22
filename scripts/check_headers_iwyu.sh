#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="build_clang"
TMP_LOG="/tmp/iwyu_current.log"

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

RUN_IWYU=true

for arg in "$@"
do
    case "${arg}" in
        --no-iwyu)
            RUN_IWYU=false
            ;;
        --iwyu)
            RUN_IWYU=true
            ;;
        *)
            echo "Unknown option: ${arg}"
            echo "Usage: $0 [--iwyu | --no-iwyu]"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Source directories
# -----------------------------------------------------------------------------

SRC_DIRS=(
    "HighMap/src/"
    # "HighMap/include/"
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
    -DHIGHMAP_ENABLE_EXAMPLES=ON \
    -DHIGHMAP_ENABLE_TESTS=ON \
    -DHIGHMAP_ENABLE_BENCHMARKS=ON

# -----------------------------------------------------------------------------
# Process files
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
    echo "FILE: ${file}"
    echo "=================================================="

    # -------------------------------------------------------------------------
    # Optionally run IWYU
    # -------------------------------------------------------------------------

    if [[ "${RUN_IWYU}" == true ]]; then

        rm -f "${TMP_LOG}"

        iwyu_tool -p "${BUILD_DIR}" "${file}" \
            -- \
            -Xiwyu --no_fwd_decls \
            -Xiwyu --transitive_includes_only \
            2>&1 \
            | grep -vE "glm/|\.inl|\.cl" \
            > "${TMP_LOG}" || true

        # Skip empty logs
        if [[ -s "${TMP_LOG}" ]]; then

            cat "${TMP_LOG}"

            # -----------------------------------------------------------------
            # Apply fixes for this file only
            # -----------------------------------------------------------------

            fix_include \
                --nocomments -b \
                --ignore_re='(glm/|\.inl|\.cl)' \
                < "${TMP_LOG}"
        else
            echo "No IWYU suggestions"
        fi

        echo ""
    fi

    # -------------------------------------------------------------------------
    # Clean-up problematic headers emitted by IWYU
    # -------------------------------------------------------------------------

    # remove existing comments after header include
    # sed -i \
    # 	-E 's|^(#include[[:space:]]+[<"][^>"]+[>"])[[:space:]]*//.*$|\1|g' \
    # 	"${file}"
    
    sed -i \
        -e 's|#include <bits/std_abs.h>|#include <cmath>|g' \
        -e 's|#include <stddef.h>|#include <cstddef>|g' \
        -e 's|#include <stdint.h>|#include <cstdint>|g' \
        -e 's|#include <math.h>|#include <cmath>|g' \
        -e 's|#include <sys/types.h>|#include <cstddef>|g' \
        -e 's|#include <stdlib.h>|#include <cstdlib>|g' \
        -e 's|#include <bits/chrono.h>|#include <chrono>|g' \
        -e 's|#include <float.h>|#include <cfloat>|g' \
	-e 's|#include <ext/type_traits.h>|#include <type_traits>|g' \
	-e 's|#include <opencv2/core/hal/interface.h>|#include <opencv2/core.hpp>|g' \
	-e 's|#include <opencv2/core/mat.hpp>|#include <opencv2/core.hpp>|g' \
	-e 's|#include <opencv2/core/matx.hpp>|#include <opencv2/core.hpp>|g' \
        "${file}"
	
    echo ""

done

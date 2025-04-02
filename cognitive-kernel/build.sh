#/bin/bash
# Phase Locked Kernel
gcc phase_locked_kernel.c -o ./bin/phase_kernel_gcc -lm -O3 -march=native -ffast-math -funroll-loops
clang -fcolor-diagnostics -fansi-escape-codes -g phase_locked_kernel.c -o ./bin/phase_kernel_clang -march=native -ffast-math -funroll-loops -fvectorize

# AVX Optimised
gcc -mavx2 -O3 avx_clone_shift.c -o ./bin/avx_shift_gcc -lm
clang -mavx2 -O3 avx_clone_shift.c -o ./bin/avx_shift_clang -lm

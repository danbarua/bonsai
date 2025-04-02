#/bin/bash
gcc phase_locked_kernel.c -o ./bin/phase_kernel_gcc -lm -O3 -march=native -ffast-math -funroll-loops
clang -fcolor-diagnostics -fansi-escape-codes -g phase_locked_kernel.c -o ./bin/phase_kernel_clang -march=native -ffast-math -funroll-loops
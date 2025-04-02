// phase_locked_kernel.c
// Benchmark: Phase-Locked vs Clone-and-Shift Implementations

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define PATCH 64
#define K 16
#define D 4
#define STEPS 10000
#define TWO_PI (2.0 * M_PI)

float* allocate_field(int multiplier) {
    return (float*)malloc(sizeof(float) * multiplier * K * PATCH * PATCH * D);
}

void init_phase_locked(float* phase) {
    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx * D;
        int shadow_base = (idx + total) * D;
        float theta = ((float)rand() / RAND_MAX) * TWO_PI;
        phase[base + 0] = theta;
        phase[shadow_base + 0] = fmodf(theta + (float)M_PI, TWO_PI);
        for (int d = 1; d < D; d++) {
            float val = ((float)rand() / RAND_MAX) * 0.1f;
            phase[base + d] = val;
            phase[shadow_base + d] = val;
        }
    }
}

__attribute__((always_inline))
void update_phase_locked(float* phase, float delta_theta) {
    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx * D;
        int shadow_base = (idx + total) * D;
        phase[base + 0] += delta_theta;
        if (phase[base + 0] > TWO_PI) phase[base + 0] -= TWO_PI;
        phase[shadow_base + 0] += delta_theta;
        if (phase[shadow_base + 0] > TWO_PI) phase[shadow_base + 0] -= TWO_PI;
    }
}

void compute_interference(float* result, float* phase) {
    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx * D;
        int shadow_base = (idx + total) * D;
        result[idx] = phase[base + 0] + phase[shadow_base + 0];
    }
}

void clone_and_shift(float* restrict primary, float* restrict shadow, float* restrict result, float delta_theta) {
    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx * D;
        float theta = primary[base + 0];
        shadow[base + 0] = fmodf(theta + (float)M_PI, TWO_PI);
        primary[base + 0] += delta_theta;
        if (primary[base + 0] > TWO_PI) primary[base + 0] -= TWO_PI;
        result[idx] = primary[base + 0] + shadow[base + 0];
    }
}

int main() {
    srand((unsigned int)time(NULL));

    // Phase-Locked Benchmark
    float* phase = allocate_field(2);
    float* result = (float*)malloc(sizeof(float) * K * PATCH * PATCH);
    init_phase_locked(phase);

    clock_t start_locked = clock();
    for (int step = 0; step < STEPS; step++) {
        update_phase_locked(phase, 0.01f);
        compute_interference(result, phase);
    }
    clock_t end_locked = clock();

    // Clone-and-Shift Benchmark
    float* primary = allocate_field(1);
    float* shadow = allocate_field(1);
    float* result2 = (float*)malloc(sizeof(float) * K * PATCH * PATCH);

    for (int idx = 0; idx < K * PATCH * PATCH * D; idx++) {
        primary[idx] = ((float)rand() / RAND_MAX) * TWO_PI;
    }

    clock_t start_clone = clock();
    for (int step = 0; step < STEPS; step++) {
        clone_and_shift(primary, shadow, result2, 0.01f);
    }
    clock_t end_clone = clock();

    double time_locked = (double)(end_locked - start_locked) / CLOCKS_PER_SEC;
    double time_clone = (double)(end_clone - start_clone) / CLOCKS_PER_SEC;

    printf("\nPhase-Locked Kernel completed %d steps in %.4f seconds\n", STEPS, time_locked);
    printf("Clone-and-Shift Kernel completed %d steps in %.4f seconds\n", STEPS, time_clone);
    printf("Speedup: %.2fx\n", time_clone / time_locked);

    free(phase);
    free(primary);
    free(shadow);
    free(result);
    free(result2);
    return 0;
}

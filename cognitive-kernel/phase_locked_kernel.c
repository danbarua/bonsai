// phase_locked_kernel.c
// Benchmark: Phase-Locked vs Clone-and-Shift Implementations

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <assert.h>

/*
θ and θ_shadow interleaved per oscillator
*/
#define CHANNELS 2 

/*
sets the width and height of a 2D oscillator field patch.
So each field layer is 64×64 = 4096 spatial units.
*/
#define PATCH 2 //64

/*
defines the number of such patches or layers, which might correspond to different feature maps, 
concept channels, or frequency bands (especially in wave or tensor models)
*/
#define K 16 //16

/*
Each oscillator in the phase field is represented using 4 float values.
These correspond to:
- Phase (θ) — the key value updated during simulation
- Amplitude (A) or placeholder for future extension
- Position or ID (x) — not explicitly used yet
- Time index or potential feature slot (t) — again, optional/future use
Right now, only the 0th index (θ) is actively used in updates and interference computations.
The other three channels (d = 1, 2, 3) are initialized but unused—likely intended for future expansion 
or compatibility with a richer cognitive kernel (e.g., [θ, A, x, t] in the 4D oscillator model).
*/
#define D 4

/*
This layout gives you:
Multi-channel support (K layers)
Spatially distributed oscillators (64×64 grid per layer)
Room for multidimensional oscillator state vectors (D=4) for future use
*/

#define STEPS 1000000
#define TWO_PI (2.0 * M_PI)

#define random_phase() (((float)arc4random() / UINT_MAX) * TWO_PI)
#define opposite_phase(X) (fmodf(X + (float)M_PI, TWO_PI))

// ---- Contiguous Implementation

float* allocate_field_contiguous(int multiplier, int dimensions) {
    // store θ, shadow_θ contiguously
    return (float*)malloc(sizeof(float) * multiplier * K * PATCH * PATCH * dimensions);
}

void init_phase_locked_contiguous(float* phase, int dimensions) {
    // Number of Oscillators = K layers of a PATCH^2 2-D Grid
    int total = K * PATCH * PATCH;

    for (int idx = 0; idx < total; idx++) {
        // index of oscillator with Dimensionality = dimensions
        int base = idx * dimensions;

        // similarly, index of shadow oscillator in memory allocated
        // at the end of the primary osccillator field
        int shadow_base = (base + total) * dimensions;

        // pick a number
        float theta = random_phase();

        // this is the phase of the oscillator
        phase[base + 0] = theta;

        // its shadow oscillator has the opposite phase
        phase[shadow_base + 0] = opposite_phase(theta);

        // populate the remaining Dimensions of the
        // oscillator vector with random values
        for (int d = 1; d < dimensions; d++) {
            float val = ((float)arc4random() / UINT_MAX) * 0.1f;
            phase[base + d] = val;
            phase[shadow_base + d] = val;
        }
    }
}

__attribute__((always_inline))
void update_phase_locked_contiguous(float* phase, int dimensions, float delta_theta) {
    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx * dimensions;
        int shadow_base = (idx + total) * dimensions;

        // phase wraparound - permitting errors in primary
        phase[base + 0] += delta_theta;
        if (phase[base + 0] > TWO_PI) phase[base + 0] -= TWO_PI;

        // phase wraparound - permitting errors in shadow
        phase[shadow_base + 0] += delta_theta;
        if (phase[shadow_base + 0] > TWO_PI) phase[shadow_base + 0] -= TWO_PI;
    }
}

void compute_interference_contiguous(float* result, int dimensions, float* phase) {
    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx * dimensions;
        int shadow_base = (idx + total) * dimensions;
        result[idx] = phase[base + 0] + phase[shadow_base + 0];
    }
}

// ---- Interleaved Implementation

float* allocate_field_interleaved(int multiplier, int dimensions) {
    // Use AoS layout (interleave θ, shadow_θ instead of storing them contiguously)
    // This gives compilers an easier time predicting memory reuse and may eliminate stall cycles.
    return (float*)malloc(sizeof(float) * K * PATCH * PATCH * multiplier * dimensions);
}

void init_phase_locked_interleaved(float* phase, int dimensions) {
    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx * dimensions;
        int shadow_base = (idx + 1) * dimensions;
        float theta = random_phase();
        phase[base] = theta;
        phase[shadow_base] = opposite_phase(theta);
    }
}

__attribute__((always_inline))
void update_phase_locked_interleaved(float* phase, int dimensions, float delta_theta) {
      int total = K * PATCH * PATCH;
      for (int idx = 0; idx < total; idx++) {
          int base = idx * dimensions;
          int shadow_base = (idx + 1) * dimensions;
    
          phase[base] += delta_theta;
          if (phase[base + 0] > TWO_PI) phase[shadow_base + 0] -= TWO_PI;
          phase[shadow_base] += delta_theta;
          if (phase[shadow_base + 0] > TWO_PI) phase[shadow_base + 0] -= TWO_PI;
      }
}

void compute_interference_interleaved(float* result, int dimensions, float* phase) {
    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx * dimensions;
        int shadow_base = (idx + 1) * dimensions;
        result[idx] = phase[base + 0] + phase[shadow_base + 0];
    }
}



// ---- Clone and Shift Implementation

void clone_and_shift(float* restrict primary, float* restrict shadow, float* restrict result, int dimensions, float delta_theta) {
    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx * dimensions;
        float theta = primary[base + 0];
        shadow[base + 0] = fmodf(theta + (float)M_PI, TWO_PI);
        primary[base + 0] += delta_theta;
        if (primary[base + 0] > TWO_PI) primary[base + 0] -= TWO_PI;
        if (shadow[base + 0] > TWO_PI) shadow[base + 0] -= TWO_PI;
        result[idx] = primary[base + 0] + shadow[base + 0];
    }
}

void benchmark(int d)
{
    // Phase-Locked Benchmark - contiguous memory allocation
    float *phase_contiguous = allocate_field_contiguous(2, d);
    float *result_phase_locked_contiguous = (float *)malloc(sizeof(float) * K * PATCH * PATCH);
    init_phase_locked_contiguous(phase_contiguous, d);

    clock_t start_locked_contiguous = clock();
    for (int step = 0; step < STEPS; step++)
    {
        update_phase_locked_contiguous(phase_contiguous, d, 0.01f);
        compute_interference_contiguous(result_phase_locked_contiguous, d, phase_contiguous);
    }
    clock_t end_locked_contiguous = clock();

    double time_contiguous = (double)(end_locked_contiguous - start_locked_contiguous) / CLOCKS_PER_SEC;
    printf("Phase-Locked Kernel (contiguous) K=%d D=%d completed %d steps in %.4f seconds\n", K, d, STEPS, time_contiguous);

    free(phase_contiguous);
    free(result_phase_locked_contiguous);

    // --------

    // Phase-Locked Benchmark - interleaved memory allocation
    float *phase_interleaved = allocate_field_interleaved(2, d);
    float *result_phase_locked_interleaved = (float *)malloc(sizeof(float) * K * PATCH * PATCH);
    init_phase_locked_interleaved(phase_interleaved, d);

    clock_t start_locked_lineterleaved = clock();
    for (int step = 0; step < STEPS; step++)
    {
        update_phase_locked_interleaved(phase_interleaved, d, 0.01f);
        compute_interference_interleaved(result_phase_locked_interleaved, d, phase_interleaved);
    }
    clock_t end_locked_interleaved = clock();
    double time_interleaved = (double)(end_locked_interleaved - start_locked_lineterleaved) / CLOCKS_PER_SEC;
    printf("Phase-Locked Kernel (interleaved) K=%d D=%d completed %d steps in %.4f seconds\n", K, d, STEPS, time_interleaved);

    free(phase_interleaved);
    free(result_phase_locked_interleaved);

    // --------

    // Clone-and-Shift Benchmark
    float *primary = allocate_field_contiguous(1, d);
    float *shadow = allocate_field_contiguous(1, d);
    float *result_clone_and_shift = (float *)malloc(sizeof(float) * K * PATCH * PATCH);

    for (int idx = 0; idx < K * PATCH * PATCH; idx++)
    {
        int base = idx * d;
        primary[base] = random_phase();
    }

    clock_t start_clone = clock();
    for (int step = 0; step < STEPS; step++)
    {
        clone_and_shift(primary, shadow, result_clone_and_shift, d, 0.01f);
    }
    clock_t end_clone = clock();

    // --------

    double time_clone = (double)(end_clone - start_clone) / CLOCKS_PER_SEC;
    printf("Clone-and-Shift Kernel K=%d D=%d completed %d steps in %.4f seconds", K, d, STEPS, time_clone);
    free(primary);
    free(shadow);
    free(result_clone_and_shift);
}

void verify_results(float* phase, int dimensions, int steps) {
    float* phase_ref = allocate_field_contiguous(2, dimensions);
    init_phase_locked_contiguous(phase_ref, dimensions);

    for (int step = 0; step < steps; step++) {
        update_phase_locked_contiguous(phase_ref, dimensions, 0.01f);
    }

    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx * dimensions;
        // add a small tolerance for floating point comparison
        assert(fabsf(phase[base + 0] - phase_ref[base + 0]) < 1e-5);
    }
    free(phase_ref);
    printf("Verification passed!\n");
}

int main() {
    srand((unsigned int)time(NULL));

    for (int d = 1; d <= D; d++){
        printf("\n--------");
        printf("\nBenchmarking with D=%d", d);
        printf("\n--------\n");
        benchmark(d);
        printf("\n--------\n");
    }

    return 0;
}
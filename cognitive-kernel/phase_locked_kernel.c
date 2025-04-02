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
#define PATCH 64 // default 64

/*
defines the number of such patches or layers, which might correspond to different feature maps, 
concept channels, or frequency bands (especially in wave or tensor models)
*/
#define K 16 // default 16

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

#define STEPS 10000
#define TWO_PI (2.0 * M_PI)

#define random_phase() (((float)arc4random() / UINT_MAX) * TWO_PI)
#define opposite_phase(X) (fmodf(X + (float)M_PI, TWO_PI))

// ---- Contiguous Implementation

float* allocate_field_contiguous(int multiplier) {
    // store θ, shadow_θ contiguously
    return (float*)malloc(sizeof(float) * multiplier * K * PATCH * PATCH);
}

void init_phase_locked_contiguous(float* phase) {
    // Number of Oscillators = K layers of a PATCH^2 2-D Grid
    int total = K * PATCH * PATCH;

    for (int idx = 0; idx < total; idx++) {
        // index of oscillator with Dimensionality = dimensions
        int base = idx;

        // similarly, index of shadow oscillator in memory allocated
        // at the end of the primary osccillator field
        int shadow_base = (base ) + total;

        // pick a number
        float theta = random_phase();

        // this is the phase of the oscillator
        phase[base] = theta;

        // its shadow oscillator has the opposite phase
        phase[shadow_base] = opposite_phase(theta);
    }
}

__attribute__((always_inline))
void update_phase_locked_contiguous(float* phase, float delta_theta) {
    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx;
        int shadow_base = (base + total);

        // phase wraparound - permitting errors in primary
        phase[base] += delta_theta;
        if (phase[base] > TWO_PI) phase[base] -= TWO_PI;

        // phase wraparound - permitting errors in shadow
        phase[shadow_base] += delta_theta;
        if (phase[shadow_base] > TWO_PI) phase[shadow_base] -= TWO_PI;
    }
}

__attribute__((always_inline))
void compute_interference_contiguous(float* result, float* phase) {
    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx;
        int shadow_base = (base + total);
        float sum = phase[base] + phase[shadow_base];// - M_PI;
        result[idx] = sum;
    }
}

// ---- Interleaved Implementation

float* allocate_field_interleaved(int multiplier) {
    // Use AoS layout (interleave θ, shadow_θ instead of storing them contiguously)
    // This gives compilers an easier time predicting memory reuse and may eliminate stall cycles.
    return (float*)malloc(sizeof(float) * K * PATCH * PATCH * multiplier);
}

void init_phase_locked_interleaved(float* phase) {
    int total = K * PATCH * PATCH * CHANNELS;
    for (int idx = 0; idx < total; idx += CHANNELS) {
        int base = idx;
        int shadow_base = (idx + 1);
        float theta = random_phase();
        phase[base] = theta;
        phase[shadow_base] = opposite_phase(theta);
    }
}

__attribute__((always_inline))
void update_phase_locked_interleaved(float* phase, float delta_theta) {
      int total = K * PATCH * PATCH * CHANNELS;
      for (int idx = 0; idx < total; idx += CHANNELS) {
          int base = idx;
          int shadow_base = (idx + 1);

          phase[base] += delta_theta;
          if (phase[base] > TWO_PI) phase[base] -= TWO_PI; // Corrected line
          phase[shadow_base] += delta_theta;
          if (phase[shadow_base] > TWO_PI) phase[shadow_base] -= TWO_PI;
      }
}

__attribute__((always_inline))
void compute_interference_interleaved(float* result, float* phase) {
    int total = K * PATCH * PATCH * CHANNELS;
    for (int idx = 0; idx < total; idx += CHANNELS) {
        int base = idx;
        int shadow_base = (idx + 1);
        result[idx / CHANNELS] = phase[base] + phase[shadow_base];
    }
}

// ---- Clone and Shift Implementation
__attribute__((always_inline))
void clone_and_shift(float* restrict primary, float* restrict shadow, float* restrict result, float delta_theta) {
    int total = K * PATCH * PATCH;
    for (int idx = 0; idx < total; idx++) {
        int base = idx;
        float theta = primary[base];
        shadow[base] = fmodf(theta + (float)M_PI, TWO_PI);
        primary[base] += delta_theta;
        if (primary[base] > TWO_PI) primary[base] -= TWO_PI;
        if (shadow[base] > TWO_PI) shadow[base] -= TWO_PI;
        result[idx] = primary[base] + shadow[base];
    }
}

int main() {
    srand((unsigned int)time(NULL));

    // Phase-Locked Benchmark - contiguous memory allocation
    float *phase_contiguous = allocate_field_contiguous(2);
    float *result_phase_locked_contiguous = (float *)malloc(sizeof(float) * K * PATCH * PATCH);
    init_phase_locked_contiguous(phase_contiguous);

    clock_t start_locked_contiguous = clock();
    for (int step = 0; step < STEPS; step++)
    {
        update_phase_locked_contiguous(phase_contiguous, 0.01f);
        compute_interference_contiguous(result_phase_locked_contiguous, phase_contiguous);
    }
    clock_t end_locked_contiguous = clock();

    double time_contiguous = (double)(end_locked_contiguous - start_locked_contiguous) / CLOCKS_PER_SEC;
    printf("Phase-Locked Kernel (contiguous) K=%d completed %d steps in %.4f seconds\n", K, STEPS, time_contiguous);

    free(phase_contiguous);
    free(result_phase_locked_contiguous);

    // --------

    // Phase-Locked Benchmark - interleaved memory allocation
    float *phase_interleaved = allocate_field_interleaved(2);
    float *result_phase_locked_interleaved = (float *)malloc(sizeof(float) * K * PATCH * PATCH);
    init_phase_locked_interleaved(phase_interleaved);

    clock_t start_locked_lineterleaved = clock();
    for (int step = 0; step < STEPS; step++)
    {
        update_phase_locked_interleaved(phase_interleaved, 0.01f);
        compute_interference_interleaved(result_phase_locked_interleaved, phase_interleaved);
    }
    clock_t end_locked_interleaved = clock();
    double time_interleaved = (double)(end_locked_interleaved - start_locked_lineterleaved) / CLOCKS_PER_SEC;
    printf("Phase-Locked Kernel (interleaved) K=%d completed %d steps in %.4f seconds\n", K, STEPS, time_interleaved);

    free(phase_interleaved);
    free(result_phase_locked_interleaved);

    // --------

    // Clone-and-Shift Benchmark
    float *primary = allocate_field_contiguous(1);
    float *shadow = allocate_field_contiguous(1);
    float *result_clone_and_shift = (float *)malloc(sizeof(float) * K * PATCH * PATCH);

    for (int idx = 0; idx < K * PATCH * PATCH; idx++)
    {
        int base = idx;
        primary[base] = random_phase();
    }

    clock_t start_clone = clock();
    for (int step = 0; step < STEPS; step++)
    {
        clone_and_shift(primary, shadow, result_clone_and_shift, 0.01f);
    }
    clock_t end_clone = clock();

    // --------

    double time_clone = (double)(end_clone - start_clone) / CLOCKS_PER_SEC;
    printf("Clone-and-Shift Kernel K=%d completed %d steps in %.4f seconds\n", K, STEPS, time_clone);
    free(primary);
    free(shadow);
    free(result_clone_and_shift);
 }
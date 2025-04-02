// avx_clone_shift.c
// AVX-optimized Clone-and-Shift kernel

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define PATCH 256
#define K 16
#define STEPS 10000
#define TWO_PI (2.0f * 3.14159265358979323846f)

#define TOTAL (K * PATCH * PATCH)
#define BLOCK_SIZE 8  // AVX processes 8 floats at a time

float* alloc_aligned_field(int size) {
    float* ptr = NULL;
    if (posix_memalign((void**)&ptr, 32, sizeof(float) * size) != 0) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }
    return ptr;
}

void init_field(float* field) {
    for (int i = 0; i < TOTAL; i++) {
        field[i] = ((float)rand() / RAND_MAX) * TWO_PI;
    }
}

void avx_clone_and_shift(float* restrict primary, float* restrict shadow, float* restrict result, float delta_theta) {
    __m256 pi_vec = _mm256_set1_ps((float)M_PI);
    __m256 two_pi_vec = _mm256_set1_ps(TWO_PI);
    __m256 delta_vec = _mm256_set1_ps(delta_theta);

    for (int i = 0; i < TOTAL; i += BLOCK_SIZE) {
        // Load 8 primary theta values
        __m256 theta = _mm256_load_ps(&primary[i]);

        // Add pi to get shadow
        __m256 theta_shadow = _mm256_add_ps(theta, pi_vec);

        // Wrap shadow if over TWO_PI
        theta_shadow = _mm256_sub_ps(theta_shadow, two_pi_vec);
        __m256 mask_shadow = _mm256_cmp_ps(theta_shadow, _mm256_setzero_ps(), _CMP_LT_OS);
        theta_shadow = _mm256_blendv_ps(theta_shadow, _mm256_add_ps(theta_shadow, two_pi_vec), mask_shadow);

        // Add delta to primary and wrap
        theta = _mm256_add_ps(theta, delta_vec);
        __m256 mask_primary = _mm256_cmp_ps(theta, two_pi_vec, _CMP_GT_OS);
        theta = _mm256_blendv_ps(theta, _mm256_sub_ps(theta, two_pi_vec), mask_primary);

        // Store updated primary
        _mm256_store_ps(&primary[i], theta);


        // // Store updated shadow
        // _mm256_store_ps(&shadow[i], theta_shadow);

        // Compute and store result
        // __m256 sum = _mm256_add_ps(theta, theta_shadow);
        // _mm256_store_ps(&result[i], sum);
    }
}

int main() {
    srand((unsigned int)time(NULL));

    float* primary = alloc_aligned_field(TOTAL);
    float* shadow = alloc_aligned_field(TOTAL);
    float* result = alloc_aligned_field(TOTAL);

    init_field(primary);

    clock_t start = clock();
    for (int step = 0; step < STEPS; step++) {
        avx_clone_and_shift(primary, shadow, result, 0.01f);
    }
    clock_t end = clock();

    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nAVX Clone-and-Shift Kernel completed %d steps in %.4f seconds\n", STEPS, duration);

    free(primary);
    free(shadow);
    free(result);
    return 0;
}

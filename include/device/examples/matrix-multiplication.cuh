/**
 * @file matrix-multiplication.cuh
 * @date 27/07/2020
 * @author Mirco De Marchi
 * @brief Header of matrix multiplication parallel and sequential algorithms.
 */

#ifndef DEVICE_EXAMPLES_MATRIX_MULTIPLICATION_H_
#define DEVICE_EXAMPLES_MATRIX_MULTIPLICATION_H_

#include <iostream>
#include <cuda_runtime.h>

#include "compare-run.cuh"
//------------------------------------------------------------------------------

typedef struct matrix_mul_params {
    uint32_t *h_matrix_a;
    uint32_t *h_matrix_b;
    uint32_t *h_matrix_res_dev;
    uint32_t *h_matrix_res_host;
    uint32_t *d_matrix_a;
    uint32_t *d_matrix_b;
    uint32_t *d_matrix_res;
} matrix_mul_params_t;
//------------------------------------------------------------------------------

/**
 * @brief Allocate and initialize all the matrix_mul_params_t fields.
 * @return The matrix_mul_params_t in a generic cast. 
 */
void *matrix_mul_init_common(const constants_t *c);
void *matrix_mul_init_device(const constants_t *c, void *params);
void *matrix_mul_init(const constants_t *c);

/**
 * @brief Calculate the Matrix Multiplication in sequential host version.
 * @param params Generic params that have to point to a matrix_mul_params_t 
 * structure.
 */
void matrix_mul_sequential(const constants_t *c, void *params, host_time_t *h_time);

/**
 * @brief Calculate the Matrix Multiplication in parallel device version.
 * @param params Generic params that have to point to a matrix_mul_params_t 
 * structure.
 */
void matrix_mul_parallel(const constants_t *c, void *params, device_time_t *d_time);

/**
 * @brief Compare the Matrix Multiplication results taken from the params in 
 * input between the parallel device version and the sequential host version.
 * @param params Generic params that have to point to a matrix_mul_params_t 
 * structure.
 */
bool matrix_mul_compare(const constants_t *c, void *params);

/**
 * @brief Erase from memory all the matrix_mul_params_t structure allocation.
 * @param params Generic params that have to point to a matrix_mul_params_t 
 * structure.
 */
void matrix_mul_free_common(void *params);
void matrix_mul_free_device(void *params);
void matrix_mul_free(void *params);

#endif  // DEVICE_EXAMPLES_MATRIX_MULTIPLICATION_H_
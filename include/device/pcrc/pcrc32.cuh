/**
 * @file pcrc32.cuh
 * @date 08/09/2020
 * @author Mirco De Marchi
 * @brief Header of 32 bit CRC parallel and sequential algorithms.
 */

#ifndef DEVICE_PCRC_PCRC32_H_
#define DEVICE_PCRC_PCRC32_H_

#include "pcrc-common.cuh"
//------------------------------------------------------------------------------

/**
 * @brief Allocate and initialize all the pcrc32_params_t fields.
 * @param c Constants for memory allocation and task management.
 * @return The pcrc32_params_t in a generic cast.
 */
void *pcrc32_init_common(const constants_t *c);
void *pcrc32_intel_init_common(const constants_t *c);
void *pcrc32_init_device(const constants_t *c, void *params);
void *pcrc32_init_device_reduction(const constants_t *c, void *params);
void *pcrc32_init_device_task_parallelism(const constants_t *c, void *params);
void *pcrc32_intel_init_device(const constants_t *c, void *params);
void *pcrc32_intel_init_device_reduction(const constants_t *c, void *params);
void *pcrc32_intel_init_device_task_parallelism(const constants_t *c, void *params);
void *pcrc32_init(const constants_t *c);
void *pcrc32_init_reduction(const constants_t *c);
void *pcrc32_init_task_parallelism(const constants_t *c);

/**
 * @brief Calculate the CRC32 in sequential host version.
 * @param c      Constants for memory allocation and task management.
 * @param params Generic params that have to point to a pcrc32_params_t 
 * structure.
 */
void pcrc32_sequential(const constants_t *c, void *params, host_time_t *h_time);
void pcrc32_sequential_bytewise(const constants_t *c, void *params, host_time_t *h_time);
void pcrc32_intel_sequential(const constants_t *c, void *params, host_time_t *h_time);
void pcrc32_intel_sequential_hw(const constants_t *c, void *params, host_time_t *h_time);

/**
 * @brief Calculate the CRC32 in parallel device version.
 * @param c      Constants for memory allocation and task management.
 * @param params Generic params that have to point to a pcrc32_params_t 
 * structure.
 */
void pcrc32_parallel(const constants_t *c, void *params, device_time_t *d_time);
void pcrc32_parallel_reduction(const constants_t *c, void *params, device_time_t *d_time);
void pcrc32_parallel_task_parallelism(const constants_t *c, void *params, device_time_t *d_time);
void pcrc32_intel_parallel(const constants_t *c, void *params, device_time_t *d_time);
void pcrc32_intel_parallel_reduction(const constants_t *c, void *params, device_time_t *d_time);
void pcrc32_intel_parallel_task_parallelism(const constants_t *c, void *params, device_time_t *d_time);

/**
 * @brief Compare the CRC results taken from the params in input between the 
 * parallel device version and the sequential host version.
 * @param params Generic params that have to point to a pcrc32_params_t 
 * structure.
 */
bool pcrc32_compare(const constants_t *c, void *params);
bool pcrc32_compare_reduction(const constants_t *c, void *params);
bool pcrc32_compare_task_parallelism(const constants_t *c, void *params);

/**
 * @brief Erase from memory all the pcrc32_params_t structure allocation.
 * @param params Generic params that have to point to a pcrc32_params_t 
 * structure.
 */
void pcrc32_free_common(void *params);
void pcrc32_free_device(void *params);
void pcrc32_free_device_reduction(void *params);
void pcrc32_free_device_task_parallelism(void *params);
void pcrc32_free(void *params);
void pcrc32_free_reduction(void *params);
void pcrc32_free_task_parallelism(void *params);

#endif  // DEVICE_PCRC_PCRC32_H_
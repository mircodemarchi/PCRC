/**
 * @file pcrc8.cuh
 * @date 01/08/2020
 * @author Mirco De Marchi
 * @brief Header of 8 bit CRC parallel and sequential algorithms.
 */

#ifndef DEVICE_PCRC_PCRC8_H_
#define DEVICE_PCRC_PCRC8_H_

#include "pcrc-common.cuh"
//------------------------------------------------------------------------------

/**
 * @brief Allocate and initialize all the pcrc8_params_t fields.
 * @param c Constants for memory allocation and task management.
 * @return The pcrc8_params_t in a generic cast. 
 */
void *pcrc8_init_common(const constants_t *c);
void *pcrc8_init_device(const constants_t *c, void *params);
void *pcrc8_init_device_reduction(const constants_t *c, void *params);
void *pcrc8_init_device_task_parallelism(const constants_t *c, void *params);
void *pcrc8_init(const constants_t *c);
void *pcrc8_init_reduction(const constants_t *c);
void *pcrc8_init_task_parallelism(const constants_t *c);

/**
 * @brief Calculate the CRC8 in sequential host version.
 * @param c      Constants for memory allocation and task management.
 * @param params Generic params that have to point to a pcrc8_params_t 
 * structure.
 */
void pcrc8_sequential(const constants_t *c, void *params, host_time_t *h_time);
void pcrc8_sequential_bytewise(const constants_t *c, void *params, host_time_t *h_time);

/**
 * @brief Calculate the CRC32 in parallel device version.
 * @param params Generic params that have to point to a pcrc8_params_t 
 * structure.
 */
void pcrc8_parallel(const constants_t *c, void *params, device_time_t *d_time);
void pcrc8_parallel_reduction(const constants_t *c, void *params, device_time_t *d_time);
void pcrc8_parallel_task_parallelism(const constants_t *c, void *params, device_time_t *d_time);

/**
 * @brief Compare the CRC results taken from the params in input between the 
 * parallel device version and the sequential host version.
 * @param c      Constants for memory allocation and task management.
 * @param params Generic params that have to point to a pcrc8_params_t 
 * structure.
 */
bool pcrc8_compare(const constants_t *c, void *params);
bool pcrc8_compare_reduction(const constants_t *c, void *params);
bool pcrc8_compare_task_parallelism(const constants_t *c, void *params);

/**
 * @brief Erase from memory all the pcrc8_params_t structure allocation.
 * @param params Generic params that have to point to a pcrc8_params_t 
 * structure.
 */
void pcrc8_free_common(void *params);
void pcrc8_free_device(void *params);
void pcrc8_free_device_reduction(void *params);
void pcrc8_free_device_task_parallelism(void *params);
void pcrc8_free(void *params);
void pcrc8_free_reduction(void *params);
void pcrc8_free_task_parallelism(void *params);

#endif  // DEVICE_PCRC_PCRC8_H_
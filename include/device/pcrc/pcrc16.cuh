/**
 * @file pcrc16.cuh
 * @date 08/09/2020
 * @author Mirco De Marchi
 * @brief Header of 16 bit CRC parallel and sequential algorithms.
 */

#ifndef DEVICE_PCRC_PCRC16_H_
#define DEVICE_PCRC_PCRC16_H_

#include "pcrc-common.cuh"
//------------------------------------------------------------------------------

/**
 * @brief Allocate and initialize all the pcrc16_params_t fields.
 * @param c Constants for memory allocation and task management.
 * @return The pcrc16_params_t in a generic cast.
 */
void *pcrc16_init_common(const constants_t *c);
void *pcrc16_init_device(const constants_t *c, void *params);
void *pcrc16_init_device_reduction(const constants_t *c, void *params);
void *pcrc16_init_device_task_parallelism(const constants_t *c, void *params);
void *pcrc16_init(const constants_t *c);
void *pcrc16_init_reduction(const constants_t *c);
void *pcrc16_init_task_parallelism(const constants_t *c);

/**
 * @brief Calculate the CRC16 in sequential host version.
 * @param c      Constants for memory allocation and task management.
 * @param params Generic params that have to point to a pcrc16_params_t 
 * structure.
 */
void pcrc16_sequential(const constants_t *c, void *params);
void pcrc16_sequential_bytewise(const constants_t *c, void *params);

/**
 * @brief Calculate the CRC32 in parallel device version.
 * @param c      Constants for memory allocation and task management.
 * @param params Generic params that have to point to a pcrc16_params_t 
 * structure.
 */
void pcrc16_parallel(const constants_t *c, void *params);
void pcrc16_parallel_reduction(const constants_t *c, void *params);
void pcrc16_parallel_task_parallelism(const constants_t *c, void *params);

/**
 * @brief Compare the CRC results taken from the params in input between the 
 * parallel device version and the sequential host version.
 * @param params Generic params that have to point to a pcrc16_params_t 
 * structure.
 */
bool pcrc16_compare(const constants_t *c, void *params);
bool pcrc16_compare_reduction(const constants_t *c, void *params);
bool pcrc16_compare_task_parallelism(const constants_t *c, void *params);

/**
 * @brief Erase from memory all the pcrc16_params_t structure allocation.
 * @param params Generic params that have to point to a pcrc16_params_t 
 * structure.
 */
void pcrc16_free_common(void *params);
void pcrc16_free_device(void *params);
void pcrc16_free_device_reduction(void *params);
void pcrc16_free_device_task_parallelism(void *params);
void pcrc16_free(void *params);
void pcrc16_free_reduction(void *params);
void pcrc16_free_task_parallelism(void *params);

#endif  // DEVICE_PCRC_PCRC16_H_
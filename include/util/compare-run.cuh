/**
 * @file compare-run.cuh
 * @date 26/07/2020
 * @author Luigi Capogrosso, Mirco De Marchi
 * @brief Header compare run functionality.
 * 
 * Compare two different version of the same algorithm, in particular, it 
 * compares an algorithm on a device (a GPU, parallel) and the other on the 
 * host (a CPU, sequential).
 * All of the necessary informations are save in a compare descriptor structure.
 */

#ifndef UTIL_COMPARE_RUN_H_
#define UTIL_COMPARE_RUN_H_

#include <iostream>

#include "Timer.cuh"

#include "log.h"
//------------------------------------------------------------------------------

/** @brief Parameter for CUDA memory allocation and thread dim. */  
typedef struct constants {
    size_t N;           ///< Size of input data.
    // size_t BLOCK_SIZE;  ///< Thread block dim.
    uint8_t STREAM_DIM; ///< Number of stream for task parallelism.
    size_t  SEG_SIZE;   ///< Segment size for task parallelism.
} constants_t;

/** 
 * @brief Algorithm function type definition that takes as parameters the input 
 * and output of the algorithm as an unique generic pointer.
 */
typedef void (*algorithm_t)(const constants_t *, void *);

/**
 * @brief Initialization function type definition that have to allocate all 
 * input and output parameters and initialize it.
 */
typedef void *(*init_t)(const constants_t *);

/**
 * @brief Initialization function type definition that takes in addition as 
 * parameter the common allocated data and have to allocate only the uncommon 
 * data and run only the specific device init steps.
 */
typedef void *(*init_device_t)(const constants_t *, void *);

/**
 * @brief Free function type definition that erase the input and output 
 * parameters allocated.
 */
typedef void (*free_t)(void *);

/**
 * @brief Free function type definition that erase only the device specific 
 * allocation.
 */
typedef void (*free_device_t)(void *);

/**
 * @brief Compare function type definition that takes the input and output 
 * of two different algoritm in an unique pointer and check if they produce the 
 * same result.
 */
typedef bool (*compare_t)(const constants_t *, void *);

/**
 * @brief Descriptor used to save all necessary data and callback used to 
 * compare the parallel and sequential implementation.
 */
typedef struct compare_descriptor {
    constants_t constants;          ///< Input params of CUDA implementation.
    init_t      init;               ///< Initialization function.
    algorithm_t host_algorithm;     ///< Host algorithm function.
    algorithm_t device_algorithm;   ///< Device algorithm function.
    free_t      free;               ///< Free function.
    compare_t   compare;            ///< Compare function.
    const char *info;               ///< String for algorithm description.
} compare_descriptor_t;

/**
 * @brief Descriptor similar to the compare_descriptor, used to save only one 
 * time all the common data and not repeating the allocations each iteration 
 * time.
 */
typedef struct compare_common_descriptor {
    init_device_t init_device;      ///< Initialization specific function.
    algorithm_t   host_algorithm;   ///< Host algorithm function.
    algorithm_t   device_algorithm; ///< Device algorithm function.
    free_device_t free_device;      ///< Free specific function.
    compare_t     compare;          ///< Compare function.
    const char   *info;             ///< String for algorithm description.
} compare_common_descriptor_t;
//------------------------------------------------------------------------------

/**
 * @brief Compare the parallel and sequential implementation of an algorithm 
 * described in the compare descriptor.
 * @param run A descriptor pointer with all the necessary informations for the 
 * sequential and parallel comparison.
 */
void compare_run(compare_descriptor_t *run);

/**
 * @brief Compare the parallel and sequential implementation of an algorithm 
 * described in the compare descriptor without repeating the allocation data.
 * @param consts Constants for memory allocation and task management.
 * @param params The common allocated data.
 * @param run A descriptor pointer with all the necessary informations for the 
 * sequential and parallel comparison but with the init function that already
 * takes the common data.
 */
void compare_common_run(const constants_t *consts, 
                        void *params, compare_common_descriptor_t *run);

#endif  // UTIL_COMPARE_RUN_H_
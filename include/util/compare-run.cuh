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

/** @brief Time statistics performed inside an host algorithm function. */
typedef struct host_time {
    bool    is_initialized; ///< If setted true, the time stats will be showed.
    float   exec_time;      ///< Sequential algorithm execution time.
} host_time_t;

/** @brief Time statistics performed inside a device algorithm function. */
typedef struct device_time {
    bool    is_initialized; ///< If setted true, the time stats will be showed.
    bool    is_task_parallelism; ///< If setted true, only kernel_time stat will be showed.
    float   htod_time;      ///< Host to device transfer data time.
    float   dtoh_time;      ///< Device to host transfer data time.
    float   kernel_time;    ///< Parallel algorithm kernel execution time.
} device_time_t;

/** 
 * @brief Host algorithm function type definition that takes as parameters 
 * the costant threads dim and memory allocation, the input and output of the 
 * algorithm as an unique generic pointer, and the time statistics structure 
 * to optionally initialize. 
 */
typedef void (*host_algorithm_t)(const constants_t *, void *, host_time_t *);


/** 
 * @brief Device algorithm function type definition that takes as parameters 
 * the costant threads dim and memory allocation, the input and output of the 
 * algorithm as an unique generic pointer, and the time statistics structure 
 * to optionally initialize. 
 */
typedef void (*device_algorithm_t)(const constants_t *, void *, device_time_t *);

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
    host_algorithm_t    host_algorithm;     ///< Host algorithm function.
    device_algorithm_t  device_algorithm;   ///< Device algorithm function.
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
    host_algorithm_t    host_algorithm;     ///< Host algorithm function.
    device_algorithm_t  device_algorithm;   ///< Device algorithm function.
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
/**
 * @file pcrc-common.cuh
 * @date 07/09/2020
 * @author Mirco De Marchi
 * @brief Header of common data for PCRC algorithms.
 */

#ifndef DEVICE_PCRC_PCRC_COMMON_H_
#define DEVICE_PCRC_PCRC_COMMON_H_

#include <iostream>
#include <cuda_runtime.h>

#include "compare-run.cuh"
//------------------------------------------------------------------------------

#define BLOCK_SIZE 128   ///< Thread block dim.

#define CHECK_BIT_IDX   8
#define CHECK_BIT       (1 << CHECK_BIT_IDX)

#define CEIL(w, d) ((((w) - 1) / (d)) + 1)
//------------------------------------------------------------------------------

typedef struct pcrc8_params {
    uint8_t *h_message;
    uint8_t *h_beta;
    uint8_t *h_crc_partial_res_dev;
    uint8_t h_crc_res_dev;
    uint8_t h_crc_res_host;
    uint8_t *d_message;
    uint8_t *d_beta;
    uint8_t *d_crc_partial_res;
    uint8_t generator;
} pcrc8_params_t;

typedef struct pcrc16_params {
    uint8_t *h_message;
    uint16_t *h_message_16;
    uint16_t *h_beta;
    uint16_t *h_crc_partial_res_dev;
    uint16_t h_crc_res_dev;
    uint16_t h_crc_res_host;
    uint16_t *d_message;
    uint16_t *d_beta;
    uint16_t *d_crc_partial_res;
    uint16_t generator;
} pcrc16_params_t;

typedef struct pcrc32_params {
    uint8_t *h_message;
    uint32_t *h_message_32;
    uint32_t *h_beta;
    uint32_t *h_crc_partial_res_dev;
    uint32_t h_crc_res_dev;
    uint32_t h_crc_res_host;
    uint32_t *d_message;
    uint32_t *d_beta;
    uint32_t *d_crc_partial_res;
    uint32_t generator;
} pcrc32_params_t;

#endif  // DEVICE_PCRC_PCRC_COMMON_H_
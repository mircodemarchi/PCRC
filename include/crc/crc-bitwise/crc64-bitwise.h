/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Header file of CRC64 bitwise implementation.
 */

#ifndef CRC_CRC_BITWISE_CRC64_BITWISE_H_
#define CRC_CRC_BITWISE_CRC64_BITWISE_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "generator.h"
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CRC64 bitwise with generator CRC64 (0x42F0E1EBA9EA3693) and 
 * initial value 0x0000000000000000.
 * @param buffer Message from which generate the CRC64 code.
 * @param length Size of the input message.
 * @return CRC64 code.
 */
uint64_t crc64_bitwise(const uint8_t *buffer, size_t length);

#ifdef __cplusplus
}
#endif

#endif // CRC_CRC_BITWISE_CRC64_BITWISE_H_

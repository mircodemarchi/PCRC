/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Header file of CRC32 bitwise implementation.
 */

#ifndef CRC_CRC_BITWISE_CRC32_BITWISE_H_
#define CRC_CRC_BITWISE_CRC32_BITWISE_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "generator.h"
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CRC32 bitwise with generator CRC32 (0x04C11DB7) and 
 * initial value 0x00000000.
 * @param buffer Message from which generate the CRC32 code.
 * @param length Size of the input message.
 * @return CRC32 code.
 */
uint32_t crc32_bitwise(const uint8_t *buffer, size_t length);

#ifdef __cplusplus
}
#endif

#endif // CRC_CRC_BITWISE_CRC32_BITWISE_H_

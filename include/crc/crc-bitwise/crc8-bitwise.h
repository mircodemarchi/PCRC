/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Header file of CRC8 bitwise implementation.
 */

#ifndef CRC_CRC_BITWISE_CRC8_BITWISE_H_
#define CRC_CRC_BITWISE_CRC8_BITWISE_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "generator.h"
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CRC8 bitwise with generator CRC8_SAE (0x1D) and initial value 0x00.
 * @param buffer Message from which generate the CRC8 code.
 * @param length Size of the input message.
 * @return CRC8 code.
 */
uint8_t crc8_bitwise(const uint8_t *buffer, size_t length);

#ifdef __cplusplus
}
#endif

#endif // CRC_CRC_BITWISE_CRC8_BITWISE_H_

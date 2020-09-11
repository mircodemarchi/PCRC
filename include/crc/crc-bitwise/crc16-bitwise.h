/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Header file of CRC16 bitwise implementation.
 */

#ifndef CRC_CRC_BITWISE_CRC16_BITWISE_H_
#define CRC_CRC_BITWISE_CRC16_BITWISE_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "generator.h"
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CRC16 bitwise with generator CRC16_CCITT (0x1012) and 
 * initial value 0x0000.
 * @param buffer Message from which generate the CRC16 code.
 * @param length Size of the input message.
 * @return CRC16 code.
 */
uint16_t crc16_bitwise(const uint8_t *buffer, size_t length);

#ifdef __cplusplus
}
#endif

#endif // CRC_CRC_BITWISE_CRC16_BITWISE_H_
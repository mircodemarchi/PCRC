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
 * @brief Standard CRC32 bitwise.
 * @param crc    CRC code initial value
 * @param buffer Message from which generate the CRC32 code.
 * @param length Size of the input message.
 * @param gen    Polynomial generator.
 * @return CRC32 code.
 */
uint32_t crc32_std_bitwise(uint32_t crc, const uint8_t *buffer, 
                           size_t length, const uint32_t gen);

/**
 * @brief Reversed CRC32 bitwise.
 * In this implementation the initial value of crc is reversed, and each byte 
 * of the message is reversed, but the bytes message order remains the same.
 * @param crc    CRC code initial value
 * @param buffer Message from which generate the CRC32 code.
 * @param length Size of the input message.
 * @param gen    Polynomial generator.
 * @return CRC32 code.
 */
uint32_t crc32_rev_bitwise(uint32_t crc, const uint8_t *buffer, 
                           size_t length, const uint32_t gen);

/**
 * @brief Reversed CRC32 bitwise using hardware instructions.
 * @param crc    CRC code initial value
 * @param buffer Message from which generate the CRC32 code.
 * @param length Size of the input message.
 * @param gen    Polynomial generator.
 * @return CRC32 code.
 */
uint32_t crc32_rev_hw_bitwise(uint32_t crc, const uint8_t *buffer, 
                              size_t length, const uint32_t gen);

/**
 * @brief CRC32 bitwise with generator CRC32 (0x04C11DB7) and 
 * initial value 0x00000000.
 * @param buffer Message from which generate the CRC32 code.
 * @param length Size of the input message.
 * @return CRC32 code.
 */
uint32_t crc32_bitwise(const uint8_t *buffer, size_t length);

/**
 * @brief CRC32 bitwise input and output reversed with generator 
 * CRC32 (0x11EDC6F41) and custom initial value.
 * @param crc    CRC code initial value.
 * @param buffer Message from which generate the CRC32 code.
 * @param length Size of the input message.
 * @return CRC32 code.
 */
uint32_t crc32_intel_bitwise(uint32_t crc, const uint8_t *buffer, 
                             size_t length);

/**
 * @brief CRC32 bitwise input and output reversed with generator 
 * CRC32 (0x11EDC6F41) and custom initial value, using hardware instructions.
 * @param crc    CRC code initial value.
 * @param buffer Message from which generate the CRC32 code.
 * @param length Size of the input message.
 * @return CRC32 code.
 */
uint32_t crc32_intel_hw_bitwise(uint32_t crc, const uint8_t *buffer, 
                                size_t length);

#ifdef __cplusplus
}
#endif

#endif // CRC_CRC_BITWISE_CRC32_BITWISE_H_

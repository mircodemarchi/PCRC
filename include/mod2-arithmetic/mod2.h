/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Header file of binary modulo 2 arithmetic remainder operation.
 */

#ifndef MOD2_ARITHMETIC_MOD2_H_
#define MOD2_ARITHMETIC_MOD2_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "generator.h"
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Binary modulo 2 arithmetic 8 bit remainder of the division operation.
 * @param divident Divident buffer.
 * @param length Size of the divident buffer.
 * @param divisor The divisor of the modulo 2 division operation in 8 bit.
 * @return Remainder of modulo 2 division operation in 8 bit.
 */
uint8_t mod2_8(const uint8_t *divident, size_t length, uint8_t divisor);

/**
 * @brief Binary modulo 2 arithmetic 16 bit remainder of the division operation.
 * @param divident Divident buffer.
 * @param length Size of the divident buffer.
 * @param divisor The divisor of the modulo 2 division operation in 16 bit.
 * @return Remainder of modulo 2 division operation in 16 bit.
 */
uint16_t mod2_16(const uint8_t *divident, size_t length, uint16_t divisor);

/**
 * @brief Binary modulo 2 arithmetic 16 bit remainder of the division operation
 * and 16 bit divident in input.
 * @param divident The divident 16 bit value.
 * @param divisor The divisor 16 bit value.
 * @return Remainder of modulo 2 division operation in 16 bit.
 */
uint16_t mod2_16_u16(uint16_t divident, uint16_t divisor);

/**
 * @brief Binary modulo 2 arithmetic 32 bit remainder of the division operation.
 * @param divident Divident buffer.
 * @param length Size of the divident buffer.
 * @param divisor The divisor of the modulo 2 division operation in 32 bit.
 * @return Remainder of modulo 2 division operation in 32 bit.
 */
uint32_t mod2_32(const uint8_t *divident, size_t length, uint32_t divisor);

/**
 * @brief Binary modulo 2 arithmetic 64 bit remainder of the division operation.
 * @param divident Divident buffer.
 * @param length Size of the divident buffer.
 * @param divisor The divisor of the modulo 2 division operation in 64 bit.
 * @return Remainder of modulo 2 division operation in 64 bit.
 */
uint64_t mod2_64(const uint8_t *divident, size_t length, uint64_t divisor);

#ifdef __cplusplus
}
#endif

#endif // MOD2_ARITHMETIC_MOD2_H_

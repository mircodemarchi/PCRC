/**
 * @author Mirco De Marchi
 * @date 01/08/2020
 * @brief Header file of binary modulo 2 arithmetic multiply operation.
 */

#ifndef MOD2_ARITHMETIC_MUL2_H_
#define MOD2_ARITHMETIC_MUL2_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "generator.h"
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Binary modulo 2 arithmetic 8 bit multiply operation.
 * @param op1 First operand of binary modulo 2 multiply.
 * @param op2 Second operand of binary modulo 2 multiply.
 * @return Result of binary modulo 2 multiply operation.
 */
uint16_t mul2_8(const uint8_t op1, const uint8_t op2);

/**
 * @brief Binary modulo 2 arithmetic 16 bit multiply operation.
 * @param op1 First operand of binary modulo 2 multiply.
 * @param op2 Second operand of binary modulo 2 multiply.
 * @return Result of binary modulo 2 multiply operation.
 */
uint32_t mul2_16(const uint16_t op1, const uint16_t op2);

/**
 * @brief Binary modulo 2 arithmetic 32 bit multiply operation.
 * @param op1 First operand of binary modulo 2 multiply.
 * @param op2 Second operand of binary modulo 2 multiply.
 * @return Result of binary modulo 2 multiply operation.
 */
uint64_t mul2_32(const uint32_t op1, const uint32_t op2);

#ifdef __cplusplus
}
#endif

#endif // MOD2_ARITHMETIC_MUL2_H_

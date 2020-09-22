/**
 * @author Mirco De Marchi
 * @date 16/09/2020
 * @brief Header file of CRC utility functions.
 */

#ifndef CRC_CRC_UTILS_H_
#define CRC_CRC_UTILS_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

extern const uint8_t reverse8_lu[];
//------------------------------------------------------------------------------

/**
 * @brief Reverse the bit sequence of a byte.
 * @param b Byte input.
 * @return Reversed byte.
 */
uint8_t reverse8(uint8_t b);

/**
 * @brief Reverse the bit sequence of a short.
 * @param b Short input.
 * @return Reversed short.
 */
uint16_t reverse16(uint16_t b);

/**
 * @brief Reverse the bit sequence of a int.
 * @param b Int input.
 * @return Reversed int.
 */
uint32_t reverse32(uint32_t b);

/**
 * @brief Initialize an array with the values of the reverse lookup table of a
 * byte.
 * @param lu Buffer to initialize.
 * @return The same input buffer initialized with the reversed values.
 */
uint8_t *generate_reverse8_lu(uint8_t *lu);

/**
 * @brief Print the reverse lookup table of a byte.
 * @param lu Reverse lookup table already initialized.
 */
void print_reverse8_lu(uint8_t *lu);

/**
 * @brief Copy the reverse version of a buffer.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param length Number of byte to copy from source buffer to destination.
 * @return Pointer of the destination buffer with the reverse array.
 */
uint8_t *memcpy_rev(uint8_t *dst, const uint8_t *src, size_t length);

#ifdef __cplusplus
}
#endif

#endif  // CRC_CRC_UTILS_H_